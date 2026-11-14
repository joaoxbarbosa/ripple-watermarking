"""llama.cpp backend against a fake llama-server (no GPU needed), plus an optional
live test against a real server when RIPPLE_FP_LLAMACPP_TEST=http://host:port is set."""

import os

import numpy as np
import pytest

from ripple_fingerprint import Watermarker
from ripple_fingerprint.llamacpp import LlamaServer, generate_ids, warp

V = 500
EOS = V - 1


class FakeServer:
    """Duck-types LlamaServer: a fixed random bigram LM over V tokens, ~4-5 bits/token."""

    def __init__(self, seed=0, n_probs_cap=V):
        rng = np.random.default_rng(seed)
        self.logits = rng.normal(0, 1.5, size=(V, V))
        self.logits[:, EOS] = -30.0  # never stop early
        self.calls = 0

    def eos_ids(self):
        return {EOS}

    def n_vocab(self):
        return V

    def next_token_logprobs(self, ids, n_probs=1000):
        self.calls += 1
        z = self.logits[ids[-1]]
        lp = z - np.logaddexp.reduce(z)
        top = np.argsort(-lp)[:n_probs]
        return top.astype(np.int64), lp[top]


def test_warp_is_a_distribution_over_candidate_ids():
    ids = np.array([7, 3, 42])
    lp = np.log(np.array([0.5, 0.3, 0.2]))
    p = warp(ids, lp, temperature=1.0)
    assert p.shape == (43,) and abs(p.sum() - 1) < 1e-12
    assert np.allclose(p[[7, 3, 42]], [0.5, 0.3, 0.2]) and p[[0, 1, 2, 41]].sum() == 0
    p = warp(ids, lp, temperature=1.0, top_p=0.6)  # keeps the top two
    assert p[42] == 0 and abs(p[7] + p[3] - 1) < 1e-12


def test_watermarked_generation_is_detected_and_plain_is_not():
    wm = Watermarker(b"llamacpp-test-key-0123456789")
    srv = FakeServer()
    prompt = [1, 2, 3, 4]
    marked = generate_ids(srv, prompt, wm, max_new_tokens=200, n_probs=200)
    plain = generate_ids(srv, prompt, wm, max_new_tokens=200, n_probs=200, watermark=False)
    assert len(marked) == len(plain) == 200 and srv.calls == 400
    assert wm.detect(marked).is_watermarked
    assert not wm.detect(plain).is_watermarked
    # Another key sees ordinary text
    assert not Watermarker(b"some-other-key-0123456789").detect(marked).is_watermarked


def test_on_step_reports_modes_and_draws():
    wm = Watermarker(b"llamacpp-test-key-0123456789")
    modes = []

    def on_step(p, history, tok, st):
        modes.append(st.last_mode)
        if st.last_mode == "keyed":
            assert st.last_draws is not None and len(st.last_draws) >= tok + 1

    generate_ids(FakeServer(), [1, 2, 3], wm, max_new_tokens=60, on_step=on_step)
    assert modes[0] == "warmup" and "keyed" in modes


LIVE = os.environ.get("RIPPLE_FP_LLAMACPP_TEST")


@pytest.mark.skipif(not LIVE, reason="set RIPPLE_FP_LLAMACPP_TEST=http://127.0.0.1:8080 to run against llama-server")
def test_live_server_story_is_detected():
    srv = LlamaServer(LIVE)
    wm = Watermarker(b"llamacpp-live-test-key-0123456789")
    ids = srv.chat_prompt_ids("Write a 200-word story about a fox who learns to sail.")
    out = generate_ids(srv, ids, wm, max_new_tokens=250)
    text = srv.detokenize(out)
    assert wm.detect(srv.tokenize(text)).p_value < 1e-5
