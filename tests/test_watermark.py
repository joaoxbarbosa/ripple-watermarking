"""Statistical tests against a toy language model (no torch needed)."""

import hashlib

import numpy as np
import pytest
from scipy import stats

from ripple_fingerprint import Watermarker

V = 2000
KEY = b"test-key-0123456789abcdef"


def toy_lm(history, alpha=0.05):
    """Deterministic next-token distribution conditioned on the last 2 tokens.

    Sparse Dirichlet draws give the uneven, fairly low-entropy distributions
    real LMs produce (a few bits per token).
    """
    h = hashlib.sha256(np.asarray(history[-2:], dtype="<i8").tobytes()).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
    return rng.dirichlet(np.full(V, alpha))


def generate(wm, prompt, n, watermark=True, seed=None):
    hist = list(prompt)
    state = wm.new_state()
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        p = toy_lm(hist)
        if watermark:
            tok = wm.sample(p, hist, state)
        else:
            tok = int(rng.choice(V, p=p))
        hist.append(tok)
        out.append(tok)
    return out


@pytest.fixture
def wm():
    return Watermarker(KEY)


def test_sampling_is_unbiased():
    """Across fresh contexts, watermarked choices follow p exactly."""
    wm = Watermarker(KEY, context_width=1, entropy_threshold=0.0)
    p = np.array([0.5, 0.25, 0.15, 0.07, 0.03])
    counts = np.zeros(len(p))
    n = 40000
    for i in range(n):
        state = wm.new_state()
        state.active = True
        counts[wm.sample(p, [10_000 + i], state)] += 1
    _, pval = stats.chisquare(counts, n * p)
    assert pval > 1e-3, (counts / n, pval)


def test_watermarked_text_is_detected(wm):
    for s in range(5):
        toks = generate(wm, [1, 2, 3, 4, s], 200)
        r = wm.detect(toks)
        assert r.is_watermarked, r


def test_null_p_values_are_uniform(wm):
    """False-positive control: unwatermarked text gives uniform p-values."""
    pvals = [wm.detect(generate(wm, [7, s], 150, watermark=False, seed=s)).p_value for s in range(200)]
    assert stats.kstest(pvals, "uniform").pvalue > 1e-3
    assert min(pvals) > 1e-5


def test_wrong_key_does_not_detect(wm):
    toks = generate(wm, [1, 2, 3], 200)
    other = Watermarker(b"another-secret-key-xyz123")
    assert not other.detect(toks).is_watermarked


def test_same_prompt_gives_diverse_outputs(wm):
    outs = {tuple(generate(wm, [5, 5, 5, 5], 40)) for _ in range(20)}
    assert len(outs) == 20


def _edit(toks, rng, sub=0.0, ins=0.0, dele=0.0):
    out = []
    for t in toks:
        r = rng.random()
        if r < dele:
            continue
        if r < dele + sub:
            out.append(int(rng.integers(V)))
        else:
            out.append(t)
        if rng.random() < ins:
            out.append(int(rng.integers(V)))
    return out


@pytest.mark.parametrize(
    "edits",
    [
        dict(sub=0.2),
        dict(ins=0.15),
        dict(dele=0.15),
        dict(sub=0.1, ins=0.05, dele=0.05),
    ],
)
def test_detection_survives_edits(wm, edits):
    rng = np.random.default_rng(0)
    hits = 0
    for s in range(10):
        toks = generate(wm, [9, 9, s], 400)
        hits += wm.detect(_edit(toks, rng, **edits)).is_watermarked
    assert hits >= 9, hits


def test_detected_when_spliced_into_human_text(wm):
    human_a = generate(wm, [3, 1], 200, watermark=False, seed=1)
    human_b = generate(wm, [4, 1], 200, watermark=False, seed=2)
    machine = generate(wm, [5, 9, 2, 6], 250)
    assert wm.detect(human_a + machine + human_b).is_watermarked
    assert not wm.detect(human_a + human_b).is_watermarked
