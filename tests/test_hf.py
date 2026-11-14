"""End-to-end tests with a real HF model. Skipped if torch/transformers are missing.

The model defaults to gpt2. Point RIPPLE_FP_TEST_MODEL at a local Ripple checkpoint
to run the same checks against Ripple.
"""

import os

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from ripple_fingerprint import Watermarker  # noqa: E402
from ripple_fingerprint.hf import detect_text, generate, watermark_generate  # noqa: E402

MODEL = os.environ.get("RIPPLE_FP_TEST_MODEL", "gpt2")
PROMPTS = [
    "The history of the printing press",
    "Here is a short story about a lighthouse keeper:",
    "The main causes of inflation are",
]


@pytest.fixture(scope="module")
def lm():
    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL).eval()
    torch.manual_seed(0)
    return model, tok


@pytest.fixture(scope="module")
def wm():
    return Watermarker(b"hf-test-key-0123456789abcdef")


@pytest.fixture(scope="module")
def samples(lm, wm):
    model, tok = lm
    return [generate(model, tok, p, wm, max_new_tokens=200, temperature=1.0, top_p=0.95) for p in PROMPTS]


def test_watermarked_generations_detected(lm, wm, samples):
    _, tok = lm
    for text in samples:
        r = detect_text(text, tok, wm)
        assert r.is_watermarked, (r, text[:200])


def test_plain_generations_not_detected(lm, wm):
    model, tok = lm
    for p in PROMPTS:
        enc = tok(p, return_tensors="pt")
        out = model.generate(**enc, do_sample=True, top_p=0.95, max_new_tokens=200, pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
        assert not detect_text(text, tok, wm).is_watermarked


def test_word_edits_survive(lm, wm, samples):
    _, tok = lm
    for text in samples:
        words = text.split(" ")
        edited = " ".join("indeed" if i % 8 == 3 else w for i, w in enumerate(words))  # 1 in 8 words replaced
        assert detect_text(edited, tok, wm).p_value < 1e-3


def test_batched_generate(lm, wm):
    model, tok = lm
    tok.padding_side = "left"
    tok.pad_token = tok.eos_token
    enc = tok(PROMPTS, return_tensors="pt", padding=True)
    out = watermark_generate(model, tok, wm, max_new_tokens=200, temperature=1.0, **enc)
    for row in out[:, enc.input_ids.shape[1]:]:
        text = tok.decode(row, skip_special_tokens=True)
        assert detect_text(text, tok, wm).is_watermarked
