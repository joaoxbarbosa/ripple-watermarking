"""Which-model attribution, against the toy LM (no torch needed)."""

import re

import numpy as np
import pytest
from scipy import stats

from ripple_fingerprint import ModelRegistry, Watermarker, derive_key
from test_watermark import _edit, generate

MASTER = b"master-secret-0123456789abcdef"
NAMES = ["ripple-65b", "ripple-8b", "ripple-8b-chat", "other-lab-model"]


@pytest.fixture
def reg():
    r = ModelRegistry(MASTER)
    for n in NAMES:
        r.register(n)
    return r


def test_derived_keys_are_stable_and_distinct():
    assert derive_key(MASTER, "a") == derive_key(MASTER, "a")
    assert derive_key(MASTER, "a") != derive_key(MASTER, "b")
    assert derive_key(MASTER, "a") != derive_key(b"another-master-key-xyz", "a")
    with pytest.raises(ValueError):
        derive_key(b"short", "a")


def test_key_id_keeps_key_across_renames():
    r = ModelRegistry(MASTER)
    a = r.register("/old/path/ripple", key_id="ripple-65b")
    b = r.register("ripple-65b")  # same label, different display name
    assert a.key == b.key
    with pytest.raises(ValueError):
        r.register("ripple-65b")


def test_each_model_is_identified(reg):
    for i, name in enumerate(NAMES):
        for s in range(3):
            toks = generate(reg.watermarker(name), [i, 7, s], 200)
            res = reg.attribute_ids(toks)
            assert res.best == name, (name, [(m.name, m.adjusted_p) for m in res.models])
            assert res.matches == [name]


def test_wrong_models_see_ordinary_text(reg):
    """Under the wrong key a model's p-values are uniform: no false attribution."""
    wrong_p = []
    for s in range(60):
        toks = generate(reg.watermarker("ripple-65b"), [3, s], 150)
        res = reg.attribute_ids(toks)
        wrong_p += [m.result.p_value for m in res.models if m.name != "ripple-65b"]
    assert stats.kstest(wrong_p, "uniform").pvalue > 1e-3
    assert min(wrong_p) * 2 * len(NAMES) > 1e-5


def test_unwatermarked_text_is_attributed_to_nobody(reg):
    for s in range(100):
        res = reg.attribute_ids(generate(reg.watermarker(NAMES[0]), [8, s], 150, watermark=False, seed=s))
        assert not res.is_attributed and res.best is None and not res.spans


def test_attribution_survives_edits(reg):
    rng = np.random.default_rng(1)
    for s in range(5):
        toks = generate(reg.watermarker("ripple-8b"), [4, 4, s], 400)
        assert reg.attribute_ids(_edit(toks, rng, sub=0.1, ins=0.05, dele=0.05)).best == "ripple-8b"


def test_mixed_document_is_split_by_source(reg):
    human1 = generate(reg.watermarker(NAMES[0]), [1, 1], 150, watermark=False, seed=1)
    a = generate(reg.watermarker("ripple-65b"), [2, 2, 2], 250)
    human2 = generate(reg.watermarker(NAMES[0]), [3, 3], 150, watermark=False, seed=2)
    b = generate(reg.watermarker("other-lab-model"), [4, 4, 4], 250)
    doc = human1 + a + human2 + b
    res = reg.attribute_ids(doc)
    assert set(res.matches) == {"ripple-65b", "other-lab-model"}

    by_model = {}
    for sp in res.spans:
        by_model.setdefault(sp.model, []).append(sp)
    assert set(by_model) == {"ripple-65b", "other-lab-model"}

    a_lo, a_hi = len(human1), len(human1) + len(a)
    b_lo = a_hi + len(human2)
    a_cover = sum(min(sp.token_end, a_hi) - max(sp.token_start, a_lo) for sp in by_model["ripple-65b"])
    b_cover = sum(min(sp.token_end, len(doc)) - max(sp.token_start, b_lo) for sp in by_model["other-lab-model"])
    assert a_cover > 0.8 * len(a) and b_cover > 0.8 * len(b)
    # spans stay (almost entirely) inside their own region
    for sp in by_model["ripple-65b"]:
        assert sp.token_start >= a_lo - 15 and sp.token_end <= a_hi + 15
    for sp in by_model["other-lab-model"]:
        assert sp.token_start >= b_lo - 15
    assert all(sp.p_value < 1e-5 for sp in res.spans)


def test_short_passage_in_long_document_matches_through_its_span(reg):
    """A whole-text test diluted by lots of human text can still match via a span."""
    human = [generate(reg.watermarker(NAMES[0]), [s, 2], 300, watermark=False, seed=10 + s) for s in range(4)]
    a = generate(reg.watermarker("ripple-8b"), [7, 7, 7], 60)
    res = reg.attribute_ids(human[0] + human[1] + a + human[2] + human[3])
    assert res.matches == ["ripple-8b"]
    assert [sp.model for sp in res.spans] == ["ripple-8b"]


class FakeFastTokenizer:
    """Space-separated integers, with HF-style offset mapping."""

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        ms = list(re.finditer(r"\d+", text))
        return {"input_ids": [int(m.group()) for m in ms], "offset_mapping": [m.span() for m in ms]}


def test_text_attribution_with_tokenizers_and_char_spans():
    r = ModelRegistry(MASTER)
    r.register("fast", FakeFastTokenizer())
    r.register("plain", encode=lambda s: [int(w) for w in s.split()])
    human = generate(r.watermarker("fast"), [5, 5], 150, watermark=False, seed=3)
    machine = generate(r.watermarker("fast"), [6, 6, 6], 250)
    prefix = " ".join(map(str, human)) + " "
    text = prefix + " ".join(map(str, machine))
    res = r.attribute(text)
    assert res.best == "fast" and res.matches == ["fast"]
    assert res["plain"].offsets is None
    (sp,) = res.spans
    assert sp.char_start is not None and abs(sp.char_start - len(prefix)) < 40
    assert sp.char_end == len(text)


def test_registry_watermarker_works_standalone(reg):
    """A model's own Watermarker still detects on its own, at the plain fpr."""
    wm = reg.watermarker("ripple-8b-chat")
    assert isinstance(wm, Watermarker)
    assert wm.detect(generate(wm, [9, 1], 200)).is_watermarked


def test_attribute_needs_tokenizers(reg):
    with pytest.raises(ValueError):
        reg.attribute("some text")
    with pytest.raises(KeyError):
        reg.attribute_ids([1, 2, 3], models=["nope"])
