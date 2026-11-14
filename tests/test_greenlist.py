import numpy as np

from ripple_fingerprint.greenlist import GreenList

KEY = "test-key-for-the-green-list-0001"


def test_green_fraction_is_gamma():
    gl = GreenList(KEY, 50_000, gamma=0.25)
    m = gl.mask(np.arange(200))
    assert abs(m.mean() - 0.25) < 0.002
    # every previous token gets its own green set
    assert (m[0] != m[1]).mean() > 0.3


def test_mask_matches_scalar():
    gl = GreenList(KEY, 5_000)
    m = gl.mask([7, 4999])
    for tok in range(0, 5_000, 97):
        assert m[0, tok] == gl.is_green(7, tok)
        assert m[1, tok] == gl.is_green(4999, tok)


def test_null_text_is_not_flagged():
    gl = GreenList(KEY, 50_000)
    rng = np.random.default_rng(0)
    zs = [gl.detect(rng.integers(0, 50_000, 300)).z_score for _ in range(200)]
    assert abs(np.mean(zs)) < 0.2 and 0.8 < np.std(zs) < 1.2


def test_biased_sampling_is_detected_and_key_specific():
    V = 2_000
    gl, other = GreenList(KEY, V), GreenList("another-key-for-the-green-list-02", V)
    rng = np.random.default_rng(1)
    logits = rng.normal(0, 1, V)
    ids = [0]
    for _ in range(200):
        z = gl.bias(logits, ids[-1])
        p = np.exp(z - z.max()); p /= p.sum()
        ids.append(int(rng.choice(V, p=p)))
    assert gl.detect(ids).p_value < 1e-5
    assert other.detect(ids).p_value > 1e-3
