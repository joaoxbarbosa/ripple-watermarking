"""Watermarker.sample must pick exactly the token of the plain full-vocabulary race,
argmin over v of -ln(u_v) / p_v, on sparse distributions like llama.cpp's top-k output."""

import numpy as np

from ripple_fingerprint import Watermarker, keyed_uniforms


def test_sample_matches_reference_race():
    wm = Watermarker(b"k" * 32)
    rng = np.random.default_rng(0)
    V = 100352  # Ripple's vocabulary
    for _ in range(300):
        p = np.zeros(V)
        k = int(rng.integers(1, 1500))
        idx = rng.choice(V, size=k, replace=False)
        p[idx] = rng.dirichlet(np.full(k, rng.choice([0.05, 0.5, 5.0])))
        ctx = [int(x) for x in rng.integers(0, V, size=3)]
        u = keyed_uniforms(wm.key, ctx, V)
        with np.errstate(divide="ignore"):
            ref = int(np.argmin(-np.log(u) / p))
        st = wm.new_state()
        st.active = True
        assert wm.sample(p, ctx, st) == ref
        assert st.last_mode == "keyed"
