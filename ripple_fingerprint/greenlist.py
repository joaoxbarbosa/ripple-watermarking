"""Green-list watermark (Kirchenbauer et al. 2023), for training into model weights.

Unlike the distortion-free scheme in ``core``, this one changes the model's
distribution: for each previous token the key marks a fraction ``gamma`` of the
vocabulary "green", and the model puts ``exp(delta)`` times more weight on green
tokens. Because it is a fixed change to the next-token distribution, a model can
learn it (watermark distillation, Gu et al. 2024) and then carry it with any
sampler, including one that the model's user controls.

Green membership is ``fmix32(s[prev] ^ r[tok]) < gamma * 2**32``, with ``r`` and
``s`` keyed 32-bit tables. That is cheap enough to compute for the whole
vocabulary at every training position.

Detection counts green tokens. Under the null (text written without this key)
each scored token is green with probability ``gamma``, so the count is
Binomial(n, gamma). Repeated (previous, token) pairs are scored once, since they
would otherwise reuse the same coin.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.stats import binom

_DOMAIN = b"ripple-fingerprint/greenlist/v1"
M1, M2 = np.uint32(0x85EBCA6B), np.uint32(0xC2B2AE35)


def _table(key: bytes, label: bytes, n: int) -> np.ndarray:
    seed = hmac.new(key, _DOMAIN + label, hashlib.sha256).digest()
    return np.frombuffer(hashlib.shake_256(seed).digest(4 * n), dtype="<u4").copy()


def fmix32(h: np.ndarray) -> np.ndarray:
    """MurmurHash3 finalizer: a bijective 32-bit mixer (uint32 arithmetic wraps)."""
    with np.errstate(over="ignore"):
        h = h ^ (h >> np.uint32(16))
        h = h * M1
        h = h ^ (h >> np.uint32(13))
        h = h * M2
        return h ^ (h >> np.uint32(16))


@dataclass
class GreenResult:
    z_score: float
    p_value: float
    green: int
    num_scored: int
    num_tokens: int


class GreenList:
    def __init__(self, key: bytes | str, vocab_size: int, gamma: float = 0.25, delta: float = 2.0):
        if isinstance(key, str):
            key = key.encode()
        if len(key) < 16:
            raise ValueError("use a secret key of at least 16 bytes")
        self.key, self.vocab_size, self.gamma, self.delta = key, vocab_size, gamma, delta
        self.r = _table(key, b"token", vocab_size)  # one per candidate token
        self.s = _table(key, b"prev", vocab_size)   # one per previous token
        self.threshold = np.uint32(min(int(gamma * 2**32), 2**32 - 1))

    def mask(self, prev_ids: Sequence[int] | np.ndarray) -> np.ndarray:
        """Boolean ``[len(prev_ids), vocab_size]``: which next tokens are green."""
        prev = np.asarray(prev_ids, dtype=np.int64).reshape(-1)
        return fmix32(self.s[prev][:, None] ^ self.r[None, :]) < self.threshold

    def is_green(self, prev: int, tok: int) -> bool:
        return bool(fmix32(np.uint32(self.s[prev] ^ self.r[tok])) < self.threshold)

    def bias(self, logits: np.ndarray, prev: int) -> np.ndarray:
        """Sampler-side watermark: add ``delta`` to green logits (the teacher distribution)."""
        return logits + self.delta * self.mask([prev])[0][: logits.shape[-1]]

    def detect(self, token_ids: Sequence[int]) -> GreenResult:
        ids = [int(t) for t in token_ids]
        seen, green = set(), 0
        for prev, tok in zip(ids, ids[1:]):
            if (prev, tok) in seen or not (0 <= prev < self.vocab_size and 0 <= tok < self.vocab_size):
                continue
            seen.add((prev, tok))
            green += self.is_green(prev, tok)
        n, g = len(seen), self.gamma
        z = (green - g * n) / np.sqrt(max(n, 1) * g * (1 - g))
        p = float(binom.sf(green - 1, n, g)) if n else 1.0  # P(count >= green)
        return GreenResult(float(z), p, green, n, len(ids))
