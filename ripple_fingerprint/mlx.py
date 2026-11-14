"""Apple MLX backend (needs ``mlx-lm``). Runs MLX models such as LM Studio's.

    from mlx_lm import load
    model, tok = load("/path/to/Some-Model-MLX-8bit")
    text = generate(model, tok, "Write a haiku.", wm)

Detection only needs the tokenizer: use ``Watermarker.detect(tok.encode(text))``
or ``ripple_fingerprint.hf.detect_text`` with a Hugging Face tokenizer loaded
from the same folder.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

from .core import Watermarker


def warp(logits: np.ndarray, temperature: float, top_k: int = 0, top_p: float = 1.0) -> np.ndarray:
    """Temperature / top-k / top-p in numpy. Returns a float64 distribution."""
    z = logits.astype(np.float64) / temperature
    if top_k and 0 < top_k < z.size:
        kth = np.partition(z, -top_k)[-top_k]
        z = np.where(z < kth, -np.inf, z)
    z -= z.max()
    p = np.exp(z)
    p /= p.sum()
    if top_p < 1.0:
        order = np.argsort(-p)
        cum = np.cumsum(p[order])
        drop = order[cum - p[order] > top_p]
        p[drop] = 0.0
        p /= p.sum()
    return p


def generate_ids(
    model,
    prompt_ids: list[int],
    watermarker: Watermarker,
    eos_ids: set[int],
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    watermark: bool = True,
    on_step=None,
) -> list[int]:
    """Token-by-token generation. ``watermark=False`` gives plain sampling with the same settings.

    ``on_step(probs, history, token, state)`` runs after each choice, before the
    token is appended to ``history`` (used by the demo's visualizer).
    """
    if temperature <= 0:
        raise ValueError("greedy decoding has no randomness to watermark; use temperature > 0")
    gpu_softmax = not top_k and top_p >= 1.0  # plain temperature: softmax on the GPU

    def forward(ids):
        """Launch the next forward pass asynchronously; returns what to read back."""
        logits = model(ids, cache=cache)[0, -1].astype(mx.float32)
        y = mx.softmax(logits / temperature) if gpu_softmax else logits
        mx.async_eval(y)
        return y

    cache = make_prompt_cache(model)
    pending = forward(mx.array(prompt_ids)[None])
    history = list(prompt_ids)
    state = watermarker.new_state()
    draws = None
    out: list[int] = []
    for _ in range(max_new_tokens):
        y = np.array(pending)  # waits for the GPU
        if gpu_softmax:  # float32 from the GPU: renormalize in float64 so it sums to 1 exactly
            p = y.astype(np.float64)
            p /= p.sum()
        else:
            p = warp(y, temperature, top_k, top_p)
        if watermark:
            tok = watermarker.sample(p, history, state, draws=draws)
        else:
            tok = int(state.rng.choice(p.size, p=p))
            state.last_mode = "plain"
        if tok in eos_ids:
            if on_step is not None:
                on_step(p, history, tok, state)
            break
        # Start the GPU on the next token, then do this step's CPU work while it runs:
        # callbacks (streaming, tracing) and the next step's keyed hash.
        pending = forward(mx.array([[tok]]))
        if on_step is not None:
            on_step(p, history, tok, state)
        history.append(tok)
        out.append(tok)
        draws = watermarker.prepare(history, state, p.size) if watermark else None
    return out


def generate(model, tokenizer, prompt: str, watermarker: Watermarker, chat: bool = True, **kw) -> str:
    """Prompt in, (watermarked) completion text out. ``tokenizer`` comes from ``mlx_lm.load``."""
    if chat and getattr(tokenizer, "chat_template", None):
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, enable_thinking=False
        )
    else:
        ids = tokenizer.encode(prompt)
    eos = set(getattr(tokenizer, "eos_token_ids", None) or [tokenizer.eos_token_id])
    return tokenizer.decode(generate_ids(model, ids, watermarker, eos, **kw))
