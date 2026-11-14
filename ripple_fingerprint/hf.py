"""Hugging Face ``transformers`` integration (needs torch + transformers).

Two ways in:

* ``watermark_generate(model, tokenizer, wm, **inputs, ...)``: a thin wrapper
  around ``model.generate`` that wires everything up. It handles batches.
* ``WatermarkLogitsProcessor``: the same thing as a ``LogitsProcessor``,
  if you build your own ``generate`` call. Pass it in ``logits_processor`` and
  set ``do_sample=False``. The processor does the (watermarked) sampling
  itself and hands ``generate`` a one-hot distribution.

Then ``detect_text(text, tokenizer, wm)`` checks any text.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import LogitsProcessor, LogitsProcessorList

from .core import DetectionResult, Watermarker


def _to_probs(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    """Row-wise temperature / top-k / top-p, the same as HF's sampling warpers."""
    logits = logits.float() / temperature
    if top_k and top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.shape[-1]), dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    if top_p < 1.0:
        sorted_p, idx = torch.sort(probs, descending=True, dim=-1)
        cum = torch.cumsum(sorted_p, dim=-1)
        sorted_p = sorted_p.masked_fill(cum - sorted_p > top_p, 0.0)
        probs = torch.zeros_like(probs).scatter_(-1, idx, sorted_p)
    return probs


class WatermarkLogitsProcessor(LogitsProcessor):
    """Watermarked sampling inside ``model.generate``.

    Use one instance per ``generate`` call (it holds per-response state).
    It needs ``do_sample=False`` and ``num_beams=1``. Put temperature/top-k/top-p
    here rather than in ``generate``. Other processors in the list (repetition
    penalty, bad-words, ...) still apply, because this one runs last.
    """

    def __init__(self, watermarker: Watermarker, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0):
        if temperature <= 0:
            raise ValueError("greedy decoding has no randomness to watermark; use temperature > 0")
        self.wm = watermarker
        self.temperature, self.top_k, self.top_p = temperature, top_k, top_p
        self.states: list | None = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.states is None:
            self.states = [self.wm.new_state() for _ in range(input_ids.shape[0])]
        if len(self.states) != input_ids.shape[0]:
            raise ValueError("batch size changed mid-generation (beam search is not supported)")
        probs = _to_probs(scores, self.temperature, self.top_k, self.top_p).double().cpu().numpy()
        hist = input_ids.tolist()
        out = torch.full_like(scores, float("-inf"))
        for b, state in enumerate(self.states):
            tok = self.wm.sample(probs[b], hist[b], state)
            out[b, tok] = 0.0
        return out


def watermark_generate(
    model,
    tokenizer,
    watermarker: Watermarker,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    logits_processor: LogitsProcessorList | None = None,
    **generate_kwargs,
) -> torch.LongTensor:
    """``model.generate`` with the watermark applied. Returns generated ids (prompt included)."""
    if generate_kwargs.get("num_beams", 1) != 1:
        raise ValueError("beam search is not supported with watermarking")
    procs = LogitsProcessorList(logits_processor or [])
    procs.append(WatermarkLogitsProcessor(watermarker, temperature, top_k, top_p))
    generate_kwargs.setdefault("pad_token_id", tokenizer.pad_token_id or tokenizer.eos_token_id)
    return model.generate(
        logits_processor=procs,
        do_sample=False,
        temperature=None,
        top_k=None,
        top_p=None,
        **generate_kwargs,
    )


def generate(model, tokenizer, prompt: str, watermarker: Watermarker, max_new_tokens: int = 256, **kw) -> str:
    """Convenience: one prompt in, watermarked completion text out."""
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    ids = watermark_generate(model, tokenizer, watermarker, max_new_tokens=max_new_tokens, **enc, **kw)
    return tokenizer.decode(ids[0, enc.input_ids.shape[1]:], skip_special_tokens=True)


def detect_text(text: str, tokenizer, watermarker: Watermarker) -> DetectionResult:
    """Re-tokenize ``text`` with the generating model's tokenizer and test it."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    return watermarker.detect(ids)
