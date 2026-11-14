"""Demo web app: generate watermarked text, edit it, and check it.

    .venv/bin/python -m demo.server            # http://localhost:8000

Environment:
    RIPPLE_FP_MODELS  comma-separated HF model ids or local paths offered in the
                      model selector (the first is the default). Defaults to
                      Qwen3-4B, two smaller Qwen chat models and gpt2. Add Ripple-65B on a
                      machine that can run it.
    RIPPLE_FP_KEY     watermark secret. If unset, a key is created once and
                      saved in demo/.demo_key (gitignored).
    PORT              default 8000

Models load on first use and stay loaded. Detection only needs the
tokenizer, so checking text never loads a model's weights.
"""

from __future__ import annotations

import os
import secrets
import threading
import time
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

from ripple_fingerprint import Watermarker
from ripple_fingerprint.hf import watermark_generate

HERE = Path(__file__).parent
DEFAULT_MODELS = "Qwen/Qwen3-4B,Qwen/Qwen3-0.6B,Qwen/Qwen2.5-0.5B-Instruct,gpt2"
MODELS = [m.strip() for m in os.environ.get("RIPPLE_FP_MODELS", DEFAULT_MODELS).split(",") if m.strip()]
# Backwards compatible: RIPPLE_FP_MODEL puts one model first.
if os.environ.get("RIPPLE_FP_MODEL"):
    MODELS = [os.environ["RIPPLE_FP_MODEL"]] + [m for m in MODELS if m != os.environ["RIPPLE_FP_MODEL"]]


def _load_key() -> str:
    if os.environ.get("RIPPLE_FP_KEY"):
        return os.environ["RIPPLE_FP_KEY"]
    path = HERE / ".demo_key"
    if not path.exists():
        path.write_text(secrets.token_hex(32))
        path.chmod(0o600)
    return path.read_text().strip()


class Registry:
    """Lazily loads tokenizers and models, one at a time."""

    def __init__(self, names: list[str]):
        self.names = names
        self.tokenizers: dict = {}
        self.models: dict = {}
        self.lock = threading.Lock()  # guards loading and generation

    def _check(self, name: str):
        if name not in self.names:
            raise HTTPException(400, f"unknown model {name!r}")

    def tokenizer(self, name: str):
        self._check(name)
        if name not in self.tokenizers:
            with self.lock:
                if name not in self.tokenizers:
                    self.tokenizers[name] = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        return self.tokenizers[name]

    def model(self, name: str):
        """Call with ``self.lock`` held."""
        self._check(name)
        if name not in self.models:
            print(f"loading {name} ...", flush=True)
            if torch.cuda.is_available():  # multi-GPU needs `accelerate`
                m = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True, dtype="auto", device_map="auto")
            elif torch.backends.mps.is_available():  # Apple GPU: bf16 halves memory (4B ≈ 8 GB)
                m = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True, dtype=torch.bfloat16).to("mps")
            else:  # bf16 is slow on CPU
                m = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True, dtype=torch.float32)
            self.models[name] = m.eval()
        return self.models[name]


reg = Registry(MODELS)
wm = Watermarker(_load_key())
reg.tokenizer(MODELS[0])  # fail fast if the default model id is wrong

app = FastAPI(title="Ripple Fingerprint demo")


class GenerateReq(BaseModel):
    model: str = MODELS[0]
    prompt: str = Field(min_length=1, max_length=4000)
    watermark: bool = True
    temperature: float = Field(0.9, gt=0, le=2)
    top_p: float = Field(0.95, gt=0, le=1)
    max_new_tokens: int = Field(200, ge=1, le=1024)


class DetectReq(BaseModel):
    model: str = MODELS[0]
    text: str = Field(max_length=100_000)


def _encode_prompt(tok, model, prompt: str) -> dict:
    if getattr(tok, "chat_template", None):
        enc = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, return_tensors="pt", return_dict=True,
            enable_thinking=False,  # Qwen3: answer directly, no <think> block; other templates ignore it
        )
        return {k: v.to(model.device) for k, v in enc.items()}
    return dict(tok(prompt, return_tensors="pt").to(model.device))


@app.get("/")
def index():
    return FileResponse(HERE / "index.html")


@app.get("/api/info")
def info():
    return {
        "models": [{"name": n, "loaded": n in reg.models} for n in reg.names],
        "default": reg.names[0],
        "context_width": wm.context_width,
        "fpr": wm.fpr,
    }


@app.post("/api/generate")
def generate(req: GenerateReq):
    tok = reg.tokenizer(req.model)
    with reg.lock, torch.no_grad():
        t_load = time.time()
        model = reg.model(req.model)
        t_load = time.time() - t_load
        enc = _encode_prompt(tok, model, req.prompt)
        n_prompt = enc["input_ids"].shape[1]
        t0 = time.time()
        if req.watermark:
            out = watermark_generate(
                model, tok, wm, temperature=req.temperature, top_p=req.top_p,
                max_new_tokens=req.max_new_tokens, **enc,
            )
        else:
            pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
            out = model.generate(
                **enc, do_sample=True, temperature=req.temperature, top_p=req.top_p, top_k=0,
                max_new_tokens=req.max_new_tokens, pad_token_id=pad_id,
            )
    text = tok.decode(out[0, n_prompt:], skip_special_tokens=True).strip()
    return {
        "text": text, "model": req.model, "watermark": req.watermark,
        "seconds": round(time.time() - t0, 2), "load_seconds": round(t_load, 1),
    }


@app.post("/api/detect")
def detect(req: DetectReq):
    tok = reg.tokenizer(req.model)
    ids = tok.encode(req.text, add_special_tokens=False)
    if not ids:
        raise HTTPException(400, "empty text")
    r = wm.detect(ids)
    scores = wm.token_scores(ids)
    pieces = [tok.decode([i]) for i in ids]
    return {
        "model": req.model,
        "is_watermarked": bool(r.is_watermarked),
        "p_value": r.p_value,
        "z_score": float(r.z_score),
        "num_scored": r.num_scored,
        "num_tokens": r.num_tokens,
        "tokens": [{"t": p, "s": s} for p, s in zip(pieces, scores)],
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8000)))
