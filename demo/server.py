"""Demo web app: generate watermarked text, edit it, and check it.

    .venv/bin/python -m demo.server            # http://localhost:8000

Environment:
    RIPPLE_FP_MODEL   HF model id or local path (default: gpt2).
                      Point it at Ripple-65B on a machine that can run it.
    RIPPLE_FP_KEY     watermark secret. If unset, a key is created once and
                      saved in demo/.demo_key (gitignored).
    PORT              default 8000
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
MODEL_NAME = os.environ.get("RIPPLE_FP_MODEL", "gpt2")


def _load_key() -> str:
    if os.environ.get("RIPPLE_FP_KEY"):
        return os.environ["RIPPLE_FP_KEY"]
    path = HERE / ".demo_key"
    if not path.exists():
        path.write_text(secrets.token_hex(32))
        path.chmod(0o600)
    return path.read_text().strip()


print(f"loading {MODEL_NAME} ...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, trust_remote_code=True, dtype="auto",
    device_map="auto" if torch.cuda.is_available() else None,  # multi-GPU needs `accelerate`
).eval()
wm = Watermarker(_load_key())
lock = threading.Lock()  # one generation at a time
pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

app = FastAPI(title="Ripple Fingerprint demo")


class GenerateReq(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    watermark: bool = True
    temperature: float = Field(0.9, gt=0, le=2)
    top_p: float = Field(0.95, gt=0, le=1)
    max_new_tokens: int = Field(200, ge=1, le=1024)


class DetectReq(BaseModel):
    text: str = Field(max_length=100_000)


def _encode_prompt(prompt: str) -> dict:
    if getattr(tok, "chat_template", None):
        ids = tok.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        return {k: v.to(model.device) for k, v in ids.items()}
    return tok(prompt, return_tensors="pt").to(model.device)


@app.get("/")
def index():
    return FileResponse(HERE / "index.html")


@app.get("/api/info")
def info():
    return {
        "model": MODEL_NAME,
        "context_width": wm.context_width,
        "fpr": wm.fpr,
        "chat": bool(getattr(tok, "chat_template", None)),
    }


@app.post("/api/generate")
def generate(req: GenerateReq):
    enc = _encode_prompt(req.prompt)
    n_prompt = enc["input_ids"].shape[1]
    t0 = time.time()
    with lock, torch.no_grad():
        if req.watermark:
            out = watermark_generate(
                model, tok, wm, temperature=req.temperature, top_p=req.top_p,
                max_new_tokens=req.max_new_tokens, **enc,
            )
        else:
            out = model.generate(
                **enc, do_sample=True, temperature=req.temperature, top_p=req.top_p, top_k=0,
                max_new_tokens=req.max_new_tokens, pad_token_id=pad_id,
            )
    text = tok.decode(out[0, n_prompt:], skip_special_tokens=True)
    return {"text": text, "watermark": req.watermark, "seconds": round(time.time() - t0, 2)}


@app.post("/api/detect")
def detect(req: DetectReq):
    ids = tok.encode(req.text, add_special_tokens=False)
    if not ids:
        raise HTTPException(400, "empty text")
    r = wm.detect(ids)
    scores = wm.token_scores(ids)
    pieces = [tok.decode([i]) for i in ids]
    return {
        "is_watermarked": bool(r.is_watermarked),
        "p_value": r.p_value,
        "z_score": float(r.z_score),
        "num_scored": r.num_scored,
        "num_tokens": r.num_tokens,
        "tokens": [{"t": p, "s": s} for p, s in zip(pieces, scores)],
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8000)))
