"""Demo web app: generate watermarked text, edit it, and check it.

    .venv/bin/python -m demo.server            # http://localhost:8000

Environment:
    RIPPLE_FP_MODELS  comma-separated HF model ids or local paths offered in the
                      model selector (the first is the default). Defaults to
                      Qwen3-4B, two smaller Qwen chat models and gpt2. Add Ripple-65B on a
                      machine that can run it.
    RIPPLE_FP_KEY     master watermark secret. Each model's key is derived from
                      it and the model id, so checking text also tells you
                      which model wrote it. If unset, a key is created once and
                      saved in demo/.demo_key (gitignored).
    PORT              default 8000

Models load on first use and stay loaded. Detection only needs the
tokenizers, so checking text (against every model) never loads weights.
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
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

from ripple_fingerprint import ModelRegistry
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
keys = ModelRegistry(_load_key())  # one derived watermark key per model
for _name in MODELS:  # tokenizers are small; loading them all up front also fails fast on a bad id
    keys.register(_name, reg.tokenizer(_name))
CONTEXT_WIDTH = keys.watermarker(MODELS[0]).context_width

app = FastAPI(title="Ripple Fingerprint demo")
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")


class GenerateReq(BaseModel):
    model: str = MODELS[0]
    prompt: str = Field(min_length=1, max_length=4000)
    watermark: bool = True
    temperature: float = Field(0.9, gt=0, le=2)
    top_p: float = Field(0.95, gt=0, le=1)
    max_new_tokens: int = Field(200, ge=1, le=1024)


class DetectReq(BaseModel):
    text: str = Field(max_length=100_000)
    focus: str | None = None  # model whose token heatmap to return; default: the best match


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
        "context_width": CONTEXT_WIDTH,
        "fpr": keys.fpr,
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
                model, tok, keys.watermarker(req.model), temperature=req.temperature, top_p=req.top_p,
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


def _pieces(text: str, ms) -> list[str]:
    """The text cut at token boundaries; joined back together they give ``text`` exactly."""
    if not ms.offsets:
        return [reg.tokenizer(ms.name).decode([i]) for i in ms.token_ids]
    out, cur = [], 0
    for _, end in ms.offsets:
        end = max(end, cur)
        out.append(text[cur:end])
        cur = end
    if out:
        out[-1] += text[cur:]
    return out


@app.post("/api/detect")
def detect(req: DetectReq):
    if not req.text.strip():
        raise HTTPException(400, "empty text")
    res = keys.attribute(req.text)
    focus = req.focus if req.focus in keys else (res.best or res.models[0].name)
    fm = res[focus]
    span_p = {}
    for sp in res.spans:
        span_p[sp.model] = min(sp.p_value, span_p.get(sp.model, 1.0))
    return {
        "best": res.best,
        "matches": res.matches,
        "fpr": res.fpr,
        "models": [
            {
                "name": m.name, "is_match": m.is_match, "p_value": m.result.p_value,
                "adjusted_p": m.adjusted_p, "evidence_p": min(m.adjusted_p, span_p.get(m.name, 1.0)), "z_score": float(m.result.z_score),
                "num_scored": m.result.num_scored, "num_tokens": m.result.num_tokens,
            }
            for m in res.models
        ],
        "spans": [
            {"model": sp.model, "start": sp.char_start, "end": sp.char_end, "p_value": sp.p_value, "num_scored": sp.num_scored}
            for sp in res.spans
        ],
        "text_length": len(req.text),
        "focus": focus,
        "tokens": [{"t": p, "s": s} for p, s in zip(_pieces(req.text, fm), fm.token_scores)],
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8000)))
