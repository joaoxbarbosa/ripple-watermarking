"""Model loading and generation for the demo, with token streaming and sampler traces.

Kept separate from ``server.py`` so the server only wires up routes:

    engine = Engine(discover_models())
    engine.run(model, prompt, watermarker, ...)          # -> result dict
    add_stream_route(app, engine, watermarker_for)       # POST /api/stream

Backends:
* MLX models (e.g. LM Studio's ~/.lmstudio/models), run with Apple MLX.
* Hugging Face models, run with torch (CUDA, Apple GPU or CPU).
"""

from __future__ import annotations

import json
import os
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from scipy.special import gammaincc
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

from ripple_fingerprint import Watermarker, keyed_uniforms
from ripple_fingerprint.hf import watermark_generate

LMSTUDIO = Path.home() / ".lmstudio" / "models"
HF_DEFAULTS = "Qwen/Qwen3-4B,Qwen/Qwen3-0.6B,Qwen/Qwen2.5-0.5B-Instruct,gpt2"
PREFERRED_DEFAULT = "Qwen3-VL-4B-Instruct-MLX-8bit"
TRACE_K = 8  # candidate tokens recorded per step for the visualizer


# --------------------------------------------------------------------------- discovery
def is_mlx(path: Path) -> bool:
    try:
        return path.is_dir() and "quantization" in json.loads((path / "config.json").read_text())
    except (OSError, ValueError):
        return False


def discover_models() -> list[dict]:
    """LM Studio MLX models plus RIPPLE_FP_MODELS (or the HF defaults). The default goes first."""
    models = []
    if LMSTUDIO.is_dir():
        for d in sorted(LMSTUDIO.glob("*/*")):
            if is_mlx(d):
                models.append({"name": str(d), "label": f"LM Studio · {d.name}", "backend": "mlx"})
    for n in os.environ.get("RIPPLE_FP_MODELS", HF_DEFAULTS).split(","):
        n = n.strip()
        if n and not any(m["name"] == n for m in models):
            local = Path(n).expanduser()
            models.append({"name": n, "label": local.name if local.is_dir() else n,
                           "backend": "mlx" if is_mlx(local) else "hf"})
    want = os.environ.get("RIPPLE_FP_DEFAULT", PREFERRED_DEFAULT)
    models.sort(key=lambda m: not (m["name"] == want or Path(m["name"]).name == want))
    return models


# --------------------------------------------------------------------------- engine
class LiveDetector:
    """The detector's whole-text test, updated one generated token at a time.

    Scores the reply on its own (no prompt), exactly as checking the text later
    would, so the running p-value converges to what /api/detect reports."""

    def __init__(self, wm: Watermarker):
        self.wm, self.ids, self.seen = wm, [], set()
        self.total, self.n = 0.0, 0

    def push(self, tok: int) -> dict:
        h, ids = self.wm.context_width, self.ids
        ids.append(int(tok))
        s = None
        if len(ids) > h:
            ctx = tuple(ids[-h - 1:-1])
            if ctx not in self.seen:
                self.seen.add(ctx)
                u = keyed_uniforms(self.wm.key, ctx, tok + 1)[tok]
                s = float(-np.log1p(-u))
                self.total += s
                self.n += 1
        p = float(gammaincc(self.n, self.total)) if self.n else 1.0
        z = (self.total - self.n) / np.sqrt(max(self.n, 1))
        return {"s": s, "p": p, "n": self.n, "z": round(float(z), 3)}


class Engine:
    """Loads tokenizers and models lazily. All model work runs on one worker thread
    (MLX streams are per-thread, and it keeps generation one at a time)."""

    def __init__(self, models: list[dict]):
        self.info = {m["name"]: m for m in models}
        self.names = [m["name"] for m in models]
        self.tokenizers: dict = {}
        self.models: dict = {}
        self.worker = ThreadPoolExecutor(max_workers=1)

    def check(self, name: str):
        if name not in self.info:
            raise HTTPException(400, f"unknown model {name!r}")

    def tokenizer(self, name: str):
        """Hugging Face tokenizer (MLX folders ship the same files). Used for detection."""
        self.check(name)
        if name not in self.tokenizers:
            self.tokenizers[name] = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        return self.tokenizers[name]

    def loaded(self, name: str) -> bool:
        return name in self.models

    def _model(self, name: str):
        if name not in self.models:
            print(f"loading {name} ...", flush=True)
            if self.info[name]["backend"] == "mlx":
                from mlx_lm import load
                self.models[name] = load(name)  # (model, mlx tokenizer)
            elif torch.cuda.is_available():  # multi-GPU needs `accelerate`
                self.models[name] = AutoModelForCausalLM.from_pretrained(
                    name, trust_remote_code=True, dtype="auto", device_map="auto").eval()
            elif torch.backends.mps.is_available():  # Apple GPU: bf16 halves memory
                self.models[name] = AutoModelForCausalLM.from_pretrained(
                    name, trust_remote_code=True, dtype=torch.bfloat16).to("mps").eval()
            else:  # bf16 is slow on CPU
                self.models[name] = AutoModelForCausalLM.from_pretrained(
                    name, trust_remote_code=True, dtype=torch.float32).eval()
        return self.models[name]

    def run(self, name: str, prompt: str, wm: Watermarker, *, watermark=True, temperature=1.0,
            top_p=1.0, max_new_tokens=300, trace=False, emit: Callable[[dict], None] | None = None) -> dict:
        """Generate (blocking). ``emit`` gets {"type": "token", "text": delta, "step": trace|None}
        for every token as it is produced."""
        self.check(name)
        return self.worker.submit(self._run, name, prompt, wm, watermark, temperature, top_p,
                                  max_new_tokens, trace, emit).result()

    def _run(self, name, prompt, wm, watermark, temperature, top_p, max_new_tokens, trace, emit):
        t_load = time.time()
        loaded = self._model(name)
        t_load = time.time() - t_load
        tok = self.tokenizer(name)
        steps: list = []
        gen_ids: list[int] = []
        state = {"text": ""}
        live = LiveDetector(wm)
        backend = self.info[name]["backend"]
        eos: set[int] = set()

        def on_step(p, history, chosen, st):
            if chosen in eos:  # end of reply: not part of the text or the visualizer
                return
            step = _trace_step(tok, wm, p, history, chosen, st) if trace else None
            if step is not None:
                steps.append(step)
            if emit is not None:
                gen_ids.append(int(chosen))
                full = tok.decode(gen_ids, skip_special_tokens=True)
                delta, state["text"] = full[len(state["text"]):], full  # incremental decode (multi-byte safe)
                emit({"type": "token", "text": delta, "step": step, "live": live.push(int(chosen))})

        t0 = time.time()
        if backend == "mlx":
            from ripple_fingerprint.mlx import generate_ids
            model, mtok = loaded
            eos = set(getattr(mtok, "eos_token_ids", None) or [mtok.eos_token_id])
            if getattr(mtok, "chat_template", None):
                ids = mtok.apply_chat_template([{"role": "user", "content": prompt}],
                                               add_generation_prompt=True, enable_thinking=False)
            else:
                ids = mtok.encode(prompt)
            out = generate_ids(model, list(ids), wm, eos, max_new_tokens=max_new_tokens,
                               temperature=temperature, top_p=top_p, watermark=watermark, on_step=on_step)
            text = mtok.decode(out)
        else:
            model = loaded
            eos = set(np.atleast_1d(model.generation_config.eos_token_id or tok.eos_token_id).tolist())
            if getattr(tok, "chat_template", None):
                enc = tok.apply_chat_template(
                    [{"role": "user", "content": prompt}], add_generation_prompt=True,
                    return_tensors="pt", return_dict=True,
                    enable_thinking=False,  # Qwen3: answer directly; other templates ignore it
                )
            else:
                enc = tok(prompt, return_tensors="pt")
            enc = {k: v.to(model.device) for k, v in dict(enc).items()}
            n_prompt = enc["input_ids"].shape[1]
            with torch.no_grad():
                out = watermark_generate(model, tok, wm, temperature=temperature, top_p=top_p,
                                         max_new_tokens=max_new_tokens, watermark=watermark,
                                         on_step=on_step, **enc)
            text = tok.decode(out[0, n_prompt:], skip_special_tokens=True)
        return {
            "text": text.strip(), "model": name, "watermark": watermark,
            "seconds": round(time.time() - t0, 2), "load_seconds": round(t_load, 1),
            "trace": {"steps": steps, "context_width": wm.context_width} if trace else None,
        }


def _trace_step(tok, wm: Watermarker, p, history, chosen, state) -> dict:
    """What the sampler saw at one step: top candidates with p and keyed u."""
    top = np.argpartition(-p, TRACE_K)[:TRACE_K]
    top = top[np.argsort(-p[top])]
    if chosen not in top:
        top = np.append(top[: TRACE_K - 1], chosen)
    u = keyed_uniforms(wm.key, list(history[-wm.context_width:]), int(max(top.max(), chosen)) + 1)
    q = p[p > 0]
    mode = state.last_mode
    return {
        "mode": mode,
        "entropy": float(-(q * np.log2(q)).sum()),
        "chosen": int(np.where(top == chosen)[0][0]),
        # The detector's score for this token (None where the detector skips it)
        "score": None if mode == "repeat" else float(-np.log1p(-u[chosen])),
        "cands": [{"t": tok.decode([int(i)]), "p": round(float(p[i]), 6), "u": round(float(u[i]), 6)} for i in top],
    }


# --------------------------------------------------------------------------- streaming route
class StreamReq(BaseModel):
    model: str = ""
    prompt: str = Field(min_length=1, max_length=4000)
    watermark: bool = True
    temperature: float = Field(1.0, gt=0, le=2)
    top_p: float = Field(1.0, gt=0, le=1)
    max_new_tokens: int = Field(300, ge=1, le=1024)
    trace: bool = True  # per-step sampler details for the visualizer (/static/sampler-demo.html)


def add_stream_route(app, engine: Engine, watermarker_for: Callable[[str], Watermarker], flag_p: float = 1e-5):
    """POST /api/stream → newline-delimited JSON events:
    {"type":"start","model":..,"loading":bool,"flag_p":..},
    {"type":"token","text":..,"step":{..}|null,"live":{"s","p","n","z"}}…,
    then {"type":"done", ...result} or {"type":"error","detail":..}.

    ``live`` is the generating model's detector test on the reply so far; ``flag_p``
    is the whole-text p-value below which the checker calls a match."""

    @app.post("/api/stream")
    def stream(req: StreamReq):
        name = req.model or engine.names[0]
        engine.check(name)
        wm = watermarker_for(name)
        events: queue.Queue = queue.Queue()

        def work():
            try:
                res = engine._run(name, req.prompt, wm, req.watermark, req.temperature, req.top_p,
                                  req.max_new_tokens, req.trace, events.put)
                res.pop("trace", None)  # steps were already streamed
                events.put({"type": "done", **res})
            except Exception as e:  # report it in the stream instead of a bare 500
                events.put({"type": "error", "detail": f"{type(e).__name__}: {e}"})
            events.put(None)

        events.put({"type": "start", "model": name, "loading": not engine.loaded(name),
                    "context_width": wm.context_width, "flag_p": flag_p})
        engine.worker.submit(work)

        def gen():
            while (ev := events.get()) is not None:
                yield json.dumps(ev) + "\n"

        return StreamingResponse(gen(), media_type="application/x-ndjson",
                                 headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})
