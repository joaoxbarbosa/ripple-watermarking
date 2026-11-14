"""Model loading and generation for the demo, with token streaming and sampler traces.

Kept separate from ``server.py`` so the server only wires up routes:

    engine = Engine(discover_models())
    engine.run(model, prompt, watermarker, ...)          # -> result dict
    add_stream_route(app, engine, watermarker_for)       # POST /api/stream

Backends:
* MLX models (e.g. LM Studio's ~/.lmstudio/models), run with Apple MLX.
* Hugging Face models, run with torch (CUDA, Apple GPU or CPU).
* llama.cpp: Ripple-65B-Preview (GGUF) served by a running ``llama-server``
  (see scripts/start_llama_server.sh). Offered automatically when the server
  at RIPPLE_FP_LLAMACPP (default http://127.0.0.1:8080) is up.
"""

from __future__ import annotations

import gc
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
HF_DEFAULTS = "Qwen/Qwen3-0.6B,Qwen/Qwen2.5-0.5B-Instruct,gpt2,openai-community/gpt2-large"
PREFERRED_DEFAULT = "Qwen3-VL-4B-Instruct-MLX-8bit"
HIDDEN_MLX = {"Qwen3-4B-Base-mlx-4Bit"}  # LM Studio models left out of the list
TRACE_K = 8  # candidate tokens recorded per step for the visualizer

# Ripple-65B through llama.cpp. The name is also the key label: keep it stable,
# or text fingerprinted earlier stops checking out.
RIPPLE_NAME = "ripple-65b-preview"
RIPPLE_TOKENIZER = "ricursiveAI/Ripple-65B-Preview"  # same ids as the GGUF's tokenizer (checked)
LLAMACPP_URL = os.environ.get("RIPPLE_FP_LLAMACPP", "http://127.0.0.1:8080")


# --------------------------------------------------------------------------- discovery
def is_mlx(path: Path) -> bool:
    try:
        return path.is_dir() and "quantization" in json.loads((path / "config.json").read_text())
    except (OSError, ValueError):
        return False


def discover_models() -> list[dict]:
    """Ripple via llama-server (if it's up), LM Studio MLX models, plus RIPPLE_FP_MODELS
    (or the HF defaults). The default goes first: Ripple when it's available."""
    models = []
    if LLAMACPP_URL.lower() not in ("", "0", "off", "none"):
        from ripple_fingerprint.llamacpp import LlamaServer
        if LlamaServer(LLAMACPP_URL, timeout=3).healthy():
            models.append({"name": RIPPLE_NAME, "label": "Ripple-65B-Preview · llama.cpp Q4_K_M (GPU)",
                           "backend": "llamacpp", "url": LLAMACPP_URL, "tokenizer": RIPPLE_TOKENIZER})
        else:
            print(f"llama-server not reachable at {LLAMACPP_URL}: Ripple-65B not offered", flush=True)
    if LMSTUDIO.is_dir():
        for d in sorted(LMSTUDIO.glob("*/*")):
            if is_mlx(d) and d.name not in HIDDEN_MLX:
                models.append({"name": str(d), "label": f"LM Studio · {d.name}", "backend": "mlx"})
    for n in os.environ.get("RIPPLE_FP_MODELS", HF_DEFAULTS).split(","):
        n = n.strip()
        if n and not any(m["name"] == n for m in models):
            local = Path(n).expanduser()
            models.append({"name": n, "label": local.name if local.is_dir() else n,
                           "backend": "mlx" if is_mlx(local) else "hf"})
    has_ripple = any(m["backend"] == "llamacpp" for m in models)
    want = os.environ.get("RIPPLE_FP_DEFAULT", RIPPLE_NAME if has_ripple else PREFERRED_DEFAULT)
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

    def __init__(self, models: list[dict], max_loaded: int | None = None):
        self.info = {m["name"]: m for m in models}
        self.names = [m["name"] for m in models]
        self.tokenizers: dict = {}
        self.models: dict = {}  # insertion order = least recently used first
        # Running every model in turn would otherwise keep them all in memory.
        self.max_loaded = max_loaded or int(os.environ.get("RIPPLE_FP_MAX_LOADED", 2))
        self.worker = ThreadPoolExecutor(max_workers=1)

    def check(self, name: str):
        if name not in self.info:
            raise HTTPException(400, f"unknown model {name!r}")

    def tokenizer(self, name: str):
        """Hugging Face tokenizer (MLX folders ship the same files). Used for detection."""
        self.check(name)
        if name not in self.tokenizers:
            info = self.info[name]
            if info["backend"] == "llamacpp":  # plain tokenizer.json: no remote code needed
                self.tokenizers[name] = AutoTokenizer.from_pretrained(info["tokenizer"])
            else:
                self.tokenizers[name] = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        return self.tokenizers[name]

    def loaded(self, name: str) -> bool:
        return name in self.models

    def unload(self, name: str):
        if self.models.pop(name, None) is None:
            return
        print(f"unloading {name}", flush=True)
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
        if self.info[name]["backend"] == "mlx":
            import mlx.core as mx
            mx.clear_cache()

    def _model(self, name: str):
        if self.info[name]["backend"] == "llamacpp":
            # Weights live in llama-server's process: nothing to load or evict here,
            # and it doesn't count against max_loaded.
            from ripple_fingerprint.llamacpp import LlamaServer
            if name not in self.models:
                self.models[name] = LlamaServer(self.info[name]["url"])
            return self.models[name]
        if name in self.models:
            self.models[name] = self.models.pop(name)  # mark as most recently used
        else:
            local = [n for n in self.models if self.info[n]["backend"] != "llamacpp"]
            while len(local) >= self.max_loaded:
                self.unload(local.pop(0))
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
        elif backend == "llamacpp":
            from ripple_fingerprint.llamacpp import generate_ids
            server = loaded
            eos = server.eos_ids()
            ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                          add_generation_prompt=True, tokenize=True, return_dict=False)
            # Ripple's template has no enable_thinking switch: prefill an empty think block so
            # it answers directly (like enable_thinking=False for the other models).
            ids = list(ids) + server.tokenize("<think>\n\n</think>\n\n", parse_special=True)
            out = generate_ids(server, ids, wm, eos, max_new_tokens=max_new_tokens,
                               temperature=temperature, top_p=top_p, watermark=watermark, on_step=on_step)
            text = tok.decode(out, skip_special_tokens=True)
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
    if state.last_draws is not None:  # keyed step: reuse the sampler's draws, u = exp(-E)
        u = np.exp(-state.last_draws)
    else:
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
