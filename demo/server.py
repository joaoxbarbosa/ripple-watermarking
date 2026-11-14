"""Demo web app: generate watermarked text, edit it, and check it.

    .venv/bin/python -m demo.server            # http://localhost:8000
                                               # results site at /results/

Environment:
    RIPPLE_FP_MODELS  comma-separated HF model ids or local paths offered in the
                      model selector, next to any LM Studio MLX models found.
                      Defaults to two small Qwen chat models and gpt2.
    RIPPLE_FP_DEFAULT model selected by default (see demo/engine.py).
    RIPPLE_FP_KEY     master watermark secret. Each model's key is derived from
                      it and the model id, so checking text also tells you
                      which model wrote it. If unset, a key is created once and
                      saved in demo/.demo_key (gitignored).
    PORT              default 8000

Models load on first use and stay loaded. Detection only needs the
tokenizers, so checking text (against every model) never loads weights.
"""

from __future__ import annotations

import json
import os
import queue
import secrets
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ripple_fingerprint import ModelRegistry

from .engine import Engine, add_stream_route, discover_models

HERE = Path(__file__).parent


def _load_key() -> str:
    if os.environ.get("RIPPLE_FP_KEY"):
        return os.environ["RIPPLE_FP_KEY"]
    path = HERE / ".demo_key"
    if not path.exists():
        path.write_text(secrets.token_hex(32))
        path.chmod(0o600)
    return path.read_text().strip()


engine = Engine(discover_models())
MODELS = engine.names
keys = ModelRegistry(_load_key())  # one derived watermark key per model
for _name in MODELS:  # tokenizers are small; loading them all up front also fails fast on a bad id
    keys.register(_name, engine.tokenizer(_name))
CONTEXT_WIDTH = keys.watermarker(MODELS[0]).context_width

app = FastAPI(title="Ripple Fingerprint demo")
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
# The results site (site/) and the images it embeds (docs/) ride on the same server.
app.mount("/results", StaticFiles(directory=HERE.parent / "site", html=True), name="results")
app.mount("/docs", StaticFiles(directory=HERE.parent / "docs"), name="docs")
# The whole-text test is flagged when p * (2 tests per model) < fpr.
add_stream_route(app, engine, keys.watermarker, flag_p=keys.fpr / (2 * len(MODELS)))


class LineupReq(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    models: list[str] | None = None  # default: every model
    watermark: bool = True
    temperature: float = Field(1.0, gt=0, le=2)
    top_p: float = Field(1.0, gt=0, le=1)
    max_new_tokens: int = Field(300, ge=1, le=1024)


@app.post("/api/lineup")
def lineup(req: LineupReq):
    """Every model answers the same prompt, one after another; each reply is then
    checked against every model's key. Newline-delimited JSON events:
    {"type":"start","models":[..]}, then per model {"type":"model","model":..} when it
    starts and {"type":"result","model":..,"text":..,"best":..,"evidence":{name: p},..},
    or {"type":"error","model":..,"detail":..}; finally {"type":"done"}."""
    names = req.models or MODELS
    for n in names:
        engine.check(n)
    events: queue.Queue = queue.Queue()

    def work():
        for n in names:
            events.put({"type": "model", "model": n, "loading": not engine.loaded(n)})
            try:
                r = engine._run(n, req.prompt, keys.watermarker(n), req.watermark, req.temperature,
                                req.top_p, req.max_new_tokens, False, None)
                res = keys.attribute(r["text"]) if r["text"].strip() else None
                span_p: dict = {}
                for sp in res.spans if res else []:
                    span_p[sp.model] = min(sp.p_value, span_p.get(sp.model, 1.0))
                events.put({
                    "type": "result", "model": n, "text": r["text"], "seconds": r["seconds"],
                    "load_seconds": r["load_seconds"], "watermark": req.watermark,
                    "best": res.best if res else None, "matches": res.matches if res else [],
                    "num_scored": res[n].result.num_scored if res else 0,
                    "z": float(res[n].result.z_score) if res else 0.0,
                    "evidence": {m.name: min(m.adjusted_p, span_p.get(m.name, 1.0)) for m in res.models} if res else {},
                })
            except Exception as e:  # one broken model shouldn't stop the lineup
                events.put({"type": "error", "model": n, "detail": f"{type(e).__name__}: {e}"})
        events.put({"type": "done"})
        events.put(None)

    events.put({"type": "start", "models": names, "fpr": keys.fpr})
    engine.worker.submit(work)

    def gen():
        while (ev := events.get()) is not None:
            yield json.dumps(ev) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


class GenerateReq(BaseModel):
    model: str = MODELS[0]
    prompt: str = Field(min_length=1, max_length=4000)
    watermark: bool = True
    temperature: float = Field(1.0, gt=0, le=2)
    top_p: float = Field(1.0, gt=0, le=1)  # top-p < 1 cuts the randomness the fingerprint lives in
    max_new_tokens: int = Field(300, ge=1, le=1024)


class DetectReq(BaseModel):
    text: str = Field(max_length=100_000)
    focus: str | None = None  # model whose token heatmap to return; default: the best match


@app.get("/")
def index():
    return FileResponse(HERE / "index.html")


@app.get("/api/info")
def info():
    return {
        "models": [{"name": n, "label": engine.info[n]["label"], "loaded": engine.loaded(n)} for n in MODELS],
        "default": MODELS[0],
        "context_width": CONTEXT_WIDTH,
        "fpr": keys.fpr,
    }


@app.post("/api/generate")
def generate(req: GenerateReq):
    """Non-streaming generation. The page uses POST /api/stream (see engine.py)."""
    res = engine.run(req.model, req.prompt, keys.watermarker(req.model), watermark=req.watermark,
                     temperature=req.temperature, top_p=req.top_p, max_new_tokens=req.max_new_tokens)
    res.pop("trace", None)
    return res


def _pieces(text: str, ms) -> list[str]:
    """The text cut at token boundaries; joined back together they give ``text`` exactly."""
    if not ms.offsets:
        return [engine.tokenizer(ms.name).decode([i]) for i in ms.token_ids]
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
