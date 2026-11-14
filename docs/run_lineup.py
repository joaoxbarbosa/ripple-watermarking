"""Run every demo model on the same prompts and check each reply against every key.

    .venv/bin/python -m demo.server &          # the demo server must be running
    .venv/bin/python docs/run_lineup.py [http://localhost:8000]
    .venv/bin/python docs/run_lineup.py --only gemma   # rerun matching models, keep the rest

Writes site/models.js (``window.LINEUP = {...}``) for site/models.html.
"""

from __future__ import annotations

import json
import math
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARGS = sys.argv[1:]
ONLY = ARGS[ARGS.index("--only") + 1] if "--only" in ARGS else None
_pos = [a for i, a in enumerate(ARGS) if not a.startswith("--") and (i == 0 or ARGS[i - 1] != "--only")]
BASE = _pos[0] if _pos else "http://localhost:8000"
PROMPTS = [
    "Write a short story about a lighthouse keeper.",
    "Explain how vaccines work.",
    "What are the main causes of inflation?",
    "Write a poem about the ocean at night.",
    "Give advice to someone starting their first job.",
    "Write a letter from a cat to its owner.",
]
SETTINGS = {"temperature": 1.0, "top_p": 1.0, "max_new_tokens": 600}


def call(path: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(BASE + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    for attempt in range(40):  # ride out a demo-server restart (up to ~4 minutes) or a stuck request
        try:
            with urllib.request.urlopen(req, timeout=240) as r:
                return json.loads(r.read())
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            if attempt == 39:
                raise
            time.sleep(6)


def log10(p: float) -> float:
    return round(math.log10(max(p, 1e-300)), 2)


def main():
    info = call("/api/info")
    models = [{"name": m["name"], "label": m["label"].removeprefix("LM Studio · ")} for m in info["models"]]
    names = [m["name"] for m in models]
    dest = ROOT / "site/models.js"
    # Each finished reply is saved here, so an interrupted run picks up where it stopped.
    partial = ROOT / "docs/.lineup-partial.json"
    sig = {"names": names, "settings": SETTINGS, "prompts": PROMPTS, "only": ONLY}
    runs = []
    if ONLY:  # keep the saved runs of every other model
        old = json.loads(dest.read_text().removeprefix("window.LINEUP = ").rstrip().rstrip(";"))
        assert [m["name"] for m in old["models"]] == names, "model list changed: run the full lineup"
        assert old["settings"] == SETTINGS, "settings changed: run the full lineup"
        runs = [r for r in old["runs"] if ONLY not in r["model"]]
    if partial.exists():
        saved = json.loads(partial.read_text())
        if saved["sig"] == sig:
            runs = saved["runs"]
            print(f"resuming: {len(runs)} replies already done", flush=True)
    done = {(r["model"], r["prompt"]) for r in runs}
    for i, m in enumerate(models):
        if ONLY and ONLY not in m["name"]:
            continue
        for j, prompt in enumerate(PROMPTS):
            if (m["name"], prompt) in done:
                continue
            g = call("/api/generate", {"model": m["name"], "prompt": prompt, "watermark": True, **SETTINGS})
            d = call("/api/detect", {"text": g["text"]})
            by = {x["name"]: x for x in d["models"]}
            runs.append({
                "model": m["name"], "prompt": prompt, "seconds": g["seconds"],
                "tokens": by[m["name"]]["num_tokens"], "scored": by[m["name"]]["num_scored"],
                "best": d["best"], "matches": d["matches"],
                "log10_p": [log10(by[n]["p_value"]) for n in names],      # raw p under each key
                "log10_adj": [log10(by[n]["evidence_p"]) for n in names],  # corrected across keys and spans
                "z": [round(by[n]["z_score"], 2) for n in names],
            })
            ok = d["best"] == m["name"] and d["matches"] == [m["name"]]
            print(f"[{i + 1}/{len(models)}] {m['label'][:34]:34} {j + 1}/{len(PROMPTS)} "
                  f"own z={runs[-1]['z'][i]:6.2f} {'named' if ok else 'MISSED'}", flush=True)
            partial.write_text(json.dumps({"sig": sig, "runs": runs}))
    runs.sort(key=lambda r: (names.index(r["model"]), PROMPTS.index(r["prompt"])))
    out = {"generated": time.strftime("%Y-%m-%d"), "fpr": info["fpr"], "settings": SETTINGS,
           "prompts": PROMPTS, "models": models, "runs": runs}
    dest.parent.mkdir(exist_ok=True)
    dest.write_text("window.LINEUP = " + json.dumps(out, separators=(",", ":")) + ";\n")
    partial.unlink(missing_ok=True)
    print("wrote", dest)


if __name__ == "__main__":
    main()
