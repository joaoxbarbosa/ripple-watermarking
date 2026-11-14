"""Compute the numbers behind the results site and the model card.

    .venv/bin/python docs/make_stats.py experiment.json

Reads an experiment file from docs/run_experiment.py and writes
site/results.js (``window.RESULTS = {...}``) so site/index.html works opened
straight from disk.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ripple_fingerprint import ModelRegistry, Watermarker  # noqa: E402

FPR = 1e-5


def main(path: str):
    exp = json.loads(Path(path).read_text())
    wm = Watermarker(exp["key"].encode())
    tok = AutoTokenizer.from_pretrained(exp["model"])
    runs = exp["runs"]
    wmk = [r for r in runs if r["watermark"]]
    plain = [r for r in runs if not r["watermark"]]
    rng = np.random.default_rng(0)
    out: dict = {
        "model": Path(exp["model"]).name,
        "fpr": FPR,
        "settings": {"temperature": 1.0, "top_p": 1.0, "max_new_tokens": 300, "context_width": 3},
        "n_prompts": len(wmk),
        "generated": time.strftime("%Y-%m-%d"),
    }

    # ---------------------------------------------------------- headline detection
    def summary(rs):
        det = [wm.detect(r["ids"]) for r in rs]
        return {
            "n": len(rs),
            "flagged": int(sum(d.is_watermarked for d in det)),
            "median_tokens": float(np.median([d.num_tokens for d in det])),
            "median_scored_share": float(np.median([d.num_scored / max(d.num_tokens, 1) for d in det])),
            "log10_p": [float(np.log10(max(d.p_value, 1e-300))) for d in det],
            "z": [float(d.z_score) for d in det],
            "tok_s": float(np.median([r["tok_s"] for r in rs])),
        }

    out["watermarked"] = summary(wmk)
    out["plain"] = summary(plain)

    # tokens needed before the first p < FPR
    def first_hit(ids):
        for L in range(10, len(ids) + 1, 5):
            if wm.detect(ids[:L]).p_value < FPR:
                return L
        return None

    hits = [first_hit(r["ids"]) for r in wmk]
    out["tokens_to_detect"] = hits
    out["watermarked_prompts"] = [r["prompt"] for r in wmk]

    # ---------------------------------------------------------- evidence curves
    N = max(len(r["ids"]) for r in runs)

    def curve(r):
        tot, n, c = 0.0, 0, []
        for v in wm.token_scores(r["ids"]):
            if v is not None:
                tot += v; n += 1
            c.append(round(tot - n, 2))
        return c

    out["evidence"] = {"watermarked": [curve(r) for r in wmk], "plain": [curve(r) for r in plain]}

    # ---------------------------------------------------------- detection vs length
    lens = list(range(10, N + 1, 10))
    out["by_length"] = {
        "tokens": lens,
        "watermarked": [float(np.mean([wm.detect(r["ids"][:L]).p_value < FPR for r in wmk]) * 100) for L in lens],
        "plain": [float(np.mean([wm.detect(r["ids"][:L]).p_value < FPR for r in plain]) * 100) for L in lens],
    }

    # ---------------------------------------------------------- robustness to edits
    fills = ["the", "very", "also", "quite", "really", "just", "some", "often"]

    def replace(text, frac):
        parts = re.split(r"(\s+)", text)
        return "".join(rng.choice(fills) if (i % 2 == 0 and p and rng.random() < frac) else p
                       for i, p in enumerate(parts))

    def delete(text, frac):
        parts = re.split(r"(\s+)", text)
        keep = [p for i, p in enumerate(parts) if not (i % 2 == 0 and p and rng.random() < frac)]
        return "".join(keep)

    fracs = [round(f, 2) for f in np.arange(0, 0.51, 0.05)]
    rob = {"fraction": fracs}
    for name, fn in (("replace", replace), ("delete", delete)):
        rw, med = [], []
        for f in fracs:
            ps = [wm.detect(tok.encode(fn(r["text"], f), add_special_tokens=False)).p_value
                  for r in wmk for _ in range(3)]
            rw.append(float(np.mean(np.array(ps) < FPR) * 100))
            med.append(float(np.log10(max(np.median(ps), 1e-300))))
        fp = [float(np.mean([wm.detect(tok.encode(fn(r["text"], f), add_special_tokens=False)).p_value < FPR
                             for r in plain]) * 100) for f in fracs]
        rob[name] = {"watermarked": rw, "median_log10_p": med, "plain": fp}
    out["robustness"] = rob

    # ---------------------------------------------------------- calibration under wrong keys
    K = 60
    pv = []
    for k in range(K):
        other = Watermarker(f"calibration-key-{k:04d}-xxxxxxxxxxxx".encode())
        pv += [other.detect(r["ids"]).p_value for r in runs]
    pv = np.array(pv)
    counts, _ = np.histogram(pv, np.linspace(0, 1, 21))
    out["calibration"] = {
        "tests": len(pv), "keys": K, "bins": counts.tolist(),
        "min_p": float(pv.min()), "share_below_01": float(np.mean(pv < 0.01)),
        "share_below_05": float(np.mean(pv < 0.05)), "flagged": int(np.sum(pv < FPR)),
    }

    # ---------------------------------------------------------- attribution with a registry
    # The experiment key plays model A. Four more labels share the tokenizer.
    reg = ModelRegistry(master_key=b"site-attribution-master-0123456789")
    labels = ["model-A", "model-B", "model-C", "model-D", "model-E"]
    for lb in labels:
        reg.register(lb, tok)
    reg._entries["model-A"].watermarker = wm  # noqa: SLF001 - the experiment was sampled with this key
    rows = []
    for r in wmk:
        res = reg.attribute_ids(r["ids"])
        rows.append({"best": res.best, "matches": res.matches})
    plain_rows = [reg.attribute_ids(r["ids"]).matches for r in plain]
    out["attribution"] = {
        "labels": labels,
        "correct": sum(x["best"] == "model-A" and x["matches"] == ["model-A"] for x in rows),
        "wrong": sum(any(m != "model-A" for m in x["matches"]) for x in rows),
        "plain_named": sum(bool(m) for m in plain_rows),
        "n": len(rows),
    }

    # ---------------------------------------------------------- mixed-document spans
    # Splice a watermarked reply between two plain replies and check the span edges.
    span_tests = []
    for i in range(min(8, len(wmk))):
        a, b, c = plain[i]["text"], wmk[i]["text"], plain[(i + 1) % len(plain)]["text"]
        doc = a + "\n\n" + b + "\n\n" + c
        lo, hi = len(a) + 2, len(a) + 2 + len(b)
        res = reg.attribute(doc)
        sp = [s for s in res.spans if s.model == "model-A"]
        if sp and sp[0].char_start is not None:
            ov = max(0, min(hi, sp[0].char_end) - max(lo, sp[0].char_start)) / (hi - lo)
            span_tests.append({"found": True, "overlap": float(ov), "others": sum(s.model != "model-A" for s in res.spans)})
        else:
            span_tests.append({"found": False, "overlap": 0.0, "others": sum(s.model != "model-A" for s in res.spans)})
    out["spans"] = span_tests

    # ---------------------------------------------------------- speed + samples
    out["speed"] = json.loads((ROOT / "docs/speed.json").read_text())
    pick = int(np.argmin([abs(len(r["ids"]) - 200) for r in wmk]))
    sample = wmk[pick]
    sc = wm.token_scores(sample["ids"])
    out["sample"] = {
        "prompt": sample["prompt"],
        "tokens": [tok.decode([t]) for t in sample["ids"]],
        "scores": [None if s is None else round(s, 3) for s in sc],
        "log10_p": float(np.log10(max(wm.detect(sample["ids"]).p_value, 1e-300))),
    }

    dest = ROOT / "site/results.js"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text("window.RESULTS = " + json.dumps(out, separators=(",", ":")) + ";\n")
    print("wrote", dest)
    print(json.dumps({k: v for k, v in out.items() if k in ("watermarked", "plain", "attribution", "calibration")},
                     default=str)[:3000])
    print("tokens_to_detect", hits)
    print("robustness replace", rob["replace"]["watermarked"])
    print("robustness delete", rob["delete"]["watermarked"])
    print("spans", span_tests)


if __name__ == "__main__":
    main(sys.argv[1])
