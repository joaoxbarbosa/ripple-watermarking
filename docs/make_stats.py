"""Compute the numbers behind the results site and the model card.

    .venv/bin/python docs/make_stats.py experiment.json
    python3 docs/make_stats.py docs/experiment-ripple-65b.json ripple-65b docs/speed-ripple-65b.json

Reads an experiment file from docs/run_experiment.py (MLX) or
docs/run_experiment_llamacpp.py and writes site/results.js
(``window.RESULTS = {...}``), or, given a slug, site/results-<slug>.js, which
adds the run to ``window.RESULTS_MORE`` for the model switcher on
site/index.html. Either way the page works opened straight from disk.
llama.cpp runs are tokenized by the running llama-server.
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


class _ServerTokenizer:
    """The two tokenizer calls this script makes, answered by llama-server."""

    def __init__(self):
        from ripple_fingerprint.llamacpp import LlamaServer
        import os
        self.srv = LlamaServer(os.environ.get("RIPPLE_FP_LLAMACPP", "http://127.0.0.1:8080"))

    def encode(self, text, add_special_tokens=False):
        return self.srv.tokenize(text)

    def decode(self, ids):
        return self.srv.detokenize(ids)

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        """HF-style call, with character offsets rebuilt from the server's token pieces."""
        toks = self.srv._post("/tokenize", {"content": text, "with_pieces": True})["tokens"]
        ids = [t["id"] for t in toks]
        if not return_offsets_mapping:
            return {"input_ids": ids}
        raw = text.encode("utf-8")
        char_at = [0] * (len(raw) + 1)  # byte position -> character position
        b = 0
        for ci, ch in enumerate(text):
            for _ in range(len(ch.encode("utf-8"))):
                char_at[b] = ci; b += 1
        char_at[len(raw)] = len(text)
        offsets, pos = [], 0
        for t in toks:
            pc = t["piece"]
            n = len(pc.encode("utf-8")) if isinstance(pc, str) else len(pc)
            offsets.append((char_at[pos], char_at[min(pos + n, len(raw))])); pos += n
        if pos != len(raw):
            raise ValueError("token pieces do not cover the text")  # registry falls back to no offsets
        return {"input_ids": ids, "offset_mapping": offsets}


def main(path: str, slug: str | None = None, speed_path: str | None = None):
    exp = json.loads(Path(path).read_text())
    wm = Watermarker(exp["key"].encode())
    llama = exp.get("backend") == "llamacpp"
    tok = _ServerTokenizer() if llama else AutoTokenizer.from_pretrained(exp["model"])
    runs = exp["runs"]
    wmk = [r for r in runs if r["watermark"]]
    plain = [r for r in runs if not r["watermark"]]
    rng = np.random.default_rng(0)
    out: dict = {
        "model": Path(exp["model"]).name,
        "backend": "llama.cpp" if llama else "MLX",
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
    out["speed"] = json.loads(Path(speed_path or ROOT / "docs/speed.json").read_text())
    pick = int(np.argmin([abs(len(r["ids"]) - 200) for r in wmk]))
    sample = wmk[pick]
    sc = wm.token_scores(sample["ids"])
    out["sample"] = {
        "prompt": sample["prompt"],
        "tokens": [tok.decode([t]) for t in sample["ids"]],
        "scores": [None if s is None else round(s, 3) for s in sc],
        "log10_p": float(np.log10(max(wm.detect(sample["ids"]).p_value, 1e-300))),
    }

    blob = json.dumps(out, separators=(",", ":"))
    if slug:
        dest = ROOT / f"site/results-{slug}.js"
        js = f"(window.RESULTS_MORE = window.RESULTS_MORE || {{}})[{json.dumps(slug)}] = {blob};\n"
    else:
        dest = ROOT / "site/results.js"
        js = "window.RESULTS = " + blob + ";\n"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(js)
    print("wrote", dest)
    print(json.dumps({k: v for k, v in out.items() if k in ("watermarked", "plain", "attribution", "calibration")},
                     default=str)[:3000])
    print("tokens_to_detect", hits)
    print("robustness replace", rob["replace"]["watermarked"])
    print("robustness delete", rob["delete"]["watermarked"])
    print("spans", span_tests)


if __name__ == "__main__":
    main(*sys.argv[1:4])
