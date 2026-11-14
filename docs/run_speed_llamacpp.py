"""Speed of fingerprinted sampling vs llama-server's own sampler, for the results site.

    python3 docs/run_speed_llamacpp.py docs/experiment-ripple-65b.json docs/speed-ripple-65b.json

Native speed: llama-server generating the experiment's prompts itself (same
no-think prefill, temperature 1.0, 300 tokens), timed by the server. The other
rows are the medians of the experiment's own runs, so run this with the GPU idle.
"""
import json, os, statistics, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ripple_fingerprint.llamacpp import LlamaServer  # noqa: E402

exp = json.loads(Path(sys.argv[1]).read_text())
srv = LlamaServer(os.environ.get("RIPPLE_FP_LLAMACPP", "http://127.0.0.1:8080"))
prompts = list(dict.fromkeys(r["prompt"] for r in exp["runs"]))[:8]
native = []
for pr in prompts:
    text = srv._post("/apply-template", {"messages": [{"role": "user", "content": pr}]})["prompt"]
    r = srv._post("/completion", {"prompt": text + "<think>\n\n</think>\n\n", "n_predict": 300,
                                  "temperature": 1.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0, "cache_prompt": False})
    native.append(r["timings"]["predicted_per_second"])
nat = statistics.median(native)
fp = statistics.median(r["tok_s"] for r in exp["runs"] if r["watermark"])
plain = statistics.median(r["tok_s"] for r in exp["runs"] if not r["watermark"])
out = {
    "model": exp["model"] + " (llama.cpp, Q4_K_M)",
    "machine": "NVIDIA RTX 6000 Ada",
    "note": f"Native: llama-server's own sampler, median of {len(native)} replies ({nat:.1f} tok/s). "
            f"Others: medians of the experiment's 16 replies each ({plain:.1f} and {fp:.1f} tok/s).",
    "rows": [
        {"name": "Native llama-server sampling", "share": 1.0, "reference": True, "tok_s": nat},
        {"name": "Python loop, no fingerprint", "share": round(plain / nat, 3), "before": True, "tok_s": plain},
        {"name": "Fingerprinted: now", "share": round(fp / nat, 3), "tok_s": fp},
    ],
}
Path(sys.argv[2]).write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
