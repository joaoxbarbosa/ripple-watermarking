"""Per-prompt generation speed with and without the fingerprint (for the README chart).

    PYTHONPATH=. .venv/bin/python docs/run_speed.py docs/speed-pairs.json [model_path] [reps]

Each prompt is generated `reps` times with and without the fingerprint, alternating,
so background load hits both equally. The chart uses the per-prompt medians.
"""
import json, sys, time
from pathlib import Path
from mlx_lm import load
from ripple_fingerprint import Watermarker
from ripple_fingerprint.mlx import generate_ids

P = sys.argv[2] if len(sys.argv) > 2 else str(Path.home() / ".lmstudio/models/lmstudio-community/Qwen3-VL-4B-Instruct-MLX-8bit")
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
m, tok = load(P)
wm = Watermarker(b"speed-pairs-key-0123456789abcdefg")
prompts = json.loads(Path(__file__).with_name("prompts.json").read_text())
eos = set(tok.eos_token_ids)
generate_ids(m, tok.encode("warm up"), wm, eos, max_new_tokens=20)  # compile kernels
rows = []
for i, pr in enumerate(prompts):
    ids = tok.apply_chat_template([{"role": "user", "content": pr}], add_generation_prompt=True, enable_thinking=False)
    runs = {"with": [], "without": []}
    for _ in range(REPS):
        for name, wmk in (("with", True), ("without", False)):
            t = time.time()
            n = len(generate_ids(m, ids, wm, eos, max_new_tokens=150, temperature=1.0, watermark=wmk))
            runs[name].append(n / (time.time() - t))
    rows.append({"prompt": pr, **runs})
    print(f"{i+1}/{len(prompts)}", flush=True)
json.dump({"model": Path(P).name, "reps": REPS, "max_new_tokens": 150, "rows": rows}, open(sys.argv[1], "w"), indent=1)
