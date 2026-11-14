"""Generate the replies behind the README charts (watermarked and plain, MLX model).

    PYTHONPATH=. .venv/bin/python docs/run_experiment.py experiment.json [model_path]
"""
import json, sys, time
from mlx_lm import load
from ripple_fingerprint import Watermarker
from ripple_fingerprint.mlx import generate_ids
P = sys.argv[2] if len(sys.argv) > 2 else str(__import__("pathlib").Path.home() / ".lmstudio/models/lmstudio-community/Qwen3-VL-4B-Instruct-MLX-8bit")
m, tok = load(P)
wm = Watermarker(b"readme-experiment-key-0123456789ab")
prompts = [
    "Write a short story about a lighthouse keeper.", "Explain how vaccines work.",
    "What are the main causes of inflation?", "Write a short history of the printing press.",
    "Describe the water cycle.", "Write a poem about the ocean at night.",
    "Give advice to someone starting their first job.", "Explain how a CPU executes instructions.",
    "Write a letter from a cat to its owner.", "Summarize the plot of a heist movie you invent.",
    "What makes a good scientific experiment?", "Describe a busy market in a city you invent.",
    "Explain the difference between weather and climate.", "Write a dialogue between two robots meeting for the first time.",
    "How do airplanes stay in the air?", "Write a recipe for a soup, with a short story about it.",
]
eos = set(tok.eos_token_ids); out = []
for i, pr in enumerate(prompts):
    ids = tok.apply_chat_template([{"role": "user", "content": pr}], add_generation_prompt=True, enable_thinking=False)
    for wmk in (True, False):
        t = time.time()
        g = generate_ids(m, ids, wm, eos, max_new_tokens=300, temperature=1.0, watermark=wmk)
        out.append({"prompt": pr, "watermark": wmk, "ids": g, "text": tok.decode(g), "tok_s": len(g) / (time.time() - t)})
    print(f"{i+1}/{len(prompts)} done", flush=True)
json.dump({"key": wm.key.decode(), "model": P, "runs": out}, open(sys.argv[1], "w"))
