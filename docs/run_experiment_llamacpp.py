"""Generate the replies behind the results site with a llama.cpp model (Ripple-65B).

    python3 docs/run_experiment_llamacpp.py docs/experiment-ripple-65b.json [label]

Same 16 prompts, key and settings as docs/run_experiment.py (the MLX run), so the
two models' numbers are directly comparable. Needs scripts/start_llama_server.sh
running. Writes after every prompt, and resumes from a partial file.
"""
import json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ripple_fingerprint import Watermarker  # noqa: E402
from ripple_fingerprint.llamacpp import LlamaServer, generate_ids  # noqa: E402

PROMPTS = [
    "Write a short story about a lighthouse keeper.", "Explain how vaccines work.",
    "What are the main causes of inflation?", "Write a short history of the printing press.",
    "Describe the water cycle.", "Write a poem about the ocean at night.",
    "Give advice to someone starting their first job.", "Explain how a CPU executes instructions.",
    "Write a letter from a cat to its owner.", "Summarize the plot of a heist movie you invent.",
    "What makes a good scientific experiment?", "Describe a busy market in a city you invent.",
    "Explain the difference between weather and climate.", "Write a dialogue between two robots meeting for the first time.",
    "How do airplanes stay in the air?", "Write a recipe for a soup, with a short story about it.",
]

dest = Path(sys.argv[1])
label = sys.argv[2] if len(sys.argv) > 2 else "Ripple-65B"
srv = LlamaServer(os.environ.get("RIPPLE_FP_LLAMACPP", "http://127.0.0.1:8080"))
assert srv.healthy(), "llama-server is not up (scripts/start_llama_server.sh)"
wm = Watermarker(b"readme-experiment-key-0123456789ab")
exp = json.loads(dest.read_text()) if dest.exists() else {
    "key": wm.key.decode(), "model": label, "backend": "llamacpp", "runs": []}
done = {(r["prompt"], r["watermark"]) for r in exp["runs"]}
eos = srv.eos_ids()
# Ripple's template has no enable_thinking switch, so prefill an empty think block:
# the replies are answers only, like the MLX run (enable_thinking=False).
NO_THINK = srv.tokenize("<think>\n\n</think>\n\n", parse_special=True)
exp["thinking"] = False
for i, pr in enumerate(PROMPTS):
    ids = srv.chat_prompt_ids(pr) + NO_THINK
    for wmk in (True, False):
        if (pr, wmk) in done:
            continue
        t = time.time()
        g = generate_ids(srv, ids, wm, eos, max_new_tokens=300, temperature=1.0, watermark=wmk)
        exp["runs"].append({"prompt": pr, "watermark": wmk, "ids": g, "text": srv.detokenize(g),
                            "tok_s": len(g) / (time.time() - t)})
        dest.write_text(json.dumps(exp))
    print(f"{i+1}/{len(PROMPTS)} done", flush=True)
