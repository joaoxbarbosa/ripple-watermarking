# Running Ripple-65B on a GPU pod (llama.cpp)

Notes for serving **Ripple-65B-Preview** (Ricursive, Apache-2.0) through llama.cpp on a
rented GPU pod, with the watermark applied in Python. Merged in from the earlier
`Ripple-Watermark` working repo.

## Restore after a pod crash

```bash
git clone https://github.com/joaoxbarbosa/llm-fingerprinting.git /workspace/llm-fingerprinting
cd /workspace/llm-fingerprinting
bash scripts/setup_pod.sh                   # ~42 GB disk; model download is resumable
nohup scripts/start_llama_server.sh > /root/llama-server.log 2>&1 &
curl -s --retry 60 --retry-delay 2 --retry-all-errors http://127.0.0.1:8080/health

python3 scripts/ripple_cli_demo.py          # one watermarked + one plain reply, detect + attribute
python3 scripts/eval_ripple.py              # story + code, watermarked vs plain (~1.5 min)
python3 -m demo.server                      # demo website on 127.0.0.1:8000
```

The demo server only listens on localhost. From your own computer, tunnel to it:

```bash
ssh -N -L 8000:localhost:8000 root@<pod public IP> -p <pod SSH port>
```

then open http://localhost:8000. If SSH asks for a password, the pod has no public key
installed (`~/.ssh/authorized_keys` reads `null`): add yours in RunPod → Settings →
SSH Public Keys before creating the pod, or append it to that file.

### The key

The master key is committed at `keys/demo-master.key` **on purpose, because this is a demo.**
Anyone with the key can detect the watermark *and forge it*, so a public key gives no real
protection. For real use, generate a private key, keep it out of git (`.gitignore` blocks other
`*.key` files), and point the scripts at it with `RIPPLE_KEY=/path/to/private.key` (the demo
server reads `RIPPLE_FP_KEY`). Text watermarked under one key is only detectable with that
same key.

## Environment it was built on

| | |
|---|---|
| GPU | NVIDIA RTX 6000 Ada, 48 GB (driver 595.91.07, CUDA 13.2) |
| Model | `AtomicChat/Ripple-65B-Preview-GGUF` @ `df550a7`, `Ripple-65B-Preview-Q4_K_M.gguf`, 38,798,820,576 bytes, sha256 `51de49f0…7b8652` |
| llama.cpp | `9183a552e5fd` (2026-11-14), built with `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89` |
| Python | 3.12.3, numpy 2.5.4, scipy 1.18.1 |
| VRAM in use | ~40 GB (weights + 8K context with `--swa-full`) |

Model architecture (`general.architecture = ricursive`): 80 layers, 5120 hidden, 80 heads / 8 KV
heads, vocab 100,352, sliding window 1024 on 4 of every 5 layers, ChatML template
(`<|im_start|>` … `<|im_end|>`).

**Thinking.** Ripple may emit a `<think>…</think>` block before answering, and its template
has no `enable_thinking` switch. The demo, the results experiment and the CLI demo prefill an
empty `<think>\n\n</think>\n\n` block, so the model answers directly and detection runs on
the visible answer.

## How the llama.cpp backend works

For each token, `ripple_fingerprint/llamacpp.py` asks llama-server for the **top 1000 raw
next-token log-probs** (`n_predict=1`, `n_probs=1000`, `post_sampling_probs=false`, prompt
cache on). It applies temperature / top-p in Python and lets `Watermarker.sample` pick the
token. One HTTP request per token over a kept-alive connection; the next step's keyed draws
are computed on a helper thread while the server runs the forward pass.

```python
from ripple_fingerprint import ModelRegistry
from ripple_fingerprint.llamacpp import LlamaServer, generate

srv = LlamaServer("http://127.0.0.1:8080")
reg = ModelRegistry(master_key_bytes)
wm  = reg.register("ripple-65b-preview", encode=srv.tokenize)
text = generate(srv, "Write a haiku.", wm, temperature=1.0, top_p=0.95)
reg.attribute(text).best          # -> "ripple-65b-preview"
```

## Speed work

| | tokens/sec |
|---|---|
| Plain Ripple (llama-server's own sampler) | ~21 |
| Watermarked, first version | 5-6 |
| Watermarked, now | **15-17.5** |

The fixes:
1. **`--swa-full` (the big one).** Without it, llama-server recomputed the last ~6 tokens on
   every request because the sliding-window cache can't be rewound (120-138 ms/request vs 48 ms).
   See `scripts/cache_probe.py`.
2. `Watermarker.sample` races only tokens with nonzero probability (~1000) instead of all
   100k. Verified identical picks in `tests/test_sampler_equivalence.py`. The scheme and keys
   are unchanged.
3. 1000 candidates instead of 2000, plus a persistent HTTP connection.

The full 16-prompt evaluation (detection, edits, false alarms, speed against the native
sampler) is in `MODEL_CARD.md` under "Ripple-65B" and on the results site (`/results/`).

### First spot checks (one sample each, temperature 1.0, top-p 0.95, ≤400 tokens, fpr = 1e-5)

| Text | Watermarked? | p-value | Detected | Attributed to Ripple |
|---|---|---|---|---|
| Story | yes | 1e-22 to 3e-25 | ✅ | ✅ |
| Story | no | 0.35 / 0.54 | ✅ not flagged | ✅ none |
| Code | yes | 3e-6 / 3e-7 | ✅ | ⚠ first run just missed (adj. p 1.08e-5), second passed |
| Code | no | 0.08 / 0.025 | ✅ not flagged | ✅ none |

A decoy key never matched anything. Detection works from the decoded text (re-tokenized),
not just the original token ids.

## Open issues / next steps

- **Code is low-entropy** (~0.3 bits/token vs ~1.1 for prose), so it needs roughly 600+ tokens
  for reliable attribution at fpr 1e-5.
- **Plain code scored 1.105 and 1.109 mean token score** in the spot checks (null expectation
  1.0, about 2σ high both times). The 16-prompt run found no false alarms in 16 plain replies
  and 1,920 wrong-key tests, but it had no code prompts; a code-specific check is still open.
- **Ripple with thinking on:** if a UI shows or hides the reasoning, decide whether detection
  should cover the visible answer only.
- **Speed:** sampling inside llama.cpp (Python bindings / C API) would remove the per-token
  HTTP round trip and get close to the native ~21 tok/s.
