"""End-to-end demo with any Hugging Face causal LM (stand-in for Ripple).

    pip install torch transformers
    RIPPLE_FP_KEY=$(python -c "import secrets;print(secrets.token_hex(32))") \
        python examples/demo_hf.py gpt2
"""

import os
import re
import sys

from transformers import AutoModelForCausalLM, AutoTokenizer

from ripple_fingerprint import Watermarker
from ripple_fingerprint.hf import detect_text, generate

model_name = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).eval()
wm = Watermarker(os.environ["RIPPLE_FP_KEY"])

prompt = "The history of the printing press"
text = generate(model, tok, prompt, wm, max_new_tokens=200, temperature=0.9, top_p=0.95)
print(text, "\n")
print("watermarked :", detect_text(text, tok, wm))

parts = re.split(r"(\s+)", text)  # keep whitespace/newlines intact
edited = "".join("indeed" if i % 10 == 4 else p for i, p in enumerate(parts))  # every 5th word replaced
print("edited (20%):", detect_text(edited, tok, wm))

plain = model.generate(**tok(prompt, return_tensors="pt"), do_sample=True, temperature=0.9, top_p=0.95,
                       max_new_tokens=200, pad_token_id=tok.eos_token_id)
print("unwatermarked:", detect_text(tok.decode(plain[0], skip_special_tokens=True), tok, wm))
