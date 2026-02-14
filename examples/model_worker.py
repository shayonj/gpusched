#!/usr/bin/env python3
"""GPU model worker for gpusched.

Loads a HuggingFace model onto the GPU, then idles as a simulated inference worker.
Perfect for testing freeze/thaw with gpusched.

Usage:
    gpusched run --name model-a -- python3 model_worker.py
    gpusched run --name model-b -- python3 model_worker.py Qwen/Qwen2.5-1.5B-Instruct
"""
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B-Instruct"

print(f"Loading {MODEL}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda()
model.eval()

mem_mb = torch.cuda.memory_allocated() / 1024**2
print(f"Model loaded: {mem_mb:.0f} MB GPU memory")
print(f"PID: {os.getpid()}")
if os.environ.get("GPUSCHED_MANAGED"):
    print("Managed by gpusched — freeze me with 'gpusched freeze'")
print("---")

# Idle loop with periodic heartbeat inference.
step = 0
while True:
    time.sleep(5)
    step += 1
    if step % 12 == 0:  # Every 60 seconds
        inputs = tokenizer("What is 2+2?", return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        print(f"[heartbeat {step}] {text[:60]}")
        sys.stdout.flush()
