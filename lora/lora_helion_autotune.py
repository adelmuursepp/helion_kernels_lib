"""
Autotunes the helion LoRA kernel for each shape in LORA_CONFIGS and saves
the best config to autotune_cache/<key>.json.

Output files:
    autotune_cache/lora_512_4096_4096_16_bfloat16.json
    autotune_cache/lora_2048_4096_4096_16_bfloat16.json
    ...

Run with GPU:
    srun --gres=gpu:h100:1 --mem=16G apptainer exec --nv ~/apptainer.sandbox \
        python3.12 lora_helion_autotune.py
"""

import os

# Run each autotune candidate in a subprocess so a crashing config
# (e.g. cudaErrorIllegalAddress or PTXAS register overflow) doesn't kill the
# whole tuner. The best non-crashing config is still saved.
os.environ.setdefault("HELION_AUTOTUNE_PRECOMPILE", "spawn")

import torch
import helion
from common import LORA_CONFIGS, config_key, lora_kernel_fn

CACHE_DIR = os.path.join(os.path.dirname(__file__), "autotune_cache")

lora_helion_autotune = helion.kernel()(lora_kernel_fn)

if __name__ == "__main__":
    os.makedirs(CACHE_DIR, exist_ok=True)

    for tokens, in_dim, out_dim, rank, dtype in LORA_CONFIGS:
        key = config_key(tokens, in_dim, out_dim, rank, dtype)
        cache_path = os.path.join(CACHE_DIR, f"{key}.json")

        if os.path.exists(cache_path):
            print(f"Skipping {key} (cache exists)")
            continue

        print(f"Autotuning {key} ...")

        x   = torch.randn(tokens,  in_dim,  device="cuda", dtype=dtype)
        W   = torch.randn(out_dim, in_dim,  device="cuda", dtype=dtype)
        A   = torch.randn(out_dim, rank,    device="cuda", dtype=dtype)
        B_T = torch.randn(in_dim,  rank,    device="cuda", dtype=dtype)  # B.T

        best_config = lora_helion_autotune.autotune((x, W, A, B_T))
        best_config.save(cache_path)

        print(f"  Saved -> {cache_path}")
