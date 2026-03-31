"""
Output files:
    autotune_cache/512_4096_bfloat16.json
    autotune_cache/2048_4096_float32.json
    ...
"""

import os
import torch
import helion
from helion_common import MATMUL_CONFIGS, config_key, swiglu_kernel_fn

CACHE_DIR = os.path.join(os.path.dirname(__file__), "autotune_cache")

swiglu_helion_autotune = helion.kernel()(swiglu_kernel_fn)

if __name__ == "__main__":
    os.makedirs(CACHE_DIR, exist_ok=True)

    for tokens, d_model, hidden_dim, dtype in MATMUL_CONFIGS:
        key = config_key(tokens, d_model,  hidden_dim, dtype)
        cache_path = os.path.join(CACHE_DIR, f"{key}.json")

        print(f"Autotuning {key} ...")

        x = torch.randn(tokens, d_model, device="cuda", dtype=dtype)
        w1 = torch.randn(hidden_dim, d_model, device="cuda", dtype=dtype)
        w2 = torch.randn(hidden_dim, d_model, device="cuda", dtype=dtype)

        best_config = swiglu_helion_autotune.autotune((x, w1, w2))
        best_config.save(cache_path)

        print(f"  Saved -> {cache_path}")

