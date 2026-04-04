"""
Output files:
    autotune_cache/512_4096_bfloat16.json
    autotune_cache/2048_4096_float32.json
    ...
"""

import os

# Run each autotune candidate in a subprocess so a crashing config (e.g.
# cudaErrorIllegalAddress or PTXAS register overflow) doesn't kill the whole
# tuner. The best non-crashing config still gets saved.
os.environ.setdefault("HELION_AUTOTUNE_PRECOMPILE", "spawn")

import torch
import helion
from common import MATMUL_CONFIGS, config_key, swiglu_kernel_fn

CACHE_DIR = os.path.join(os.path.dirname(__file__), "autotune_cache")

swiglu_helion_autotune = helion.kernel()(swiglu_kernel_fn)

if __name__ == "__main__":
    os.makedirs(CACHE_DIR, exist_ok=True)

    for tokens, d_model, hidden_dim, dtype in MATMUL_CONFIGS:
        key = config_key(tokens, d_model,  hidden_dim, dtype)
        cache_path = os.path.join(CACHE_DIR, f"{key}.json")

        if os.path.exists(cache_path):
            print(f"Skipping {key} (cache exists)")
            continue

        print(f"Autotuning {key} ...")

        x = torch.randn(tokens, d_model, device="cuda", dtype=dtype)
        w1 = torch.randn(hidden_dim, d_model, device="cuda", dtype=dtype)
        w2 = torch.randn(hidden_dim, d_model, device="cuda", dtype=dtype)

        best_config = swiglu_helion_autotune.autotune((x, w1, w2))
        best_config.save(cache_path)
        print(f"  Saved config -> {cache_path}")

        triton_path = os.path.join(CACHE_DIR, f"{key}_triton.py")
        try:
            triton_code = helion.kernel(config=best_config)(swiglu_kernel_fn).bind(x, w1, w2).to_triton_code()
            with open(triton_path, "w") as f:
                f.write(triton_code)
            print(f"  Saved Triton -> {triton_path}")
        except Exception as e:
            print(f"  Could not save Triton code: {e}")

