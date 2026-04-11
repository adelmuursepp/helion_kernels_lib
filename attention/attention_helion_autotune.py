import os


import torch
import helion
from common import ATTENTION_CONFIGS, SAFE_CONFIGS
from helion_common import config_key, VARIANTS

CACHE_DIR = os.path.join(os.path.dirname(__file__), "autotune_cache")


if __name__ == "__main__":
    os.makedirs(CACHE_DIR, exist_ok=True)

    for variant_name, kernel_fn in VARIANTS:
        # Pass configs= so Helion uses FiniteSearch over our safe list only,
        # instead of random population that explores tile_m=256/512.
        # force=False is required to actually use FiniteSearch — the default
        # force=True overrides configs= and runs the full random search.
        autotune_kernel = helion.kernel(static_shapes=True, configs=SAFE_CONFIGS)(kernel_fn)

        for batch, num_heads, seq_len, head_dim, dtype in ATTENTION_CONFIGS:
            key = config_key(batch, num_heads, seq_len, head_dim, dtype, variant_name)
            cache_path = os.path.join(CACHE_DIR, f"{key}.json")

            if os.path.exists(cache_path):
                print(f"Skipping {key} (cache exists)")
                continue

            print(f"Autotuning {key} ...")

            q = torch.randn(batch, num_heads, seq_len, head_dim, device="cuda", dtype=dtype)
            k = torch.randn(batch, num_heads, seq_len, head_dim, device="cuda", dtype=dtype)
            v = torch.randn(batch, num_heads, seq_len, head_dim, device="cuda", dtype=dtype)

            best_config = autotune_kernel.autotune((q, k, v), force=False)
            best_config.save(cache_path)
            print(f"  Saved config -> {cache_path}")

            triton_path = os.path.join(CACHE_DIR, f"{key}_triton.py")
            try:
                triton_code = helion.kernel(config=best_config, static_shapes=True)(kernel_fn).bind((q, k, v)).to_triton_code()
                with open(triton_path, "w") as f:
                    f.write(triton_code)
                print(f"  Saved Triton -> {triton_path}")
            except Exception as e:
                print(f"  Could not save Triton code: {e}")
