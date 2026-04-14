# Helper script for reproducing failing configs in Helion to Triton
import os
import torch
import helion
from common import ATTENTION_CONFIGS
from helion_common import config_key, VARIANTS

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "crash_repro")
config_for_repro = helion.Config(
    block_sizes=[1, 64, 16],
    indexing=['pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'],
    l2_groupings=[16],
    load_eviction_policies=['last', 'first', 'last'],
    loop_orders=[[0, 1]],
    num_stages=6,
    num_warps=16,
    pid_type='flat',
    range_flattens=[None, True],
    range_multi_buffers=[None, None],
    range_num_stages=[0, 4],
    range_unroll_factors=[0, 3],
    range_warp_specializes=[])

config_idx_to_repro = 3

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for variant_name, kernel_fn in VARIANTS:

        for i, (batch, num_heads, seq_len, head_dim, dtype) in enumerate(ATTENTION_CONFIGS):
            if i != config_idx_to_repro:
                continue

            key = config_key(batch, num_heads, seq_len, head_dim, dtype, variant_name)
            print(f"Running {key} with failing config ...")

            q = torch.randn(batch, num_heads, seq_len, head_dim, device="cuda", dtype=dtype)
            k = torch.randn(batch, num_heads, seq_len, head_dim, device="cuda", dtype=dtype)
            v = torch.randn(batch, num_heads, seq_len, head_dim, device="cuda", dtype=dtype)

            # Save Triton code first (compile only, no GPU run)
            try:
                triton_code = helion.kernel(config=config_for_repro, static_shapes=True)(kernel_fn).bind((q, k, v)).to_triton_code()
                triton_path = os.path.join(OUTPUT_DIR, f"{key}_triton.py")
                with open(triton_path, "w") as f:
                    f.write(triton_code)
                print(f"  Saved Triton -> {triton_path}")
            except Exception as e:
                print(f"  Could not save Triton code: {e}")

            # Then actually run it to trigger the crash
            try:
                helion.kernel(config=config_for_repro, static_shapes=True)(kernel_fn)(q, k, v)
                print(f"  Ran successfully (no crash)")
            except Exception as e:
                print(f"  Crashed on GPU run: {e}")
