import torch
import helion

# Configs aligned with profile_script.py:
# https://github.com/dungwoong/fused_kernels_cutedsl/blob/main/flashattention/attempt6_epi_pipeline/profile_script.py
# total_tokens = 16384 fixed, batch = total_tokens // seq_len, num_heads = 16

# (batch, num_heads, seq_len, head_dim, dtype)
ATTENTION_CONFIGS = [
    (32, 16,  512,  64, torch.bfloat16),
    (32, 16,  512, 128, torch.bfloat16),
    (16, 16, 1024,  64, torch.bfloat16),
    (16, 16, 1024, 128, torch.bfloat16),
    ( 8, 16, 2048,  64, torch.bfloat16),
    ( 8, 16, 2048, 128, torch.bfloat16),
    ( 4, 16, 4096,  64, torch.bfloat16),
    ( 4, 16, 4096, 128, torch.bfloat16),
    ( 2, 16, 8192,  64, torch.bfloat16),
    ( 2, 16, 8192, 128, torch.bfloat16),
]

# Constrained search space to avoid register overflow.
# The attention accumulator is [tile_b, tile_m, head_dim] float32.
# Unlike SwiGLU whose accumulator is 2D, head_dim is a compile-time constant
# (via hl.specialize), so the full tile_m x head_dim block lives in registers.
# Helion's default random search tries tile_m=256/512 which at head_dim=128
# requires 256 registers/thread — H100's per-thread limit — causing
# PassManager::run failed or PTXAS register overflow for nearly every candidate.
# Keeping tile_m <= 64 caps the accumulator at 64 registers/thread (feasible).
# block_sizes = [tile_b, tile_m, tile_n] matching the two hl.tile() calls.
SAFE_CONFIGS = [
    helion.Config(block_sizes=[1, 64,  64]),
    helion.Config(block_sizes=[1, 64, 128]),
    helion.Config(block_sizes=[1, 32,  64]),
    helion.Config(block_sizes=[1, 32, 128]),
    helion.Config(block_sizes=[2, 64,  64]),
    helion.Config(block_sizes=[2, 32,  64]),
]
