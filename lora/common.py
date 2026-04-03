import torch
import helion.language as hl


# Typical LLM LoRA shapes:
#   tokens  = sequence length
#   in_dim  = input hidden size  (e.g. 4096 for LLaMA-7B)
#   out_dim = output hidden size (Q/K/V/O projections, MLP gates)
#   rank    = LoRA rank (small, e.g. 16 or 64)
LORA_CONFIGS = [
    (512,   4096, 4096, 16,  torch.bfloat16),
    (2048,  4096, 4096, 16,  torch.bfloat16),
    (2048,  4096, 4096, 64,  torch.bfloat16),
    (8192,  4096, 4096, 64,  torch.bfloat16),
    (8192,  8192, 8192, 64,  torch.bfloat16),
    (32768, 4096, 8192, 64,  torch.bfloat16),
]


def config_key(tokens, in_dim, out_dim, rank, dtype):
    return f"lora_{tokens}_{in_dim}_{out_dim}_{rank}_{str(dtype).split('.')[-1]}"


def lora_kernel_fn(
    x: torch.Tensor,
    W: torch.Tensor,
    A: torch.Tensor,
    B_T: torch.Tensor,
) -> torch.Tensor:
    """
    Fused LoRA forward: output = x @ W.T + (x @ B.T) @ A.T

    Inputs
    ------
    x   : [tokens,  in_dim]   – input activations
    W   : [out_dim, in_dim]   – frozen weight (stored row-major as usual for F.linear)
    A   : [out_dim, rank]     – LoRA up-projection
    B_T : [in_dim,  rank]     – LoRA down-projection transposed (B.T, contiguous)

    The kernel tiles over (tokens, out_dim) and iterates the K (in_dim) loop once,
    simultaneously accumulating:
      acc1[tile_m, tile_n]  +=  x_tile @ W_tile.T        (W contribution)
      acc2[tile_m, rank]    +=  x_tile @ B_T_tile         (B contribution, rank-sized)

    After the K-loop, acc2 holds x @ B.T for this m-tile. We then fuse:
      acc1 += acc2 @ A[tile_n, :].T

    This avoids writing the intermediate [tokens, rank] tensor x@B.T to global memory.
    rank must be a compile-time constant (hl.specialize) so acc2 can have a fixed shape.
    """
    tokens, in_dim = x.shape
    out_dim = W.shape[0]
    rank = hl.specialize(B_T.size(1))  # rank is small and fixed — make it compile-time

    out = torch.empty(tokens, out_dim, dtype=x.dtype, device=x.device)

    for tile_m, tile_n in hl.tile([tokens, out_dim]):
        acc1 = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        acc2 = hl.zeros([tile_m, rank],   dtype=torch.float32)

        for tile_k in hl.tile(in_dim):
            x_tile  = x[tile_m, tile_k].to(torch.float32)
            # W[tile_n, tile_k]: [tile_n, tile_k] -> .T -> [tile_k, tile_n]
            acc1 = acc1 + x_tile @ W[tile_n, tile_k].to(torch.float32).T
            # B_T[tile_k, :]: [tile_k, rank] — same pattern as y[tile_k, :] in matmul_layernorm
            acc2 = acc2 + x_tile @ B_T[tile_k, :].to(torch.float32)

        # A[tile_n, :]: [tile_n, rank] -> .T -> [rank, tile_n]
        # acc2 [tile_m, rank] @ [rank, tile_n] = [tile_m, tile_n]
        acc1 = acc1 + acc2 @ A[tile_n, :].to(torch.float32).T

        out[tile_m, tile_n] = acc1.to(x.dtype)

    return out
