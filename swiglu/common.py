import torch
import helion.language as hl

MATMUL_CONFIGS = [
    (512,   4096, 4096, torch.bfloat16),
    (2048,  4096, 4096, torch.bfloat16),
    (8192,  4096, 4096, torch.bfloat16),
    (8192,  8192, 8192, torch.bfloat16),
    (32768, 4096, 8192, torch.bfloat16),
    (65536, 4096, 8192, torch.bfloat16),
]


def config_key(tokens, d_model, hidden_dim, dtype):
    return f"matmul_{tokens}_{d_model}_{hidden_dim}_{str(dtype).split('.')[-1]}"


def swiglu_kernel_fn(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    tokens = x.shape[0]
    hidden_dim = w1.shape[0]
    out = torch.empty(tokens, hidden_dim, dtype=x.dtype, device=x.device)
    for tile_i, tile_j in hl.tile([tokens, hidden_dim]):
        gate_acc = hl.zeros([tile_i, tile_j], dtype=torch.float32)
        up_acc   = hl.zeros([tile_i, tile_j], dtype=torch.float32)
        for tile_k in hl.tile(x.shape[1]):
            x_tile   = x[tile_i, tile_k].to(torch.float32)
            gate_acc = hl.dot( x_tile, w1[tile_j, tile_k].to(torch.float32).T, acc=gate_acc)
            up_acc   = hl.dot(x_tile, w2[tile_j, tile_k].to(torch.float32).T, acc=up_acc)
        silu_gate = gate_acc * torch.sigmoid(gate_acc)
        out[tile_i, tile_j] = (silu_gate * up_acc).to(x.dtype)
    return out
