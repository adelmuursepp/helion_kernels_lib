import torch
import torch.nn.functional as F


def lora_pytorch(x, W, A, B):
    """
    Unfused LoRA: output = x @ W.T + (x @ B.T) @ A.T

    Three separate kernel launches:
      1. x @ W.T         -> [tokens, out_dim]  written to gmem
      2. x @ B.T         -> [tokens, rank]     written to gmem
      3. tmp @ A.T + base -> [tokens, out_dim]  written to gmem
    """
    base = F.linear(x, W)        # [tokens, out_dim]
    lora = F.linear(x, B)        # [tokens, rank]
    lora = F.linear(lora, A)     # [tokens, out_dim]
    return base + lora
