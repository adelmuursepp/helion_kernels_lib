import torch
import torch.nn.functional as F


@torch.compile
def lora_pytorch_compile(x, W, A, B):
    """
    torch.compile version of LoRA: output = x @ W.T + (x @ B.T) @ A.T

    Inductor can fuse the elementwise add and potentially the small B/A matmuls,
    but the two large matmuls (x@W.T and x@B.T) will still be separate kernel
    launches since they are compute-bound GEMMs.
    """
    base = F.linear(x, W)
    lora = F.linear(x, B)
    lora = F.linear(lora, A)
    return base + lora
