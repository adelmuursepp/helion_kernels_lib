import torch
import torch.nn.functional as F

@torch.compile
def swiglu_pytorch_compile(x1, x2):
    return F.silu(x1) * x2
