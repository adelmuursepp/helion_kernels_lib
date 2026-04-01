import torch
import torch.nn.functional as F

@torch.compile
def swiglu_pytorch_compile_stacked(x, W):
    both = F.linear(x, W)
    gate = both[:, :both.shape[1]//2]
    up = both[:, both.shape[1]//2:]
    return F.silu(gate) * up

@torch.compile
def swiglu_pytorch_compile_separate(x, W1, W2):
    gate = F.linear(x, W1)
    up = F.linear(x, W2)
    return F.silu(gate) * up
