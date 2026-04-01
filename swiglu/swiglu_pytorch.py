import torch
import torch.nn.functional as F

def swiglu_pytorch(x, w1, w2):
    gate = F.linear(x, w1)
    up = F.linear(x, w2)
    return F.silu(gate) * up
    
