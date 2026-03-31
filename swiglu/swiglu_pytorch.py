import torch
import torch.nn.functional as F

def swiglu_pytorch(x1, x2):
    both = F.linear(x, W)
    gate = both[:, :hidden_dim]
    up = both[:, hidden_dim:]
    return F.silu(gate) * up
    
