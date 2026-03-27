import torch
import torch.nn.functional as F

def swiglu_pytorch(x1, x2):
    return F.silu(x1) * x2
    
