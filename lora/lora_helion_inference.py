import os
import torch
import helion
from helion import Config
from common import config_key, lora_kernel_fn

CACHE_DIR = os.path.join(os.path.dirname(__file__), "autotune_cache")

_kernel_cache = {}


def _make_helion_kernel(config):
    return helion.kernel(config=config)(lora_kernel_fn)


def lora_helion(x: torch.Tensor, W: torch.Tensor, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Public API matches all other lora_* functions: (x, W, A, B).
    Internally transposes B to B_T = [in_dim, rank] so the helion kernel
    can index it as B_T[tile_k, :] — the same proven pattern as y[tile_k, :]
    in helion's matmul_layernorm example.
    """
    tokens, in_dim = x.shape
    out_dim = W.shape[0]
    rank = B.shape[0]
    key = config_key(tokens, in_dim, out_dim, rank, x.dtype)

    if key not in _kernel_cache:
        cache_path = os.path.join(CACHE_DIR, f"{key}.json")
        _kernel_cache[key] = _make_helion_kernel(Config.load(cache_path))

    B_T = B.T.contiguous()  # [in_dim, rank]
    return _kernel_cache[key](x, W, A, B_T)
