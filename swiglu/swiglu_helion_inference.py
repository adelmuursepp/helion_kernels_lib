import os
import torch
import helion
from helion import Config
from common import config_key, swiglu_kernel_fn

CACHE_DIR = os.path.join(os.path.dirname(__file__), "autotune_cache")

_kernel_cache = {}


def _make_helion_kernel(config):
    return helion.kernel(config=config)(swiglu_kernel_fn)


def swiglu_helion(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    tokens, d_model = x.shape
    hidden_dim = w1.shape[0]
    key = config_key(tokens, d_model, hidden_dim, x.dtype)

    if key not in _kernel_cache:
        cache_path = os.path.join(CACHE_DIR, f"{key}.json")
        _kernel_cache[key] = _make_helion_kernel(Config.load(cache_path))

    return _kernel_cache[key](x, w1, w2)
