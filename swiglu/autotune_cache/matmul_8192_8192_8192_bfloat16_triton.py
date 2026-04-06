from __future__ import annotations

import torch
import helion
import triton
import triton.language as tl
from helion.runtime import default_launcher as _default_launcher

_BLOCK_SIZE_0 = tl.constexpr(128)
_BLOCK_SIZE_1 = tl.constexpr(32)
_BLOCK_SIZE_2 = tl.constexpr(32)
# src[common.py:18]: def swiglu_kernel_fn(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
# src[common.py:19]:     tokens = x.shape[0]
# src[common.py:20]:     hidden_dim = w1.shape[0]
# src[common.py:18-31]: ...
helion.runtime.set_triton_allocator()

@triton.jit
def _helion_swiglu_kernel_fn(x, w1, w2, out):
    # src[common.py:26]: x_tile   = x[tile_i, tile_k].to(torch.float32)
    x_desc = tl.make_tensor_descriptor(x, [8192, 8192], [8192, 1], [_BLOCK_SIZE_0, _BLOCK_SIZE_2])
    # src[common.py:28]: up_acc   = hl.dot(x_tile, w2[tile_j, tile_k].to(torch.float32).T, acc=up_acc)
    w2_desc = tl.make_tensor_descriptor(w2, [8192, 8192], [8192, 1], [_BLOCK_SIZE_1, _BLOCK_SIZE_2])
    # src[common.py:30]: out[tile_i, tile_j] = (silu_gate * up_acc).to(x.dtype)
    out_desc = tl.make_tensor_descriptor(out, [8192, 8192], [8192, 1], [_BLOCK_SIZE_0, _BLOCK_SIZE_1])
    # src[common.py:22]: for tile_i, tile_j in hl.tile([tokens, hidden_dim]):
    num_pid_m = tl.cdiv(8192, _BLOCK_SIZE_0)
    num_pid_n = tl.cdiv(8192, _BLOCK_SIZE_1)
    inner_2d_pid = tl.program_id(0)
    num_pid_in_group = 8 * num_pid_n
    group_id = inner_2d_pid // num_pid_in_group
    first_pid_m = group_id * 8
    group_size_m = min(num_pid_m - first_pid_m, 8)
    pid_0 = first_pid_m + inner_2d_pid % num_pid_in_group % group_size_m
    pid_1 = inner_2d_pid % num_pid_in_group // group_size_m
    offset_0 = pid_0 * _BLOCK_SIZE_0
    offset_1 = pid_1 * _BLOCK_SIZE_1
    indices_1 = (offset_1 + tl.arange(0, _BLOCK_SIZE_1)).to(tl.int32)
    # src[common.py:23]: gate_acc = hl.zeros([tile_i, tile_j], dtype=torch.float32)
    gate_acc = tl.full([_BLOCK_SIZE_0, _BLOCK_SIZE_1], 0.0, tl.float32)
    # src[common.py:24]: up_acc   = hl.zeros([tile_i, tile_j], dtype=torch.float32)
    up_acc = tl.full([_BLOCK_SIZE_0, _BLOCK_SIZE_1], 0.0, tl.float32)
    # src[common.py:25]: for tile_k in hl.tile(x.shape[1]):
    # src[common.py:26]:     x_tile   = x[tile_i, tile_k].to(torch.float32)
    # src[common.py:27]:     gate_acc = hl.dot( x_tile, w1[tile_j, tile_k].to(torch.float32).T, acc=gate_acc)
    # src[common.py:25-28]: ...
    for offset_2 in tl.range(0, 8192, _BLOCK_SIZE_2, loop_unroll_factor=1, num_stages=4, disallow_acc_multi_buffer=False, flatten=False):
        indices_2 = offset_2 + tl.arange(0, _BLOCK_SIZE_2).to(tl.int32)
        gate_acc_copy = gate_acc
        up_acc_copy = up_acc
        gate_acc_copy_0 = gate_acc_copy
        up_acc_copy_0 = up_acc_copy
        # src[common.py:26]: x_tile   = x[tile_i, tile_k].to(torch.float32)
        load = x_desc.load([offset_0, offset_2])
        v_0 = tl.cast(load, tl.float32)
        # src[common.py:27]: gate_acc = hl.dot( x_tile, w1[tile_j, tile_k].to(torch.float32).T, acc=gate_acc)
        load_1 = tl.load(w1 + (indices_1[:, None] * 8192 + indices_2[None, :] * 1), None)
        v_1 = tl.cast(load_1, tl.float32)
        permute = tl.permute(v_1, [1, 0])
        gate_acc = tl.dot(tl.cast(v_0, tl.float32), tl.cast(permute, tl.float32), acc=gate_acc_copy_0, input_precision='tf32', out_dtype=tl.float32)
        # src[common.py:28]: up_acc   = hl.dot(x_tile, w2[tile_j, tile_k].to(torch.float32).T, acc=up_acc)
        load_2 = w2_desc.load([offset_1, offset_2])
        v_2 = tl.cast(load_2, tl.float32)
        permute_1 = tl.permute(v_2, [1, 0])
        up_acc = tl.dot(tl.cast(v_0, tl.float32), tl.cast(permute_1, tl.float32), acc=up_acc_copy_0, input_precision='tf32', out_dtype=tl.float32)
    # src[common.py:29]: silu_gate = gate_acc * torch.sigmoid(gate_acc)
    v_3 = tl.cast(gate_acc, tl.float32)
    v_4 = tl.sigmoid(v_3)
    v_5 = gate_acc * v_4
    # src[common.py:30]: out[tile_i, tile_j] = (silu_gate * up_acc).to(x.dtype)
    v_6 = v_5 * up_acc
    v_7 = tl.cast(v_6, tl.bfloat16)
    out_desc.store([offset_0, offset_1], tl.cast(v_7, tl.bfloat16))

def swiglu_kernel_fn(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, *, _launcher=_default_launcher):
    # src[common.py:19]: tokens = x.shape[0]
    tokens = x.shape[0]
    # src[common.py:20]: hidden_dim = w1.shape[0]
    hidden_dim = w1.shape[0]
    # src[common.py:21]: out = torch.empty(tokens, hidden_dim, dtype=x.dtype, device=x.device)
    out = torch.empty(tokens, hidden_dim, dtype=x.dtype, device=x.device)
    # src[common.py:22]: for tile_i, tile_j in hl.tile([tokens, hidden_dim]):
    _BLOCK_SIZE_0 = 128
    _BLOCK_SIZE_1 = 32
    # src[common.py:22]: for tile_i, tile_j in hl.tile([tokens, hidden_dim]):
    # src[common.py:23]:     gate_acc = hl.zeros([tile_i, tile_j], dtype=torch.float32)
    # src[common.py:24]:     up_acc   = hl.zeros([tile_i, tile_j], dtype=torch.float32)
    # src[common.py:22-30]: ...
    _launcher(_helion_swiglu_kernel_fn, (triton.cdiv(8192, _BLOCK_SIZE_0) * triton.cdiv(8192, _BLOCK_SIZE_1),), x, w1, w2, out, num_warps=2, num_stages=7)
    # src[common.py:31]: return out
    return out