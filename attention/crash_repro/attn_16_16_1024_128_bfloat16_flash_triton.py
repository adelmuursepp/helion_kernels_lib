from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_compat import libdevice
from helion.runtime import default_launcher as _default_launcher

_BLOCK_SIZE_1 = tl.constexpr(64)
_BLOCK_SIZE_0 = tl.constexpr(1)
_BLOCK_SIZE_3 = tl.constexpr(16)

@triton.jit
def _helion_attention_kernel_fn(q_view, k_view, v_view, out, _RDIM_SIZE_2: tl.constexpr):
    # src[helion_common.py:30]: for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
    num_blocks_0 = 256
    pid_0 = tl.program_id(0) % num_blocks_0
    pid_1 = tl.program_id(0) // num_blocks_0
    offset_0 = pid_0
    indices_0 = offset_0 + tl.zeros([1], tl.int32)
    offset_1 = pid_1 * _BLOCK_SIZE_1
    indices_1 = (offset_1 + tl.arange(0, _BLOCK_SIZE_1)).to(tl.int32)
    indices_2 = tl.arange(0, _RDIM_SIZE_2).to(tl.int32)
    # src[helion_common.py:31]: m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
    m_i = tl.full([_BLOCK_SIZE_0, _BLOCK_SIZE_1], float('-inf'), tl.float32)
    # src[helion_common.py:32]: l_i = torch.full_like(m_i, 1.0)
    l_i = tl.full([_BLOCK_SIZE_0, _BLOCK_SIZE_1], 1.0, tl.float32)
    # src[helion_common.py:33]: acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
    acc = tl.full([_BLOCK_SIZE_0, _BLOCK_SIZE_1, 128], 0.0, tl.float32)
    # src[helion_common.py:34]: q = q_view[tile_b, tile_m, :]
    q = tl.load(q_view + (indices_0[:, None, None] * 131072 + indices_1[None, :, None] * 128 + indices_2[None, None, :] * 1), None, eviction_policy='evict_last')
    # src[helion_common.py:36]: for tile_n in hl.tile(v_view.size(1)):
    # src[helion_common.py:37]:     k = k_view[tile_b, :, tile_n]
    # src[helion_common.py:38]:     qk = torch.bmm(q, k)
    # src[helion_common.py:36-49]: ...
    for offset_3 in tl.range(0, 1024, _BLOCK_SIZE_3, loop_unroll_factor=3, num_stages=1, flatten=True):
        indices_3 = offset_3 + tl.arange(0, _BLOCK_SIZE_3).to(tl.int32)
        q_copy = q
        m_i_copy = m_i
        l_i_copy = l_i
        acc_copy = acc
        q_copy_0 = q_copy
        m_i_copy_0 = m_i_copy
        l_i_copy_0 = l_i_copy
        acc_copy_0 = acc_copy
        # src[helion_common.py:37]: k = k_view[tile_b, :, tile_n]
        k = tl.load(k_view + (indices_0[:, None, None] * 131072 + indices_2[None, :, None] * 1 + indices_3[None, None, :] * 128), None, eviction_policy='evict_last')
        # src[helion_common.py:38]: qk = torch.bmm(q, k)
        qk = tl.cast(tl.reshape(tl.dot(tl.reshape(tl.cast(q_copy_0, tl.bfloat16), [_BLOCK_SIZE_1, 128]), tl.reshape(tl.cast(k, tl.bfloat16), [128, _BLOCK_SIZE_3]), input_precision='tf32', out_dtype=tl.float32), [_BLOCK_SIZE_0, _BLOCK_SIZE_1, _BLOCK_SIZE_3]), tl.bfloat16)
        # src[helion_common.py:39]: m_ij = torch.maximum(m_i, torch.amax(qk, -1) * qk_scale)
        amax = tl.cast(tl.max(qk, 2), tl.bfloat16)
        v_0 = tl.full([], 0.12751743074602467, tl.float32)
        v_1 = tl.cast(amax * v_0, tl.bfloat16)
        v_2 = tl.cast(v_1, tl.float32)
        v_3 = triton_helpers.maximum(m_i_copy_0, v_2)
        # src[helion_common.py:40]: qk = qk * qk_scale - m_ij[:, :, None]
        v_4 = tl.full([], 0.12751743074602467, tl.float32)
        v_5 = tl.cast(qk * v_4, tl.bfloat16)
        subscript = v_3[:, :, None]
        v_6 = tl.cast(v_5, tl.float32)
        v_7 = v_6 - subscript
        # src[helion_common.py:41]: p = torch.exp2(qk)
        v_8 = libdevice.exp2(v_7)
        # src[helion_common.py:42]: l_ij = torch.sum(p, -1)
        l_ij = tl.cast(tl.sum(v_8, 2), tl.float32)
        # src[helion_common.py:43]: alpha = torch.exp2(m_i - m_ij)
        v_9 = m_i_copy_0 - v_3
        v_10 = libdevice.exp2(v_9)
        # src[helion_common.py:44]: l_i = l_i * alpha + l_ij
        v_11 = l_i_copy_0 * v_10
        l_i = v_11 + l_ij
        # src[helion_common.py:45]: acc = acc * alpha[:, :, None]
        subscript_1 = v_10[:, :, None]
        v_13 = acc_copy_0 * subscript_1
        # src[helion_common.py:46]: v = v_view[tile_b, tile_n, :]
        v = tl.load(v_view + (indices_0[:, None, None] * 131072 + indices_3[None, :, None] * 128 + indices_2[None, None, :] * 1), None, eviction_policy='evict_first')
        # src[helion_common.py:47]: p = p.to(v.dtype)
        v_14 = tl.cast(v_8, tl.bfloat16)
        # src[helion_common.py:48]: acc = torch.baddbmm(acc, p, v)
        acc = tl.reshape(tl.dot(tl.reshape(tl.cast(v_14, tl.bfloat16), [_BLOCK_SIZE_1, _BLOCK_SIZE_3]), tl.reshape(tl.cast(v, tl.bfloat16), [_BLOCK_SIZE_3, 128]), acc=tl.reshape(v_13, [_BLOCK_SIZE_1, 128]), input_precision='tf32', out_dtype=tl.float32), [_BLOCK_SIZE_0, _BLOCK_SIZE_1, 128])
        # src[helion_common.py:49]: m_i = m_ij
        m_i = v_3
    # src[helion_common.py:52]: acc = acc / l_i[:, :, None]
    subscript_2 = l_i[:, :, None]
    v_15 = acc / subscript_2
    # src[helion_common.py:53]: out[tile_b, tile_m, :] = acc.to(out.dtype)
    v_16 = tl.cast(v_15, tl.bfloat16)
    tl.store(out + (indices_0[:, None, None] * 131072 + indices_1[None, :, None] * 128 + indices_2[None, None, :] * 1), v_16, None)

def attention_kernel_fn(q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor, *, _launcher=_default_launcher):
    # src[helion_common.py:16]: m_dim = q_in.size(-2)
    m_dim = q_in.size(-2)
    # src[helion_common.py:17]: n_dim = k_in.size(-2)
    n_dim = k_in.size(-2)
    # src[helion_common.py:18]: head_dim = hl.specialize(q_in.size(-1))
    head_dim = 128
    # src[helion_common.py:20]: q_view = q_in.reshape([-1, m_dim, head_dim])
    q_view = q_in.reshape([-1, m_dim, head_dim])
    # src[helion_common.py:21]: v_view = v_in.reshape([-1, n_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    # src[helion_common.py:22]: k_view = k_in.reshape([-1, n_dim, head_dim]).transpose(1, 2)
    k_view = k_in.reshape([-1, n_dim, head_dim]).transpose(1, 2)
    # src[helion_common.py:23]: out = torch.empty_like(q_view)
    out = torch.empty_like(q_view)
    # src[helion_common.py:30]: for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
    _BLOCK_SIZE_1 = 64
    _RDIM_SIZE_2 = 128
    # src[helion_common.py:30]: for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
    # src[helion_common.py:31]:     m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
    # src[helion_common.py:32]:     l_i = torch.full_like(m_i, 1.0)
    # src[helion_common.py:30-53]: ...
    _launcher(_helion_attention_kernel_fn, (256 * triton.cdiv(1024, _BLOCK_SIZE_1),), q_view, k_view, v_view, out, _RDIM_SIZE_2, num_warps=32, num_stages=1)
    # src[helion_common.py:55]: return out.view(q_in.size())
    return out.view(q_in.size())