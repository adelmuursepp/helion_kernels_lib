import torch
import cutlass
import cutlass.cute as cute
from cutlass import Float32, BFloat16, Int32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack


@dsl_user_op
def silu_f32(x: Float32, *, loc=None, ip=None) -> Float32:
    """
    SiLU(x) = x * sigmoid(x), computed entirely in PTX.

    sigmoid(x) = 1 / (1 + exp(-x))
               = 1 / (1 + 2^(-x * log2(e)))   [since exp(a) = 2^(a * log2(e))]

    PTX instruction ex2.approx.f32 computes 2^x (fast approximation).
    -log2(e) = -1.4426950... stored as IEEE 754 hex: 0fBFB8AA3B
    1.0 stored as IEEE 754 hex: 0f3F800000
    """
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            # $0 = output register, $1 = input x (read-only)
            "mul.f32 $0, $1, 0fBFB8AA3B;\n\t"  # $0 = -x * log2(e)
            "ex2.approx.f32 $0, $0;\n\t"        # $0 = 2^(-x*log2e) = exp(-x)
            "add.f32 $0, $0, 0f3F800000;\n\t"   # $0 = 1 + exp(-x)
            "rcp.approx.f32 $0, $0;\n\t"        # $0 = sigmoid(x)
            "mul.f32 $0, $0, $1;",              # $0 = x * sigmoid(x) = silu(x)
            "=&f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.kernel
def swiglu_kernel_bf16(x1: cute.Tensor, x2: cute.Tensor, out: cute.Tensor, N: Int32):
    tid = cute.arch.thread_idx()[0] + cute.arch.block_idx()[0] * cute.arch.block_dim()[0]
    if tid < N:
        x = Float32(x1[tid])
        y = Float32(x2[tid])
        out[tid] = BFloat16(silu_f32(x) * y)


@cute.kernel
def swiglu_kernel_f32(x1: cute.Tensor, x2: cute.Tensor, out: cute.Tensor, N: Int32):
    tid = cute.arch.thread_idx()[0] + cute.arch.block_idx()[0] * cute.arch.block_dim()[0]
    if tid < N:
        x = Float32(x1[tid])
        y = Float32(x2[tid])
        out[tid] = silu_f32(x) * y


@cute.jit
def run_swiglu_bf16(x1: cute.Tensor, x2: cute.Tensor, out: cute.Tensor, N: Int32):
    BLOCK = 256
    grid = ((N + BLOCK - 1) // BLOCK, 1, 1)
    swiglu_kernel_bf16(x1, x2, out, N).launch(grid=grid, block=(BLOCK, 1, 1))


@cute.jit
def run_swiglu_f32(x1: cute.Tensor, x2: cute.Tensor, out: cute.Tensor, N: Int32):
    BLOCK = 256
    grid = ((N + BLOCK - 1) // BLOCK, 1, 1)
    swiglu_kernel_f32(x1, x2, out, N).launch(grid=grid, block=(BLOCK, 1, 1))


_compiled_cache = {}


def swiglu_cutedsl(x1_torch: torch.Tensor, x2_torch: torch.Tensor) -> torch.Tensor:
    N = x1_torch.numel()
    out_torch = torch.empty_like(x1_torch)

    x1 = from_dlpack(x1_torch.view(-1))
    x2 = from_dlpack(x2_torch.view(-1))
    out = from_dlpack(out_torch.view(-1))

    cache_key = (N, x1_torch.dtype)
    if cache_key not in _compiled_cache:
        if x1_torch.dtype == torch.bfloat16:
            fn = cute.compile(run_swiglu_bf16, x1, x2, out, Int32(N))
        else:
            fn = cute.compile(run_swiglu_f32, x1, x2, out, Int32(N))
        _compiled_cache[cache_key] = fn

    _compiled_cache[cache_key](x1, x2, out, Int32(N))
    return out_torch
