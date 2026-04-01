import torch
import cutlass
import cutlass.cute as cute
from cutlass import Float32, BFloat16
from cutlass.cute.runtime import from_dlpack
import cutlass.cute.sm90_utils as sm90_utils

#Activation function
def silu_f32(x: Float32) -> Float32:
    """SiLU(x) = x / (1 + exp(-x)), using fast 2^x hardware instruction."""
    LOG2_E = Float32(1.4426950408889634)
    exp_neg_x = cute.math.exp2(x * Float32(-1.0) * LOG2_E, fastmath=True)
    return x / (Float32(1.0) + exp_neg_x)

class SwiGLUHopperNoPipeline:
    """
    One warpgroup (128 threads) per block.
    K-loop: cp.async loads X/W1/W2 tiles → syncthreads → WGMMA gate+up.
    Epilogue: SiLU(gate) * up written directly from registers to gmem.
    """
    def __init__(self, tile_m: int = 64, tile_n: int = 128, tile_k: int = 64):
        self.tile_m = tile_m # constrained by the instruction below to 64
        self.tile_n = tile_n # constrained by instruction below to 128
        self.tile_k = tile_k

        # Major.K for both will do transpose during matmul internally
        self.tiled_mma = sm90_utils.make_trivial_tiled_mma(
            cute.SM90_64x128x16_F32BF16BF16_SS(
                cute.GMMA.Major.K, cute.GMMA.Major.K
            ),
            # standard warps layout: cooperate on columns but handle diff rows
            warp_group_thread_layout=cute.make_layout((4,1)),
        )
        # swizzle, informed by the mma instruction for spreading across banks
        # for X
        self.smem_layout_a = sm90_utils.make_smem_layout_a(self.tiled_mma,
            (tile_m, tile_k), cute.BF16)
        # for W1/W2
        self.smem_layout_b = sm90_utils.make_smem_layout_b(self.tiled_mma,
            (self.tile_n, self.tile_k), cute.BF16)


    @cute.jit
    def __call__(
            self, 
            mX: cute.Tensor, 
            mW1: cute.Tensor,
            mW2: cute.Tensor, 
            mOut: cute.Tensor, 
            stream,
            ):
        tokens = mX.shape[0]
        hidden_dim = mW1.shape[0]

        # grid per output tile
        grid = (
            (tokens + self.tile_m - 1) // self.tile_m,
            (hidden_dim + self.tile_n - 1) // self.tile_n, 
            1
        )
        block = (128, 1, 1) # 4 warps per block

        smem_a = self.tile_m * self.tile_k * 2 
        smem_b = self.tile_n * self.tile_k * 2
        smem_bytes = smem_a + smem_b + smem_b # bytes of x, W1, W2

        # pass default pytorch stream so it stays in sync with pytorch ops
        # instantiate and launch
        self.kernel(
                mX, mW1, mW2, mOut,
                self.tiled_mma, 
                self.smem_layout_a, self.smem_layout_b, 
                self.tile_m, self.tile_n, self.tile_k, 
        ).launch(grid=grid, block=block, smem=smem_bytes, stream=stream)

    @cute.kernel
    def kernel(self, mX, mW1, mW2, mOut, tiled_mma, smem_layout_a, smem_layout_b, 
               TILE_M, TILE_N, TILE_K):
        bm = cute.arch.block_idx()[0]
        bn = cute.arch.block_idx()[1]
        tid = cute.arch.thread_idx()[0]

        # Iterator over tile k
        gX = cute.local_tile(mX, (TILE_M, TILE_K), (bm, cute.UNDERSCORE))
        gW1 = cute.local_tile(mW1, (TILE_N, TILE_K), (bn, cute.UNDERSCORE))
        gW2 = cute.local_tile(mW2, (TILE_N, TILE_K), (bn, cute.UNDERSCORE))

        # Output tile is fixed
        gOut = cute.local_tile(mOut, (TILE_M, TILE_N), (bm, bn))


        smem_a_bytes = TILE_M * TILE_K * 2
        smem_b_bytes = TILE_N * TILE_K * 2
        sX  = cute.make_tensor(
            cute.make_smem_ptr(cute.BF16, 0), smem_layout_a)
        sW1 = cute.make_tensor(
            cute.make_smem_ptr(cute.BF16, smem_a_bytes), smem_layout_b)
        sW2 = cute.make_tensor(
            cute.make_smem_ptr(cute.BF16, smem_a_bytes + smem_b_bytes), smem_layout_b)

        # each thread owns (TILE_M * TILE_N) / 128 = 64 float32 values in registers
        # cute.clear zeroes them: cute.gemm adds into existing register values
        # two accumulators: gate for X @ W1.T, up for X @ W2.T, both built across k-loop
        gate_acc = cute.partition_fragment_C(tiled_mma, (TILE_M, TILE_N))
        up_acc   = cute.partition_fragment_C(tiled_mma, (TILE_M, TILE_N))
        cute.clear(gate_acc)
        cute.clear(up_acc)

        g2s_tiled = cute.make_tiled_copy(
            cute.Copy_Atom(cute.SM80_CP_ASYNC_CACHEGLOBAL(), cute.BF16),
            cute.make_layout((16, 8)),  # thread layout: 16 rows x 8 threads/row = 128 threads
            cute.make_layout((1, 8)),   # val layout: each thread loads 1 row x 8 bf16 = 128 bits
        )
        thr_copy = g2s_tiled.get_slice(tid)

        k_tiles = gX.shape[2]
        for k in range(k_tiles):
            # partition_S: slice of gmem this thread reads from
            # partition_D: slice of smem this thread writes into
            cute.copy(g2s_tiled, thr_copy.partition_S(gX[:, :, k]),  thr_copy.partition_D(sX))
            cute.copy(g2s_tiled, thr_copy.partition_S(gW1[:, :, k]), thr_copy.partition_D(sW1))
            cute.copy(g2s_tiled, thr_copy.partition_S(gW2[:, :, k]), thr_copy.partition_D(sW2))
            cute.arch.cp_async_commit_group()  # mark end of this batch of async copies
            cute.arch.cp_async_wait_all()      # stall until copies land in smem
            cute.arch.syncthreads()            # all threads see smem filled by all others

            # WGMMA: gate_acc += X_tile @ W1_tile.T
            #        up_acc   += X_tile @ W2_tile.T
            tAs  = cute.partition_A(tiled_mma, sX)
            tBs1 = cute.partition_B(tiled_mma, sW1)
            tBs2 = cute.partition_B(tiled_mma, sW2)
            cute.gemm(tiled_mma, gate_acc, tAs, tBs1, gate_acc)
            cute.gemm(tiled_mma, up_acc,   tAs, tBs2, up_acc)
            cute.arch.syncthreads()  # protect smem before next k iteration's load

        # partition_C maps each register in gate_acc/up_acc to its output element in gOut
        tOut = cute.partition_C(tiled_mma, gOut)
        for i in range(cute.size(gate_acc)):
            tOut[i] = BFloat16(silu_f32(gate_acc[i]) * up_acc[i])


# ── Python wrapper ────────────────────────────────────────────────────────────

_kernel_cache = {}

def swiglu_cutedsl(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    tokens, d_model = x.shape
    hidden_dim = w1.shape[0]
    out = torch.empty(tokens, hidden_dim, dtype=x.dtype, device=x.device)

    mX   = from_dlpack(x)
    mW1  = from_dlpack(w1)
    mW2  = from_dlpack(w2)
    mOut = from_dlpack(out)
    stream = torch.cuda.current_stream().cuda_stream

    key = (tokens, d_model, hidden_dim, x.dtype)
    if key not in _kernel_cache:
        op = SwiGLUHopperNoPipeline()
        _kernel_cache[key] = cute.compile(op, mX, mW1, mW2, mOut, stream)

    _kernel_cache[key](mX, mW1, mW2, mOut, stream)
    return out
