from typing import Any, Dict, Tuple

import torch
import vpex

import triton
from triton import language as tl

import triton_vsi_backend


@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    m: int,
    k: int,
    n: int,
    stride_a_m: int,
    stride_a_k: int,
    stride_b_k: int,
    stride_b_n: int,
    stride_out_m: int,
    stride_out_n: int,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    IS_EVEN: tl.constexpr,
):
    pid_m = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=0)

    offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets_k = tl.arange(0, BLOCK_SIZE_K)

    offsets_a = offsets_m[:, None] * stride_a_m + \
        offsets_k[None, :] * stride_a_k
    offsets_b = offsets_k[:, None] * stride_b_k + \
        offsets_n[None, :] * stride_b_n

    if IS_EVEN:
        a = tl.load(a_ptr + offsets_a)
        b = tl.load(b_ptr + offsets_b)
    else:
        mask_a = (offsets_m < m)[:, None]
        mask_b = (offsets_n < n)[None, :]
        a = tl.load(a_ptr + offsets_a, mask_a, other=0)
        b = tl.load(b_ptr + offsets_b, mask_b, other=0)

    out = tl.dot(a, b, out_dtype=a.dtype)

    offsets_out = offsets_m[:, None] * stride_out_m + \
        offsets_n[None, :] * stride_out_n

    if IS_EVEN:
        tl.store(out_ptr + offsets_out, out)
    else:
        mask_out = mask_a & mask_b
        tl.store(out_ptr + offsets_out, out, mask_out)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.device == b.device, "Operands must be on the same device."
    assert len(a.shape) == 2 and len(b.shape) == 2, "Must be 2D tensor."
    assert a.shape[1] == b.shape[0], "Incompatible contracting dimensions."

    m, k = a.shape
    k, n = b.shape

    out = torch.empty(size=(m, n), dtype=a.dtype, device=a.device)

    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    IS_EVEN = (m % BLOCK_SIZE_M == 0) and (n % BLOCK_SIZE_N == 0)

    def grid(meta: Dict[str, Any]) -> Tuple[int, ...]:
        num_blocks_m = triton.cdiv(m, BLOCK_SIZE_M)
        num_blocks_n = triton.cdiv(n, BLOCK_SIZE_N)
        return (num_blocks_n, num_blocks_m)

    # fmt: off
    matmul_kernel[grid](
        a, b, out,
        m, k, n,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, k,
        IS_EVEN)
    # fmt: on
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("vsi", 0)

    m, k, n = 128, 32, 64
    a = torch.randn(size=(m, k), dtype=torch.float16)
    b = torch.randn(size=(k, n), dtype=torch.float16)

    out_golden = a @ b

    a = a.to(device)
    b = b.to(device)
    out = matmul(a, b)

    out = out.cpu()
    torch.testing.assert_close(out, out_golden)
