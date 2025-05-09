from typing import Any, Dict, Tuple

import torch
import vpex

import triton
from triton import language as tl
from triton.language.extra import libdevice

import triton_vsi_backend


@triton.jit
def elementwise_binary_kernel(
    lhs_ptr,
    rhs_ptr,
    out_ptr,
    num_elements: int,
    BLOCK_SIZE: tl.constexpr,
    IS_EVEN: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if IS_EVEN:
        lhs = tl.load(lhs_ptr + offsets)
        rhs = tl.load(rhs_ptr + offsets)
    else:
        mask = offsets < num_elements
        lhs = tl.load(lhs_ptr + offsets, mask)
        rhs = tl.load(rhs_ptr + offsets, mask)

    diff = lhs - rhs
    relu = tl.maximum(diff, 0)
    out = libdevice.pow(relu * -0.25, 2)

    if IS_EVEN:
        tl.store(out_ptr + offsets, out)
    else:
        tl.store(out_ptr + offsets, out, mask)


def elementwise_binary(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    assert lhs.device == rhs.device, "Operands must be on the same device."
    assert lhs.numel() == rhs.numel(), "Must have the same number of elements."
    assert lhs.dtype == rhs.dtype, "Must have the same data type."
    assert lhs.is_contiguous() and rhs.is_contiguous(
    ), "Operands must be contiguous."

    num_elements = lhs.numel()
    out = torch.empty_like(lhs)

    BLOCK_SIZE = 256
    IS_EVEN = num_elements % BLOCK_SIZE == 0

    def grid(meta: Dict[str, Any]) -> Tuple[int, ...]:
        num_blocks = triton.cdiv(num_elements, BLOCK_SIZE)
        return (num_blocks,)

    elementwise_binary_kernel[grid](lhs, rhs, out, num_elements, BLOCK_SIZE,
                                    IS_EVEN)
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("vsi", 0)

    a = torch.randn(size=(4, 1024), dtype=torch.float32)
    b = torch.randn(size=(4, 1024), dtype=torch.float32)

    out_golden = (torch.relu(a - b) * -0.25) ** 2

    a = a.to(device)
    b = b.to(device)
    out = elementwise_binary(a, b)

    out = out.cpu()
    torch.testing.assert_close(out, out_golden)
