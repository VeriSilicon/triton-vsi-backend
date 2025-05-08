import torch._inductor.lowering  # Fix circular import.
from torch._inductor import config as inductor_config

# TODO: Find proper matmul kernel configs for TensorCore.

# The matmul kernel configs are as follows:
# (BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps)

mm_platform_configs = [
    (16, 16, 64, 1, 1),
    (16, 16, 128, 1, 1),
    (32, 32, 64, 1, 1),
    (32, 32, 128, 1, 1),
]

extra_mm_platform_configs = [
    (64, 64, 128, 1, 1),
]

int8_mm_platform_configs = [
    (32, 32, 64, 1, 1),
]

# Mixed precision kernel configs for small sizes of m for mm's like (16, 8192) x (8192, 8192).

mixed_mm_platform_configs_small_m = [
    (16, 32, 128, 1, 1),
]

mixed_mm_platform_configs = (mm_platform_configs +
                             mixed_mm_platform_configs_small_m
                             if inductor_config.max_autotune_gemm_search_space
                             != "EXHAUSTIVE" else mm_platform_configs)

persistent_mm_platform_configs = [
    (128, 128, 64, 1, 1),
]

scaled_mm_platform_configs = [
    (64, 128, 32, 1, 1),
]
