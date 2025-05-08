# Override upstream matmul kernel configs.
from .mm_common import (
    mm_platform_configs,
    extra_mm_platform_configs,
    int8_mm_platform_configs,
    persistent_mm_platform_configs,
    scaled_mm_platform_configs,
)


def register_vsi_kernel():
    from torch._inductor.kernel import mm_common as inductor_mm_common
    inductor_mm_common.mm_configs.keywords["configs"] = \
        mm_platform_configs
    inductor_mm_common.extra_mm_configs.keywords["configs"] = \
        extra_mm_platform_configs
    inductor_mm_common.int8_mm_configs.keywords["configs"] = \
        int8_mm_platform_configs
    inductor_mm_common.persistent_mm_configs.keywords["configs"] = \
        persistent_mm_platform_configs
    inductor_mm_common.scaled_mm_configs.keywords["configs"] = \
        scaled_mm_platform_configs
