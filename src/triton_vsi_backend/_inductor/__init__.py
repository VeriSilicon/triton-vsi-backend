from .codegen import register_vsi_codegen
from .kernel import register_vsi_kernel


def register_vsi_torch_inductor_backend():
    from torch._inductor import utils as inductor_utils
    inductor_utils.GPU_TYPES.append("vsi")
    inductor_utils.GPU_KERNEL_BIN_EXTS["vsi"] = ".spv"

    # HACK: Bypass upstream check to use triton template for vsi device.
    inductor_utils._use_template_for_gpu = \
        lambda layout, allowed_dtypes: layout.device.type == "vsi"

    register_vsi_codegen()
    register_vsi_kernel()
