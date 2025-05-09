from torch._inductor.codegen.triton import TritonScheduling
from torch._inductor.codegen.wrapper import PythonWrapperCodegen
from torch._inductor.codegen.cpp_wrapper_gpu import CppWrapperGpu
from torch._inductor.codegen.common import (
    register_backend_for_device,
    register_device_op_overrides,
)

from .device_op_overrides import VsiDeviceOpOverrides


def register_vsi_codegen():
    register_backend_for_device("vsi", TritonScheduling, PythonWrapperCodegen,
                                CppWrapperGpu)
    register_device_op_overrides("vsi", VsiDeviceOpOverrides())
