from .backend import register_vsi_backend
from .language.vsi import register_vsi_extern_ops

register_vsi_backend()
register_vsi_extern_ops()

from .backend.driver import VSIDriver
if VSIDriver.is_active():
    from ._dynamo import register_vsi_device_interface
    from ._inductor import register_vsi_torch_inductor_backend

    register_vsi_device_interface()
    register_vsi_torch_inductor_backend()
