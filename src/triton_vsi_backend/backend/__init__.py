from .driver import VSIDriver
from .compiler import VSIBackend

def register_vsi_backend():
    from triton.backends import Backend, backends
    backends["vsi"] = Backend(VSIBackend, VSIDriver)
