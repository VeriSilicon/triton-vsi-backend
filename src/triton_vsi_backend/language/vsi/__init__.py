from importlib.util import module_from_spec
import sys

from . import libdevice


def register_vsi_extern_ops():
    module = module_from_spec(__spec__)
    sys.modules["triton.language.extra.vsi"] = module


__all__ = ["libdevice"]
