import torch
from torch._dynamo.device_interface import register_interface_for_device

from .device_interface import VsiInterface


def register_vsi_device_interface():
    register_interface_for_device("vsi", VsiInterface)
    for i in range(torch.vsi.device_count()):
        register_interface_for_device(f"vsi:{i}", VsiInterface)
