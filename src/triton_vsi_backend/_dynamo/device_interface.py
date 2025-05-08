import torch
from torch._dynamo.device_interface import (
    DeviceInterface,
    _device_t,
    caching_worker_current_devices,
    caching_worker_device_properties,
)


class VsiInterface(DeviceInterface):
    device = torch.vsi.device
    Event = torch.vsi.Event
    Stream = torch.vsi.Stream

    class Worker:

        @staticmethod
        def set_device(device: int):
            caching_worker_current_devices["vsi"] = device

        @staticmethod
        def current_device() -> int:
            if "vsi" in caching_worker_current_devices:
                return caching_worker_current_devices["vsi"]
            return torch.vsi.current_device()

        @staticmethod
        def get_device_properties(device: _device_t = None):
            if device is not None:
                if isinstance(device, str):
                    device = torch.device(device)
                    assert device.type == "vsi"
                if isinstance(device, torch.device):
                    device = device.index
            if device is None:
                device = VsiInterface.Worker.current_device()

            if "vsi" not in caching_worker_device_properties:
                device_prop = [
                    torch.vsi.get_device_properties(i)
                    for i in range(torch.vsi.device_count())
                ]
                caching_worker_device_properties["vsi"] = device_prop

            return caching_worker_device_properties["vsi"][device]

    is_available = staticmethod(torch.vsi.is_available)
    device_count = staticmethod(torch.vsi.device_count)
    current_device = staticmethod(torch.vsi.current_device)
    set_device = staticmethod(torch.vsi.set_device)
    get_compute_capability = staticmethod(torch.vsi.get_compute_capability)
    stream = staticmethod(torch.vsi.stream)
    current_stream = staticmethod(torch.vsi.current_stream)
    set_stream = staticmethod(torch.vsi.set_stream)
    # _set_stream_by_id = staticmethod(torch.vsi._set_stream_by_id)
    get_raw_stream = staticmethod(torch.vsi._get_current_raw_stream)
    synchronize = staticmethod(torch.vsi.synchronize)

    @staticmethod
    def maybe_exchange_device(device: int) -> int:
        # TODO: Impl.
        return device

    @staticmethod
    def exchange_device(device: int) -> int:
        # TODO: Impl.
        return device

    @staticmethod
    def memory_allocated(device: _device_t = None) -> int:
        raise NotImplementedError
