from torch._inductor.codegen.common import DeviceOpOverrides


class VsiDeviceOpOverrides(DeviceOpOverrides):

    def import_get_raw_stream_as(self, name: str) -> str:
        return f"from torch.vsi import _get_current_raw_stream as {name}"

    def set_device(self, device_idx) -> str:
        return f"torch.vsi.set_device({device_idx})"

    def synchronize(self) -> str:
        return f"torch.vsi.synchronize()"

    def device_guard(self, device_idx) -> str:
        return f"torch.vsi._DeviceGuard({device_idx})"
