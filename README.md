# Triton VeriSilicon Device Backend

Triton backend plugin for VeriSilicon (vsi) GPGPU/NPU device.

## Build

### Requirements

1. Install LLVM and Clang toolchain, version `17` is recommended. See [LLVM Debian/Ubuntu packages](https://apt.llvm.org).
2. Install [VPEX](https://github.com/VeriSilicon/VPEX) PyTorch backend for `vsi` device.
3. Get prebuilt SDK from this repository.

### Build wheel

```sh
export VSI_ZEN_COMPILER_PATH=${path/to/zen-compiler}
export VSI_ZEN_TC_BRIDGE_PATH=${path/to/libZenTCBridge.so}
cd triton_vsi_backend
pip3 wheel --no-build-isolation .
```

### Develop

```sh
pip3 install --no-build-isolation --editable .
```

## Usage

### Setup

Set these environment variables:

- `VSI_SDK_DIR`: The path to VeriSilicon GPGPU/NPU SDK dir.
- `LD_LIBRARY_PATH`: The path for linker to find vsi driver libraries, usually set to `${VSI_SDK_DIR}/drivers`.
- `CC`: The C compiler, `clang` is recommended.

To use the vsi backend:

```python
import vpex
import triton
import triton_vsi_backend

# ... your triton kernel and other code
```

### Use with TorchInductor

Currently PyTorch has no out-of-tree registration mechanism for adding a new triton backend for the Inductor. In order to use our plugin with Inductor, you need to modify the source code in `torch/utils/_triton.py`:

```python
# Locate the `has_triton` method:

@functools.lru_cache(None)
def has_triton() -> bool:
    # ...
    # Around line 82:
    triton_supported_devices = {
        "cuda": cuda_extra_check,
        "xpu": _return_true,
        "cpu": cpu_extra_check,
        "vsi": _return_true, # Add this entry to the dict.
    }
    # ...
```

If your platform has CUDA devices, you need to set env var `CUDA_VISIBLE_DEVICES=""` to disable them, otherwise there will be conflict when running some Inductor passes.

### Dump standalone launcher for debug

For single triton kernel, You can dump a C++ standalone launcher that is capable to run the compiled kernel without the triton python runtime.

Set the kernel metadata parameter `dump_standalone=True` and env var `VSI_DUMP_DIR` to dump the standalone launcher source code and compile artifacts (IRs, binaries, etc) to dir `${VSI_DUMP_DIR}/${kernel_name}`.

Compile the dumped standalone launcher:

```sh
clang++ -Og -glldb -std=c++17 -stdlib=libc++ -fuse-ld=lld ${kernel_name}_standalone.cpp -o ${kernel_name}_standalone -I ${VSI_SDK_INCLUDE_DIR} -L ${VSI_SDK_LIB_DIR} -lOpenCL
```

Run standalone launcher:

```sh
# Prepare your input kernel args (both tensors and scalars) to ./inputs/arg_{i}.npy
export LD_LIBRARY_PATH=${VSI_SDK_LIB_DIR}
./${kernel_name}_standalone -g grid_x,grid_y,grid_z
# If runs OK, the output args (only tensors) are dumped to ./outputs/arg_{i}.npy
```
