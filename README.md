# Triton Verisilicon Device Backend
This repo is the plugin backend of triton, just like amd/nvidia backend,  
to support compile and execute triton kernels on Verisilicon's device.

## Installation
### Requirements
1. Before install the plugin, you need install llvm tools at first. Recommend the llvm-17 version.  
2. Triton use torch tensor as input, and this backend take "vsi" tensor only, make sure pytorch [vpex](https://github.com/VeriSilicon/VPEX) plugin installed.
3. Install [ZenCompiler](https://github.com/VeriSilicon/ZenCompiler.git) as the triton IR compiler.
4. And the plugin use [cnpy](https://github.com/rogersce/cnpy) as the numpy file reader/writer in C/C++ code, if you need debug the dumped standalone code, install cnpy at first.

### Install with soft link
Assume that you've install python triton package in your environment, you can just install the plugin by create a soft link.  
``` bash
cd <your_env/lib/python3.x/site-packages/triton/backends>
ln -s <this_repo_root/backend> vsi
```

## Usage
### Environment variables
- VSI_DRIVER_PATH: the path to verisilicon sdk drivers
- TC_TOOLKITS_PATH: the path to verisilicon TensorCore toolkits
- ZEN_COMPILER_PATH: the path to zen_compiler executable
- CC: the C/C++ compiler used of backend, need use clang++ in vsi backend
``` python
import torch
import triton
import vpex
from triton.backends.vsi.driver import VSIDriver
triton.runtime.driver.set_active(VSIDriver())
# ... your triton kernel and other code
```

### Dump standalone for debug
For single triton kernel, You can dump the triton_launcher C/C++ code as a compile-able standalone source code.  
Set the meta parameter `dump_standalone=True` to dump the source code.
And do not specialize those kernel arguments equal to one like this:
``` python
@triton.jit(do_not_specialize=["arg0", "arg1", ...])
# ... your kernel
```

Compile standalone source like:  
``` bash
export LD_LIBRARY_PATH=<verisilicon_drivers_path>
clang++ -g <source_path> -O0 -fPIC -o <out_name> -l OpenCL -l cnpy -I <verisilicon_herders_path>
```
