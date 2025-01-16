from triton.backends.compiler import BaseBackend, GPUTarget
from triton._C.libtriton import ir, passes
from dataclasses import dataclass
from typing import Any, Dict, Tuple
from types import ModuleType
import hashlib
import tempfile
import os
import re
import subprocess
import functools
from pathlib import Path

def  _get_zen_compiler_path(bin_name: str) -> str:
    path = os.getenv("ZEN_COMPILER_PATH", "")
    if path == "":
        raise Exception("ZEN_COMPILER_PATH is not set.")
    return os.path.join(path, bin_name)

def _get_tc_toolkits_path() -> str:
    path = os.getenv("TC_TOOLKITS_PATH", "")
    if path == "":
        raise Exception("TC_TOOLKITS_PATH is not set.")
    return path

def _ttir_to_spv(ttir: str, metadata, option):
    # Get Triton-MLIR as string
    with tempfile.TemporaryDirectory() as tmpdir:
        import shutil
        tt_path = os.path.join(tmpdir, "tt.ttir")
        zen_spv_path = os.path.join(tmpdir, "zen.spv")
        Path(tt_path).write_text(str(ttir))
        zen_compiler_path = _get_zen_compiler_path("zen-compiler")

        match = re.search(r"tt.func public\s+@(\w+)\(", Path(tt_path).read_text())
        if match:
            metadata["name"] = match.group(1)
        else:
            raise Exception("cannot match kernel name")
        if option.dump_standalone:
            if not os.path.exists(metadata["name"]):
                os.makedirs(metadata["name"])
            tt = os.path.join(metadata["name"], metadata["name"] + ".ttir")
            shutil.copy(tt_path, tt)

        # Currently VSI not handle multiple local threads
        work_group_size = "1,1,1"
        subprocess.check_call([zen_compiler_path,
            # "--disable-linalg-to-tc-pipeline",
            "--triton-kernel-compile",
            "--enable-verify-after-passes",
            "--workgroup-size-options=" + work_group_size,
            "--spirv-target-options=caps=GenericPointer,Kernel,Addresses,TensorCoreVSI,Vector32,Vector16,Int8,Float16,Int64 exts=SPV_KHR_storage_buffer_storage_class,SPV_VSI_TC_inst module=.*",
            tt_path,
            "-o",
            zen_spv_path], env={"LD_LIBRARY_PATH": f"{_get_tc_toolkits_path()}"})

        if option.dump_standalone:
            if not os.path.exists(metadata["name"]):
                os.makedirs(metadata["name"])
            spv_path = os.path.join(metadata["name"], metadata["name"] + ".spv")
            shutil.copy(zen_spv_path, spv_path)

        with open(zen_spv_path, 'rb') as f:
            content = f.read()
        return content


@dataclass(frozen=True)
class VSIOptions:
    debug: bool = False
    arch: str = None
    num_warps: int = 4
    num_ctas: int = 1
    num_stages: int = 3
    enable_warp_specialization: bool = False
    enable_fp_fusion: bool = False
    extern_libs = None
    cluster_dims: tuple = (1, 1, 1)
    shared: bool = False
    allow_fp8e4nv: bool = False
    backend_name: str = 'vsi'
    dump_standalone: bool = False
    allowed_dot_input_precisions: Tuple[str] = ("ieee", )


    def __post_init__(self):
        pass

    def hash(self):
        key = '_'.join([f'{name}-{val}' for name, val in self.__dict__.items()])
        return hashlib.md5(key.encode("utf-8")).hexdigest()


class VSIBackend(BaseBackend):
    binary_ext = 'spv'

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == 'vsi'

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)

    def parse_options(self, opts) -> Any:
        args = {'arch': self.target.arch}
        args.update({k: opts[k] for k in VSIOptions.__dataclass_fields__.keys() if k in opts})
        return VSIOptions(**args)

    def get_codegen_implementation(self):
        codegen_fns = {"min_dot_size": lambda lhsType, rhsType: (1, 1, 1)}
        return codegen_fns

    def pack_metadata(self, metadata):
        return (
            metadata.num_warps,
            metadata.num_ctas,
            metadata.shared,
            metadata.cluster_dims[0],
            metadata.cluster_dims[1],
            metadata.cluster_dims[2],
        )

    def load_dialects(self, ctx):
        return

    @staticmethod
    def make_ttir(mod, metadata, opt):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_combine(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_licm(pm)
        passes.common.add_symbol_dce(pm)
        pm.run(mod)
        return mod

    def add_stages(self, stages, options):
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["spv"] = lambda src, metadata: _ttir_to_spv(src, metadata, options)


    @functools.lru_cache()
    def hash(self):
        return self.target


    def get_module_map(self) -> Dict[str, ModuleType]:
        return {}
