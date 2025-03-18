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
import math
from pathlib import Path


def _get_zen_compiler_path() -> Path:
    path = os.getenv("ZEN_COMPILER_PATH")
    if not path:
        raise Exception("ZEN_COMPILER_PATH is not set.")
    return Path(path)


def _get_cmdgen_path() -> Path:
    # TODO(): Put libcmdgen.so in tmpdir.
    path = os.getenv("CMD_GEN_PATH")
    if not path:
        raise Exception("CMD_GEN_PATH is not set.")
    return Path(path) / "libcmdgen.so"


def _get_tc_bridge_path() -> Path:
    # TODO(): Bundle libZenTCBridge with plugin.
    path = os.getenv("TC_BRIDGE_PATH")
    if not path:
        raise Exception("TC_BRIDGE_PATH is not set.")
    return Path(path) / "libZenTCBridge.so"


def _get_dump_dir() -> Path:
    path = os.getenv("DUMP_DIR")
    if not path:
        raise Exception("DUMP_DIR is not set.")
    return Path(path)


def _is_enable_tc() -> bool:
    is_enable_tc = int(os.getenv("ENABLE_TC", "0")) == 1
    return is_enable_tc


def _is_enable_multi_engines() -> bool:
    is_enable_multi_engines = int(os.getenv("ENABLE_MULTI_ENGINES", "0")) == 1
    return is_enable_multi_engines


def _is_enable_multi_threads() -> bool:
    is_enable_multi_threads = int(os.getenv("ENABLE_MULTI_THREADS", "0")) == 1
    return is_enable_multi_threads


def _ttir_to_spv(mod: ir.module, metadata: Dict[str, Any], option):
    # TODO(): Use mod.get_entry_func_name() in triton 3.3.0.
    ttir_asm = str(mod)
    match = re.search(r"tt.func public\s+@(\w+)\(", ttir_asm)
    if match:
        kernel_name = match.group(1)
        metadata["name"] = kernel_name
    else:
        raise Exception("cannot match kernel name")

    # Dump triton IR asm to file.
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        ttir_path = tmpdir / f"{kernel_name}.ttir"
        zen_spv_path = tmpdir / f"{kernel_name}.spv"

        ttir_path.write_text(ttir_asm)

        if option.dump_standalone:
            dump_dir = _get_dump_dir()
            dump_dir.mkdir(parents=True, exist_ok=True)
            dump_ttir_path = dump_dir / f"{kernel_name}.ttir"
            dump_ttir_path.write_text(ttir_asm)

        zen_compiler_path = _get_zen_compiler_path()

        cluster_dims: Tuple[int, ...] = metadata["cluster_dims"]
        work_group_size = ",".join(map(str, cluster_dims))
        threads_per_workgroup = math.prod(cluster_dims)

        zen_compiler_args = [
            zen_compiler_path,
            "--triton-kernel-compile",
            "--enable-verify-after-passes",
            f"--workgroup-size-options={work_group_size}",
            f"--threads-per-workgroup={threads_per_workgroup}",
            "--spirv-target-options=caps=GenericPointer,Kernel,Addresses,TensorCoreVSI,Vector32,Vector16,Int8,Int16,Float16,Int64 exts=SPV_KHR_storage_buffer_storage_class,SPV_VSI_TC_inst module=.*",
            ttir_path,
            "-o",
            zen_spv_path,
            "-out-lib",
            _get_cmdgen_path(),
            "-tc-bridge",
            _get_tc_bridge_path()
        ]
        if not _is_enable_tc():
            zen_compiler_args.append(
                "--disable-linalg-to-tc-pipeline")
        if not _is_enable_multi_engines():
            zen_compiler_args.append(
                "--disable-multi-engine-scheduling-pipeline")
        if not _is_enable_multi_threads():
            zen_compiler_args.append(
                "--disable-threads-scheduling-pipeline")

        subprocess.check_call(
            args=zen_compiler_args,
            executable=zen_compiler_path
        )

        if option.dump_standalone:
            dump_dir = _get_dump_dir()
            dump_dir.mkdir(parents=True, exist_ok=True)
            dump_spv_path = dump_dir / f"{kernel_name}.spv"
            import shutil
            shutil.copy(zen_spv_path, dump_spv_path)

        with open(zen_spv_path, "rb") as f:
            content = f.read()
        return content


@dataclass(frozen=True)
class VSIOptions:
    debug: bool = False
    arch: str = None
    num_warps: int = 4
    num_ctas: int = 1
    num_stages: int = 3
    num_buffers_warp_spec: int = 0
    num_consumer_groups: int = 0
    reg_dec_producer: int = 0
    reg_inc_consumer: int = 0
    enable_warp_specialization: bool = False
    enable_fp_fusion: bool = False
    extern_libs = None
    cluster_dims: Tuple[int, ...] = (1, 1, 1)
    shared: bool = False
    allow_fp8e4nv: bool = False
    backend_name: str = "vsi"
    dump_standalone: bool = False
    allowed_dot_input_precisions: Tuple[str] = ("ieee", )
    sanitize_overflow: bool = True

    def __post_init__(self):
        pass

    def hash(self):
        key = "_".join([f"{name}-{val}" for name,
                       val in self.__dict__.items()])
        return hashlib.md5(key.encode("utf-8")).hexdigest()


class VSIBackend(BaseBackend):
    binary_ext = "spv"

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == "vsi"

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)

    def parse_options(self, opts) -> Any:
        args = {"arch": self.target.arch}
        args.update(
            {k: opts[k] for k in VSIOptions.__dataclass_fields__.keys() if k in opts})
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
    def make_ttir(mod: ir.module, metadata: Dict[str, Any], opt):
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
        stages["ttir"] = lambda src, metadata: self.make_ttir(
            src, metadata, options)
        stages["spv"] = lambda src, metadata: _ttir_to_spv(
            src, metadata, options)

    @functools.lru_cache()
    def hash(self):
        return self.target

    def get_module_map(self) -> Dict[str, ModuleType]:
        from triton.language.extra.vsi import libdevice
        return {"triton.language.extra.libdevice": libdevice}
