import os
import functools
import hashlib
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Tuple, NamedTuple, Callable

from triton._C.libtriton import ir, passes
from triton.backends.compiler import BaseBackend, GPUTarget
from triton.runtime.cache import get_cache_manager

_BACKEND_ROOT_DIR = Path(__file__).parent.resolve()


@dataclass(frozen=True)
class VSIOptions:
    backend_name: str = "vsi"
    arch: str = None
    num_warps: int = 2
    num_ctas: int = 1
    shared: int = 0
    num_stages: int = 3
    num_buffers_warp_spec: int = 0
    num_consumer_groups: int = 0
    reg_dec_producer: int = 0
    reg_inc_consumer: int = 0
    enable_warp_specialization: bool = False
    enable_fp_fusion: bool = False
    extern_libs = None
    cluster_dims: Tuple[int, ...] = (1, 1, 1)
    supported_fp8_dtypes: Tuple[str] = ("fp8e5",)
    deprecated_fp8_dtypes: Tuple[str] = ()
    default_dot_input_precision: str = "ieee"
    allowed_dot_input_precisions: Tuple[str] = ("ieee",)
    sanitize_overflow: bool = True
    debug: bool = False
    dump_standalone: bool = False

    def __post_init__(self):
        pass

    def hash(self) -> str:
        key = "_".join([f"{name}-{val}" for name, val in self.__dict__.items()])
        return hashlib.md5(key.encode("utf-8")).hexdigest()


def get_dump_dir(kernel_name: str) -> Path:
    path = os.getenv("VSI_DUMP_DIR")
    if not path:
        raise Exception("VSI_DUMP_DIR is not set.")
    dump_dir = Path(path) / kernel_name
    dump_dir.mkdir(parents=True, exist_ok=True)
    return dump_dir


def _get_zen_tc_bridge_path() -> Path:
    path = os.getenv("VSI_ZEN_TC_BRIDGE_PATH")
    if path:
        return Path(path)
    return _BACKEND_ROOT_DIR / "lib" / "libZenTCBridge.so"


def _get_zen_compiler_path() -> Path:
    path = os.getenv("VSI_ZEN_COMPILER_PATH")
    if path:
        return Path(path)
    return _BACKEND_ROOT_DIR / "bin" / "zen-compiler"


def _is_enable_tc() -> bool:
    is_enable_tc = int(os.getenv("VSI_ENABLE_TC", "0")) == 1
    return is_enable_tc


def _is_enable_multi_engines() -> bool:
    is_enable_multi_engines = int(os.getenv("VSI_ENABLE_MULTI_ENGINES",
                                            "0")) == 1
    return is_enable_multi_engines


def _is_enable_multi_threads() -> bool:
    is_enable_multi_threads = int(os.getenv("VSI_ENABLE_MULTI_THREADS",
                                            "0")) == 1
    return is_enable_multi_threads


def _ttir_to_spv(mod: ir.module, metadata: Dict[str, Any],
                 options: VSIOptions) -> bytes:
    kernel_name = mod.get_entry_func_name()
    metadata["name"] = kernel_name

    hash = metadata["hash"]
    cache = get_cache_manager(hash)

    # Dump triton IR asm to file.
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        ttir_path = tmp_dir / f"{kernel_name}.ttir"
        spv_path = tmp_dir / f"{kernel_name}.spv"
        cmd_gen_path = tmp_dir / f"{kernel_name}_cmd_gen.so"

        ttir_asm = str(mod)
        ttir_path.write_text(ttir_asm)

        if options.dump_standalone:
            dump_dir = get_dump_dir(kernel_name)
            dump_ttir_path = dump_dir / f"{kernel_name}.ttir"
            shutil.copy(ttir_path, dump_ttir_path)

        zen_compiler_path = _get_zen_compiler_path()

        if _is_enable_multi_threads():
            warp_size = metadata["target"].warp_size
            num_warps = metadata["num_warps"]
            num_threads_per_workgroup = num_warps * warp_size
        else:
            metadata["num_warps"] = 0
            num_threads_per_workgroup = 1

        spirv_capabilities = [
            "GenericPointer",
            "Kernel",
            "Addresses",
            "Vector16",
            "Vector32",
            "Int8",
            "Int16",
            "Int64",
            "Float16",
            "TensorCoreVSI",
        ]
        spirv_extensions = [
            "SPV_KHR_storage_buffer_storage_class",
            "SPV_VSI_TC_inst",
        ]

        # fmt: off
        zen_compiler_args = [
            zen_compiler_path,
            "--triton-kernel-compile",
            "--enable-verify-after-passes",
            f"--workgroup-size-options={num_threads_per_workgroup},1,1",
            f"--threads-per-workgroup={num_threads_per_workgroup}",
            "--spirv-target-options="
            f"caps={','.join(spirv_capabilities)} exts={','.join(spirv_extensions)} module=.*",
            ttir_path,
            "-o",
            spv_path,
            "-out-lib",
            cmd_gen_path,
            "-tc-bridge",
            _get_zen_tc_bridge_path()
        ]
        # fmt: on
        if not _is_enable_tc():
            zen_compiler_args.append("--disable-linalg-to-tc-pipeline")
        if not _is_enable_multi_engines():
            zen_compiler_args.append(
                "--disable-multi-engine-scheduling-pipeline")
        if not _is_enable_multi_threads():
            zen_compiler_args.append("--disable-threads-scheduling-pipeline")

        subprocess.check_call(args=zen_compiler_args,
                              executable=zen_compiler_path)

        if options.dump_standalone:
            dump_dir = get_dump_dir(kernel_name)
            dump_spv_path = dump_dir / f"{kernel_name}.spv"
            dump_cmd_gen_path = dump_dir / f"{kernel_name}_cmd_gen.so"

            shutil.copy(spv_path, dump_spv_path)
            shutil.copy(cmd_gen_path, dump_cmd_gen_path)

        cached_cmd_gen_path = cache.put(data=cmd_gen_path.read_bytes(),
                                        filename=f"{kernel_name}_cmd_gen.so",
                                        binary=True)
        metadata["cmd_gen_path"] = cached_cmd_gen_path

        spv_bc = spv_path.read_bytes()
        return spv_bc


class VSIBackend(BaseBackend):
    # TODO(): Generate machine code instead of SPIR-V bytecode.
    binary_ext = "spv"

    @staticmethod
    def supports_target(target: GPUTarget) -> bool:
        return target.backend == "vsi"

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)

    def parse_options(self, opts: Dict[str, Any]) -> VSIOptions:
        args = {"arch": self.target.arch}
        args.update({
            k: opts[k]
            for k in VSIOptions.__dataclass_fields__.keys()
            if k in opts
        })
        return VSIOptions(**args)

    def pack_metadata(self, metadata: NamedTuple) -> Tuple[int, ...]:
        return (
            metadata.num_warps,
            metadata.num_ctas,
            metadata.shared,
            metadata.cluster_dims[0],
            metadata.cluster_dims[1],
            metadata.cluster_dims[2],
        )

    def get_codegen_implementation(self,
                                   options: VSIOptions) -> Dict[str, Callable]:
        codegen_fns = {"min_dot_size": lambda lhs_dtype, rhs_dtype: (16, 16, 8)}
        return codegen_fns

    def load_dialects(self, ctx: ir.context) -> None:
        pass

    @staticmethod
    def make_ttir(mod: ir.module, metadata: Dict[str, Any],
                  options: VSIOptions) -> ir.module:
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

    def add_stages(self, stages: Dict[str, Callable],
                   options: VSIOptions) -> None:
        stages["ttir"] = lambda src, metadata: self.make_ttir(
            src, metadata, options)
        stages["spv"] = lambda src, metadata: _ttir_to_spv(
            src, metadata, options)

    @functools.lru_cache()
    def hash(self) -> str:
        return str(self.target)

    def get_module_map(self) -> Dict[str, ModuleType]:
        from triton_vsi_backend.language.vsi import libdevice
        return {"triton.language.extra.libdevice": libdevice}
