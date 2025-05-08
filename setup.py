import os
import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.install_lib import install_lib


class PluginInstallLib(install_lib):

    def copy_zen_compiler(self):
        src_path = Path(os.getenv("VSI_ZEN_COMPILER_PATH", ""))
        if not src_path.is_file():
            raise RuntimeError(
                "Must set `VSI_ZEN_COMPILER_PATH` to valid zen-compiler executable path"
            )

        dst_path = self.build_backend_dir / "bin" / src_path.name
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src_path, dst_path)

    def copy_zen_tc_bridge(self):
        src_path = Path(os.getenv("VSI_ZEN_TC_BRIDGE_PATH", ""))
        if not src_path.is_file():
            raise RuntimeError(
                "Must set `VSI_ZEN_TC_BRIDGE_PATH` to valid TC bridge library path"
            )

        dst_path = self.build_backend_dir / "lib" / src_path.name
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src_path, dst_path)

    def run(self):
        self.build_backend_dir = \
            Path(self.build_dir) / "triton_vsi_backend" / "backend"
        self.copy_zen_compiler()
        self.copy_zen_tc_bridge()
        super().run()


setup(cmdclass={"install_lib": PluginInstallLib})
