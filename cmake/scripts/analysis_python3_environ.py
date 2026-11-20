#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Python3 environment analysis (single probe point for the CMake configure stage).

Probes the build interpreter for torch / torch_npu facts and writes a generated
.cmake file consumed by cmake/config.cmake:

    PYTHON3_VERSION_ID                     interpreter major.minor
    PY3_MOD_TORCH_VERSION                  torch.__version__ (raw, local part kept)
    PY3_MOD_TORCH_ROOT_PATH                torch package root
    PY3_MOD_TORCH_CMAKE_DIR                torch.utils.cmake_prefix_path
    PY3_MOD_TORCH_C_GLIBCXX_USE_CXX11_ABI  torch._C._GLIBCXX_USE_CXX11_ABI (0/1)
    PY3_MOD_TORCH_NPU_VERSION              torch_npu.__version__
    PY3_MOD_TORCH_NPU_ROOT_PATH            torch_npu package root

Missing modules yield empty values (probe/policy separation): whether torch or
torch_npu is REQUIRED is decided by the CMake side according to
ENABLE_TESTS / ENABLE_TORCH_BINDINGS. Modeled after pypto's
cmake/scripts/analysis_python3_environ.py.
"""

import argparse
import logging
import os
from pathlib import Path
import sys


class Analysis:
    def __init__(self, args):
        self.output: Path = Path(args.output).resolve()
        self.interpreter_version: str = f"{sys.version_info.major}.{sys.version_info.minor}"
        self.py_mod_torch_version: str = ""
        self.py_mod_torch_root_dir: str = ""
        self.py_mod_torch_cmake_dir: str = ""
        self.py_mod_torch_c_use_cxx11_abi: int = 1
        self.py_mod_torch_npu_version: str = ""
        self.py_mod_torch_npu_root_dir: str = ""
        self._init_torch_param()
        self._init_torch_npu_param()

    def __str__(self) -> str:
        desc = "\nEnviron"
        desc += f"\n  Output  : {self.output}"
        desc += f"\n  Python3 : {sys.executable} ({self.interpreter_version})"
        desc += "\n    torch"
        desc += f"\n      Version                   : {self.py_mod_torch_version}"
        desc += f"\n      CMake_Dir                 : {self.py_mod_torch_cmake_dir}"
        desc += f"\n      Root_Dir                  : {self.py_mod_torch_root_dir}"
        desc += f"\n      _C._GLIBCXX_USE_CXX11_ABI : {self.py_mod_torch_c_use_cxx11_abi}"
        desc += "\n    torch_npu"
        desc += f"\n      Version                   : {self.py_mod_torch_npu_version}"
        desc += f"\n      Root_Dir                  : {self.py_mod_torch_npu_root_dir}"
        return desc

    @staticmethod
    def main():
        """Main processing flow"""
        parser = argparse.ArgumentParser(description="Python3-Environ Analysis.", epilog="Best Regards!")
        parser.add_argument(
            "-o", "--output", type=Path, required=True, help="Specify output .cmake file path.")
        ctrl = Analysis(args=parser.parse_args())
        logging.info("%s", ctrl)
        ctrl.analysis()

    def analysis(self):
        """Write the generated .cmake consumed by cmake/config.cmake."""
        lines = [
            '\n# Python3 Version',
            f'\nset(PYTHON3_VERSION_ID "{self.interpreter_version}")',
            '\nmessage(STATUS "PYTHON3_VERSION_ID=${PYTHON3_VERSION_ID}")',
            '\n',
            '\n# Python3 module torch',
            f'\nset(PY3_MOD_TORCH_VERSION "{self.py_mod_torch_version}")',
            f'\nget_filename_component(PY3_MOD_TORCH_ROOT_PATH "{self.py_mod_torch_root_dir}" REALPATH)',
            f'\nget_filename_component(PY3_MOD_TORCH_CMAKE_DIR "{self.py_mod_torch_cmake_dir}" REALPATH)',
            f'\nset(PY3_MOD_TORCH_C_GLIBCXX_USE_CXX11_ABI {self.py_mod_torch_c_use_cxx11_abi})',
            '\nmessage(STATUS "PY3_MOD_TORCH_VERSION=${PY3_MOD_TORCH_VERSION}")',
            '\nmessage(STATUS "PY3_MOD_TORCH_CMAKE_DIR=${PY3_MOD_TORCH_CMAKE_DIR}")',
            '\n',
            '\n# Python3 module torch_npu',
            f'\nset(PY3_MOD_TORCH_NPU_VERSION "{self.py_mod_torch_npu_version}")',
            f'\nget_filename_component(PY3_MOD_TORCH_NPU_ROOT_PATH "{self.py_mod_torch_npu_root_dir}" REALPATH)',
            '\nmessage(STATUS "PY3_MOD_TORCH_NPU_VERSION=${PY3_MOD_TORCH_NPU_VERSION}")',
            '\nmessage(STATUS "PY3_MOD_TORCH_NPU_ROOT_PATH=${PY3_MOD_TORCH_NPU_ROOT_PATH}")',
            '\n',
        ]
        self.output.parent.mkdir(parents=True, exist_ok=True)
        with open(str(self.output), "w", encoding="utf-8") as f:
            f.writelines(lines)

    def _init_torch_param(self):
        os_env = os.environ.copy()
        try:
            # Keep `import torch` side-effect free: do not autoload device backends.
            os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
            import torch

            self.py_mod_torch_version = str(torch.__version__)
            self.py_mod_torch_root_dir = str(Path(torch.__file__).parent)
            self.py_mod_torch_cmake_dir = str(Path(torch.utils.cmake_prefix_path).resolve())
            self.py_mod_torch_c_use_cxx11_abi = int(torch._C._GLIBCXX_USE_CXX11_ABI)
        except ImportError:
            pass
        finally:
            os.environ.clear()
            os.environ.update(os_env)

    def _init_torch_npu_param(self):
        if not self.py_mod_torch_version:
            return
        try:
            import torch_npu

            self.py_mod_torch_npu_version = str(getattr(torch_npu, "__version__", ""))
            self.py_mod_torch_npu_root_dir = str(Path(torch_npu.__file__).parent)
        except ImportError:
            pass


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s', level=logging.INFO)
    Analysis.main()
