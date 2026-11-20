#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""PyPTO-GYM setuptools configuration module

Two wheel forms, selected by exactly one environment variable (both produced
by build_ci.py; bare `python -m build` is rejected with a pointer):

- GYM_WHL_STAGE=<dir>  community full wheel: Python shell + _bindings materials
  (csrc/frontends/manifest) + _opp vendor tree (precompiled kernels / tiling /
  aclnn). Tag py3-none-<plat>: platform binaries aboard, but no Python/torch
  ABI lock (bindings are JIT compiled at runtime, decision 28). The plat tag
  defaults to linux_<arch>; pass a manylinux plat via build_ci.py --plat_name
  (forwarded as `bdist_wheel --plat-name=...`) for PyPI-compatible tags.
  Version <base>.
- GYM_WHL_SHELL=<dir>  run-embedded shell wheel: Python shell + _bindings only,
  no vendor binaries. Tag py3-none-any (fully platform independent, shared by
  x86/aarch64 run packages). Version <base>.

The staged tree layout (materialized by `cmake --install`; the repo source tree
under python/pypto_gym/ mirrors it 1:1) is <dir>/pypto_gym/{__init__.py,
_loader/,_bindings/{operators.json,csrc/,frontends/}[,_opp/]}.
"""

import logging
import os
from pathlib import Path
from typing import List
from datetime import datetime, timezone

import setuptools
from setuptools.dist import Distribution
from setuptools.command.egg_info import egg_info as _egg_info

try:
    from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel
except ImportError:  # older setuptools: fall back to the wheel package
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel


class BinaryDistribution(Distribution):
    """Force a platform-specific wheel (Root-Is-Purelib: false).

    The full form ships the vendor tree (kernel .o, host libraries) as package
    data; without this, bdist_wheel would tag the wheel py3-none-any and pip
    would install it on incompatible platforms.
    """

    def has_ext_modules(self):
        return True


class Py3NonePlatformWheel(_bdist_wheel):
    """Tag the full form py3-none-<plat>.

    The binaries aboard are Python/torch ABI free (device kernels + C-API host
    libraries; bindings are JIT compiled against the runtime torch), so the
    cpXX-cpXX interpreter lock of the default platform tag would wrongly
    reject compatible interpreters (decision 22/28).
    """

    def finalize_options(self):
        super().finalize_options()
        self.python_tag = "py3"

    def get_tag(self):
        _python, _abi, plat = super().get_tag()
        return "py3", "none", plat


class CustomEggInfo(_egg_info):
    """Inject X-Build-Timestamp into PKG-INFO (bdist_wheel copies it to METADATA).

    setuptools has no public API for custom METADATA fields; following pypto's
    CustomEggInfo, the field is inserted into the RFC 5322 header region (before
    the blank-line separator) so importlib.metadata can parse it. The JIT cache
    keys on this build identity: same-version rebuilds must not reuse a stale
    binding cache. The timestamp comes from GYM_BUILD_TIMESTAMP (build_ci
    injects one per build round, shared by the full and shell wheels); a manual
    `python -m build` generates one locally.
    """

    def run(self):
        super().run()
        self._inject_build_timestamp()

    def _inject_build_timestamp(self):
        metadata_file = Path(self.egg_info) / "PKG-INFO"
        if not metadata_file.exists():
            # 兼容部分 setuptools 版本使用 METADATA 文件名
            metadata_file = Path(self.egg_info) / "METADATA"
        if not metadata_file.exists():
            # -e 模式下没有 METADATA/PKG-INFO 文件
            logging.warning("Can't get METADATA in %s", self.egg_info)
            return
        timestamp = os.environ.get("GYM_BUILD_TIMESTAMP", "").strip()
        if not timestamp:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        field = f"X-Build-Timestamp: {timestamp}\n"
        content = metadata_file.read_text(encoding="utf-8")
        separator = "\n\n"
        idx = content.find(separator)
        if idx == -1:
            # 无空行分隔符(异常情况), 追加到末尾
            metadata_file.write_text(content + field, encoding="utf-8")
        else:
            metadata_file.write_text(content[:idx] + "\n" + field + content[idx + 1:], encoding="utf-8")


class SetupCtrl:
    """Setuptools configuration flow controller (staged-tree packaging only)."""

    _LICENSE_EXPR: str = "LicenseRef-CANN-Open-Software-License-Agreement-Version-2.0"
    _LICENSE_FILES: List[str] = ["LICENSE"]
    _PACKAGE: str = "pypto_gym"
    _PACKAGE_DATA_COMMON: List[str] = [
        "_bindings/operators.json",
        "_bindings/frontends/*.py",
        "_bindings/csrc/*",
    ]
    _PACKAGE_DATA_FULL: List[str] = _PACKAGE_DATA_COMMON + [
        "_opp/vendors/**/*",
    ]

    def __init__(self):
        self.src_root: Path = Path(__file__).parent.resolve()
        stage = os.environ.get("GYM_WHL_STAGE", "")
        shell = os.environ.get("GYM_WHL_SHELL", "")
        if bool(stage) == bool(shell):
            raise RuntimeError(
                "exactly one of GYM_WHL_STAGE (community full wheel) or GYM_WHL_SHELL "
                "(run-embedded shell wheel) must be set; build wheels via "
                "build_ci.py --pkg-type whl|run")
        self.shell_mode: bool = bool(shell)
        root = Path(shell if self.shell_mode else stage).resolve()
        if not (root / self._PACKAGE / "__init__.py").is_file():
            raise RuntimeError(f"staged tree {root} misses {self._PACKAGE}/__init__.py")
        self.stage: Path = root
        self.packages: List[str] = [self._PACKAGE, f"{self._PACKAGE}._loader"]
        # package_dir must be relative to the setup.py directory (setuptools requirement).
        self.package_dir = {
            self._PACKAGE: os.path.relpath(root / self._PACKAGE, self.src_root)
        }

    @classmethod
    def main(cls):
        ts = datetime.now(tz=timezone.utc)
        try:
            cls._main()
        except KeyboardInterrupt:
            logging.error("Operation cancelled by user")
            raise
        # Calculate total duration
        duration = int((datetime.now(tz=timezone.utc) - ts).seconds)
        logging.info("Setuptools Finish, Duration %s secs.", duration)

    @classmethod
    def _main(cls):
        ctrl = SetupCtrl()
        logging.info("packaging %d packages (shell=%s)", len(ctrl.packages), ctrl.shell_mode)
        kwargs = dict(
            # 1. SPDX expression; newer setuptools maps it to the License-Expression field
            # 2. license_files packs LICENSE into dist-info/licenses across setuptools versions
            license=cls._LICENSE_EXPR,
            license_files=cls._LICENSE_FILES,
            packages=ctrl.packages,
            package_dir=ctrl.package_dir,
            zip_safe=False,
        )
        if ctrl.shell_mode:
            kwargs.update(
                package_data={cls._PACKAGE: cls._PACKAGE_DATA_COMMON},
                cmdclass={"egg_info": CustomEggInfo},
            )
        else:
            kwargs.update(
                package_data={cls._PACKAGE: cls._PACKAGE_DATA_FULL},
                distclass=BinaryDistribution,
                cmdclass={"bdist_wheel": Py3NonePlatformWheel, "egg_info": CustomEggInfo},
            )
        setuptools.setup(**kwargs)


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s', level=logging.INFO)
    SetupCtrl.main()
