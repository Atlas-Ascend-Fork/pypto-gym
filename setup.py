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

Source tree to install tree path mapping is controlled here:

    ops -> pypto_gym.ops    # operator library: pro/tensor paradigms map level by level

Only packages under ops/pro and ops/tensor are discovered; tests/op_host
directories inside operator trees and non-.py files (.md/.txt etc.) are
excluded from the bdist.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime, timezone

import setuptools


class SetupCtrl:
    """Setuptools configuration flow controller

    Discovers source packages and the operator tree, generates the
    packages/package_dir mapping, and starts the setuptools build.
    """

    _LICENSE_EXPR: str = "LicenseRef-CANN-Open-Software-License-Agreement-Version-2.0"
    _LICENSE_FILES: List[str] = ["LICENSE"]

    # Paradigm dirs under ops/ that participate in packaging
    _PARADIGM_DIRS = ("pro", "tensor")

    # Dir names inside operator trees excluded from packaging
    _EXCLUDED_DIRS = {"tests", "op_host", "__pycache__"}

    def __init__(self):
        self.src_root: Path = Path(__file__).parent.resolve()
        self.packages, self.package_dir = self._discover_packages()

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
        logging.info("packaging %d packages", len(ctrl.packages))
        setuptools.setup(
            # 1. SPDX expression; newer setuptools maps it to the License-Expression field
            # 2. license_files packs LICENSE into dist-info/licenses across setuptools versions
            license=cls._LICENSE_EXPR,
            license_files=cls._LICENSE_FILES,
            packages=ctrl.packages,
            package_dir=ctrl.package_dir,
        )

    def _discover_packages(self) -> Tuple[List[str], Dict[str, str]]:
        """Aggregate the packaging scope (currently the ops tree only)."""
        packages: List[str] = []
        package_dir: Dict[str, str] = {}
        ops_packages, ops_package_dir = self._discover_ops_packages()
        packages += ops_packages
        package_dir.update(ops_package_dir)
        return packages, package_dir

    def _discover_ops_packages(self) -> Tuple[List[str], Dict[str, str]]:
        """Discover the ops tree and generate the packages list and package_dir mapping.

        1. Only packages under the _PARADIGM_DIRS trees (pro/tensor) are discovered:
           ops/<paradigm> maps to pypto_gym.ops.<paradigm>, with network/operator
           dirs mapping level by level;
        2. Legacy trees outside pro/tensor (e.g. pypto_pro/pypto_tensor) are not
           packaged until they are migrated to the pro/tensor layout;
        3. tests/op_host dirs inside operator trees are skipped, so they never
           enter the bdist;
        4. package_dir values must be paths relative to the repo root (the
           setup.py directory).
        """
        ops_root = self.src_root / "ops"
        if not ops_root.is_dir():
            logging.warning("Ops tree not found: %s", ops_root)
            return [], {}
        all_dirs = []
        for path in ops_root.iterdir():
            if path.is_dir() and not path.name.startswith((".", "__")):
                all_dirs.append(path.name)
        ignored = sorted(d for d in all_dirs if d not in self._PARADIGM_DIRS)
        if ignored:
            logging.warning("Skip non-paradigm dirs under ops: %s", ignored)
        packages: List[str] = ["pypto_gym.ops"]
        for paradigm in self._PARADIGM_DIRS:
            paradigm_root = ops_root / paradigm
            if not paradigm_root.is_dir():
                logging.warning("Paradigm tree not found: %s", paradigm_root)
                continue
            pkg_root = f"pypto_gym.ops.{paradigm}"
            packages.append(pkg_root)
            for path in sorted(paradigm_root.rglob("*")):
                if not path.is_dir():
                    continue
                rel = path.relative_to(paradigm_root)
                if any(part in self._EXCLUDED_DIRS for part in rel.parts):
                    continue
                if not path.name.isidentifier():
                    logging.warning("skip non-identifier package dir: %s", path)
                    continue
                packages.append(f"{pkg_root}." + ".".join(rel.parts))
        return packages, {"pypto_gym.ops": "ops"}


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s', level=logging.INFO)
    SetupCtrl.main()
