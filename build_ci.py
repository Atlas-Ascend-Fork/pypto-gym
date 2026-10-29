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
"""PyPTO-GYM build CI scenario control entry point

This file provides a unified build entry point for PyPTO-GYM project, supporting dual product modes:

    whl: jit direct-call mode, python -m build -> build_out/whl/<soc>/*.whl
    run: built-in mode, CMake full chain -> build_out/run/<soc>/ producing custom run packages
         configure -> build -> package (.run)
    stest: -s/--stest runs on-board system tests (whl -> jit / run -> aclnn)

Multi-soc semantics: --soc accepts a comma-separated list; each soc runs a FULL
build round in its own binary dir (build/<soc>), producing one .run per soc
(chip in package name, e.g. cann-950-pypto_gym_9.2.0_linux-aarch64.run).
The output dir (build_out/) accumulates chip-named .run files; CPack staging is
rebuilt per round inside each binary dir, so .run content is always single-soc.

Main features:
    - Dual product modes (whl / run), one .run per soc
    - Build environment precheck with fail-fast (enforced on the CMake side)
    - Build timeout control and automatic subprocess-group cleanup after timeout

Usage:
    Configure build options via command-line arguments, then run the script to trigger the build process:

        python build_ci.py [options]

    Common options:
        --pkg-type: Specify product type (whl/run)
        --ops: Specify operators to build (comma separated, default ALL)
        --soc: Specify soc version (currently ascend950 only;
               comma-separated multi-value activates when whitelist expands)
        --build_type: Specify build type (Debug/Release/MinSizeRel/RelWithDebInfo, run mode)
        --asan: Enable AddressSanitizer (run mode, host side)
        --ubsan: Enable UndefinedBehaviorSanitizer (run mode, host side)
        -j/--jobs: Specify build parallelism
        -c/--clean: Clean build directory and install directory
        --timeout: Total timeout in seconds
        -s/--stest: Run on-board system tests (optional value is a pytest -k filter)

Examples:
    # Build whl (jit mode)
    python build_ci.py --pkg-type whl

    # Build run package for flex_attention on ascend950
    python build_ci.py --pkg-type run --ops flex_attention --soc ascend950

    # Debug build with AddressSanitizer on host side
    python build_ci.py --pkg-type run --build_type Debug --asan

    # Clean and rebuild with timeout control
    python build_ci.py --pkg-type run -c --timeout 3600 -j 16
"""

import abc
import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
import logging
import os
from pathlib import Path
import shutil
import signal
import shlex
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

from packaging import requirements


class CMakeParam(abc.ABC):
    """CMake parameter abstract base class

    Defines the common interface for all parameter classes that need to pass Options to CMake.
    Subclasses must implement reg_args() to register CLI arguments and get_cfg_cmd() to generate
    CMake configuration commands.
    """

    @staticmethod
    @abc.abstractmethod
    def reg_args(parser):
        """Register command-line arguments

        Registers the command-line arguments supported by the current class to the argument parser.

        :param parser: ArgumentParser instance
        """
        pass

    @abc.abstractmethod
    def get_cfg_cmd(self) -> str:
        """Generate CMake Configure command

        :return: CMake configuration parameter string
        """
        pass


class FeatureParam(CMakeParam):
    """Feature parameter class

    Manages feature options during the build process: product type (whl/run),
    operator scope and soc versions for the run mode.

    Multi-soc semantics: one .run package per soc, each soc runs a full isolated
    build round.
    """

    def __init__(self, args):
        """Initialize FeatureParam instance

        :param args: Parsed command-line arguments
        """
        self.pkg_type: str = args.pkg_type  # Product type: whl (jit) / run (built-in)
        self.ops: str = args.ops or "ALL"  # Operators to build (comma separated)
        self.soc: str = args.soc or "ascend950"  # Soc versions (comma separated)
        self.whl_isolation: bool = args.isolation  # Compile whl package in isolation mode

    def __str__(self) -> str:
        """Return string representation of feature parameters

        :return: Formatted feature parameter string
        :rtype: str
        """
        desc = "\nFeature"
        desc += f"\n    PackageType             : {self.pkg_type}"
        desc += f"\n    Ops                     : {self.ops}"
        run_count = len(self.soc_list)
        desc += f"\n    Soc                     : {self.soc} ({run_count} run{'s' if run_count > 1 else ''})"
        desc += f"\n    Isolation               : {self.whl_isolation}"
        return desc

    @property
    def soc_list(self) -> List[str]:
        """Normalized soc list (lowercase, stripped, deduplicated, order preserved)

        CMake side matching is case-sensitive, so lowercase normalization is done here.

        :return: Soc list
        :rtype: List[str]
        """
        seen = []
        for s in self.soc.split(","):
            s = s.strip().lower()
            if s and s not in seen:
                seen.append(s)
        return seen or ["ascend950"]

    @property
    def pkg_type_run(self) -> bool:
        """Whether current build is run (built-in) mode

        :return: True if pkg_type is run
        :rtype: bool
        """
        return self.pkg_type == "run"

    @staticmethod
    def reg_args(parser):
        parser.add_argument(
            "--pkg-type",
            choices=["whl", "run"],
            default="whl",
            help="package type: whl (jit direct call) / run (built-in custom run package).",
        )
        parser.add_argument(
            "--ops",
            nargs="?",
            type=str,
            default="ALL",
            help="operators to build (comma separated), e.g. flex_attention. Default ALL.",
        )
        parser.add_argument(
            "--soc",
            nargs="?",
            type=str,
            default="ascend950",
            choices=["ascend950"],
            help="soc version, one .run per soc (comma-separated multi-value activates when whitelist expands)."
        )
        parser.add_argument(
            "--isolation",
            action="store_true",
            default=False,
            help="isolation, build whl in an isolated virtual environment (default: no isolation, "
            "build dependencies must be installed in the current environment).",
        )

    def get_cfg_cmd(self) -> str:
        """Generate CMake Configure command for feature options

        Note: ASCEND_COMPUTE_UNIT is NOT included here — it is a per-soc build-round
        parameter passed by BuildCtrl.cmake_configure(soc), not a global feature.

        :return: CMake configuration parameter string
        :rtype: str
        """
        cmd = f" -DPACKAGE_TYPE={self.pkg_type}"  # Control the type of product package, such as run/rpm/deb
        cmd += f" -DASCEND_OP_NAME={self.ops}"  # Control the scope of compilation operators
        return cmd


class BuildParam(CMakeParam):
    """Build parameter class

    Manages build process parameters for the run mode: build type and sanitizer
    switches. Sanitizers apply to host side targets only (kernel .o is compiled
    by the opc/bisheng toolchain where sanitizers do not apply).
    """

    def __init__(self, args):
        """Initialize BuildParam instance

        :param args: Parsed command-line arguments
        """
        self.build_type: str = args.build_type or "Release"  # CMake build type
        self.asan: bool = args.asan  # Enable AddressSanitizer
        self.ubsan: bool = args.ubsan  # Enable UndefinedBehaviorSanitizer

    def __str__(self) -> str:
        """Return string representation of build parameters

        :return: Formatted build parameter string
        :rtype: str
        """
        desc = "\nBuild"
        desc += f"\n    BuildType               : {self.build_type}"
        desc += f"\n    ASan                    : {self.asan}"
        desc += f"\n    UbSan                   : {self.ubsan}"
        return desc

    @staticmethod
    def reg_args(parser):
        parser.add_argument(
            "--build_type",
            "--build-type",
            nargs="?",
            type=str,
            default="Release",
            choices=["Debug", "Release", "MinSizeRel", "RelWithDebInfo"],
            help="build type (run mode).",
        )
        parser.add_argument("--asan",
                            action="store_true", default=False,
                            help="Enable AddressSanitizer (run mode).")
        parser.add_argument("--ubsan",
                            action="store_true", default=False,
                            help="Enable UndefinedBehaviorSanitizer (run mode)."
        )

    def get_cfg_cmd(self) -> str:
        """Generate CMake Configure command for build parameters

        :return: CMake configuration parameter string
        :rtype: str
        """
        cmd = f" -DCMAKE_BUILD_TYPE={self.build_type}"
        cmd += f" -DENABLE_ASAN={'ON' if self.asan else 'OFF'}"
        cmd += f" -DENABLE_UBSAN={'ON' if self.ubsan else 'OFF'}"
        return cmd


class TestParam(CMakeParam):
    """System test parameter (on-board pytest execution).

    --stest without a value runs all selected operator tests; with a value the
    value is used as a pytest -k expression (comma-separated entries are joined
    with "or"). UTest is intentionally not registered yet.
    """

    def __init__(self, args):
        """Initialize TestParam instance.

        :param args: Parsed command-line arguments
        """
        self.stest = args.stest  # False (not requested) / True (no filter) / str (filter)

    @property
    def enabled(self) -> bool:
        """Whether system test execution is requested."""
        return self.stest is not False

    @property
    def filter_expr(self) -> str:
        """Return the pytest -k expression derived from --stest=<value>."""
        if not isinstance(self.stest, str) or not self.stest.strip():
            return ""
        parts = [p.strip() for p in self.stest.split(",") if p.strip()]
        return " or ".join(parts)

    def __str__(self) -> str:
        if not self.enabled:
            return ""
        desc = "\nTests"
        desc += f"\n    STest                   : {'ON' if not self.filter_expr else self.filter_expr}"
        return desc

    @staticmethod
    def reg_args(parser):
        parser.add_argument(
            "-s",
            "--stest",
            nargs="?",
            const=True,
            default=False,
            help="run on-board system tests; optional value is a pytest -k filter "
                 "(comma-separated entries are ORed), e.g. --stest=test_fa_mask_info",
        )

    def get_cfg_cmd(self) -> str:
        """Enable test targets in CMake when system tests are requested."""
        return " -DENABLE_TESTS=ON" if self.enabled else ""


class BuildCtrl(CMakeParam):
    """Build process control class

    This class contains control flags/parameters specified or parsed from the command line
    to control build process execution. It serves as the entry point and controller for the
    entire build flow, responsible for coordinating the whole build process.
    """

    def __init__(self, args):
        """Initialize BuildCtrl instance

        Parses and initializes all build-related configuration from command-line arguments.

        :param args: Parsed command-line arguments
        """
        self.clean: bool = args.clean  # Force clean Build-Tree and Install-Tree marker
        # Timeout duration
        self.origin_timeout: Optional[int] = args.timeout if args.timeout and args.timeout > 0 else None
        self.remain_timeout: Optional[int] = self.origin_timeout
        # Path
        self.src_root: Path = Path(__file__).parent.resolve()
        self._build_root: Path = Path(Path.cwd(), "build")  # CMake binary dir root
        self.install_root: Path = Path(self._build_root.parent, "build_out")  # Install dir root
        # 3rd lib cache path: CLI > env CANN_3RD_LIB_PATH > None (cmake falls back to repo third_party/)
        third_party_path = args.cann_3rd_lib_path or os.environ.get("CANN_3RD_LIB_PATH")
        self.third_party_path: Optional[Path] = Path(third_party_path).resolve() if third_party_path else None
        # Others
        self.feature: FeatureParam = FeatureParam(args=args)
        self.build: BuildParam = BuildParam(args=args)
        self.tests: TestParam = TestParam(args=args)
        self.jobs: Optional[int] = args.jobs
        self.verbose: bool = args.verbose
        self.cmake: Optional[Path] = self.which_cmake()
        if not self.cmake:
            raise RuntimeError("Can't find system cmake (pip cmake package is excluded, see which_cmake)")

    def __str__(self) -> str:
        """Return string representation of build control parameters

        :return: Formatted build control parameter string
        :rtype: str
        """
        py3_ver = sys.version_info
        desc = ""
        desc += "\nEnviron"
        desc += f"\n    Python3                 : {sys.executable} ({py3_ver.major}.{py3_ver.minor}.{py3_ver.micro})"
        desc += "\nPath"
        desc += f"\n    Source  Dir             : {self.src_root}"
        desc += f"\n    Build   Root Dir        : {self._build_root}"
        desc += f"\n    Install Dir             : {self.install_root}"
        desc += f"\n    3rd     Dir             : {self.third_party_path}"
        desc += "\nFlag"
        desc += f"\n    Clean                   : {self.clean}"
        desc += f"\n    Verbose                 : {self.verbose}"
        desc += f"\n    Jobs                    : {self.jobs}"
        desc += "\nOthers"
        desc += f"\n    Timeout                 : {self.origin_timeout}"
        desc += f"{self.feature}"
        desc += f"{self.build}"
        desc += f"{self.tests}"
        desc += "\n"
        return desc

    @staticmethod
    def which_cmake() -> Optional[Path]:
        """Find system-level CMake executable path

        Excludes interference from the cmake pip package (whose cmake entry is a
        Python script, not an ELF binary): walks PATH directories one by one and
        returns the first cmake whose first 4 bytes are the ELF magic.

        :return: Absolute path to system-level cmake executable, returns None if not found
        :rtype: Optional[Path]
        """
        path_dir_lst = [d.strip() for d in os.environ.get("PATH", "").split(os.pathsep) if d.strip()]
        valid_path_lst = []
        for path_dir in path_dir_lst:
            if path_dir in valid_path_lst:  # deduplicate PATH entries
                continue
            valid_path_lst.append(path_dir)
            # Limit shutil.which to the current single directory
            cmake_str = shutil.which("cmake", path=path_dir)
            if not cmake_str:
                continue
            cmake_file = Path(cmake_str).resolve()
            if not cmake_file.exists() or not cmake_file.is_file():
                continue
            if cmake_file.stat().st_size <= 4:
                continue
            with open(cmake_file, "rb") as fh:
                if fh.read(4) != b"\x7fELF":  # ELF magic check excludes pip cmake script entries
                    continue
            return cmake_file
        return None

    @staticmethod
    def reg_args(parser):
        parser.add_argument(
            "-c",
            "--clean",
            action="store_true",
            default=False,
            help="clean, clean Build-Tree and Install-Tree before build.",
        )
        parser.add_argument("--timeout", nargs="?", type=int, default=None, help="Total timeout.")
        parser.add_argument(
            "-j",
            "--jobs",
            nargs="?",
            type=int,
            default=None,
            help="build parallelism, default is cpu count.",
        )
        parser.add_argument(
            "--cann_3rd_lib_path",
            "--third_party_path",
            nargs="?",
            type=str,
            default="",
            dest="cann_3rd_lib_path",
            help="Specify 3rd libraries cache path (cann-cmake / makeself); "
            "default: <repo>/third_party/",
        )
        parser.add_argument("--verbose", action="store_true", default=False, help="verbose, enable verbose output.")

    @classmethod
    def check_pip_dependencies(cls, deps: Dict[str, str], raise_err: bool = False, log_err: bool = True) -> bool:
        """Check whether pip packages meet the version requirements

        :param deps: Package dependencies, e.g. {"build": ">=1.0.3"}
        :param raise_err: Raise RuntimeError when check fails
        :param log_err: Log error details when check fails
        :return: True if all dependencies are satisfied
        :rtype: bool
        """
        info_lst = []
        for pkg, ver in deps.items():
            info = cls._check_pip_pkg(pkg=pkg, ver=ver)
            info_lst.extend(info)
        if info_lst:
            if log_err:
                logging.error("%s", info_lst)
                install_cmd = " ".join([f"{pkg}{deps[pkg]}" for pkg in deps])
                logging.error("Please install the missing dependencies first [%s]", install_cmd)
            if raise_err:
                raise RuntimeError("\n".join(info_lst))
            return False
        return True

    @classmethod
    def main(cls):
        """Main entry point with exception handling and duration statistics"""
        ts = datetime.now(tz=timezone.utc)
        try:
            cls._main()
        except KeyboardInterrupt as e:
            logging.error("Operation cancelled by user")
            raise e
        except subprocess.TimeoutExpired as e:
            logging.error("Operation timeout, %s", e)
            raise e
        # Calculate total duration
        duration = int((datetime.now(tz=timezone.utc) - ts).seconds)
        logging.info("Build[CI] Finish, Duration %s secs.", duration)

    @classmethod
    def _main(cls):
        """Main processing flow"""
        parser = argparse.ArgumentParser(description="PyPTO-GYM Build Ctrl.", epilog="Best Regards!")
        # Register arguments
        FeatureParam.reg_args(parser=parser)
        BuildParam.reg_args(parser=parser)
        TestParam.reg_args(parser=parser)
        BuildCtrl.reg_args(parser=parser)

        # Parse arguments
        args = parser.parse_args()
        ctrl = BuildCtrl(args=args)
        # Process flow
        if ctrl.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        logging.info("%s", ctrl)
        if ctrl.tests.enabled and len(ctrl.feature.soc_list) != 1:
            raise ValueError("--stest supports exactly one soc; pass a single --soc")
        if ctrl.feature.pkg_type_run:
            soc_list = ctrl.feature.soc_list
            logging.info("PackageType(run), start process with CMake, %s soc(s): %s", len(soc_list), soc_list)
            ctrl.cmake_clean()
            # Build environment precheck is enforced on the CMake configure side
            # (cmake/config.cmake), covering every entry point (build_ci.py / manual cmake / CLion).

            # Multi-soc = multiple full build rounds: each soc gets its own binary dir and
            # produces an independent .run (chip in package name); the output dir is shared.
            for i, soc in enumerate(soc_list, start=1):
                logging.info("===== Soc round (%s/%s): %s =====", i, len(soc_list), soc)
                ctrl.cmake_configure(soc=soc)
                ctrl.cmake_build(soc=soc)
                ctrl.cmake_package(soc=soc)
            ctrl.run_stest()
        else:
            logging.info("PackageType(whl), start process")
            ctrl.cmake_clean()
            ctrl.py_build()
            ctrl.run_stest()

    @classmethod
    def _check_pip_pkg(cls, pkg: str, ver: str) -> List[str]:
        """Check whether a single pip package satisfies the version requirement

        :param pkg: Package name
        :param ver: Version requirement string, e.g. ">=1.0.3"
        :return: Empty list when satisfied, otherwise a list of problem descriptions
        :rtype: List[str]
        """
        info_lst = []
        requirement_str = f"{pkg}{ver}"
        try:
            req = requirements.Requirement(requirement_str)
            try:
                installed_version = metadata.version(pkg)
                if ver and not req.specifier.contains(installed_version, prereleases=True):
                    info_lst.append(f"{pkg}: version {installed_version} not satisfy {ver}")
            except metadata.PackageNotFoundError:
                info_lst.append(f"package {pkg} has not been installed")
        except Exception as e:
            info_lst.append(f"package {pkg} check fail {e}")
        return info_lst

    def get_cfg_cmd(self, soc: str) -> str:
        """Generate CMake Configure command for BuildCtrl's own -D parameters

        :param soc: Soc name of current build round
        :return: CMake configuration parameter string
        :rtype: str
        """
        # No standalone install step: install() rules are consumed by the CPack staging
        # inside the package target; CMAKE_INSTALL_PREFIX here only decides where the
        # .run files land (set_cann_cpack_config -> CPACK_CMAKE_INSTALL_PREFIX).
        cmd = f" -DCMAKE_INSTALL_PREFIX={self._run_dir(soc)}"
        cmd += f" -DPython3_EXECUTABLE={sys.executable}"
        cmd += f" -DASCEND_COMPUTE_UNIT={soc}"
        cmd += self.tests.get_cfg_cmd()
        if self.third_party_path:
            cmd += f" -DCANN_3RD_LIB_PATH={self.third_party_path}"
        return cmd

    def cmake_clean(self):
        """Clean CMake build intermediate results

        Cleans build tree and output tree contents. Only executes when clean flag is True.
        """
        if not self.clean:
            return
        if self._build_root.exists():
            logging.info("Clean Build-Root-Tree(%s)", self._build_root)
            shutil.rmtree(self._build_root)
        if self.install_root.exists():
            logging.info("Clean Output-Tree(%s)", self.install_root)
            shutil.rmtree(self.install_root)

    def py_build(self):
        """whl package compilation (jit direct call mode)"""
        # The build-system.requires check in pyproject.toml is implemented by the build
        # package itself and thus cannot fail early; validate the build package version here.
        self.check_pip_dependencies(deps={"build": ">=1.0.3"}, raise_err=True)
        whl_dir = self._whl_dir()
        whl_dir.mkdir(parents=True, exist_ok=True)
        cmd = f"{sys.executable} -m build --outdir={whl_dir}"
        cmd += " --no-isolation" if not self.feature.whl_isolation else ""
        logging.info("Build whl, Cmd: %s, Timeout: %s secs", cmd, self.remain_timeout)
        _, duration = self._run_build_cmd(cmd=cmd, pg_desc="build")
        logging.info("Build whl success, %s", duration)

    def cmake_configure(self, soc: str):
        """Execute CMake Configure stage for one soc build round

        Each soc gets an isolated binary dir (build/<soc>). ASCEND_COMPUTE_UNIT is
        single-valued per round. Python3_EXECUTABLE is passed explicitly so that the
        CMake toolchain (python3: pypto codegen, pypto_pro probe) uses the same
        python interpreter as this script.

        :param soc: Soc name of current build round
        """
        cmd = f"{self.cmake} -S {self.src_root} -B {self._build_dir(soc)}"
        cmd += self.get_cfg_cmd(soc)
        cmd += self.feature.get_cfg_cmd()
        cmd += self.build.get_cfg_cmd()
        # Execute
        logging.info("CMake Configure[%s], Cmd: %s, Timeout: %s secs", soc, cmd, self.remain_timeout)
        _, duration = self._run_build_cmd(cmd=cmd)
        logging.info("CMake Configure[%s] success, %s", soc, duration)

    def cmake_build(self, soc: str):
        """Execute CMake Build stage for one soc build round

        :param soc: Soc name of current build round
        """
        jobs = self.jobs or (os.cpu_count() or 8)
        cmd = f"{self.cmake} --build {self._build_dir(soc)} -j {jobs}"
        cmd += " --verbose" if self.verbose else ""
        logging.info("CMake Build[%s], Cmd: %s, Timeout: %s secs", soc, cmd, self.remain_timeout)
        _, duration = self._run_build_cmd(cmd=cmd)
        logging.info("CMake Build[%s] success, %s", soc, duration)

    def cmake_package(self, soc: str):
        """Execute CMake Package stage for one soc build round (set_cann_cpack_config -> .run)

        The .run name carries the chip (COMPUTE_UNIT -> chip name, e.g.
        cann-950-pypto_gym_9.2.0_linux-aarch64.run), so multi-soc rounds never collide.

        :param soc: Soc name of current build round
        """
        cmd = f"{self.cmake} --build {self._build_dir(soc)} --target package"
        cmd += " --verbose" if self.verbose else ""
        logging.info("CMake Package[%s], Cmd: %s, Timeout: %s secs", soc, cmd, self.remain_timeout)
        _, duration = self._run_build_cmd(cmd=cmd, pg_desc="cmake-package")
        for run_file in sorted(self._run_dir(soc).glob("*.run")):
            logging.info("run package: %s", run_file)
        logging.info("CMake Package[%s] success, %s", soc, duration)

    def _pkg_dir(self, soc: str) -> Path:
        """Per-package-type/per-soc output dir under build_out.

        whl: build_out/whl/<soc> (wheel + site-packages + ops registry)
        run: build_out/run/<soc> (run package + vendors after test install)
        """
        return self.install_root / self.feature.pkg_type / soc

    def _whl_dir(self) -> Path:
        """Output dir for whl artifacts and the test install tree."""
        return self._pkg_dir(self.feature.soc_list[0])

    def _run_dir(self, soc: str) -> Path:
        """Output dir for one soc's run package and its isolated test install."""
        return self._pkg_dir(soc)

    def _build_dir(self, soc: str) -> Path:
        """Per-soc CMake binary dir (isolated per soc round to avoid cross-round artifacts)

        :param soc: Soc name of current build round
        :return: Per-soc binary dir, e.g. build/ascend950
        :rtype: Path
        """
        return self._build_root / soc

    @staticmethod
    def _prepend_env(name: str, value: str) -> str:
        """Prepend a path to an existing environment variable value."""
        old = os.environ.get(name, "")
        return f"{value}:{old}" if old else value

    @staticmethod
    def _find_artifact(directory: Path, pattern: str) -> Path:
        """Return the newest artifact matching pattern in directory."""
        files = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        if not files:
            raise RuntimeError(f"no artifact matching {pattern!r} under {directory}")
        return files[0]

    def _discover_op_dirs(self) -> Dict[str, Path]:
        """Discover operator dirs that have a tests/ directory (ops/{pro,tensor} only)."""
        found: Dict[str, Path] = {}
        for paradigm in ("pro", "tensor"):
            root = self.src_root / "ops" / paradigm
            for tests_dir in sorted(root.glob("**/tests")):
                op_dir = tests_dir.parent
                if not (op_dir / "op_host").is_dir():
                    continue
                found[op_dir.name] = op_dir
        return found

    def _selected_op_dirs(self) -> List[Path]:
        """Resolve --ops into operator dirs for stest."""
        found = self._discover_op_dirs()
        if self.feature.ops.strip().upper() == "ALL":
            selected = sorted(found.values(), key=lambda p: p.name)
            if not selected:
                raise RuntimeError("no operator tests found under ops/{pro,tensor}/**/tests")
            return selected
        names = [n.strip() for n in self.feature.ops.split(",") if n.strip()]
        missing = [n for n in names if n not in found]
        if missing:
            raise RuntimeError(
                f"operator tests not found for {missing}; available: {sorted(found)}")
        return [found[n] for n in names]

    def _pytest_filter_args(self) -> List[str]:
        expr = self.tests.filter_expr
        return ["-k", expr] if expr else []

    def _run_pytest(self, op_dirs: List[Path], mode: str, soc: str,
                    update_env: Optional[Dict[str, str]]) -> None:
        """Run pytest in a child process with an isolated environment."""
        args = [sys.executable, "-m", "pytest"]
        args += [str(op_dir / "tests") for op_dir in op_dirs]
        args += ["--mode", mode, "--rootdir", str(self.src_root), "-q"]
        if mode == "aclnn":
            args += ["--aclnn-lib-dir", str(self._build_dir(soc) / "tests")]
            args += ["-m", "not jit_only"]
        args += self._pytest_filter_args()
        cmd = " ".join(shlex.quote(a) for a in args)
        logging.info("pytest(%s), Cmd: %s, Timeout: %s secs", mode, cmd, self.remain_timeout)
        _, duration = self._run_build_cmd(cmd=cmd, update_env=update_env, pg_desc=f"pytest-{mode}")
        logging.info("pytest(%s) success, %s", mode, duration)

    def _install_whl_for_test(self) -> Path:
        """Install the built whl into build_out/whl/<soc>/site-packages."""
        whl_dir = self._whl_dir()
        whl = self._find_artifact(whl_dir, "*.whl")
        site_dir = whl_dir / "site-packages"
        if site_dir.exists():
            shutil.rmtree(site_dir)
        site_dir.mkdir(parents=True, exist_ok=True)
        cmd = (f"{sys.executable} -m pip install --target {shlex.quote(str(site_dir))} "
               f"--no-deps --no-compile {shlex.quote(str(whl))}")
        logging.info("Install whl for test, Cmd: %s, Timeout: %s secs", cmd, self.remain_timeout)
        _, duration = self._run_build_cmd(cmd=cmd, pg_desc="pip-install-whl")
        logging.info("Install whl for test success, %s", duration)
        return site_dir

    def _write_ops_registry(self, op_dirs: List[Path]) -> Path:
        """Write op -> installed package module mapping for jit stest."""
        mapping = {}
        for op_dir in op_dirs:
            rel = op_dir.relative_to(self.src_root)
            pkg = "pypto_gym." + ".".join(rel.parts)
            mapping[op_dir.name] = f"{pkg}.op_kernel.{op_dir.name}"
        registry = self._whl_dir() / "ops_registry.json"
        registry.write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")
        logging.info("ops registry: %s", registry)
        return registry

    def _run_stest_jit(self, op_dirs: List[Path]) -> None:
        site_dir = self._install_whl_for_test()
        registry = self._write_ops_registry(op_dirs)
        update_env = {
            "PYTHONPATH": self._prepend_env("PYTHONPATH", str(site_dir)),
            "GYM_OPS_REGISTRY": str(registry),
        }
        self._run_pytest(op_dirs, mode="jit", soc=self.feature.soc_list[0], update_env=update_env)

    def _run_stest_aclnn(self, op_dirs: List[Path], soc: str) -> None:
        run_dir = self._run_dir(soc).resolve()
        run_pkg = self._find_artifact(run_dir, "*.run")
        install_path = run_dir
        uninstall_cmd = (f"bash {shlex.quote(str(run_pkg))} --uninstall "
                         f"--install-path={shlex.quote(str(install_path))}")
        self._run_build_cmd(cmd=uninstall_cmd, check=False, pg_desc="run-uninstall")
        install_cmd = (f"bash {shlex.quote(str(run_pkg))} --install-path="
                       f"{shlex.quote(str(install_path))}")
        logging.info("Install run for test, Cmd: %s, Timeout: %s secs", install_cmd, self.remain_timeout)
        _, duration = self._run_build_cmd(cmd=install_cmd, pg_desc="run-install")
        logging.info("Install run for test success, %s", duration)
        vendor = install_path / "vendors" / "pypto_gym"
        if not vendor.is_dir():
            raise RuntimeError(f"test install did not produce vendor dir: {vendor}")
        update_env = {
            "ASCEND_CUSTOM_OPP_PATH": self._prepend_env("ASCEND_CUSTOM_OPP_PATH", str(vendor)),
            "LD_LIBRARY_PATH": self._prepend_env("LD_LIBRARY_PATH", str(vendor / "op_api" / "lib")),
        }
        self._run_pytest(op_dirs, mode="aclnn", soc=soc, update_env=update_env)

    def run_stest(self) -> None:
        """Execute system tests for the selected operators."""
        if not self.tests.enabled:
            return
        op_dirs = self._selected_op_dirs()
        logging.info("STest operators: %s", [d.name for d in op_dirs])
        if self.feature.pkg_type_run:
            self._run_stest_aclnn(op_dirs, self.feature.soc_list[0])
        else:
            self._run_stest_jit(op_dirs)

    def _duration(self, ts: datetime) -> str:
        """Calculate stage duration and update remaining timeout

        :param ts: Stage start timestamp
        :return: Duration description string
        :rtype: str
        """
        duration = int((datetime.now(tz=timezone.utc) - ts).seconds)
        duration_str = f"Duration {duration} secs"
        if self.remain_timeout:
            self.remain_timeout = max(self.remain_timeout - duration, 0)
            duration_str += f" Remain {self.remain_timeout} secs"
        return duration_str

    def _run_build_cmd(self, cmd: str, update_env: Optional[Dict[str, str]] = None,
                      check: bool = True, pg_desc: str = "CMake") -> Tuple[subprocess.CompletedProcess, str]:
        """Execute a specific build command

        This function is used instead of calling subprocess.run directly for the following reasons:
            1. Supports multi-stage builds, where build duration shares a common timeout configuration;
            2. In the run build scenario, the process invocation chain is:
                   build_ci.py(main process) -> process1(CMake) -> process2(CMake Generator, make/ninja)
                   -> process3(Python: codegen/opc) -> process4(compiler)
               If process1 times out, it needs to notify all child/grandchild processes to terminate

        :param cmd: Build command
        :param update_env: Environment variables (additional updates)
        :param check: Check return value
        :param pg_desc: Process Group Description
        """

        def _stop_pg(_msg: str, _p: subprocess.Popen):
            """Notify all child/grandchild processes to terminate via SIGINT signal"""
            _pgid = os.getpgid(_p.pid)
            logging.info("%s. Send terminate event to %s[%s]", _msg, pg_desc, _pgid)
            os.killpg(_pgid, signal.SIGINT)

        ts = datetime.now(tz=timezone.utc)
        stdout = None
        stderr = None
        env = os.environ.copy()
        env.update(update_env if update_env else {})
        with subprocess.Popen(
            shlex.split(cmd), env=env, text=True, encoding="utf-8", start_new_session=True
        ) as process:
            try:
                stdout, stderr = process.communicate(timeout=self.remain_timeout)
            except subprocess.TimeoutExpired as e:
                _stop_pg(_msg=f"Timeout({self.remain_timeout})", _p=process)
                raise e
            except KeyboardInterrupt as e:
                _stop_pg(_msg="KeyboardInterrupt", _p=process)
                raise e
            except Exception as e:
                process.kill()
                raise e
            finally:
                stdout = stdout or ""
                stderr = stderr or ""
            ret_code = process.poll()
            if check and ret_code:
                raise subprocess.CalledProcessError(ret_code, process.args, output=stdout, stderr=stderr)
        # Update remaining timeout
        duration = self._duration(ts=ts)
        return subprocess.CompletedProcess(process.args, ret_code, stdout, stderr), duration


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s", level=logging.INFO)
    BuildCtrl.main()
