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
"""
PyPTO kernel codegen driver.

Generate AI Core binary headers or a native AICPU fast-launch library for CMake.
Generated artifacts go into --out-dir.
"""

import argparse
import importlib.util
import logging
import re
import shutil
import sys
from pathlib import Path
from unittest.mock import MagicMock


def _install_build_only_torch_stub():
    """Stub runtime-only dependencies that binary codegen does not use."""
    sys.modules["torch"] = MagicMock(name="torch")
    sys.modules["torch_npu"] = MagicMock(name="torch_npu")


# gym: soc -> pypto arch token, 显式传参(不依赖 prepare_binary_headers 的 3510 默认值)
_ARCH_MAP = {
    "ascend950": "3510",
}


def _load_legacy_kernel(py_file: Path):
    """Load the sole @pl.jit kernel for the legacy header-generation API."""
    from pypto_pro.runtime.jit import _TileJitKernel

    spec = importlib.util.spec_from_file_location("_pypto_codegen_kernel_mod", py_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load kernel module from {py_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    kernels = [
        value for value in vars(module).values() if isinstance(value, _TileJitKernel)
    ]
    if len(kernels) != 1:
        raise RuntimeError(
            f"kernel module '{py_file}' must define exactly one @pl.jit kernel, found {len(kernels)}"
        )
    return kernels[0]


def _prepare_binary_headers(py_file: Path, arch: str) -> Path:
    try:
        from pypto_pro.runtime.opc import prepare_binary_headers

        return Path(prepare_binary_headers(str(py_file), arch)).resolve()
    except Exception as error:
        logging.warning(
            "prepare_binary_headers failed, falling back to generate_binary_headers: %s",
            error,
        )

    import pypto_pro.runtime.opc.pypto_compile as pto_compile

    return Path(
        pto_compile.generate_binary_headers(_load_legacy_kernel(py_file), arch)
    ).resolve()


def _build_native_aicpu(py_file: Path, out_dir: Path, op_file: str, arch: str):
    """Compile the beta AICPU launcher and emit its C ABI declaration."""
    from pypto_pro import ir
    from pypto_pro.runtime.aicpu_codegen import generate_aicpu_bare_body, generate_aicpu_sources
    from pypto_pro.runtime.aicpu_compile import build_aicpu_jit_so
    from pypto_pro.runtime.opc import pypto_aicpu_codegen as offline

    spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    kernel = getattr(module, op_file)
    arch = offline._setup_arch_env(offline.normalize_arch(arch))
    definition = kernel.to_kernel_def(None, None)
    program, _ = definition.parse_target_program(ir.SectionKind.Vector, bound_signature=None)
    params = offline._extract_param_specs(program, definition.last_param_directions)
    body = generate_aicpu_bare_body(program, arch, kernel.__name__)
    sources = generate_aicpu_sources(body, params, kernel.__name__)

    args_header = out_dir / f"{op_file}_aicpu_args.h"
    kernel_source = out_dir / f"{op_file}_aicpu.cpp"
    host_source = out_dir / f"{op_file}_aicpu_host.cpp"
    args_header.write_text(sources["args_header"], encoding="utf-8")
    kernel_source.write_text(sources["kernel_source"], encoding="utf-8")
    host_source.write_text(sources["host_source"], encoding="utf-8")
    declaration = re.search(r'extern "C" uint32_t \w+Run\([^{}]+\)', sources["host_source"])
    if declaration is None:
        raise RuntimeError("native AICPU codegen did not emit its Run ABI")
    (out_dir / f"{op_file}_native.h").write_text(
        "// Generated from PyPTO beta's native host ABI.\n#pragma once\n#include <stdint.h>\n"
        + declaration.group() + ";\n", encoding="utf-8")
    build_aicpu_jit_so(str(out_dir), op_file, kernel_source, host_source,
                       out_dir / f"lib{op_file}.so", arch, 600)


def main():
    parser = argparse.ArgumentParser(description="PyPTO kernel codegen driver")
    parser.add_argument("--py-file", required=True, help="kernel python source file")
    parser.add_argument(
        "--out-dir", required=True, help="directory to place generated artifacts"
    )
    parser.add_argument(
        "--op-file", required=True, help="op file stem (e.g. flash_attention_score_apt)"
    )
    parser.add_argument(
        "--soc", required=True, help="soc version (e.g. ascend950), mapped to pypto arch"
    )
    parser.add_argument("--aicpu-native", action="store_true", help="build a native AICPU library")
    args = parser.parse_args()

    arch = _ARCH_MAP.get(args.soc)
    if arch is None:
        raise SystemExit(f"[pypto_codegen] unsupported soc: {args.soc} (supported: {sorted(_ARCH_MAP)})")

    py_file = Path(args.py_file).resolve()
    if not py_file.is_file():
        raise SystemExit(f"[pypto_codegen] py-file not found: {py_file}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.aicpu_native:
        _build_native_aicpu(py_file, out_dir.resolve(), args.op_file, arch)
        return

    _install_build_only_torch_stub()
    binary_dir = _prepare_binary_headers(py_file, arch)
    shutil.copytree(binary_dir, out_dir, dirs_exist_ok=True)
    copied = sorted(
        path.relative_to(binary_dir).as_posix()
        for path in binary_dir.rglob("*")
        if path.is_file()
    )

    logging.info("%s: generated %s into %s", args.op_file, copied, out_dir)


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s][%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    main()
