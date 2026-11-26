# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Build the QLI planner's native AICPU library and matching host declaration."""
import argparse
import re
from pathlib import Path

from pypto_pro.runtime.aicpu_compile import (
    _parse_aicpu_program,
    build_aicpu_jit_so,
    generate_aicpu_bare_body,
    generate_aicpu_sources,
)
from quant_lightning_indexer_metadata import quant_lightning_indexer_metadata


def build(out_dir: Path) -> None:
    kernel = quant_lightning_indexer_metadata
    name = kernel.__name__
    _, program, params = _parse_aicpu_program(kernel)
    body = generate_aicpu_bare_body(program, kernel.arch, name)
    sources = generate_aicpu_sources(body, params, name)

    out_dir.mkdir(parents=True, exist_ok=True)
    kernel_source = out_dir / f"{name}_aicpu.cpp"
    host_source = out_dir / f"{name}_aicpu_host.cpp"
    kernel_source.write_text(sources["kernel_source"], encoding="utf-8")
    host_source.write_text(sources["host_source"], encoding="utf-8")
    declaration = re.search(rf'extern "C" uint32_t {name}Run\([^{{}}]+\)', sources["host_source"])
    if declaration is None:
        raise RuntimeError(f"PyPTO did not generate the {name}Run declaration")
    (out_dir / f"{name}_native.h").write_text(
        "// Generated from the PyPTO native host ABI.\n#pragma once\n#include <stdint.h>\n"
        + declaration.group() + ";\n",
        encoding="utf-8",
    )
    build_aicpu_jit_so(str(out_dir), name, kernel_source, host_source,
                       out_dir / f"lib{name}.so", kernel.arch, 600)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    build(parser.parse_args().out_dir.resolve())
