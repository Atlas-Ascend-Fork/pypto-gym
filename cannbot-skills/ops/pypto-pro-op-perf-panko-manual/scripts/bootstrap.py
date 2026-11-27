# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Locate the canonical core in a checkout or its installed snapshot."""
import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parent
KERNEL_PROFILER = SCRIPTS.parents[1] / "pypto-pro-kernel/scripts/profile_kernel.py"


def core_path():
    for path in (SCRIPTS / "_core", SCRIPTS.parents[1] / "pypto-op-perf-panko-manual" / "scripts"):
        if (path / "panko" / "backend.py").is_file():
            return path
    raise RuntimeError("PANKO core missing; run the PyPTO-Pro orchestrator init again")


def activate():
    path = core_path()
    if str(path) not in sys.path:
        if str(SCRIPTS) not in sys.path:
            sys.path.insert(0, str(SCRIPTS))
        sys.path.insert(sys.path.index(str(SCRIPTS)) + 1, str(path))
    # Bind Pro tile modules before the standalone PyPTO core adjusts its import path.
    import bayesian_optimization
