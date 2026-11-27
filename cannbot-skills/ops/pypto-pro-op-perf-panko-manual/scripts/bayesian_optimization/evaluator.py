# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Process-group-aware command runner for the Pro STOP device probe."""
import os
import shutil
import subprocess

from evaluation.process import run_captured
from panko_core.bayesian_optimization import evaluator as core_evaluator
from panko_core.bayesian_optimization.evaluator import DEVICE_CONTEXT_MARKERS, DEVICE_FAULT_MARKERS


def _run_one(step, cwd, timeout, env=None):
    """Resolve the executable and run one parsed step; return combined output and exit code."""
    argv, step_env = step
    run_env = {**(env or os.environ), **step_env}
    directory = os.path.abspath(cwd or os.curdir)
    command = os.path.join(directory, argv[0]) if os.path.dirname(argv[0]) else argv[0]
    search_path = os.pathsep.join(os.path.join(directory, entry) for entry in os.get_exec_path(run_env))
    exe = shutil.which(command, path=search_path)
    if not exe:
        return f"{argv[0]}: not found on PATH\n", 127
    try:
        r = run_captured([os.path.abspath(exe), *argv[1:]], cwd=cwd, env=run_env, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        return e.output, 124
    except OSError as exc:
        return f"command invocation failed: {exc}\n", 126
    return (r.stdout or ""), r.returncode


def run_command(cmd, timeout, env=None):
    return core_evaluator.run_command(cmd, timeout, env, run_step=_run_one)
