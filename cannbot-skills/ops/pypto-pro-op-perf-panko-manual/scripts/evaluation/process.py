# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Bound an acquisition and clean up its process group before returning control."""
import math
import os
import signal
import subprocess

DEFAULT_TIMEOUT_S = 600


def run_captured(argv, *, timeout=DEFAULT_TIMEOUT_S, cwd=None, env=None):
    timeout = DEFAULT_TIMEOUT_S if timeout is None else float(timeout)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Acquisition timeout must be finite and positive.")
    proc = subprocess.Popen(argv, cwd=cwd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        output, _ = proc.communicate(timeout=timeout)
    except BaseException as exc:
        # Killing only msprof leaves its compiler/application holding the NPU
        # and the output pipe. The new session confines cleanup to this run.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        output, _ = proc.communicate()
        if isinstance(exc, subprocess.TimeoutExpired):
            exc.output = output
        raise
    return subprocess.CompletedProcess(argv, proc.returncode, output)
