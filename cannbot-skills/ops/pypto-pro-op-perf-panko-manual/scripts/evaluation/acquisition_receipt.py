# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Completion evidence for a profiled correctness loop, never a timing cache."""

import json
import os
import time
import uuid
from pathlib import Path

PATH_ENV = "PANKO_ACQUISITION_RECEIPT"
TOKEN_ENV = "PANKO_ACQUISITION_TOKEN"


def prepare(directory, env):
    """Use a fresh identity for each acquisition, including profile retries."""
    token = uuid.uuid4().hex
    path = Path(directory) / ("correctness-" + token + ".json")
    return path, token, dict(env, **{PATH_ENV: str(path), TOKEN_ENV: token})


def capture(run, script, args, launches, block_dim):
    """Only the NPU command-line runner enables this receipt; SIM is unchanged."""
    path, token = os.environ.pop(PATH_ENV, None), os.environ.pop(TOKEN_ENV, None)
    if not path or not token:
        return run(script, args, launches, block_dim=block_dim)
    timing = {}
    started = time.perf_counter()
    code = run(script, args, launches, block_dim=block_dim, timing=timing)
    receipt = {"version": 1, "token": token, "requested": launches,
               "completed": timing.pop("completed", 0), "exit_code": int(code),
               "timing_s": dict(timing, child_total=time.perf_counter() - started)}
    temporary = Path(path + ".tmp")
    temporary.write_text(json.dumps(receipt), encoding="utf-8")
    os.replace(temporary, path)
    return code


def validate(path, token, launches, returncode):
    """A profiler exit code or valid CSV alone cannot prove correctness."""
    try:
        receipt = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        receipt = None
    valid = (isinstance(receipt, dict) and receipt.get("version") == 1
             and receipt.get("token") == token
             and all(type(receipt.get(key)) is int
                     for key in ("requested", "completed", "exit_code"))
             and isinstance(receipt.get("timing_s", {}), dict))
    if not valid:
        return {"correctness_passed": False, "p": None, "samples": [], "sigma_log": None,
                "p_error": "profiled correctness completion receipt missing or invalid"}
    passed = (returncode == 0 and receipt.get("exit_code") == 0
              and receipt.get("completed") == launches and receipt.get("requested") == launches)
    result = {"correctness_passed": passed, "timing_s": receipt.get("timing_s", {}),
              "correctness_launches": receipt.get("completed", 0),
              "acquisition_strategy": "joint_correctness_profile"}
    if not passed:
        result.update(p=None, samples=[], sigma_log=None,
                      p_error="profiled correctness loop did not complete successfully")
        # A failed frozen test is a candidate failure. A missing receipt is an
        # acquisition failure instead; it must not become negative evidence.
        if receipt.get("exit_code") != 0:
            result["test_failed"] = True
    return result
