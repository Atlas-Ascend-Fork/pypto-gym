# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Host-only CLI frontend for the registered-backend campaign test."""
import shutil
import statistics
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
from panko import backend
import feasibility
import predicates

catalog = str(SCRIPTS.parent / "references/action_catalog.json")
defaults = {"bayesian_optimization_tile_search": False, "success_stop": False}
default_reference = (1.0, "no_target:1us")
EXPORT_OUTCOMES = False


def bayesian():
    import bayesian_optimization
    return bayesian_optimization


def revision():
    return "host-test"


def state_metadata(args):
    return {"backend": "registered-host"}


def verify_initialization(args, state):
    if state.get("backend") != "registered-host":
        raise ValueError("Expected the registered-host backend after initialization")


def wrapper_signature(source):
    return source.split("@pypto.frontend.jit()", 1)[0]


def debug_options_signature(source):
    return None


def tile_constants(state, path, source):
    return {}


def entry_reads(entry):
    samples = entry.get("samples") or []
    return list(entry.get("reads") or ([statistics.median(samples)] if samples else []))


def write_source(path, source):
    Path(path).write_bytes(source.encode("utf-8"))


export_source = shutil.copyfile


def reading_interval(xs, ys, alpha=None):
    from panko.measurement import bootstrap_speedup
    return bootstrap_speedup(xs, ys, alpha=alpha)


def verification_policy():
    from panko.delivery import VerificationPolicy
    from panko.state import file_hash
    return VerificationPolicy(warmup=3, readings=lambda arm: arm.get("samples", []), interval=reading_interval,
                              matches=lambda op_dir, path, expected: file_hash(path) == expected)


if __name__ == "__main__":
    backend.register(sys.modules[__name__])
    import panko_harness
    panko_harness.main()
