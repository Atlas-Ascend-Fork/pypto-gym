# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Select a DSL adapter before importing the command modules.

The native entrypoint never registers one and retains its existing defaults.
A CLI process uses exactly one backend; command modules are not monkey-patched.
"""
import sys

_frontend = None


def register(frontend):
    """Install one DSL adapter for this process before importing panko_harness or panko.config.

    At import time the core reads the adapter's catalog path, defaults mapping,
    predicates and feasibility modules, and calls bayesian() for its tile-search API.
    A second or late registration raises RuntimeError. Classic uses the built-in defaults.

    INIT uses revision() for code provenance, default_reference as a (value, provenance)
    pair, state_metadata(args) to extend campaign state, and verify_initialization(args, state).

    Search uses wrapper_signature(source) and debug_options_signature(source) to check edits.
    If owned bindings were recorded, wrapper_signature also takes context and owned bindings.
    tile_constants(state, path, source) supplies tile-search keyword arguments.
    EXPORT_OUTCOMES controls whether outcome records also go to the shared prior store.

    entry_reads(row) returns independent readings. reading_interval(xs, ys, alpha=...)
    returns ratio bounds (ci_low and ci_high) or None. write_source(path, text) stages
    or restores code, export_source(src, dst) exports its snapshot, and verification_policy()
    returns the policy for the final ABBA comparison.
    """
    global _frontend
    if _frontend is not None or __package__ + ".config" in sys.modules:
        raise RuntimeError("select the PANKO backend once, before importing commands")
    _frontend = frontend


def current():
    return _frontend
