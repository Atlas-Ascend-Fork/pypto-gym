#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import predicates as shared_predicates
from evaluation import workspace_probe
from panko_core.pypto_ast import pypto_roots as core_roots

satisfies = shared_predicates.satisfies
CUBE_CALLS = {"matmul", "mmad", "cube_tile", "batch_matmul", "grouped_matmul", "quant_matmul"}

CUBE_SPACES = {"Left", "Right", "Acc"}

BROADCAST_CALLS = {"expand", "expand_clone", "expand_mul", "expand_add", "broadcast", "broadcast_to", "combine_axis"}

WRITEBACK_CALLS = {"assemble", "copy", "copy_out", "store", "scatter"}


def pypto_roots(tree):
    return core_roots(tree, packages=("pypto_pro", "pypto"),
                      seeds=("pypto_pro", "pypto", "pl", "vf"), include_from=True)


def _is_loop(call, roots):
    # A bare Python range is a host loop.
    return (shared_predicates.is_pypto_call(call, roots) and shared_predicates.call_name(call) in {"loop", "range"})


def _declares_cube_space(src):
    try:
        from source_analysis import tile_sites
        return any((site.space or "") in CUBE_SPACES for site in tile_sites.read_sites(src))
    except (SyntaxError, ImportError, ValueError):
        return False


RULES = shared_predicates.PredicateRules(pypto_roots, CUBE_CALLS, BROADCAST_CALLS, WRITEBACK_CALLS,
                                         _is_loop, _declares_cube_space)


def compute(impl_path, op_dir=None):
    return shared_predicates.compute(impl_path, op_dir, rules=RULES, gate=workspace_probe)
