# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Run shape-only TPE after a tile action or structural edit. Charge measurements to the block."""
import math

from panko_core.bayesian_optimization import block as shared_block
from panko import measurement as _measure

from . import apply
from . import domain as domain_mod
from . import driver, lever, policy
from . import seed as seed_mod
from . import space


BlockOptions = shared_block.BlockOptions


def _block_start_points(src, hw, opts, sites):
    """Admit warm seeds without dropping the measured incumbent."""
    warm, measured, incumbent = shared_block.split_seed_points(lever.seed_points(
        src, hw, opts.dtype_bytes, opts.current_latency,
        widths=(opts.domain or {}).get("vec_dtype_bytes"), dom=opts.domain))
    groups = policy.shared_shape_groups(sites, domain=opts.domain) if opts.startup_cover else None
    return shared_block.complete_warm_starts(warm, measured, incumbent, sites, hw, opts,
                                             seed_mod, domain_mod, groups=groups)


def _tile_sites(src, hw, opts):
    sites = policy.shape_sites(apply.tunable_sites(src))
    if sites:
        return sites, None
    refused = apply.refusals(src)
    if refused:
        details = "; ".join("%s (%s)" % pair for pair in sorted(refused.items()))
        return [], "no tunable tile site: all %d pl.make_tile_group site(s) were refused -- %s" % (len(refused),
                                                                                                   details)
    return [], "no tunable tile site; this kernel writes no pl.make_tile_group(type=pl.TileType(shape=[...]))"


def _search(objective, sites, hw, opts, warm, measured, incumbent):
    options = shared_block.search_options(driver.SearchOptions, opts, warm, incumbent,
                                          apply.tie_groups(sites, incumbent), observed=measured, sites_per_draw=1)
    return driver.run_bo(objective, sites, hw, options)


def run(op_file, objective, hw, opts=None, *, confirm=None):
    opts = opts or BlockOptions()
    return shared_block.run_search(op_file, objective, hw, opts, discover=_tile_sites, seeds=_block_start_points,
                                   search=_search, prepare=lambda src,
                                   hw: apply.tile_program(src, getattr(hw, "extra_consts", None)), keep_ties=True,
                                   confirm=confirm or confirm_block_winner)


# Assumed log-difference noise for confirmation sample sizing.
CONFIRM_SIGMA_D = 0.02

CONFIRM_RESOLVE = 0.02

#: The one-sided z the pair count is sized at. 1.645 = 95 %.
CONFIRM_Z = 1.645

CONFIRM_PAIRS = max(2, math.ceil((CONFIRM_Z * CONFIRM_SIGMA_D / CONFIRM_RESOLVE) ** 2))


def confirm_block_winner(objective, incumbent, winner, seed=0):
    return _measure.confirm_block_winner(objective, incumbent, winner, seed, pairs=CONFIRM_PAIRS)


def apply_best(src, best_config, extra_consts=None):
    """Apply all winning site edits and return (source, changed-site count)."""
    if not best_config:
        return src, 0
    return apply.apply(src, best_config, extra_consts)


def clusters_for(src, extra_consts=None, tune_launch=False, domain=None):
    """Size the sweep using the sampler’s tied groups and active case constants."""
    try:
        program = apply.tile_program(src, extra_consts)
        sites = policy.shape_sites(apply.tunable_sites(program))
    except SyntaxError:
        return []
    incumbent = apply.incumbent_config(program)
    groups = policy.shared_shape_groups(sites, domain=domain)
    tie = policy.expanded_ties(sites, apply.tie_groups(sites, incumbent), groups)
    return space.clusters(sites, groups, tie)


def hw_for(op_file, live_tiles=None, envelope=None, author_src=None, extra_consts=None):
    """Return the hardware envelope and derived domain used for static checks.

    live_tiles describes simultaneous residency after structural edits. The caller supplies
    the same chip capacities as the feasibility gate. Calibration requires design-specific L1 data.
    Determine address ownership from the baseline author_src; use op_file when it is the baseline.
    """
    env = envelope or {}
    program, domain = domain_mod.analyze_file(op_file, live_tiles, env.get("ub_kb"), extra_consts)
    kw = {field: env[key] for field, key in shared_block.HW_FROM_ENVELOPE if key in env}
    consts = dict(domain.get("extra_consts") or {})
    sites = program.active_sites if program else ()
    author = sites if program and author_src == program.src else space.author_sites_of(author_src, consts)
    hw = space.HW(calibrated=False,
                  dtype_bytes=domain.get("dtype_bytes", 4),
                  vec_dtype_bytes=domain.get("vec_dtype_bytes"),
                  vec_trailing_ok=domain.get("vec_trailing"),
                  n_live_tiles=domain.get("live_tiles", 0),
                  sites=sites, extra_consts=consts,
                  # Resolve baseline sites using the same case constants as the candidate.
                  author_sites=author,
                  **kw)
    return hw, domain
