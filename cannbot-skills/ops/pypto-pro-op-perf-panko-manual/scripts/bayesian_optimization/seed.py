# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Choose feasible, diverse shape configurations for Pro warm starts."""
from dataclasses import dataclass
from typing import Any

from panko_core.bayesian_optimization import seed as _core_seed

from . import domain as tile_domain
from . import space


def _vec_features(v, hw, domain, site_id):
    return _core_seed.vector_features(v, hw, domain, site_id, fields=space.vec_fields(v),
                                     cores=getattr(hw, "vector_cores", None) or _core_seed.VECTOR_CORES)


def features(cfg, hw, domain=None):
    """Flat feature vector for a per-site config. Keys are stable, values finite."""
    return _core_seed.site_features(cfg, hw, domain, space.vec_fields, _vec_features)


def rand_vec(site_id, a, domain, rng):
    """A vec draw, or {} when the incumbent names no fields to draw."""
    return _core_seed.draw_vector(site_id, a, domain, rng, space)


def n_dims(sites, anchor=None):
    return _core_seed.count_dimensions(sites, anchor, {"global": 1, "launch": 1, "depth": 1}, space.vec_fields)


@dataclass
class CoverOptions(_core_seed.CoverOptions):
    """Classic cover options plus the Pro shape groups."""
    groups: Any = None


def cover(sites, hw, anchor, k, opts=None):
    """Choose feasible, distinct seeds far from each other and the measured incumbent."""
    opts = opts or CoverOptions()
    setup = _core_seed.cover_setup(sites, anchor, k, opts, n_dims, _core_seed.MIN_DIMS)
    if setup is None:
        return []
    rng, seen = setup

    rows = _core_seed.draw_pool(sites, anchor, opts.pool, lambda s, a: rand_vec(s.id, a, opts.domain, rng))
    if opts.groups:
        rows = (space.unflatten(space.flatten(cfg, sites, opts.groups), sites, opts.groups) for cfg in rows)

    def admitted(cfg):
        return (space.static_feasible(cfg, hw)[0] and (not opts.domain or tile_domain.verify(opts.domain, cfg)[0]))

    candidates = _core_seed.feasible_rows(rows, seen, admitted)
    return _core_seed.select_cover(candidates, anchor, k, hw, opts.domain, features)
