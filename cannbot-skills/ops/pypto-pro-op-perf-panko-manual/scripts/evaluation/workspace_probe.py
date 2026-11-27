#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Measure source-declared on-chip high-water bytes for the Pro memory objective."""
from __future__ import annotations

import os
from pathlib import Path

from bayesian_optimization.space import HW
from memory import onchip_layout
from source_analysis import tile_sites

#: The spaces a figure is summed over, in report order.
SPACES = ("Vec", "Mat", "Left", "Right", "Acc")


def validate_capacity(capacity):
    """Require a positive byte count for every reported on-chip memory space."""
    if (not isinstance(capacity, dict) or capacity.keys() != set(SPACES)
            or any(type(value) is not int or value <= 0 for value in capacity.values())):
        raise ValueError(f"memory-capacity needs positive integer bytes for {', '.join(SPACES)}")


def high_water(src, budget=None, extra_consts=None, addr_budget=None):
    """
    Return (known per-space bytes, complete layout or None, unsized site ids, error).

    budget maps memory spaces to ceilings, with None meaning unknown. This reads
    the kernel as written, so addr_budget may allow its existing physical address
    range while reporting Layout.over_soft. New candidate placement keeps the
    software-usable bound. Unresolved regions are reported instead of sized at zero.
    """
    try:
        sites = tile_sites.TileProgram(src, extra_consts or {}, include_single_tiles=True).active_sites
    except SyntaxError as exc:
        return {}, None, [], "the kernel does not parse: %s" % (exc,)
    if not sites:
        return {}, None, [], None
    unsized = []
    try:
        groups = [onchip_layout.declared_regions(group) for group in onchip_layout.allocation_groups(sites)]
        unsized = [sid for _, unreadable in groups for sid in unreadable]
        plans = []
        for regions, _ in groups:
            if regions:
                plans.append(onchip_layout.plan(regions, dict(budget or {}), width=onchip_layout.width_of_text,
                                                addr_budget=dict(addr_budget or {})))
        layout = onchip_layout.combine_layouts(plans)
    except onchip_layout.LayoutRejected as exc:
        return {}, None, unsized, "the declared layout does not fit: %s" % (exc,)
    except (onchip_layout.LayoutInvalid, onchip_layout.LayoutOverlap) as exc:
        return {}, None, unsized, "%s: %s" % (type(exc).__name__, exc)
    if unsized:
        return dict(layout.used), None, unsized, "unresolved declared on-chip allocation: " + ", ".join(unsized)
    return dict(layout.used), layout, unsized, None


def _read_layout(path, budget=None, extra_consts=None, address_limits=None):
    if not path:
        return {}, None, [], "no kernel source to read: pass the implementation path"
    if not os.path.isfile(path):
        return {}, None, [], "no such kernel file: %r" % (path,)
    try:
        source = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {}, None, [], "cannot read %s: %s" % (path, exc)
    constants = {**tile_sites.sibling_constants(source, path), **(extra_consts or {})}
    return high_water(source, budget, constants, address_limits)


def validate_address_limits(limits):
    if limits is None:
        return
    if (not isinstance(limits, dict) or not limits.keys() <= set(SPACES)
            or any(type(value) is not int or value <= 0 for value in limits.values())):
        raise ValueError("address-limits needs positive integer byte limits for known on-chip spaces")


def profile_workspace(impl, budget, extra_consts=None, *, address_limits=None):
    """Read the same source footprint during acquisition and receipt validation."""
    validate_address_limits(address_limits)
    used, layout, unsized, error = _read_layout(impl, budget, extra_consts, address_limits)
    result = {"workspace_source": "declared_tile_addresses", "workspace_unsized": unsized,
              "workspace_b": None, "workspace_state": "unavailable"}
    if layout is None:
        if unsized:
            result["workspace_known_by_space"] = used
        return dict(result, workspace_error=error or "nothing was read")
    by_space = {space: int(used[space]) for space in SPACES if space in used}
    result.update(workspace_by_space=by_space, workspace_ungated=list(layout.ungated),
                  workspace_headroom={space: layout.headroom(space) for space in by_space})
    if layout.ungated:
        return dict(result, workspace_error=(
            "no capacity for %s on this arch, so the high-water in it "
            "has no ceiling behind it; the per-space figures are in "
            "workspace_by_space" % ", ".join(layout.ungated)))
    total = sum(by_space.values())
    return dict(result, workspace_b=total, workspace_state="measured" if total else "zero")


def check(path, hw):
    try:
        src = Path(path).read_bytes().decode("utf-8")
    except OSError:
        return True, ""
    _, _, _, error = high_water(src, hw.budget(), tile_sites.sibling_constants(src, path))
    if error and error.startswith(("the declared layout does not fit", "LayoutOverlap", "LayoutInvalid")):
        return False, error
    return True, ""


def _occupancy(path, key):
    # Occupancy symptoms require the same chip capacities used by the active run.
    from chip_profile import resolve
    envelope, _ = resolve()
    if not envelope or not envelope.get(key):
        return None
    space = "Vec" if key == "ub_kb" else "Mat"
    used, layout, _, _ = _read_layout(path)
    if layout is not None and space in used:
        return round(used[space] / (int(envelope[key]) * 1024), 4)
    return None


def ub_occupancy(path):
    return _occupancy(path, "ub_kb")


def l1_occupancy(path):
    return _occupancy(path, "l1_kb")
