# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Hypervolume and each point's exclusive contribution, for a small front.

Everything here is in MINIMISATION space: the caller negates `max` objectives
first. A point counts only where it is strictly better than the reference on
every axis, so a point ON the reference contributes nothing -- which is what a
reference point means.

Exact, by slicing along the last axis and recursing (Fonseca, Paquete and
Lopez-Ibanez's HSO). Exponential in the number of objectives, which is two or
three here, and quadratic-ish in the number of points, which is a front of a
few tens at most. Nothing is approximated, so two runs that hold the same
front always choose the same base.
"""

from __future__ import annotations


def hypervolume(points, ref):
    """Volume dominated by `points` and bounded by `ref` (minimisation)."""
    d = len(ref)
    pts = [tuple(p) for p in points if all(p[i] < ref[i] for i in range(d))]
    if not pts:
        return 0.0
    if d == 1:
        return float(ref[0] - min(p[0] for p in pts))
    pts.sort(key=lambda p: p[-1])
    vol = 0.0
    for k, p in enumerate(pts):
        upper = pts[k + 1][-1] if k + 1 < len(pts) else ref[-1]
        h = upper - p[-1]
        if h > 0.0:
            vol += h * hypervolume([q[:-1] for q in pts[:k + 1]], ref[:-1])
    return vol


def contributions(points, ref):
    """Each point's exclusive hypervolume: HV(all) - HV(all but it)."""
    total = hypervolume(points, ref)
    out = []
    for i in range(len(points)):
        rest = points[:i] + points[i + 1:]
        out.append(max(0.0, total - hypervolume(rest, ref)))
    return out


def relative_gain(before, after, ref):
    """(gain, hv_before, hv_after): how much `after` adds, as a fraction.

    `None` for the gain when `before` encloses no volume at all -- the front
    came into existence rather than grew, and there is nothing to take a
    fraction of. The caller decides what that means.
    """
    hb = hypervolume(before, ref)
    ha = hypervolume(after, ref)
    if hb <= 0.0:
        return None, hb, ha
    return (ha - hb) / hb, hb, ha
