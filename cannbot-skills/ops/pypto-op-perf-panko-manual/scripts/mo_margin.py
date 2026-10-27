# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
The margin a candidate clears the ARCHIVE by, in the currency the scalar
ratchet already uses.

`_apply_ratchet` accepts on `j > bar` and SPRT reads `log(J/bar)`: one scalar,
one incumbent. The objectives a run declares never reach either -- they decide
what the archive RECORDS, not what the search ACCEPTS -- so phase 1 is a
single-objective search whatever `config.objectives` says.

Both generalise to a front without moving anything downstream, because
`log(J/bar)` IS the one-objective case of the multiplicative epsilon indicator.

For minimisation objectives `f` and an archive `A`, the factor by which a
candidate `c` clears archive point `a` is

    d(a, c) = max_i  f_i(a) / f_i(c)

-- the axis on which `c` most beats `a`. `d <= 1` exactly when `a` dominates
`c`, since then every ratio is at most one. Over the whole archive,

    margin(A, c) = log( min_{a in A} d(a, c) )

so `margin > 0` reads "no archive point dominates c", which for any fixed
reference point is also "c increases the hypervolume".

What this is NOT is a replacement for the SPRT margin, and the tests say so.
`max_i` reads the axis on which `c` is CLOSEST to `a`, so the margin is decided
entirely by the candidate's BEST axis and every other one is invisible: against
`(100, 100)`, the candidates `(101, 101)` and `(400, 101)` score the same
number, because both sit one percent of workspace from the front. Exact ties
make it starker -- where the front's points carry the same workspace to the
byte, the margin is zero however slow the candidate is.

That is the right answer to "how much improvement would clear the front", which
is the question this function asks, and the wrong input to "is this action worth
another evaluation", which is what SPRT asks. Acceptance and the stopping margin
are two different quantities and only the first is here. Wiring this into
a mean-based stopping test would make a four-times regression and a near miss the same
evidence.

With one objective and an archive holding only the incumbent it reduces, term
for term, to what the harness computes today:

    margin = log(p_best / p_c) = log(J_c / J_bar)

`tests/front/multi_objective/test_mo_margin.py` pins that equality. It is the safety net for moving
acceptance to the front: the stopping rules keep reading the same number.

Deterministic only. The noisy counterpart belongs with `dominates_measured`,
whose `undetermined` verdict means something different once it decides
ACCEPTANCE rather than recording -- see the noise note in the TODO.
"""

import math

INF = float("inf")


def _ratio(direction, a_v, c_v):
    """How much `c` beats `a` on ONE axis, as a factor, or None if unusable.

    A ratio, not a difference, so the two directions cannot share a sign flip
    the way `dominates` does: "higher is better" inverts to `c / a` rather than
    negating. Zero is a real reading here -- a kernel that reserves no workspace
    reports it -- so it is answered rather than refused: zero against a positive
    is an infinite win on that axis, and zero against zero is a tie.
    """
    if a_v is None or c_v is None:
        return None
    a_v, c_v = float(a_v), float(c_v)
    if a_v < 0.0 or c_v < 0.0:
        return None
    num, den = (c_v, a_v) if direction == "max" else (a_v, c_v)
    if den == 0.0:
        return INF if num > 0.0 else 1.0
    return num / den


def epsilon_margin(objs, archive_values, cand_values):
    """`log` of the factor by which `cand_values` clears every archive point.

    `objs` and the value dicts are the shapes `dominates` takes: a list of
    `{name, direction}` and plain `{objective: value}` mappings.

    Returns:
        > 0   no archive point dominates the candidate
        <= 0  some archive point dominates it, by that factor
        inf   the archive is empty -- there is nothing yet to clear
        None  an objective is missing or negative on either side

    `None` rather than a guess: a front built on a value nobody read is the
    failure `_refuse_objectives` exists to stop.
    """
    if not objs:
        return None
    best = INF
    for a in archive_values:
        worst_axis = 0.0
        for o in objs:
            name = o["name"]
            if name not in a or name not in cand_values:
                return None
            r = _ratio(o.get("direction", "min"), a[name], cand_values[name])
            if r is None:
                return None
            if r > worst_axis:
                worst_axis = r
        if worst_axis < best:
            best = worst_axis
    if best == INF:
        return INF
    if best == 0.0:
        return -INF
    return math.log(best)


def epsilon_margin_rows(objs, archive_rows, cand_values):
    """
    `epsilon_margin` against archive ROWS, which carry their values under
    `values` -- the shape `_dominates_row` unwraps.
    """
    return epsilon_margin(objs, [r["values"] for r in archive_rows], cand_values)


def extends_front(objs, archive_values, cand_values):
    """Does the candidate add to the front? (bool, margin).

    The acceptance predicate the ratchet would read in place of `j > bar`.
    `(False, None)` when the margin could not be computed, so a caller that
    cannot tell never accepts.
    """
    m = epsilon_margin(objs, archive_values, cand_values)
    if m is None:
        return False, None
    return m > 0.0, m
