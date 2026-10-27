#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Action selection as a posterior, not a product of hand-set factors.

The product score `w_stage * prior_gain * novelty * V` never learned anything:
`prior_gain` is transcribed from the catalogue, `novelty` counts attempts
without looking at what they measured, and nothing in the score moves with what
an action turned out to be worth. A low prior therefore keeps an action out of
the selection however well it measures, and the intents the optimizer inserts
are rarely selected at all.

Here each action is a Bernoulli arm: a measured candidate is kept (it beat the
running best) or it is not. The chance of keeping has a Beta posterior built
from three sources, weakest first:

  1. a base rate for the action's class -- the historical table when one is
     learned table when this machine has built one, else the fallback below;
  2. the model's belief V about THIS program, as `v_strength` pseudo-candidates
     -- it has read the source, which no table has;
  3. this run's own measurements of the same catalogue action.

Selection is Thompson sampling: draw p from each posterior and take the largest
`w_stage * p` (optionally `* g`, the expected log gain when kept -- see
`select` for why that is off by default). Wide posteriors -- actions
with little evidence -- are drawn high often enough to be tried, which is the
exploration the product lacked, and it stops as evidence accumulates.

Within one run an action node is consumed by its first close, so the run's own
counts move the posterior only for catalogue actions seen more than once. The
learning that matters is across campaigns, which is why the table exists.

Deterministic: the draws come from a generator seeded by (seed, select count),
so the same state selects the same action.
"""

import json
import math
import os
import random
import typing

# Weak base rates, used until this machine has measured enough to have a table
# of its own. They are deliberately weak: a population prior worth a handful of
# candidates, not a claim about any operator.
FALLBACK = {
    "global_kept_rate": 0.13,           # roughly one candidate in eight is kept
    "insert_kept_rate": 0.18,           # an inserted intent does a little better
    "global_log_gain": math.log(1 / (1 - 0.04)),     # a kept candidate gains a few percent
    "prior_gain_mean": 0.6,             # catalogue stars 0.3 / 0.6 / 0.9
}
MIN_CELL = 5          # a table cell below this many measurements is not used
TABLE_CAP = 50        # the table never speaks louder than this many candidates
SHRINK_K = 10         # pseudo-candidates pulling a sparse GAIN estimate to its parent
EB_MIN_N = 20         # cells used to estimate how much actions differ from each other
RATE_FLOOR, RATE_CEIL = 0.005, 0.95


def _clamp(x, lo=RATE_FLOOR, hi=RATE_CEIL):
    return max(lo, min(hi, x))


def load_table(path):
    """
    The prior table at `path`, or None. Never raises: a missing or broken table
    means the fallback, and the report says which one was used.
    """
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            t = json.load(f)
    except (OSError, ValueError):
        return None
    return t if isinstance(t, dict) and "global" in t else None


def empirical_bayes(table):
    """
    (a0, b0): the Beta prior of an action NOBODY has measured, fitted to how
    the measured actions' kept rates spread around their mean.

    Method of moments over cells with at least EB_MIN_N measurements, with the
    binomial sampling variance subtracted, so the spread is between actions and
    not within them. An unmeasured action is then believed to be an ordinary
    one, to within the spread actions actually show. A hand-set strength of 2
    instead makes every unmeasured action as likely to be drawn high as a
    measured winner, so the unknowns on the frontier take most of the draws.
    """
    cells = [c for c in (table.get("by_catalog") or {}).values()
             if (c.get("n_meas") or 0) >= EB_MIN_N]
    g = table["global"]
    base = _clamp(g["n_kept"] / max(1, g["n_meas"]))
    if len(cells) < 5:
        return 2 * base, 2 * (1 - base)
    rates = [c["n_kept"] / c["n_meas"] for c in cells]
    mu = sum(rates) / len(rates)
    var = sum((r - mu) ** 2 for r in rates) / len(rates)
    sampling = sum(mu * (1 - mu) / c["n_meas"] for c in cells) / len(cells)
    between = max(var - sampling, 1e-6)
    strength = max(2.0, min(200.0, mu * (1 - mu) / between - 1))
    mu = _clamp(mu)
    return mu * strength, (1 - mu) * strength


def _gain(cell, parent_gain):
    k = cell.get("n_kept", 0) or 0
    sg = cell.get("sum_log_gain_kept", 0.0) or 0.0
    return (sg + SHRINK_K * parent_gain) / (k + SHRINK_K)


def class_key(node):
    """The catalogue id, or `insert` for an action the optimizer added."""
    return node.get("catalog_id") or "insert"


def catalog_prior_gain(node):
    """The human hint: the catalogue's star.

    An optimizer insert has none, and gets the neutral 0.6: the model's opinion
    of its own intent is V, a separate factor, and letting its self-reported
    `prior_gain` also set g would hand the model two of the factors the core is
    meant to own. The insert CLASS still has a measured record in the table.

    Read from `prior_gain_catalog` when present, because `prior_gain` itself may
    by then hold the LEARNED value (see `learned_prior_gain`), and reading it
    here would count the table twice.
    """
    if not node.get("catalog_id"):
        return FALLBACK["prior_gain_mean"]
    v = node.get("prior_gain_catalog", node.get("prior_gain"))
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return FALLBACK["prior_gain_mean"]


def human_prior(node, a0, b0):
    """The population prior, re-centred on the human hint.

    The star is a HINT about which actions work, so it sets where belief starts
    and the measurements decide where it ends: the population's strength is
    kept, and only its centre moves -- by the star's ratio to the catalogue's
    modal 0.6. An action nobody has measured is ordered by its star; an action
    measured many times is ordered by what it did. The two disagree often
    enough to matter: a low-starred action can hold the best kept rate on
    record while a high-starred one has never been kept at all.
    """
    strength = a0 + b0
    mean = _clamp(a0 / strength * catalog_prior_gain(node) / FALLBACK["prior_gain_mean"])
    return mean * strength, (1 - mean) * strength


def _betacf(a, b, x):
    """Continued fraction for the incomplete beta (Numerical Recipes, Lentz)."""
    tiny, eps = 1e-300, 3e-14
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def beta_cdf(x, a, b):
    """Regularised incomplete beta I_x(a, b). No scipy: this runs in the core."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbt = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
           + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbt) * _betacf(b, a, 1.0 - x) / b


def beta_ppf(q, a, b):
    """The q-quantile of Beta(a, b), by bisection -- deterministic to 1e-10."""
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if beta_cdf(mid, a, b) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def population_prior(table):
    """(a0, b0): the prior of an action nobody has measured."""
    if table:
        return empirical_bayes(table)
    m = FALLBACK["global_kept_rate"]
    return 2 * m, 2 * (1 - m)


def learned_prior_gain(node, table, predicates, stats=None, quantile=None):
    """g: the catalogue star updated by evidence, on the catalogue's own scale.

    `1 - 0.4 ** (rate / reference)`: an ordinary action reads 0.6 (the
    catalogue's modal star), twice the reference 0.84, half of it 0.37, never 0
    unless it never works and never above 1. Bounded, so it stands in for
    `prior_gain` in `V_sel` without breaking the argument that assumes [0, 1].

    `rate` is the posterior over this action's kept rate -- the population
    prior re-centred on the star, the table's record, and `stats`, this run's
    own measurements of it (so an action that failed here, and was revived by a
    structure change, carries that failure; this is what `novelty` was for).

    `quantile` makes g OPTIMISTIC under uncertainty (Bayes-UCB, Kaufmann et al.
    2012): the rate read is the posterior's q-quantile rather than its mean,
    and the reference is the same quantile of the population prior. An action
    measured hundreds of times has a narrow posterior and reads about its mean;
    an action nobody has measured reads high, because it MIGHT be good. That is
    the exploration the selection score needs, inside the score: `V_sel`
    stays an argmax. None = the posterior mean.
    """
    a, b, _, _ = base_prior(node, table, predicates)
    s = (stats or {}).get(stats_key(node)) or {}
    a += s.get("kept", 0)
    b += max(0, s.get("n", 0) - s.get("kept", 0))
    a0, b0 = population_prior(table)
    if quantile is None:
        rate, ref = a / (a + b), a0 / (a0 + b0)
    else:
        rate, ref = beta_ppf(quantile, a, b), beta_ppf(quantile, a0, b0)
    return round(1.0 - 0.4 ** max(0.0, rate / ref), 4)


def base_prior(node, table, predicates):
    """(alpha, beta, expected log gain, source) before this run's evidence."""
    key = class_key(node)
    if table:
        a0, b0 = human_prior(node, *empirical_bayes(table))
        g = table["global"]
        g_gain = (g.get("sum_log_gain_kept", 0.0) / g["n_kept"]
                  if g.get("n_kept") else FALLBACK["global_log_gain"])
        cell = (table.get("by_catalog") or {}).get(key)
        if not (cell and (cell.get("n_meas") or 0) >= MIN_CELL):
            return a0, b0, g_gain, "table:global"
        n, k = cell["n_meas"], cell["n_kept"]
        # Down-weighted, not trusted in full: the table is old code, and every
        # count in it was selected by the product score it replaces.
        w = min(1.0, TABLE_CAP / n)
        alpha, beta = a0 + w * k, b0 + w * (n - k)
        strength = alpha + beta
        mean = alpha / strength
        # Each predicate cell describes the SAME measurements as the action's
        # marginal, split one way, so the only information it adds is how its
        # rate DIFFERS from the marginal's. It enters as a lift against the
        # marginal's raw rate, shrunk towards 1 by the population strength, and
        # the lifts are averaged geometrically, never multiplied. Folding the
        # cells in as data counts those measurements once per predicate, which
        # can send one sparsely measured action -- identical in every cell --
        # to most of the draws. As a lift, each split moves the action's rate
        # once.
        raw = k / n
        s_eb = a0 + b0
        by_pred = table.get("by_catalog_predicate") or {}
        logs = []
        for p, v in sorted((predicates or {}).items()):
            if not isinstance(v, bool) or p == "determined":
                continue
            pc = by_pred.get(f"{key}|{p}={v}")
            if pc and (pc.get("n_meas") or 0) >= MIN_CELL and raw > 0:
                r = (pc["n_kept"] + s_eb * raw) / (pc["n_meas"] + s_eb)
                logs.append(math.log(max(r, RATE_FLOOR) / raw))
        if logs:
            mean = mean * math.exp(sum(logs) / len(logs))
        mean = _clamp(mean)
        return mean * strength, (1 - mean) * strength, _gain(cell, g_gain), f"table:{key}"
    if key == "insert":
        m = FALLBACK["insert_kept_rate"]
        return 2 * m, 2 * (1 - m), FALLBACK["global_log_gain"], "fallback:insert"
    pg = catalog_prior_gain(node)
    m = _clamp(FALLBACK["global_kept_rate"] * pg / FALLBACK["prior_gain_mean"])
    return 2 * m, 2 * (1 - m), FALLBACK["global_log_gain"], "fallback:prior_gain"


def global_rate(table):
    """The base kept rate every other rate is read against."""
    if table:
        g = table["global"]
        return _clamp(g["n_kept"] / max(1, g["n_meas"]))
    return FALLBACK["global_kept_rate"]


def posterior(node, stats, table, predicates, v_strength=2.0):
    """Beta(alpha, beta) for the node's kept rate, its expected gain, and why."""
    alpha, beta, gain, src = base_prior(node, table, predicates)
    base = global_rate(table)
    # A V of one half reads as "an ordinary action": the base rate, and a V
    # of one doubles it.
    # Only a V the model WROTE is evidence about this program. At seeding V is
    # a copy of the catalogue star, which the base prior has already used.
    if node.get("V_source") != "catalog":
        v = max(0.0, min(1.0, float(node.get("V", 0.5) or 0.0)))
        mv = _clamp(2.0 * base * v)
        alpha += v_strength * mv
        beta += v_strength * (1 - mv)
    s = (stats or {}).get(stats_key(node)) or {}
    kept, n = s.get("kept", 0), s.get("n", 0)
    alpha += kept
    beta += max(0, n - kept)
    if kept and s.get("sum_log_gain"):
        gain = (s["sum_log_gain"] + SHRINK_K * gain) / (kept + SHRINK_K)
    return alpha, beta, gain, src


def stats_key(node):
    """
    Where a node's measurements are pooled: per catalogue action, or per
    inserted node (an insert has no class to share with).
    """
    return node.get("catalog_id") or node["id"]


def observe(stats, node, improved, log_gain):
    """Fold one measured candidate into the run's counts."""
    s = stats.setdefault(stats_key(node), {"n": 0, "kept": 0, "sum_log_gain": 0.0})
    s["n"] += 1
    if improved:
        s["kept"] += 1
        s["sum_log_gain"] = round(s["sum_log_gain"] + max(0.0, log_gain), 6)


class DrawContext(typing.NamedTuple):
    """What one Thompson draw reads: the run's counts, the prior table, predicates, seed and knobs."""
    stats: typing.Any
    table: typing.Any
    predicates: typing.Any
    seed: typing.Any
    draw_no: typing.Any
    v_strength: typing.Any = 2.0
    use_gain: typing.Any = False


def select(frontier, weight, ctx):
    """(chosen node, draws) -- Thompson over the frontier.

    `weight(node)` is the stage weight. Nodes are visited in id order so the
    generator's sequence, and so the choice, depends on the state alone.

    `use_gain` multiplies in the expected log gain g. Off by default: the only
    g available is a MEAN over kept candidates, and those means are set by the
    first moves of a campaign -- an action selected early averages a large cut
    while its median is small -- so g mostly measures how early an action tends
    to be selected. With it on, one early action takes most of the draws and
    the inserts stop being selected. g is still computed and reported.
    """
    rng = random.Random(f"{ctx.seed}:{ctx.draw_no}")
    draws = []
    for n in sorted(frontier, key=lambda x: x["id"]):
        a, b, g, src = posterior(n, ctx.stats, ctx.table, ctx.predicates, ctx.v_strength)
        p = rng.betavariate(a, b)
        draws.append({"id": n["id"], "class": class_key(n), "p": round(p, 6),
                      "mean": round(a / (a + b), 6), "g": round(g, 6),
                      "w": weight(n),
                      "score": round(weight(n) * p * (g if ctx.use_gain else 1.0), 8),
                      "prior": src})
    best = max(draws, key=lambda d: (d["score"], d["id"]))
    chosen = next(n for n in frontier if n["id"] == best["id"])
    return chosen, sorted(draws, key=lambda d: -d["score"])


def propensity(frontier, weight, ctx, chosen_id, m=200):
    """Monte-Carlo estimate of the probability that `select` chooses `chosen_id`.

    Recorded with each outcome so the table rebuild can correct for which
    actions the search chose to measure (inverse-propensity weighting). Uses its
    own generator namespace, so it never disturbs the draw that was made.
    """
    rng_seed = f"prop:{ctx.seed}:{ctx.draw_no}"
    hits = 0
    for i in range(m):
        draw = DrawContext(ctx.stats, ctx.table, ctx.predicates, rng_seed, i,
                           ctx.v_strength, ctx.use_gain)
        ch, _ = select(frontier, weight, draw)
        hits += ch["id"] == chosen_id
    return round(max(hits, 0.5) / m, 4)
