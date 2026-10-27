# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""BO driver (Optuna TPE) for the numeric config subspace, correctness-constrained.

The objective is the SAME frozen evaluator E(x) that PANKO uses, wired in by the caller:

    objective(cfg) -> (s, p)
        s in {0,1} : correctness = golden_compare AND layout_check AND lint  (hard constraint)
        p          : measured latency (or geomean latency across shapes)

Correctness is a HARD constraint, exactly like PANKO's J = s * P_ref / p: an incorrect config
can NEVER win. A config that BO returns as best is therefore drop-in for PANKO cmd_record /
cmd_close — same referee, same J.

Three layers of infeasibility, cheapest first:
    1. static_feasible()  -> compiler-rule reject, no eval at all
    2. objective s==0     -> compile fail or golden mismatch (numerical)
Both are scored with a large penalty so TPE's surrogate learns to avoid that region
(e.g. "nbuffer=2 breaks this recurrent kernel", "this tile overflows L1") instead of re-drawing it.

No LLM runs inside this loop: suggest -> static check -> deterministic apply -> E(x). Fully
reproducible for a given seed.
"""

import json
import re

from dataclasses import dataclass
from functools import partial
from typing import Any

import optuna

# Aliased: `domain` is a parameter of `run_bo`, and a parameter shadows a
# module for the whole function body.
from . import evaluator, space
from . import domain as domain_mod
from .lever import _dist

# Penalty returned for infeasible / incorrect trials. Any real latency is far below this, so the
# best feasible trial is simply argmin over trials with value < PENALTY.
# A trial that produced a wrong kernel, however it was wrong.
_FAILURE_STATUSES = ("incorrect", "incorrect_capacity")

PENALTY = 1e12

# `ErrCode: F4FFFF! Enum: InternalError::PASS_INNER_ERROR. [PadLocalBuffer.Tensor]:...`
# Everything after the pass name is per-instance -- tensor ids, PIDs, timestamps,
# byte counts -- so only these three fields identify the KIND of refusal.
_ERRCODE_RE = re.compile(r"ErrCode:\s*([A-Za-z0-9]+)")
_ENUM_RE = re.compile(r"Enum:\s*([A-Za-z_]+::[A-Za-z_]+)")
# `[Pass.Thing]:message` -- the colon is required, because without it this also
# matches a pytest parametrize id such as `test_op[torch.float16]`.
_PASS_RE = re.compile(r"\[([A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*)\]:")


# Enums that name a memory-allocation refusal. These are facts about how much
# room the PROGRAM left, so they move when the program's residency or view
# extents move -- exactly what a structural delta changes. `evaluator`'s
# CAPACITY_MARKERS carries the UB wording and some generic ones, but nothing
# that matches `exceeds MEM_L1 size`, so an L1 or L0 overflow arrives here
# classified `other` and would otherwise be carried forever, permanently hiding
# a tile that fits the next program. Refuse to carry any of them and let
# `ceiling` (which generalises far better) do the work it can.
#
# Substring match on the enum, which is SCREAMING_CASE in every pypto error
# observed. A differently-spelled allocation enum would slip through.
_ALLOCATION_ENUM = ("MEMORY", "ALLOCATION", "OUT_OF_RANGE")


def error_signature(detail):
    """The kind of a compiler refusal, stripped of everything per-instance.

    A signature is produced ONLY when the text carries a pypto ErrCode AND a
    pypto Enum. Requiring both is not fussiness: the pass-name pattern alone
    matches any bracketed dotted identifier, and pytest writes parametrize ids
    like `test_op[torch.float16]` into exactly the output a golden mismatch
    hands back -- so the looser form promoted a NUMERICAL failure into a
    permanent cross-structure "compiler refusal". Golden and lint failures are
    structure-dependent by construction and must never be carried.

    "" when the text is not a pypto front-end error, and an empty signature
    never matches anything.

    NOTE what this does NOT establish: that two refusals sharing a signature
    share a cause. `F4FFFF` is the generic FeError code and this repo's own
    experience table maps it to several. The claim is only that the signature is
    stable across occurrences of the SAME refusal, which is what carrying one
    forward requires.
    """
    if not detail:
        return ""
    code = _ERRCODE_RE.search(detail)
    enum = _ENUM_RE.search(detail)
    if not (code and enum):
        return ""
    if any(t in enum.group(1) for t in _ALLOCATION_ENUM):
        return ""
    parts = [code.group(1), enum.group(1)]
    pas = _PASS_RE.search(detail)
    if pas:
        parts.append(pas.group(1))
    return "|".join(parts)


def _refusal_key(params):
    return json.dumps(params, sort_keys=True)


def _dedup_refusals(rows):
    """One entry per (params, signature), first occurrence wins."""
    out, seen = [], set()
    for r in rows:
        if not r.get("params") or not r.get("sig"):
            continue
        k = (_refusal_key(r["params"]), r["sig"])
        if k in seen:
            continue
        seen.add(k)
        out.append({"params": r["params"], "sig": r["sig"]})
    return out


def _dedup(*groups):
    """One entry per (params, value), first occurrence wins, order preserved.

    A block that resumes replays its predecessor's observations into the study,
    so the study and the memory it came from overlap. Union them rather than
    concatenate.
    """
    out, seen = [], set()
    for g in groups:
        for t in g:
            k = (json.dumps(t.get("params"), sort_keys=True), t.get("value"))
            if k in seen:
                continue
            seen.add(k)
            out.append(t)
    return out


def site_widths(opts):
    """Per-site (element width, residency) from the derived domain, or {}.

    Both, because both are how the buffer is actually spent. `domain.verify`
    charges a site `elements x width x residency`; this charged `elements x
    width` and took the max over sites. On a kernel whose vec#1 holds 3 tiles
    and vec#3 holds 4, a draw that overflows because of vec#3 sets the ceiling
    from whichever site is largest in that wrong currency -- so the innocent
    site is retired at exactly its own legal maximum and the site that caused
    the refusal stays under the ceiling and is drawn again until the block's
    failure limit closes it.
    """
    dom = getattr(opts, "domain", None) or {}
    widths = dom.get("vec_dtype_bytes") or {}
    res = dom.get("vec_residency") or {}
    keys = set(widths) | set(res)
    if not keys:
        return {}
    default = dom.get("dtype_bytes") or getattr(opts, "dtype_bytes", 4)
    # `live_for`, not the raw walk: the ceiling has to be in the same currency
    # the gate spends, and the gate charges the capped residency.
    return {
        sid: (widths.get(sid) or default) * max(1, domain_mod.live_for(dom, sid))
        for sid in keys
    }


def draw_tile_bytes(cfg, dtype_bytes, widths=None):
    """Bytes in the largest tile of `cfg`. 0 when the config has no tile site.

    `widths` is the per-site BUFFER COST of one tile element -- the element width
    times the site's residency, as `site_widths` builds it from the derived domain.
    It matters because the ceiling is a claim about SIZE and is compared across
    sites, so it has to be in the currency the buffer is actually spent in. With
    one width for the whole kernel and no residency at all, the number was an
    element count wearing a byte label: in a kernel whose `dtype_bytes` came out
    1, a refusal at an fp32 site retires int8 tiles of the same element count
    that would have fitted four times over, and a refusal caused by a 4-live
    site is charged to whichever site has the most elements.
    """
    widths = widths or {}
    n = 0
    for sid, c in cfg.items():
        db = widths.get(sid) or dtype_bytes
        dims = [v for k, v in c.items() if k.startswith("d")]
        if dims:
            p = 1
            for d in dims:
                p *= d
            n = max(n, p * db)
        elif "mL0" in c:
            n = max(n, c["mL0"] * c["kL0"] * db)
    return n


# The `Invalid L1/L0 relation` error class -- and the machinery that used to learn
# it from the device -- is gone. It only ever occurred because the config carried
# L0 alone and `apply` wrote the three cube lists as operand pairs, so the L1
# the compiler paired against L0 was one BO could neither see nor set. Now L1 is a
# real per-axis knob (mL1/kL1/nL1), `suggest` draws it, and `space._cube_feasible`
# enforces `0 < L0 <= L1 and L1 % L0 == 0` statically, before the device. There is
# nothing left to learn: an illegal pair is rejected for free at the gate.


@dataclass
class SearchOptions:
    """Everything about HOW the tile search runs, as one value.

    `run_bo` takes what it searches -- an objective, the sites, the envelope --
    and this. Nineteen parameters made every call site a wall of keywords that
    could not be read against the signature, and made adding one knob a change
    in two files.

    The defaults are the production ones; a caller overrides what it means to
    and says nothing about the rest.
    """

    warm_start: Any = None  # configs measured before TPE draws
    n_trials: int = 40
    seed: int = 0
    groups: Any = None  # site ids tied to one knob
    on_trial: Any = None  # callback(record) after each trial
    stagnation_limit: int = 10
    domain: Any = None
    n_startup_trials: int = 3
    dtype_bytes: int = 4
    fault_limit: int = 3
    failure_limit: int = 6
    memory: Any = None
    anchor: Any = None  # the config the kernel is running
    startup_min_feasible: int = 5
    refusals: Any = None  # compiler refusals already earned
    # The smallest footprint the compiler has refused on THIS TILE DOMAIN, from
    # any earlier block. Separate from `memory` because it is a different kind of
    # fact: `memory.trials` are latencies, comparable only within one program
    # shape, while the ceiling says which tiles the compiler will accept, which
    # depends on the derived domain and nothing else. Sharing one key made the
    # ceiling inherit the narrower lifetime -- see `block.ceiling_key`.
    ceiling: Any = None
    tie: Any = None
    repair: bool = True  # shrink a too-big draw instead of rejecting it


def _emit(ctx, rec, trial, status, reason=None):
    """Record one trial outcome: the history, the caller's callback, the study's attrs."""
    rep = ctx.get("repair")
    if rep:
        # `config` is what was MEASURED; the draw it stands for is kept beside it.
        rec = dict(
            rec,
            repaired_from=rep["from"],
            repair_steps=rep["steps"],
            repair_reason=rep["why"],
        )
    ctx["history"].append(rec)
    if ctx["opts"].on_trial:
        ctx["opts"].on_trial(rec)
    trial.set_user_attr("status", status)
    if reason is not None:
        trial.set_user_attr("reason", reason)


def _known_value(ctx, cfg, key, trial):
    """What this block, or an earlier one, already knows about these exact params.

    A configuration already measured in this block is not a new experiment.
    Nothing here prevented TPE from drawing the same categorical point twice, and
    it does: with most of the space statically rejected the acquisition collapses
    onto the handful of cells that returned a real value and re-draws them. One
    recorded block spent SEVEN of its twelve device trials -- 64 % of what it
    charged, about twenty minutes of device time -- re-measuring one identical
    configuration, whose seven latencies agreed to 0.38 %.

    Returning the stored value rather than skipping keeps the study honest: the
    sampler is told exactly what it would have been told by the device, so the
    stagnation counter still advances and the block still stops on its own terms.
    It just stops without paying.

    Returns the value to hand the study, or None when the draw must go on.
    """
    if key in ctx["state"]["seen"]:
        v = ctx["state"]["seen"][key]
        _emit(
            ctx,
            {
                "config": cfg,
                "status": "duplicate_config",
                "objective_us": None,
                "reason": f"already measured in this block ({v})",
            },
            trial,
            "duplicate_config",
        )
        return v
    if key in ctx["refused"]:
        _emit(
            ctx,
            {
                "config": cfg,
                "status": "known_compile_failure",
                "objective_us": None,
                "reason": f"the compiler refused this configuration before "
                f"({ctx['refused'][key]}); carried across structures",
            },
            trial,
            "known_compile_failure",
        )
        return PENALTY
    return None


def _largest_site(cfg, dtype_bytes, widths=None):
    best, nb = None, 0
    for sid, c in cfg.items():
        b = draw_tile_bytes({sid: c}, dtype_bytes, widths)
        if b > nb:
            best, nb = sid, b
    return best, nb


def static_reason(ctx, cfg):
    """(status, reason) for a draw the free gates reject, else None.

    Pure: the caller decides whether to emit it or to repair the draw.
    """
    opts, hw = ctx["opts"], ctx["hw"]
    nb = draw_tile_bytes(cfg, opts.dtype_bytes, site_widths(opts))
    # `> floor`, not just `< ceiling`: a ceiling is only actionable while it sits
    # ABOVE the footprint the kernel is running at. `_lower_ceiling` checks that
    # when it LEARNS one, and that was enough while a ceiling died with the
    # action that bought it. It no longer does -- `block.ceiling_key` carries it
    # across every block on the same tile domain -- and the incumbent moves
    # underneath it: a later block on the same domain can be running a smaller
    # footprint than the block a ceiling was legitimately learned against, so
    # the carried ceiling sits below the running value. Acting on it would
    # retire the ladder including the value already
    # in the kernel, which is the undetectable narrowing this design is most
    # exposed to. So the guard is applied again here, where it is USED.
    if nb and nb >= ctx["state"]["ceiling"] and nb > ctx["state"]["floor"]:
        sid, _ = _largest_site(cfg, opts.dtype_bytes, site_widths(opts))
        return (
            "over_learned_ceiling",
            f"{sid}: largest tile {nb}B >= {ctx['state']['ceiling']}B, "
            f"which the device refused",
        )
    ok, reason = space.static_feasible(cfg, hw)
    if ok and opts.domain:
        ok, reason = domain_mod.verify(opts.domain, cfg, hw.dtype_bytes)
    return None if ok else ("infeasible_static", reason)


def _try_repair(ctx, cfg):
    """The nearest feasible config below `cfg`, or None. See `space.repair`."""
    if not ctx["opts"].repair:
        return None, 0

    def check(c):
        bad = static_reason(ctx, c)
        return (True, "") if bad is None else (False, bad[1])

    return space.repair(
        cfg,
        check,
        space.RepairContext(ctx["anchor"], ctx["opts"].domain, ctx["opts"].tie),
    )


def _static_verdict(ctx, cfg, key, trial):
    """The gates that reject a draw without touching the device.

    Returns PENALTY when the draw is rejected, or None when it must be measured.
    `key` is unused here and kept for one signature across the gates.
    """
    del key
    opts, hw = ctx["opts"], ctx["hw"]
    nb = draw_tile_bytes(cfg, opts.dtype_bytes, site_widths(opts))
    # `> floor` for the same reason as in `static_reason`: a carried ceiling can
    # sit below a later incumbent on the same domain, and retiring the running
    # value is the one narrowing that leaves no trace.
    if nb and nb >= ctx["state"]["ceiling"] and nb > ctx["state"]["floor"]:
        _emit(
            ctx,
            {
                "config": cfg,
                "status": "over_learned_ceiling",
                "objective_us": None,
                "reason": f"{nb}B >= {ctx['state']['ceiling']}B, "
                f"which the device refused",
            },
            trial,
            "over_learned_ceiling",
        )
        return PENALTY
    ok, reason = space.static_feasible(cfg, hw)
    if ok and opts.domain:
        # The derived domain is re-checked per candidate, which is what makes
        # erring WIDE safe: a cell the ladder allows but the program cannot hold
        # is rejected here, for free, instead of on the device.
        ok, reason = domain_mod.verify(opts.domain, cfg, hw.dtype_bytes)
    if not ok:
        _emit(
            ctx,
            {
                "config": cfg,
                "status": "infeasible_static",
                "reason": reason,
                "objective_us": None,
            },
            trial,
            "infeasible_static",
            reason,
        )
        return PENALTY
    return None


def _record_fault(ctx, cfg, exc, trial):
    """Not a verdict on this configuration.

    Record it, tell the study nothing, and let `_stopper` decide whether the
    device is unusable. `detail` and `kind` are carried out, not just counted. A
    run that ends on env faults ends on THIS record and nothing else, and the
    count alone is unusable: a handful of faults can abandon most of a budget,
    with no way afterwards to tell whether the device had been held or the
    command mistyped.
    """
    rec = {
        "config": cfg,
        "status": "env_fault",
        "reason": exc.marker,
        "detail": getattr(exc, "detail", ""),
        "kind": getattr(exc, "kind", "device"),
        "retries": getattr(exc, "retries", 0),
        "objective_us": None,
    }
    _emit(ctx, rec, trial, "env_fault")
    ctx["state"]["faults"] += 1
    ctx["state"]["fault_kind"] = rec["kind"]


def _lower_ceiling(ctx, cfg):
    """A configuration the device refused is evidence about SIZE, and the evidence
    is monotone: nothing larger can fit either.

    TPE cannot see that, because every tile dimension is a CATEGORICAL parameter
    and the ladder carries no order -- a failure at 8192 says nothing to it about
    12288. So the ceiling is recorded here and enforced by the free static gate,
    which turns one paid failure into the whole upper half of the ladder instead
    of one rung.

    Never below the footprint the incumbent is running at. That program executes,
    so a ceiling at or under its size is incoherent, and acting on it would retire
    the entire ladder including the value already in the kernel -- the
    undetectable narrowing this design is most exposed to. A capacity marker on a
    tile smaller than the running one means the marker matched something else.
    """
    state = ctx["state"]
    nb = draw_tile_bytes(cfg, ctx["opts"].dtype_bytes, site_widths(ctx["opts"]))
    if nb and nb < state["ceiling"] and nb > state["floor"]:
        state["ceiling"] = nb


def _record_refusal(ctx, cfg, key, outcome, trial):
    """The device ran the candidate and refused it.

    `kind is not None` matters: the objective contract allows a 2-tuple return,
    which binds `kind = None`, and `None` means UNCLASSIFIED, not "definitely not
    capacity". Reading it as the latter both skipped the `ceiling` update and
    filed a genuine UB overflow as a permanent cross-structure refusal -- wrong
    twice in the same branch. An unclassified refusal is carried nowhere.
    """
    kind, detail = outcome
    if kind == "capacity":
        _lower_ceiling(ctx, cfg)
    elif kind is not None:
        # A front-end pass refusing this tile call's own arguments, so it is a
        # statement about the tile rather than about how much room the program
        # left. Record it for every LATER block, whatever the structure becomes.
        sig = error_signature(detail)
        if sig:
            ctx["new_refusals"].append({"params": json.loads(key), "sig": sig})
    _emit(
        ctx,
        {
            "config": cfg,
            "status": "incorrect_capacity" if kind == "capacity" else "incorrect",
            "reason": detail or "s=0",
            "objective_us": None,
        },
        trial,
        "incorrect",
    )
    # A refusal is a measurement too: re-running it costs the same evaluation and
    # returns the same PENALTY.
    ctx["state"]["seen"][key] = PENALTY
    if ctx.get("repair"):
        ctx["state"]["seen"][ctx["repair"]["drawn_key"]] = PENALTY
    return PENALTY


# Consecutive unevaluated trials a block tolerates before it gives up. A
# compile that will not finish inside the budget usually will not finish for the
# next few draws either -- the space is continuous and TPE draws near what it
# just drew -- so an unbounded retry would spend the whole block on timeouts.
UNEVALUATED_LIMIT = 3


def record_unevaluated(ctx, cfg, detail, trial):
    """The device never returned a number, and that is not the candidate's fault.

    `_record_refusal` does four things that are all WRONG for a compile killed by
    the budget, and each one is a lasting error rather than a lost trial:

      - it lowers the `ceiling` on a "capacity" kind, teaching a size limit that
        was never demonstrated;
      - it files an `error_signature` as a cross-structure refusal, so a timeout
        would free-reject this tile in every LATER block;
      - it writes `state["seen"][key] = PENALTY`, so the study remembers the
        configuration as measured-and-terrible;
      - it returns PENALTY as the objective value, which tells TPE this point is
        bad when the truth is that there is no observation here at all.

    So: none of those. The caller raises `optuna.TrialPruned`, which is what
    "no observation" means to the study, and the trial does not count toward
    stagnation either -- `stagnation` counts measurements that failed to
    improve, and this measured nothing.
    """
    ctx["state"]["unevaluated"] += 1
    _emit(
        ctx,
        {
            "config": cfg,
            "status": "unevaluated",
            "reason": detail or "compile exceeded eval_timeout_s",
            "objective_us": None,
        },
        trial,
        "unevaluated",
    )


def _register_measured(ctx, cfg, p, trial):
    """File a repaired trial's value under the params that were MEASURED.

    A repaired draw is two different points: the one TPE drew, which the device
    cannot run, and the one we shrank it to, which we measured. Reporting the
    measured value for the DRAWN point tells the surrogate that the infeasible
    region is fine -- so it keeps drawing there, and since every such draw is now
    repaired and measured, it keeps costing an evaluation. Unbounded, a run can
    spend most of its device trials re-measuring repairs of draws from the same
    infeasible region, without any action closing.

    So the value goes where it was measured, as its own completed trial, and the
    drawn point gets the PENALTY a static rejection would have given it. The
    measurement is still kept and can still win the block -- what changes is the
    coordinates it teaches.

    Returns True when the measured point was filed; False leaves the caller on
    the old path, because a study that will not accept the trial must not cost a
    block its measurement.
    """
    study = ctx.get("study")
    params = (ctx.get("repair") or {}).get("measured_params")
    if study is None or not params:
        return False
    try:
        study.add_trial(
            optuna.trial.create_trial(
                params=params,
                distributions=dict(trial.distributions),
                value=float(p),
                user_attrs={
                    "status": "ok",
                    "measured_params": params,
                    "repaired_into": True,
                },
            )
        )
    except (ValueError, TypeError, RuntimeError):
        return False
    return True


def record_ok(ctx, cfg, key, p, trial):
    """A measured latency.

    The diagnostics the evaluator just read are carried with it. They are not part
    of the objective -- the study still optimises `p` alone -- but the winning
    trial is later promoted into the tree, and a spine node whose `util` reads 0.0
    because nobody passed it along is indistinguishable from a kernel that
    genuinely idled.
    """
    # `objective_us` is what this trial was RANKED on, which is `p`. The field
    # used to be called `latency`, which named no instrument. The launch's other
    # readings ride on `metrics` below.
    rec = {"config": cfg, "status": "ok", "reason": "", "objective_us": float(p)}
    met = getattr(ctx["objective"], "last_metrics", None)
    if met:
        rec["metrics"] = dict(met)
    samples = getattr(ctx["objective"], "last_samples", None)
    if samples:
        rec["samples"] = list(samples)
    _emit(ctx, rec, trial, "ok")
    ctx["state"]["seen"][key] = float(p)
    if ctx.get("repair"):
        # The drawn point is infeasible whatever the repaired one measured.
        if _register_measured(ctx, cfg, p, trial):
            ctx["state"]["seen"][ctx["repair"]["drawn_key"]] = PENALTY
            return PENALTY
        ctx["state"]["seen"][ctx["repair"]["drawn_key"]] = float(p)
    return float(p)


def _objective(ctx, trial):
    """One trial: draw a configuration, reject it for free if we can, else measure it."""
    opts, sites = ctx["opts"], ctx["sites"]
    cfg = space.suggest(trial, sites,
                        space.DrawOptions(groups=opts.groups, anchor=ctx["anchor"],
                                          domain=opts.domain, tie=opts.tie))
    key = json.dumps(space.flatten(cfg, sites, opts.groups, opts.tie), sort_keys=True)
    ctx["repair"] = None
    known = _known_value(ctx, cfg, key, trial)
    if known is not None:
        return known
    bad = static_reason(ctx, cfg)
    if bad is not None:
        fixed, steps = _try_repair(ctx, cfg)
        if fixed is None:
            return _static_verdict(ctx, cfg, key, trial)
        # Measure the repaired point; the value is reported for the drawn one.
        measured_params = space.flatten(fixed, sites, opts.groups, opts.tie)
        ctx["repair"] = {"from": cfg, "steps": steps, "why": bad[1][:200],
                         "drawn_key": key, "measured_params": measured_params}
        trial.set_user_attr("measured_params", measured_params)
        cfg = fixed
        key = json.dumps(space.flatten(cfg, sites, opts.groups, opts.tie), sort_keys=True)
        known = _known_value(ctx, cfg, key, trial)
        if known is not None:
            ctx["state"]["seen"][ctx["repair"]["drawn_key"]] = known
            return known
    try:
        # An objective may return (s, p) or (s, p, kind). `kind` is "capacity"
        # only when the failure is monotone evidence about size.
        got = ctx["objective"](cfg)
        s, p, kind = (got if len(got) == 3 else (got[0], got[1], None))
        detail = getattr(ctx["objective"], "last_detail", "")
    except evaluator.EnvironmentFault as e:
        _record_fault(ctx, cfg, e, trial)
        raise optuna.TrialPruned()
    ctx["state"]["faults"] = 0
    if s:
        ctx["state"]["failures"] = 0
        ctx["state"]["unevaluated"] = 0
        return record_ok(ctx, cfg, key, p, trial)
    if kind == "unevaluated":
        record_unevaluated(ctx, cfg, detail, trial)
        raise optuna.TrialPruned()
    ctx["state"]["unevaluated"] = 0
    ctx["state"]["failures"] += 1
    return _record_refusal(ctx, cfg, key, (kind, detail), trial)


def _abort(ctx, study, why, fault_kind=None):
    """End the block on the device rather than on the search."""
    ctx["stag"]["aborted"] = why
    if fault_kind is not None:
        ctx["stag"]["fault_kind"] = fault_kind
    study.stop()


def _stopper(ctx, study, trial):
    """Early stop: halt once `stagnation_limit` trials in a row fail to improve the
    best FEASIBLE latency.

    Infeasible / incorrect / worse trials all count as no-improvement; a genuine
    feasible improvement resets the counter. Same idea as PANKO's global
    stagnation.
    """
    state, stag, opts = ctx["state"], ctx["stag"], ctx["opts"]
    if state["faults"] >= opts.fault_limit:
        # The device, not the search, is the problem. Stop rather than spend the
        # rest of the block proving it.
        _abort(
            ctx,
            study,
            f"{state['faults']} consecutive environment faults ({state['fault_kind']})",
            state["fault_kind"],
        )
        return
    if opts.failure_limit and state["failures"] >= opts.failure_limit:
        # The device ran and refused, over and over. Removing failures from the
        # stagnation counter was right -- "seven trials without an improvement" is
        # a claim about seven MEASUREMENTS -- but it left a block with no brake at
        # all: the first run under that fix spent two dozen device trials and
        # measured exactly one of them. This is the brake that should have come
        # with it, counted separately so a block that is searching badly is
        # distinguishable from one that cannot search at all.
        _abort(
            ctx,
            study,
            f"{state['failures']} consecutive candidates the device would not run",
        )
        return
    if state["unevaluated"] >= UNEVALUATED_LIMIT:
        # A brake of the same shape as the two above, for the third thing that
        # measures nothing. Named separately in the abort reason because the
        # remedy is different: faults mean fix the device, failures mean the
        # search is drawing badly, and this one means `eval_timeout_s` is too
        # small for the programs this action produces.
        _abort(
            ctx,
            study,
            f"{state['unevaluated']} consecutive compiles past eval_timeout_s",
        )
        return
    v = trial.value
    if v is None or v >= PENALTY:
        # Nothing was measured. "Seven trials without an improvement" is a
        # statement about seven MEASUREMENTS; counting an unmeasurable trial
        # towards it closes a block that never got to look. The first recorded
        # block died this way at eight trials with one measurement.
        return
    if v < stag["best"] - 1e-12:
        stag["best"], stag["since"] = v, 0
    else:
        stag["since"] += 1
    # A floor on grounded data before the block may give up. The concern the
    # `--live-tiles`/startup discussion raised is real: a block that stops on
    # stagnation having measured only a couple of feasible tiles has fitted its
    # verdict to almost nothing, and one recorded block did exactly that --
    # stopped at ~14 trials with the ladder's high end never drawn, and the
    # winning tile was at that high end. Counting FEASIBLE trials (not draws,
    # which are mostly rejected) and refusing to stop below the floor lets TPE
    # keep exploring; measured 8/8 seeds finding the optimum where the early stop
    # found 4/8. Rejections do not count, so a hard kernel is not forced to spin
    # -- it simply has few feasible trials to hold back for.
    if _n_feasible(study) < opts.startup_min_feasible:
        return
    if opts.stagnation_limit and stag["since"] >= opts.stagnation_limit:
        stag["stopped"] = True
        study.stop()


def _n_feasible(study):
    """Trials that returned a real latency."""
    return sum(1 for t in study.trials if t.value is not None and t.value < PENALTY)


def _make_study(opts):
    """The study, with 3 startup trials rather than the default 10.

    Two of the three warm-start seeds are already observations by the time the
    sampler is consulted, and blocks in the recorded runs converge in nine to
    twenty trials -- ten random startup draws would spend most of a block warming
    up. TPE's own interleaved exploration is kept: replacing it with pure
    random-until-N-feasible was measured to converge WORSE (it modelled greedily
    on a few random points and stagnated before exploring the high end of the
    ladder). The floor on initial measurements is enforced on the STOPPER instead
    -- a block may not early-stop until `startup_min_feasible` feasible trials
    exist.
    """
    return optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=opts.seed, n_startup_trials=opts.n_startup_trials
        ),
    )


def _replay(study, sites, opts, anchor, memory):
    """Replay the earlier block's measurements before anything new is drawn."""
    replayed = 0
    for t in memory.get("trials", []):
        if t.get("value") is None:
            continue
        try:
            study.add_trial(
                optuna.trial.create_trial(
                    params=t["params"],
                    distributions={
                        k: _dist(k, sites, opts.groups, anchor) for k in t["params"]
                    },
                    value=float(t["value"]),
                    # A repaired observation keeps what was measured, or a resumed
                    # block would report the unrepaired draw as its best config.
                    user_attrs=(
                        {"measured_params": t["measured"]} if t.get("measured") else {}
                    ),
                )
            )
            replayed += 1
        except Exception:  # a param that no longer exists on this program
            continue
    return replayed


def _warm_start(study, sites, opts, memory):
    """Try known-good configs before the sampler explores.

    Skipped for anything already replayed -- re-measuring it would spend an
    evaluation to learn what the study was just handed.
    """
    seen = set()
    for t in memory.get("trials", []):
        seen.add(json.dumps(t.get("params"), sort_keys=True))
    for cfg in opts.warm_start or []:
        flat = space.flatten(cfg, sites, opts.groups, opts.tie)
        if json.dumps(flat, sort_keys=True) in seen:
            continue
        study.enqueue_trial(flat)


def initial_state(memory, anchor, opts):
    """The block's running facts about the device.

    `ceiling`: the smallest tile footprint the device has refused in this block.
    `faults` : consecutive ENVIRONMENT faults -- the device was unusable.
    `failures`: consecutive CANDIDATE failures -- the device ran and refused.

    The last two are counted separately because they mean different things and
    want different limits, and neither is a stagnation: stagnation counts
    MEASUREMENTS that failed to improve, and none of these measured anything.
    """
    # The domain-level ceiling wins when the caller supplies one; `memory` is
    # the fallback so a state file written before the split still resumes.
    return {
        "ceiling": float(opts.ceiling or memory.get("ceiling") or float("inf")),
        "faults": 0,
        "failures": 0,
        "fault_kind": None,
        # Consecutive trials whose COMPILE ran past the budget. A fourth
        # kind alongside faults, failures and stagnation, for the same
        # reason the docstring gives: it measured nothing, so it is neither
        # a verdict about the tile nor a stagnation.
        "unevaluated": 0,
        "floor": draw_tile_bytes(anchor, opts.dtype_bytes, site_widths(opts))
        if anchor
        else 0,
        # flattened params -> what the device returned. Seeded from the
        # predecessor block's observations so a resumed block does not pay
        # again for what its predecessor already measured.
        "seen": {
            json.dumps(t["params"], sort_keys=True): float(t["value"])
            for t in memory.get("trials", [])
            if t.get("value") is not None
        },
    }


def _carried_refusals(opts):
    """Refusals the COMPILER made, carried across structures.

    Unlike `memory`, which is keyed by structure and rightly forgotten when the
    program changes. A run can pay for one deterministic compile error several
    times over. An oversized `set_vec_tile_shapes` raises

        F4FFFF | InternalError::PASS_INNER_ERROR | PadLocalBuffer.Tensor

    in the block for every view it is tried on -- several structures, several
    device trials, one fact. Nothing kept it: `state["seen"]`
    dies with the block, and the memory written at the end filters to
    `value < PENALTY`, so refusals are dropped on purpose.

    ONLY non-capacity refusals are carried. A capacity refusal depends on the
    program's residency and view extents, which a structural delta changes --
    carrying it could permanently hide a tile that fits the new program, and it is
    already generalised far better by `ceiling`, which retires the whole upper
    half of the ladder rather than one rung. A front-end pass rejecting the tile
    call's own arguments is a different kind of fact: it is about the tile, and it
    survives the restructure.
    """
    return {
        _refusal_key(r["params"]): r.get("sig", "")
        for r in (opts.refusals or [])
        if r.get("params")
    }


def _result(study, ctx, memory, replayed):
    """What the block returns to its caller."""
    opts, state, stag = ctx["opts"], ctx["state"], ctx["stag"]
    history = ctx["history"]
    feasible = [t for t in study.trials if t.value is not None and t.value < PENALTY]
    best = min(feasible, key=lambda t: t.value) if feasible else None
    ceiling = None if state["ceiling"] == float("inf") else int(state["ceiling"])
    return {
        # The MEASURED params: a repaired trial's own params are the draw, which
        # the static rules reject and which was never run.
        "best_config": (space.unflatten(best.user_attrs.get("measured_params", best.params),
                                        ctx["sites"], opts.groups, opts.tie)
                        if best else None),
        "n_repaired": sum(1 for h in history if h.get("repaired_from") is not None),
        "best_objective_us": best.value if best else None,
        # Trials the study RAN, not trials it holds. `enqueue_trial` creates a
        # WAITING trial immediately, so `len(study.trials)` counts points that
        # were queued and never reached because the budget ran out first -- a
        # block granted 1 trial reported 2, and with covering seeds enqueued it
        # would report 5. The number is logged as the block's cost and read as
        # "how much did this action explore", so it has to be what happened.
        # (Charging was never affected: that path counts `device_trials`.)
        "n_trials": sum(1 for t in study.trials
                        if t.state != optuna.trial.TrialState.WAITING),
        "n_feasible": len(feasible),
        "stopped_early": stag["stopped"],
        "aborted": stag["aborted"],
        "fault_kind": stag.get("fault_kind"),
        "learned_ceiling": ceiling,
        "replayed": replayed,
        # The replayed trials are already in `study.trials`, so appending the
        # study to the memory it was built from stored each of them twice --
        # and a block resumed three times stored them four times. A recorded run
        # shows it plainly: one node on one structure carries the same measured
        # point as two separate observations. TPE weights an observation by how
        # often it
        # appears, so the duplicates quietly biased the surrogate toward
        # whatever the interrupted block happened to measure first.
        "memory": {"trials": _dedup([dict({"params": t.params, "value": t.value},
                                          **({"measured": t.user_attrs["measured_params"]}
                                             if "measured_params" in t.user_attrs else {}))
                                     for t in feasible],
                                    memory.get("trials", [])),
                   "ceiling": ceiling},
        # DOMAIN-level, so it is returned separately from `memory` for the caller
        # to store under the domain alone. It stays inside `memory` too, so a
        # caller that has not been taught the new key keeps working.
        "ceiling": ceiling,
        # Structure-INDEPENDENT, so it is returned separately from `memory` and
        # the caller stores it at run level rather than under the structure key.
        "refusals": _dedup_refusals((opts.refusals or []) + ctx["new_refusals"]),
        "env_faults": sum(1 for h in history if h.get("status") == "env_fault"),
        "failures": sum(1 for h in history
                        if h.get("status") in _FAILURE_STATUSES),
        "history": history,
    }


def run_bo(objective, sites, hw, opts=None):
    """Run TPE over the PER-SITE tile-shape space.

    objective   : site_configs -> (s, p). Wire the frozen E(x) here on the NPU; a mock in tests.
    sites        : the tunable Site list (apply.tunable_sites(impl)); defines the space.
    hw          : space.HW envelope (or {site_id: HW}) for the static pre-check.
    warm_start  : list of per-site config dicts to evaluate first (e.g. apply.uniform_config
                  from the current production tiles).
    groups      : optional list of id-lists to tie sites (one knob per group); default per-site.
    on_trial    : optional callback(record_dict) after each trial (logging / stream to PANKO).
    n_trials    : the UPPER BOUND on trials.
    stagnation_limit : early-stop after this many consecutive trials with no improvement to the best
                  feasible latency (mirrors PANKO's global_stagnation_limit; 0/None disables it).
                  Small spaces (e.g. an elementwise op's 2 knobs) converge and stop well under the cap.

    Returns dict: best_config (per-site), best_objective_us, n_trials, n_feasible, stopped_early, history.
    """
    opts = opts or SearchOptions()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # `memory` carries what an earlier block on THIS SAME program measured:
    # {"trials": [{"params": ..., "value": ...}], "ceiling": bytes}. The caller
    # keys it by structure, so a program that has changed shape arrives with
    # nothing and starts from zero -- observations taken on the old structure
    # are not evidence about the new one. What it does prevent is a block that
    # was cut short by a busy device paying again for everything the first
    # attempt already learned -- without it each retry begins with
    # `ceiling=None` and re-derives the same footprint bound from scratch.
    memory = opts.memory or {}
    study = _make_study(opts)
    # Anchor the domains on the INCUMBENT so its values are always legal choices -- real kernels
    # use values outside the static sets, and Optuna rejects an enqueued warm-start value that
    # isn't in the categorical domain. The anchor is also the reference for the capacity `floor`
    # and for the learned L1 relation, and all three want the program that is actually running.
    #
    # It used to default to `warm_start[0]`, which is not the incumbent: the incumbent is the one
    # seed that carries a latency, so `block.seeds_for` routes it to `measured` and the anchor
    # became the FLOOR seed -- the smallest scaling of the current shape that clears 16 KB. See
    # the note in `block.seeds_for` for what that cost. The fallback is kept for callers that
    # pass no incumbent (tests, and any block on a kernel with no measured latency yet).
    anchor = opts.anchor or (opts.warm_start[0] if opts.warm_start else None)
    replayed = _replay(study, sites, opts, anchor, memory)
    _warm_start(study, sites, opts, memory)
    ctx = {
        "objective": objective,
        "sites": sites,
        "hw": hw,
        "opts": opts,
        "study": study,
        "anchor": anchor,
        "history": [],
        "new_refusals": [],
        "state": initial_state(memory, anchor, opts),
        "refused": _carried_refusals(opts),
        "stag": {"best": float("inf"), "since": 0, "stopped": False, "aborted": None},
    }
    study.optimize(
        partial(_objective, ctx),
        n_trials=opts.n_trials,
        callbacks=[partial(_stopper, ctx)] if opts.stagnation_limit else None,
    )
    return _result(study, ctx, memory, replayed)
