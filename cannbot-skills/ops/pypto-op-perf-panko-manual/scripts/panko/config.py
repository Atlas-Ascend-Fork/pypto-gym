#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Defaults, shared types, and what this installation has available.

Nothing in the package may be imported from here, so this module imports
nothing from it: it is the bottom of the stack. The optional dependencies
are resolved once, here, so every guard downstream reads one name.
"""

import logging
import os
import sys
import typing

import jsonio


# The package needs `optuna`, which is a real dependency of the tile search and
# of nothing else here. Absent, the search still runs with
# `bayesian_optimization_tile_search` off -- that is the ablation arm, and the
# only supported way to run without it. `cmd_init` refuses the combination of
# the flag ON and the package missing, so the two can never disagree silently.
#
# ONLY ModuleNotFoundError, and only for optuna. `except Exception` swallowed a
# circular import, a syntax error in this package, and an optuna API change
# alike, set every module to None, and let the run continue reporting
# `tile_search: true` while the numbers were being chosen by the model instead.
# Those are bugs and they must reach a traceback.
OPTUNA_MIN_VERSION = (2, 0)             # `create_trial` / `add_trial`; developed against 4.9


# Diagnostics only, never the protocol: `out` owns stdout, and this channel
# carries the best-effort notes that must not be able to change an answer. It
# gets its own handler with `propagate` off for the reason jsonio states: a root
# logger configured by an embedding process must not reformat either stream.
_diag_log = logging.getLogger("pypto.panko.stderr")


if not _diag_log.handlers:
    _diag_handler = logging.StreamHandler(sys.stderr)
    _diag_handler.setFormatter(logging.Formatter("%(message)s"))
    _diag_log.addHandler(_diag_handler)
    _diag_log.setLevel(logging.WARNING)
_diag_log.propagate = False


def is_missing_optuna(exc):
    """Is this import failure the optional dependency, or a bug in this package?

    Only a ModuleNotFoundError naming optuna is the former. A circular import, a
    syntax error in this package and an optuna API change all raise ImportError
    or a ModuleNotFoundError naming something else, and each of those is a
    defect that has to reach a traceback rather than silently turn the tile
    search off.
    """
    return (isinstance(exc, ModuleNotFoundError)
            and (getattr(exc, "name", "") or "").split(".")[0] == "optuna")


try:
    import bayesian_optimization as bayesian
    _BAYESIAN_UNAVAILABLE = None
except ModuleNotFoundError as _e:       # pragma: no cover
    if not is_missing_optuna(_e):
        raise                           # this package is broken, not a missing optional dep
    # One name for the whole package, so the guards downstream are `bayesian is
    # None` rather than eleven separate module flags. The previous form bound
    # `= None, shutil` -- a TUPLE -- to three of them, so every `is None` guard
    # read False on the baseline arms.
    bayesian = None
    _BAYESIAN_UNAVAILABLE = f"optuna is not importable ({_e})"


def bayesian_capability():
    """Whether the tile search can actually run, and on what.

    Reported at INIT and persisted, so a run's report names the algorithm that
    ran rather than the one that was configured.
    """
    if bayesian is None:
        return {"available": False, "optuna_version": None,
                "reason": _BAYESIAN_UNAVAILABLE}
    try:
        import optuna                                        # noqa: PLC0415
        version = getattr(optuna, "__version__", "")
        parts = tuple(int(x) for x in version.split(".")[:2] if x.isdigit())
    except Exception as e:                                   # pragma: no cover
        return {"available": False, "optuna_version": None,
                "reason": f"optuna imported but unusable: {e}"}
    if parts and parts < OPTUNA_MIN_VERSION:
        return {"available": False, "optuna_version": version,
                "reason": (f"optuna {version} < "
                           f"{'.'.join(str(x) for x in OPTUNA_MIN_VERSION)}; "
                           "the tile search needs trial.create_trial / study.add_trial")}
    return {"available": True, "optuna_version": version, "reason": None}


try:                                   # optional: absent -> the gate is a no-op
    import chip_profile
    import feasibility
    import predicates
except Exception:                      # pragma: no cover
    # BOTH names, or the "gate is a no-op" fallback is a NameError instead: the
    # first reference is `cmd_init`'s `if (predicates and ...)`, so the run would
    # fail to start on exactly the configuration this handler exists for.
    chip_profile = feasibility = predicates = None

STAGE_ORDER = ["frontend", "swimlane", "incore"]


STAGE_ORDER = ["frontend", "swimlane", "incore"]


# How many DISTINCT structures a refusal must be seen on before it is carried to
# a structure it has not been seen on. 1 restores the old behaviour (carry from
# the first sighting); a value above 2 buys more evidence at the price of paying
# for the refusal on that many structures first.
REFUSAL_GLOBAL_AFTER = 2


# The chip envelope every static rule is validated against: buffer capacities in
# KB and the core counts the seed generator uses to ask "does every core get
# work". ONE dict, because there were two.
#
# `feasibility.HW` and `bayesian_optimization.space.HW` each carried their own
# defaults and they disagreed: L1 was 512 in one and 192 in the other, and
# space.py's own comment says the 192 "was an old conservative estimate that has
# no basis now that the real buffer is checked". Only the feasibility side was
# reachable from the CLI, so `--l1-kb` moved one envelope while the tile search
# kept using the other. On A3, today, a legal cube tile can be rejected by the
# stale number.
#
# TRANSCRIBED, and no longer the default: `init` DERIVES the envelope from the
# platform ini (see chip_profile.py) and refuses rather than falling back here.
# These remain for one case only -- a state file written before the field
# existed, where nothing better is knowable -- and `source` says so wherever the
# envelope is reported.
#
# They were transcribed faithfully from one SKU, which is not the point: they are
# right for that SKU only, and already wrong on another one in the same family,
# which carries these same five buffers and a different core count. Nor would a
# per-generation constant help -- two SKUs of one generation can share every
# buffer size and differ several fold in core count.
CHIP_ENVELOPE = {
    "ub_kb": 192,
    "l1_kb": 512,
    "l0a_kb": 64,
    "l0b_kb": 64,
    "l0c_kb": 128,
    "cube_cores": 20,
    "vector_cores": 40,
    # The chip's memory, in MiB. It is the hypervolume's reference point on a
    # byte-valued objective: nothing larger than this can run, so a program at
    # or past it encloses no volume -- the same thing a constraint violation
    # means. Derived from the platform ini like the rest; None when that SKU's
    # ini does not carry it, and a run whose objectives include a byte axis then
    # needs `--chip-memory-mib`.
    "memory_mib": 65536,
    "source": "transcribed:910B3",      # becomes the ini path once derived
}


DEFAULT_CONFIG = {
    # Refinement allowance per action: how many candidates may fail to beat the
    # running global best before the action is closed. This is a hyperparameter
    # of the search loop, not a claim either method makes, so both arms must use
    # the same value or the comparison measures the setting. 7 is the PANKO
    # baseline's own default; adopting it rather than our own removes any charge
    # that it was tuned in our favour.
    #
    # It was 3, which favours breadth. On the level-1 operators, where the
    # payoff is concentrated in one direction that needs many attempts to tune,
    # 3 closed the productive action after three near misses: several separate
    # actions can each produce a candidate within 10% of the best and be retired
    # anyway.
    "stagnation_K": 7,
    # Candidates the readings cannot order against the front. `stagnation_K`
    # counted them as failures, and a near-miss is not one: the search proposes
    # them whatever the noise is, so an action could be closed on almost
    # nothing. They advance this counter instead, and its own bound closes
    # the action -- seven candidates nobody can tell apart from the front is as
    # much a reason to stop as seven that clearly failed, and the fix for it is
    # more readings per measurement, not more candidates.
    "undetermined_K": 7,
    # Static rejections are counted separately from stagnation and bounded
    # separately. `stagnation_K` counts MEASUREMENTS that failed to improve;
    # a rejection is not a measurement. This bound exists only so that an action
    # whose whole neighbourhood is infeasible terminates: it spends no budget,
    # so eval_budget would never end it. Set well above the five rejections that
    # is the most any single action accumulated across the recorded runs, because
    # a rejection costs one dispatch rather than one device run.
    # Proposals that never reached the device: a static rejection, a semantic
    # no-op, a byte-duplicate, a compile that ran past `eval_timeout_s`. One
    # counter, because they are one fact about the action -- it is proposing
    # things that produce no measurement -- and the per-kind bounds this
    # replaces were set at different times for different reasons rather than
    # from evidence, and most of them never fired at all.
    #
    # 10, from the duplicates, which are the only kind that fires in volume. An
    # action can re-propose the SAME bytes many times over, interleaved, so the
    # consecutive-streak bound this also replaces never saw three in a row.
    # None of it costs `eval_budget`; what it costs is one LLM round trip each,
    # spent to produce no measurement.
    "unmeasured_K": 10,
    # `eval_timeout_s` is DERIVED from the preopt's compile time when `init` is
    # given one (`--preopt-compile-s`), because the right value is a property of
    # the operator and not of the method: compile time rises steeply with the
    # candidate's workspace, and candidates grow. A single
    # transcribed number either burns a hung compile's full budget on a small
    # kernel or truncates a large one -- the same failure the chip envelope had
    # when its core counts were transcribed rather than derived.
    "eval_timeout_k": 4,          # multiple of the preopt compile
    "eval_timeout_floor_s": 300,  # never below the old default
    "eval_timeout_cap_s": 1800,   # wall clock is the real budget; see `eval_timeout_source`
    "staged_prior": True,       # soft-staging toggle (staged-prior vs free-tree)
    # 24 h of the time the campaign was WORKED, which is what `wall_clock_s`
    # now holds -- see `accumulate_wall_clock`. It used to be elapsed time since
    # init, which on an orchestrated campaign is mostly time nobody was driving
    # it.
    "wall_clock_limit_s": 86400,
    # An interval between two harness invocations longer than this is idleness,
    # not work. 2 h, against a longest observed legitimate cycle of 3360 s.
    "idle_grace_s": 7200,
    "eval_timeout_s": 300,        # 5 min per candidate (enforced by the optimizer's `timeout`)
    # Evaluations per operator; None means unlimited. Raised from 90: the recorded
    # campaigns kept ending on the budget rather than on a search decision, so
    # the number was measuring the cap and not the method. 300 lets the
    # deterministic stops (stagnation, frontier exhaustion, wall clock) be the
    # ones that actually fire.
    "eval_budget": 300,
    # None = DISABLED. The counter still advances and is still reported; it simply
    # no longer halts the search. Rationale: in the 2026-07 campaign this limit
    # halted every PANKO run at 23-67 evaluations of a 120 budget while the
    # PANKO baseline ran on, which makes the budget accounting incomparable
    # across methods. Set an int to restore the old behaviour (Sec. 7.4 ablation).
    "global_stagnation_limit": None,
    # False = advance stages cyclically but never halt on a no-gain cycle. With
    # stage_patience=2 over three stages, `converged` is reachable in six
    # unproductive closes (~18 evaluations), well inside the budget.
    "stage_convergence": False,
    # End-of-run verification (`verify`): the delivered kernel and the preopt,
    # re-measured in an ABBA order at `verify_runs` readings each.
    "end_verify": True,
    # How `select` chooses. "product" is the original w_stage * prior_gain *
    # novelty * V; "thompson" samples each action's kept-rate posterior (see
    # action_posterior.py). The default stays "product" until replaying the
    # recorded campaigns shows thompson does not select worse.
    "selection": "product",
    "selection_seed": 0,
    "thompson_v_strength": 2.0,
    "thompson_use_gain": False,
    # Historical base rates; empty = this machine's learned table if one has
    # been built, else the fallback inside action_posterior.
    "action_prior_table": "",
    # Where a seeded node's `prior_gain` comes from. "learned" = the catalogue
    # star updated by the table (`action_posterior.learned_prior_gain`), which
    # the product score then uses in place of the star; "catalog" = the star as
    # written, the original behaviour. Falls back to the star when no table
    # can be read. The star is always kept as `prior_gain_catalog`.
    "prior_gain_source": "learned",
    # Rebuild $PANKO_HOME/action_prior_table.json from every campaign's
    # outcomes after a real stop (see prior_store). Off for experiments, which
    # also pin `action_prior_table` so a run reads the table it was designed on.
    "prior_auto_update": True,
    # Optimism of g under uncertainty: g reads this quantile of the action's
    # kept-rate posterior (Bayes-UCB) rather than its mean, so an action with
    # little evidence -- an insert, a barely-measured catalogue action -- scores
    # as what it MIGHT be. None reads the mean. See learned_prior_gain.
    "prior_gain_quantile": 0.8,
    # Snapshot EVERY measured candidate under nodes/<hash>.py, not only the ones
    # the ratchet keeps. A front point that trades latency for workspace is a
    # DELIVERABLE, and without this its source is gone: only the kept
    # candidates survive, and a reverted one cannot be recovered from the
    # repository at all.
    "snapshot_all_candidates": True,
    # What the search optimises, in order. The FIRST is the primary objective;
    # the front is sorted by the others first. `J`, the ratchet and `impl.py`
    # follow `p`, which is msprof's per-call MIX_AIC time (measure_latency). A name
    # from OBJECTIVE_ALIASES, or a dict {name, from, direction} -- `from` is
    # `p` (the measured latency) or `m.<key>`, so a user's own objective needs
    # no code change, only the value in `m` (record --metrics).
    # `mix_aic` is still an objective a run may DECLARE, and its readings are
    # carried on `m` either way -- it is not in the default because a campaign
    # measured what it costs here. It is the noisier of the two instruments by
    # an order of magnitude in sigma_log, and two readings of the SAME bytes
    # move it, so with `undetermined` a tie it mostly ties and adds
    # little to the front -- while a declared objective that cannot be read HALTS
    # the run, and msprof's per-call accounting breaks down at small workspaces,
    # which is exactly the region a workspace objective explores.
    #
    # The DEFAULT is one objective, and one objective is the single-objective
    # search: with nothing to trade against, the front holds one point and the
    # chosen point IS the global best, so the ratchet, `impl.py` and `J` behave
    # as they did before any of the front machinery existed. Naming a second
    # objective -- `["aicore_e2e", "workspace"]` -- is what turns the
    # multi-objective mode on; see SKILL.md (Two modes). A run that names a
    # byte-valued objective needs the chip's memory (`--chip-memory-mib` when
    # the SKU's ini does not carry it), because that figure is the
    # hypervolume's reference point on that axis; one objective needs none.
    "objectives": ["aicore_e2e"],
    # Dominance on the MEASURED objective is decided by a percentile bootstrap
    # of the difference of medians, not by the medians themselves: two points
    # 0.3% apart on a kernel whose sigma_log is 2% are not ordered, and a front
    # that says they are is a claim the readings do not support. Undetermined
    # means NEITHER dominates, so both stay -- the front errs toward keeping a
    # trade-off rather than toward a confident ordering. The structural
    # objectives (workspace) are compared exactly; they have no noise.
    # `false` restores the plain median comparison.
    "archive_bootstrap": True,
    "archive_alpha": 0.05,
    # The run's convergence test, on the HYPERVOLUME the front encloses.
    # `front_stagnation` counts measurements since the archive last accepted a
    # point, which answers "did the front move" but not "did it move enough":
    # a point 0.1% better than its neighbour resets it exactly like a point
    # that doubles the front. These two put a size and a significance on it.
    #
    # A gain counts when the bootstrap interval of the RELATIVE gain clears
    # `hv_min_gain` -- so a front that grows only by what the measurement noise
    # can produce is not growth. The readings are the archive's own
    # (`samples_by_objective`), resampled per PROGRAM so the same draw is used
    # on both sides of the comparison.
    "hv_min_gain": 0.01,
    # Halt when the primary objective reaches `P_ref`. True is the ordinary
    # case: `P_ref` is a target -- the reference kernel's number -- and there is
    # nothing left to ask for once it is met.
    #
    # False makes `P_ref` a reporting baseline and nothing else, for a run whose
    # preopt ALREADY meets it. `init` refuses that configuration rather than
    # halting at evaluation 0 (see `_refuse_met_target`), and this is how the
    # caller says the preopt being faster than the reference is expected. A
    # recorded campaign had a `P_ref` its preopt already beat, so `j0` started
    # above 100 and the run ended before it measured anything.
    "success_stop": True,
    # The other half of the convergence test: an axis RECORD that moved by more
    # than this, as a fraction of its own previous value. The hypervolume
    # cannot see every real improvement whatever its reference -- a 8%
    # workspace cut is 0.14% of volume at the chip's memory -- and the records
    # cannot see a point that fills the middle of the front. The test is the OR,
    # and these thresholds are in the unit a person reasons about.
    "axis_min_gain": 0.01,
    # Byte-valued axes separately: a 1% workspace cut is not a result, and
    # alignment makes the achievable steps coarse anyway.
    "axis_min_gain_bytes": 0.05,
    # Measurements since the last such gain. Deliberately above one tile block
    # (24 device trials at most), so a block that produces no front point
    # cannot end the run in the middle of an action; about seven actions at
    # `stagnation_K` 7. None turns the halt off and leaves the measurement.
    "hv_stagnation_limit": 48,
    # Verify EVERY retained front point against the preopt, not only the one
    # `impl.py` holds. A front is handed over as a set of choices, and a point
    # nobody re-measured is a search-time point estimate -- the same thing
    # `verify` exists to stop the run reporting about its winner. Each point
    # costs one more ABBA pass (4 stages) and no `eval_budget`. False verifies
    # the delivered point alone.
    "verify_front": True,
    "prior_half_life": 50,
    # The chip that outcome rows written BEFORE the per-chip split were
    # recorded on. Those rows carry no `soc`, so they are dropped from every
    # chip's table unless this names theirs. Left None, a new chip's table
    # starts from this machine's own history alone, and nothing is attributed
    # to hardware it may not have run on.
    "prior_legacy_soc": None,
    # alpha is the chance of closing an action that could still have beaten
    # the bar by `stage_min_gain`.
    "verify_runs": 30,
    "stage_min_gain": 0.01,       # a close is "productive" only if it cuts best latency by >= 1%
    # Firings of a tile BO lever that are exempt from the novelty penalty. A
    # sweep is not repetition. UNUSED since novelty was removed from V_sel; kept
    # so recorded configs still load. 0 restores the old behaviour and
    # is the setting for the bayesian_optimization_tile_search=false ablation arm.
    "bayesian_optimization_free_trials": 0,
    # A tile BO block explores at random until this many FEASIBLE configurations
    # are measured, then hands the search to TPE. Counting feasible measurements
    # rather than trials keeps TPE's surrogate off the 1-2 real points a
    # mostly-rejected block used to model on.
    "bayesian_optimization_startup_min_feasible": 5,
    # Shrink a tile draw the static rules reject as too big, one ladder rung at
    # a time, and measure the first point that fits (bayesian_optimization.space.repair).
    # False restores rejection -- the ablation arm.
    "bayesian_optimization_repair": True,
    # Distinct structures a compiler refusal must be observed on before it is
    # carried to one it has not been seen on. See `refusals_for_structure`:
    # the signature is stable across occurrences of the same refusal, which is
    # not the same as establishing that two refusals sharing it share a cause.
    # 1 restores the old carry-from-first-sighting behaviour.
    "bayesian_optimization_refusal_global_after": REFUSAL_GLOBAL_AFTER,
    "stage_patience": 2,          # advance to the next stage after this many non-productive closes
    # Tile shapes are the one lever where the number, not the structure, decides
    # the outcome, and the number has always been the model's guess. A generated
    # kernel carries whatever tile DESIGN.md wrote down, and the right value can
    # be orders of magnitude away from it, yet a model-driven search can spend a
    # whole budget without touching it -- because nothing in a profiler trace
    # says "this tile is 256 bytes". False restores that: the model proposes
    # numbers again.
    "bayesian_optimization_tile_search": True,
    "symptom_boost": 1.3,         # >1 enables deterministic symptom->action V re-boost (1.0 = off)
    "static_feasibility": True,   # reject hardware-infeasible candidates before the device (False = off)
    "action_preconditions": True,  # hold back actions whose structural precondition is unmet (False = off)
    "conversion_actions": True,   # include the actions that CREATE preconditions (False = off)
    # Fraction of the run budget the tile BO blocks may consume, counted across
    # all of them. It exists so blocks cannot crowd out structural search.
    #
    # It was disabled (None) on the argument that tile numbers are the one lever
    # the model cannot propose, and that a share retires blocks while they are
    # still improving. Unbounded, and with tile repair on, the blocks of a
    # SINGLE action can take most of a run's evaluations without any action
    # closing, and the run then delivers its preopt. An action the search never
    # reaches contributes nothing at all.
    #
    # None restores the unbounded behaviour.
    "block_budget_share": 0.5,
    # ② An ABSOLUTE cap on one block, independent of any run budget.
    # `failure_limit` (6 consecutive device refusals) is a brake for a block
    # that cannot search; a block that CAN search had no ceiling at all. The
    # share above is the wrong instrument for that: it bounds the run's total
    # tile spend, so a single block is free to take all of it. Unbounded, a
    # handful of blocks can take half a run's evaluations between them and
    # return almost nothing, with neither the share nor `failure_limit` binding
    # any of it -- each simply runs to its own abort. None disables the cap.
    "block_max_trials": 12,
}


# Deterministic symptom index (a subset of the existing Stage-7 index) over metrics we ALREADY
# measure per eval (core-util %, bubble %). At INIT and at every close we re-point open-node
# beliefs (V) at the CURRENT bottleneck -- no extra eval -- complementing the optimizer's evolve.
# The `bubble` threshold is the one number here with almost no evidence behind
# it. It was chosen while the metric was unobtainable: `bubble_analysis.log` was
# never written (a local `sys.exit(0)` in pypto's draw_swim_lane.py sat in front
# of the writer), the stdout fallback never matches, and every recorded value
# was a 0.0 the harness itself had supplied. So 10.0 was never tested against a
# reading.
#
# The first genuine readings, taken after that patch was moved, sit BELOW the
# threshold -- so on a well-packed kernel the symptom would still not fire. That
# may be correct (such a kernel need not be scheduler-bound) or the threshold
# may simply be too high; a handful of points cannot tell. Collect readings
# across operators before moving it, and do not treat 10.0 as validated.
SYMPTOM_THRESH = {"bubble": 10.0, "util": 50.0,   # bubble% ABOVE 10 ; core-util% BELOW 50
                  "occupancy": 0.5,               # UB fraction BELOW 0.5
                  "l1_occupancy": 0.5,            # L1 fraction BELOW 0.5
                  # --- per-pipe, from measure_latency._pipes_from_log ---
                  # `util` above is a mean over AIC and AIV together, and with 20
                  # cube cores against 40 vector ones the vector pipe carries two
                  # thirds of it. Measured kernels sit outside every threshold
                  # here while a pipe is the bottleneck. One whose vector pipe is
                  # saturated and whose cube pipe is idle reads above 50 and
                  # raises nothing at all; one where both pipes are idle raises
                  # `util`, whose action set is about occupancy and contains no
                  # way to overlap two pipes -- S-15 lives under `bubble`, and
                  # `bubble` counts only wait_schedule while nearly all of that
                  # stall is wait_predecessor.
                  # Two of these four come from the master Stage-7 skill, which
                  # grades the aggregate utilisation on a scale this project
                  # already uses elsewhere: below 50 is a symptom worth acting on
                  # (analyze_perf.py, and optimization_catalog's 症状B, which is
                  # where `util: 50.0` above came from), 70 and up is "scheduling
                  # and utilisation are about right", and 80 is 分满核 -- the
                  # cores are full (tileshape-deep-tuning §1.1, and the "利用率 >
                  # 80% 且 气泡率 < 10%" that the perf-tune skill repeats three
                  # times). Applying its own scale per pipe leaves the measured
                  # kernels classified exactly as before -- a saturated vector
                  # pipe is full by any of these lines and an idle one is idle by
                  # any of them -- so this replaces a guess with a cited number
                  # and changes no verdict.
                  "pipe_saturated": 80.0,  # ... while aiv_util is AT OR ABOVE this
                  "pipe_idle": 50.0,       # BOTH pipes below this ...
                  # The other two are ours: master measures neither a per-pipe
                  # ratio nor the predecessor wait, so there is nothing to cite
                  # and these stay open until campaign data settles them. Note
                  # pred_stall reads 75-91% on both kernels measured so far, a
                  # very different range from the bubble it mirrors, so borrowing
                  # bubble's 10/20 bands for it would fire on nearly everything.
                  #
                  # STILL UNTESTED, and the record says so precisely: across the
                  # three per-pipe readings taken to date NEITHER of these two has
                  # decided a verdict. Every ratio measured falls on the same side
                  # of 0.5, and every predecessor wait measured falls far above
                  # 50, so `pipe_saturated` and `pipe_idle` -- the two that ARE
                  # cited -- did all the separating. Moving either number on this
                  # evidence would be inventing precision. What would settle them
                  # is a kernel whose ratio lands between 0.3 and 0.7, or whose
                  # predecessor wait lands between 30 and 70; until one appears,
                  # treat a verdict that turns on these as unsupported.
                  "cube_starved": 0.5,     # aic_util / aiv_util BELOW this ...
                  "pred_stall": 50.0}      # ... and predecessor-wait% ABOVE this


SYMPTOM_ACTIONS = {
    # A symptom names an AXIS, not a direction. Where a presence directive and its
    # removal counterpart both bear on the diagnosis, BOTH are boosted and the
    # device decides which way to move: REMOVING submit_before_loop is the
    # large win on one kernel while SETTING it is a heavy loss on another.
    # Boosting only the positive prescribes a direction the evidence does not
    # support.
    # bubble high -> fill idle bubbles: larger tiles / unroll, merge loops, stitch, vec-merge, sched, fuse/pipeline
    "bubble": {"F-2", "F-3", "F-6", "F-8", "S-4", "S-5", "S-20", "S-9", "S-10", "S-14", "S-15"},
    # util low -> raise occupancy: more tasks (granularity/tiling), core-fill, cube merges, memory cuts
    "util": {"F-1", "F-7", "F-9", "F-10", "S-1", "S-2", "S-6", "S-18", "S-7", "S-19",
             "S-11", "S-12", "S-16", "S-17", "F-17", "F-18"},
    # UB under-occupied -> make the tile bigger. Only the actions that resize a
    # tile or the loop it implies; nothing here is about scheduling or fusion,
    # because the diagnosis is specifically "the buffer is nearly empty".
    "occupancy": {"F-3", "F-9", "F-10", "F-11", "F-19", "S-11", "S-12"},
    # L1 under-occupied -> the cube operands are far below what fits. Three
    # kinds of response and nothing else: enlarge the cube tile (F-9, S-11),
    # make the residency worth more by reusing or merging (S-6, S-7, S-8), or
    # concede that this shape does not belong on the Cube pipe at all (I-11,
    # "the Cube tile is mostly padding"). S-2 is deliberately ABSENT: it shrinks
    # L0/L1 tiles to raise the task count, which is the opposite prescription.
    "l1_occupancy": {"F-9", "S-6", "S-18", "S-7", "S-19", "S-8", "S-11", "I-11"},
    # Cube idle while Vector is saturated -> the work is on the wrong pipe, and
    # no amount of retiling moves it. Either put some of it on the Cube (F-20
    # turns a reduction into a matmul against a ones vector) or make the vector
    # side cheaper: halve its bytes (F-18), stop materialising what it has to
    # walk (F-17, I-10), or keep the intermediate on-chip instead of round-
    # tripping it (S-14, S-17). I-11 is deliberately ABSENT and is the inverse
    # prescription -- it moves a matmul ONTO the pipe that is already full.
    #
    # F-22 used to be here, and the note said it was the sharpest evidence the
    # set had: a kernel diagnosed cube_starved correctly, fixed by moving a
    # per-tile layout copy off the vector pipe and ONTO THE HOST. That reading
    # does not survive the scope this search now has. The copy did not get
    # cheaper; it left the region `p` measures. Against a frozen wrapper the
    # move is not available, and the gain was a boundary effect rather than a
    # kernel one.
    # The diagnosis stays right; the prescription for it is not PANKO's to give.
    "cube_starved": {"F-20", "F-18", "F-17", "I-10", "S-14", "S-17"},
    # Both pipes idle and each waiting on the other -> a serial recurrence, not
    # an occupancy problem. Overlap it (S-15, I-8), give the Cube something to
    # hold across the wait (S-7, S-8), or cut the chains into more independent
    # work (F-7, S-2). Note S-2 shrinks tiles to raise the task count, which is
    # right here and wrong under `l1_occupancy`; the two never fire together.
    # I-12 sits beside I-8 deliberately: on a pipe-serialized kernel the
    # winning move can be to REMOVE the directive, so naming only I-8 here
    # would boost the harmful direction on exactly the kernels this symptom is
    # written for.
    "pipe_serialized": {"S-15", "I-8", "I-12", "S-7", "S-19", "S-8", "F-7", "S-2"},
}


# The per-pipe readings, carried on `m` beside `util`/`bubble`. Absent means not
# measured: the installed draw_swim_lane.py has a latency-only fast path that
# exits before writing bubble_analysis.log, so these are missing far more often
# than `util` is, and a missing reading must never read as a zero.
PIPE_METRICS = ("aic_util", "aiv_util", "pred_stall",
                "aic_bubble", "aiv_bubble")


# `bo` was the package name and the key prefix until the abbreviation was spelled
# out. State files outlive a rename -- a search in progress is resumed by the next
# `select` -- so the old spellings are accepted on read and written back in the new
# one by the next `save`. Dropped once no in-flight run can predate the rename.
_RENAMED_STATE_KEYS = {
    "bo_memory": "bayesian_optimization_memory",
    "bo_refusals": "bayesian_optimization_refusals",
    "bo_studies": "bayesian_optimization_studies",
}


_RENAMED_CONFIG_KEYS = {
    "tile_bo": "bayesian_optimization_tile_search",
    "bo_free_trials": "bayesian_optimization_free_trials",
    "bo_startup_min_feasible": "bayesian_optimization_startup_min_feasible",
}


# Where this skill actually is, resolved from this file rather than from the
# caller's working directory.
#
# SKILL.md used to spell the harness and the action catalogue as
# `cannbot-skills/ops/pypto-op-perf-panko-manual/...`, which resolves only when the cwd
# happens to be a pypto-gym clone. Quickstart installs the plugin into any
# project, where the skill arrives as `.opencode/skills/pypto-op-perf-panko-manual` (a
# symlink, so the files are reachable) and no `cannbot-skills/ops/` tree exists
# at all. The agent could find the skill and then fail on its first harness call.
#
# `realpath` follows that symlink, so the layout is the source tree's either
# way, and neither the client's directory convention nor the cwd is hardcoded.
_PACKAGE_DIR = os.path.dirname(os.path.realpath(__file__))


SCRIPTS_DIR = os.path.dirname(_PACKAGE_DIR)


SKILL_ROOT = os.path.dirname(SCRIPTS_DIR)


def skill_path(*parts):
    """A path inside this skill, wherever it was installed."""
    return os.path.join(SKILL_ROOT, *parts)


DEFAULT_CATALOG = skill_path("references", "action_catalog.json")


def out(obj):
    """One JSON directive on the protocol channel. See jsonio."""
    jsonio.emit(obj)


class Refusal(Exception):
    """A command refusing to proceed, carrying the exit code it ends on.

    Every refusal in this harness has already written its JSON -- the directive
    the optimizer reads -- and what is left is the process's exit code. Raising
    `SystemExit` from the command itself would bury an exit inside a function
    that this module's own tests import and call, where stopping the interpreter
    is not what the caller asked for. The code travels as data instead, and
    `main` is the single place that turns it into an exit.
    """

    def __init__(self, code):
        super().__init__(f"refused with exit code {code}")
        self.code = code


def w_stage(stage, active_stage, staged_prior):
    """Annealed stage prior. Not a hard gate: cross-stage moves stay legal, just
    down-weighted. Favors the active stage; upstream (already-explored) < downstream.
    """
    if not staged_prior:
        return 1.0
    d = STAGE_ORDER.index(stage) - STAGE_ORDER.index(active_stage)
    return {0: 1.0, 1: 0.6, 2: 0.3, -1: 0.4, -2: 0.25}.get(d, 0.2)


def clamp_belief(v):
    """Belief into [0, 1], which is where every argument about selection assumes
    it lives.

    `symptom_boost` was clamped here from the start; the model's own `insert`
    and `update` were not, and a recorded run carried V up to 1.4. That matters
    because the staged prior is an ordering claim: an action outside the active
    stage scores at most 0.6 * 0.9 = 0.54 against 0.9 for an untried action of
    maximal prior inside it, and the claim holds only while belief is bounded by
    one. Unbounded, a belief above 1.667 inverts it. The bound is now enforced
    where the value enters rather than assumed everywhere it is read.
    """
    try:
        x = float(v)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, round(x, 6)))


FREE = {"infeasible_static": "static", "over_learned_ceiling": "ceiling",
        "duplicate_config": "dup",
        # A configuration the compiler refused in an EARLIER block, on a
        # different structure. Free by the same argument as `ceiling`: the
        # device already answered this question and the answer does not depend
        # on the program shape that asked it.
        "known_compile_failure": "refused"}


class SelectionContext(typing.NamedTuple):
    """The stage and budget state a selection is scored in."""
    active: typing.Any
    staged: typing.Any
    free: typing.Any
    thompson: typing.Any = None


class MeasurementVerdict(typing.NamedTuple):
    """The fields that say how a candidate scored against its bar."""
    improved: typing.Any
    s_ok: typing.Any
    j: typing.Any
    bar: typing.Any


UNMEASURED_FIELDS = ("n_static", "n_noop", "n_dup", "n_timeout")


def _bound(st, key):
    return st["config"].get(key, DEFAULT_CONFIG[key])


class FrontContext(typing.NamedTuple):
    """The front state a rejection is re-examined against."""
    objs: typing.Any
    others: typing.Any
    prim: typing.Any
    bound: typing.Any
    within: typing.Any


class RatchetMeasurement(typing.NamedTuple):
    """The values the ratchet reads for one candidate."""
    j: typing.Any
    m_eval: typing.Any
    accepted: typing.Any = None
    clarity: typing.Any = None


# ----------------------------------------------------- end-of-run verification
# Search-time acceptance is a point comparison of two median-of-9 readings, by
# design: requiring significance there would reject most of the small gains a
# greedy run is built from. The price is that the reported speedup is the
# maximum over many noisy comparisons, which is biased upwards (the winner's
# curse). This is where that is paid for -- once, at the end, on the two
# programs the run actually hands over.
VERIFY_ORDER = ("preopt", "delivered", "delivered", "preopt")   # ABBA: cancels linear drift


# --------------------------------------------------------- the non-dominated archive
# The ratchet keeps ONE program: whatever is fastest. A candidate that is slower
# but needs less workspace is measured and then thrown away, though nothing
# dominates it. The archive keeps every such point, and it is maintained in
# EVERY run, including a purely single-objective one: the search does not read
# it there, so recording it costs nothing and the front readout reads it.
#
# Latency is compared as measured (median-of-9); workspace exactly, because it is
# a structural quantity with no noise (`AlignUp(..., 32KB) * parallelism`). A tie
# on both keeps the first point seen.
# `latency` is `p` itself; everything else is a key of the measurement's `m`.
# A user adds an objective by recording its value (`record --metrics`) and
# naming it in `config.objectives`; nothing here needs to change.
# `samples_from` is where THIS objective's individual readings live, and it is
# what decides whether the archive judges it on an interval or on a number.
# Nothing here is special-cased by name: an objective that declares a sample
# source is compared with the bootstrap, one that does not is compared exactly.
# That is the whole extension point -- a future objective becomes "measured" by
# naming where its readings are, and `workspace` stays exact by naming nothing.
#   "samples"      -- the row's own list, which is `p`'s
#   "m.<key>"      -- a list carried on the measurement under that key
# There is deliberately no `latency` here. It used to mean `p`, and `p` has been
# two different instruments over this repo's life, so a config that said
# `latency` did not say what it optimised and a front labelled `latency` did not
# say what it was a front of. Every name below is an instrument.
OBJECTIVE_ALIASES = {
    # `p` IS the on-device profiler's "AICore End-to-End Time" (see
    # measure_latency), written only at `runtime_debug_mode: 1`. It is the
    # run's primary latency: `--p`, `--samples`, `J`, the ratchet, SPRT and
    # `verify` all mean this reading.
    "aicore_e2e": {"from": "p", "samples_from": "samples", "unit": "us",
                   "direction": "min"},
    # msprof's per-launch MIX_AIC `aicore_time`: a SECOND reading of the same
    # launch, not a transform of `p`. The two rank programs differently on a
    # substantial share of pairs, which is why both are defaults.
    "mix_aic": {"from": "m.mix_aic", "samples_from": "m.mix_aic_samples",
                "unit": "us", "direction": "min"},
    # Structural (`AlignUp(..., 32KB) * parallelism`), so no sample source and
    # no interval: it is compared as the number it is.
    "workspace": {"from": "m.workspace_b", "direction": "min"},
}


PARETO_SCHEMA = 2        # 1 wrote aicore_e2e under `mix_aic_us` as well


# ------------------------------------------------------------------- scoring
NOISE_WINDOW = 32


# The search-time protocol of `measure_latency.py`. Recorded with the estimate,
# because a sigma measured under one protocol says nothing exact about another.
SEARCH_PROTOCOL = "warmup=1,runs=9,agg=median"


J_OBJECTIVE = "aicore_e2e"


# The percentile bootstrap's parameters, shared by end-of-run `verify` and the
# archive's dominance test -- one estimator, one seed, so a front and a
# verification never disagree about the same two sample sets.
VERIFY_BOOT = 2000


VERIFY_SEED = 0          # fixed: the same readings always give the same interval


VERIFY_ALPHA = 0.05


# The prior an inserted intent gets when the model does not supply a readable
# one. Deliberately not 0.0: a zero prior is absorbing in the selection product
# and would delete the intent the model just asked for. Deliberately not 0.9
# either -- an action the catalogue does not contain has no measured standing.
CATALOG_PRIOR_DEFAULT = 0.5


# V of a seeded catalogue node before the model has said anything about it,
# when g is learned (see _seed_frontier).
SEED_V = 0.5


def clamp_stage(v):
    """A stage name the rest of the harness can index, and (value, note).

    The third model-supplied field on an `evolve --insert` node, and the only
    one that was taken raw. `V` and `prior_gain` are both bounded where they
    enter (`clamp_belief`, `clamp_prior`) for the reason those docstrings give;
    `stage` was not, and `w_stage` indexes `STAGE_ORDER` with it. An insert
    naming a stage outside the vocabulary -- which SKILL.md never lists -- was
    accepted and saved, and then every later `select` raised ValueError out of
    `pypto_action_priority`. The node sits in `open` with no `pruned_on`, so
    `live_frontier` keeps returning it and the run cannot select again at all.

    Falls back to the first stage rather than refusing the whole `evolve`: the
    insert carries a real delta the model wants tried, and a mis-named stage is
    a weight, not a claim about the program.
    """
    stage = str(v) if v is not None else ""
    if stage in STAGE_ORDER:
        return stage, None
    return STAGE_ORDER[0], (f"stage {stage!r} is not one of "
                            f"{'/'.join(STAGE_ORDER)}; read as {STAGE_ORDER[0]}")


def clamp_prior(v):
    """
    `prior_gain` into [0, 1]. Returns (value, note); note is None when the
    input was already a prior.

    `clamp_belief` bounds the model's V because the staged prior is an ORDERING
    claim that holds only while every factor in the product is bounded by one.
    `prior_gain` is the same kind of factor in the same product, and it was left
    unbounded on the one path where the model supplies it: `evolve`'s `insert`.
    A catalogue node's prior is read from `action_catalog.json` and is 0.9, 0.6
    or 0.3; an inserted node's came straight from the model's JSON through a bare
    `float()`.

    `prior_gain: 10` scores that node at 10.0 against the 0.9 ceiling every
    catalogue action shares, and `novelty = 1/(1+tried)` needs ten firings to
    pull it back under -- so a single insert could hold the top of the frontier
    for ten selections. The figure's claim that the core supplies three of the
    four factors and the model contributes only V holds only with this bound in
    place.

    An unreadable or NaN prior falls back to `CATALOG_PRIOR_DEFAULT` rather than
    to zero, and says so, because the failure mode of a silent zero is an intent
    that exists in the state and can never be selected.
    """
    try:
        x = float(v)
    except (TypeError, ValueError):
        return CATALOG_PRIOR_DEFAULT, f"prior_gain {v!r} is not a number"
    # NaN. Every comparison against it is False, so the node would lose every
    # selection and look merely unlucky while doing it.
    if x != x:
        return CATALOG_PRIOR_DEFAULT, "prior_gain was NaN"
    x = round(x, 6)
    c = max(0.0, min(1.0, x))
    return c, (None if c == x else f"prior_gain {x} clamped to {c}")


WORKSPACE_STATES = ("measured", "zero", "unavailable")


# Keys of a `measure_latency` result that are NOT readings: they have their own
# CLI flags and their own meaning, so carrying them into `m` would duplicate or
# contradict what the caller passed.
_MEASUREMENT_NOT_METRICS = ("p", "samples", "unevaluated")


# Keys a supplied `--chip-envelope` may leave out. `memory_mib` is a budget
# rather than a capacity -- the ini does not always carry it (Ascend950DT_950x)
# and a caller naming the buffers has no reason to name it -- so its absence is
# an answer. `hv_reference_point` returns None without it and the convergence test
# falls back to the axis records.
OPTIONAL_ENVELOPE = ("source", "memory_mib")


# --------------------------------------------------------------- the trajectory
# The trajectory file is how a run is READ. One long key=value line per event
# put every field on the same footing and lost the only structure that matters:
# which action the events belong to. So the file is written as SECTIONS -- a
# heading per selected action, its candidates as a table under it, and the
# notes that belong to it indented beneath. Machines read `search_state.json`
# and `selects.jsonl`; this file is for a person.
# The candidate table is written as GitHub-flavoured Markdown, so the file a
# person opens in a preview shows a table rather than a wall of aligned spaces.
# That has one consequence worth stating: a Markdown table ends at the first
# line that is not a row, so a note cannot sit BETWEEN two candidates as
# free text without splitting the table in half. Notes are therefore rows
# themselves, marked `·` in the eval column -- which keeps the one property
# that matters about this file, that its events are in the order they happened.
LOG_NOTE_MARK = "·"


NOTE_CHARS = 200


class CandidateRow(typing.NamedTuple):
    """The cells of one row of an action's table."""
    vals: typing.Any
    ratchet: typing.Any
    archive: typing.Any
    n: typing.Any
    note: str = ""
