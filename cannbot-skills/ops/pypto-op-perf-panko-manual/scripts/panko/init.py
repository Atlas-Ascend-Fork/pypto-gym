#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""`init` and `normalize`: everything that happens before the first action.

The baseline, the envelope, the catalogue, and the refusals that keep a run
from starting on a question nobody can answer.
"""

import json
import os
import time

import action_posterior
import evaluation_session

from .config import (DEFAULT_CONFIG, OPTUNA_MIN_VERSION, Refusal, SEARCH_PROTOCOL, SEED_V, SKILL_ROOT, _diag_log,
                     bayesian, bayesian_capability, out, predicates)
from .state import (_apply_chip_memory, _code_version, _domain_state, _pypto_defaults, _resolve_envelope,
                    confirm_chip, debug_options_signature, file_hash, next_id, npath, op_file_path, read_text, save,
                    snapshot, wrapper_signature)
from .measurement import (_preopt_metrics, cli_pipes, cli_samples, compute_j, derive_eval_timeout, objective_values,
                          objectives_of, observe_noise, parse_objectives)
from .front import _hv_reference_report, archive_record, refuse_no_hv_reference
from .log import _init_baseline_log, append_log
from .selection import _prior_table_path, _snapshot_prior_table, _symptom_reboost


def cmd_normalize(a):
    """Static verdict on a kernel normalisation, before any device time.

    Normalisation runs at INIT, before preopt is measured, and it is the step
    that decides whether the deterministic core owns the tile at all: `bayesian.apply`
    tunes a call only when every argument is an integer literal, so a kernel
    written `set_vec_tile_shapes(TILE_B, TILE_D)` produces zero BO trials no
    matter what else is fixed.

    Three gates decide admissibility. This is the one that costs nothing:

        this command      values unchanged, no call added or removed,
                          tunable sites actually increased
        the golden test   semantics preserved
        preopt neutrality same latency within the noise floor

    It is deliberately not a semantics check -- the golden test already answers
    that exactly, and constant folding cannot. It is the check that the rewrite
    did not TUNE while it refactored, which nothing else can see: a normalisation
    that quietly changes 64 to 8192 would pass both other gates and would move
    the baseline every speedup in the run is divided by.
    """
    if bayesian is None:
        out({"ok": False, "reasons": ["bayesian_optimization package not available on this arm"]})
        raise Refusal(2)
    for path in (a.before, a.after):
        if not os.path.exists(path):
            out({"ok": False, "reasons": [f"not found: {path}"]})
            raise Refusal(2)
    before = read_text(a.before)
    after = read_text(a.after)
    report = bayesian.normalize.check(before, after)
    # The verdict goes out FIRST. It was emitted after the log line, so when the
    # log write raised -- which it did on every fresh op dir, see `append_log` --
    # the answer this command exists to produce was computed and then thrown
    # away, and the caller got a traceback instead of `ok: false`. A record of
    # what happened must not be able to destroy what happened.
    out(report)
    if a.op_dir and os.path.isdir(a.op_dir):
        try:
            append_log(a.op_dir, a.op or "",
                       f"- normalize: ok={report['ok']} "
                       f"tunable {report.get('before', {}).get('tunable')} -> "
                       f"{report.get('after', {}).get('tunable')}"
                       + ("" if report["ok"] else " | " + "; ".join(report["reasons"])))
        except OSError as e:
            # Best effort, for the same reason as the ordering above.
            _diag_log.warning("[normalize] could not append to the run log: %s", e)
    if not report["ok"]:
        raise Refusal(1)


def _init_code_version(a):
    """Provenance is a precondition, not a nicety.

    A state file with no code_version cannot be attributed to a build, so two
    runs cannot be compared and neither can be reported. Refuse to start rather
    than produce a log that is silently unusable.
    """
    cv = _code_version()
    if cv is None and not getattr(a, "allow_unversioned", False):
        out({"initialized": False, "error": "code_version_unavailable",
             "hint": "run the script from inside its git work tree, or write a "
                     "VERSION file next to it, or pass --allow-unversioned"})
        raise Refusal(2)
    return cv


def _init_capability(cfg):
    """BO capability preflight.

    The flag says which algorithm this run claims to use, so it must not be
    possible to claim TPE and get the model's guesses: with the flag ON and the
    package missing, every tile proposal would come back empty and the coder would
    pick the numbers, while the config, the log and the report all still said the
    search was on.

    Refusing is right and stopping here is not. Without the tile search this is no
    longer PANKO, and WHICH of the two ways out to take -- install the dependency,
    or run the other method deliberately -- is the user's call, not the harness's
    and not the agent's. So the refusal carries the question to put to them.
    `ask_user` is the marker the skill keys off; an agent that answers it on the
    user's behalf has substituted its own judgement for a decision about what the
    run is.
    """
    capability = bayesian_capability()
    if not cfg.get("bayesian_optimization_tile_search", True) or capability["available"]:
        return capability
    floor = ".".join(str(x) for x in OPTUNA_MIN_VERSION)
    out({"initialized": False, "refused": True,
         "error": "bayesian_optimization_unavailable",
         "bayesian_optimization": capability,
         "ask_user": {
             "question": (
                 f"PANKO needs optuna (>= {floor}) to search tile values, and it is "
                 f"not importable here ({capability['reason']}). Tile numbers are the "
                 "one lever the model cannot propose, so without it this is a "
                 "different method rather than a slower one. How do you want to "
                 "proceed?"),
             "options": [
                 {"id": "install",
                  "label": "Install optuna and run PANKO",
                  "action": f"python3 -m pip install 'optuna>={floor}', then re-run init"},
                 {"id": "ablation",
                  "label": "Run Stage 7 without the tile search",
                  "action": ('re-run init with --config '
                             '\'{"bayesian_optimization_tile_search": false}\' -- the '
                             "structural search still runs; tile values stay as the "
                             "kernel has them unless the coder changes them")},
                 {"id": "stepwise",
                  "label": "Abandon PANKO and use the stepwise Stage 7 path",
                  "action": "report back to the orchestrator; no PANKO state is written"},
             ]},
         "hint": "relay the question to the user; do not choose for them"})
    raise Refusal(3)


def _init_envelope(a):
    """The chip envelope is DERIVED, not assumed.

    Every static rule -- the UB footprint bound, the L0/L1 capacity checks, the
    core-count seeding -- is validated against it, so running one chip's numbers
    on another rejects legal candidates before any device sees them, and leaves no
    evidence that it did. The review is explicit that silently using the A3
    defaults is the thing not to do.
    """
    envelope, why = _resolve_envelope(a)
    if envelope is not None:
        envelope, why = _apply_chip_memory(a, envelope)
    if envelope is not None:
        return envelope
    out({"initialized": False, "refused": True,
         "error": "chip_envelope_unresolved", "reason": why,
         "ask_user": {
             "question": (
                 "PANKO validates every candidate against this chip's buffer "
                 "capacities and core counts, and could not derive them here: "
                 f"{why}. Another chip's numbers would reject legal tiles "
                 "before the device ever saw them, so there is no safe "
                 "default. How do you want to proceed?"),
             "options": [
                 {"id": "cann_env",
                  "label": "Make the platform ini reachable",
                  "action": "source the CANN set_env.sh, or set "
                            "ASCEND_TOOLKIT_HOME, then re-run init"},
                 {"id": "name_soc",
                  "label": "Name the SoC explicitly",
                  "action": "PANKO_SOC_NAME=<your SoC name>; the "
                            "harness then reads <platform_config>/<name>.ini"},
                 {"id": "explicit",
                  "label": "Supply the envelope yourself",
                  "action": '--chip-envelope \'{"ub_kb":192,"l1_kb":512,'
                            '"l0a_kb":64,"l0b_kb":64,"l0c_kb":128,'
                            '"cube_cores":20,"vector_cores":40}\' -- recorded '
                            "as explicit:cli so the report says the numbers "
                            "were given rather than read"},
             ]},
         "hint": "relay the question to the user; do not choose for them"})
    raise Refusal(3)


def _init_baseline_hash(a, op_file):
    """The baseline hash is COMPUTED from the file, not taken on the caller's word.

    Everything the run does afterwards is anchored to this snapshot: `select`
    restores it, `record` rolls back to it, `stop` delivers from it. A hash that
    names no bytes turns that chain into "whatever is in the working file right
    now", and which program a candidate was built on then depends on the caller's
    path and on residue from earlier runs.
    """
    preopt_hash = file_hash(op_file)
    if preopt_hash is None:
        out({"initialized": False, "refused": True, "error": "op_file_unreadable",
             "op_file": op_file,
             "hint": "PANKO snapshots and restores this file on every cycle; it "
                     "must exist and be readable before the search starts. Pass "
                     "--op-file if the kernel is not at <op_dir>/<op>_impl.py."})
        raise Refusal(3)
    if a.preopt_hash and a.preopt_hash != preopt_hash:
        # Not a warning. The caller believes it measured one program and the file
        # on disk is another, so the preopt latency this run normalises every
        # speedup against was taken on a kernel that is not the baseline.
        out({"initialized": False, "refused": True, "error": "preopt_hash_mismatch",
             "op_file": op_file, "given": a.preopt_hash, "actual": preopt_hash,
             "hint": "--preopt-hash does not match the bytes of op_file. Re-measure "
                     "the file as it stands, or point --op-file at the kernel that "
                     "was measured. Omit --preopt-hash to let the harness derive it."})
        raise Refusal(3)
    return preopt_hash


def _init_progress(a, opening, preopt_m):
    """The progress counters as `init` starts them."""
    return {"active_stage": "frontend", "stage_stagnation": 0,
            "evals_used": 1, "wall_clock_s": 0, "_start_ts": time.time(),
            "last_seen_ts": time.time(),
            "global_stagnation": 0, "best_J": opening["j0"],
            # `J`'s instrument, and what the headline speedup is a
            # ratio of. `*_mix_aic_us` is the SAME program read on the
            # other clock, kept because the tile block ranks its own
            # trials on MIX_AIC and its bar has to be in MIX_AIC too.
            "preopt_aicore_e2e_us": a.preopt_p,
            "best_aicore_e2e_us": a.preopt_p,
            "preopt_mix_aic_us": preopt_m.get("mix_aic"),
            "best_mix_aic_us": preopt_m.get("mix_aic"),
            "best_speedup_vs_preopt": 1.0,
            "cycle_productive": False, "converged": False,
            # The volume the front encloses, and measurements since it
            # last grew by more than the readings' noise.
            "hv": None, "hv_stagnation": 0,
            "static_rejected": 0, "duplicate_rejected": 0,
            # Candidates whose compile ran past `eval_timeout_s`.
            # Counted, never charged: see `_reject_unevaluated`.
            "unevaluated_rejected": 0,
            # The front point the next action is applied to, chosen
            # at every close by `_advance_front_base`. None until the
            # first close, which means the global best.
            "base_code_hash": None,
            "front_stagnation": 0,
            "frontier_emptied_by_prune": 0, "expansion_requests": 0,
            "expansion_required": False,
            "stop_reason": None,
            "refine": None}


def _init_nodes(a, opening, preopt_m):
    """The node tree at init: the preopt program, already measured and closed."""
    return {"closed": [{"id": "root", "parent": None, "stage": "frontend",
                        "code_hash": opening["preopt_hash"], "J": opening["j0"],
                        "eval": {"s": a.preopt_s, "p": a.preopt_p,
                                 "m": preopt_m}}],
            "open": [], "pruned": []}


def _init_state(a, cfg, opening):
    """The state file as `init` writes it, before any action is seeded into it."""
    preopt_m = _preopt_metrics(a)
    # Derived here rather than at first use, so the number a campaign ran under is
    # in the state file next to the rest of the run parameters. `source`
    # distinguishes a derived value from the configured default the way
    # `chip_envelope.source` distinguishes a derived envelope from a supplied one.
    timeout_s, timeout_src = derive_eval_timeout(cfg, getattr(a, "preopt_compile_s", None))
    cfg = dict(cfg, eval_timeout_s=timeout_s, eval_timeout_source=timeout_src)
    st = {
        "op": a.op, "device_id": a.device,
        # Which entry of the kernel's per-case config table this campaign tunes.
        # None for an ordinary kernel. The key also lives in the kernel as a
        # `panko:tune` comment, which is what every later command reads; this is
        # the record of what `init` was asked for.
        "config_key": opening.get("config_key"),
        # `metric` names the instrument `P_ref` is in, because `J = P_ref / x`
        # and the two have to be the same clock. See `J_OBJECTIVE`.
        "target": {"metric": "aicore_e2e_us", "P_ref": a.p_ref,
                   # `preopt` means no reference kernel was named, so there is
                   # no target and every `J` is a percentage of the baseline.
                   "P_ref_via": opening.get("p_ref_via", "explicit:cli"),
                   "lower_is_better": True},
        "config": cfg, "_seq": 0,
        "progress": _init_progress(a, opening, preopt_m),
        "nodes": _init_nodes(a, opening, preopt_m),
        "best_node": "root",
        "op_file": (a.op_file or None),   # working <op>_impl.py the harness snapshots/restores
        # The preopt reading was accepted at INIT, used once for symptom anchoring
        # and then dropped, so every later close re-read an empty `m` on the root.
        "eval_cache": {opening["preopt_hash"]: {"s": a.preopt_s, "p": a.preopt_p,
                                                "m": preopt_m}},
        "code_version": opening["cv"],
        # What actually ran, not what was asked for. A report that names the
        # algorithm has to read this rather than the config flag.
        "bayesian_optimization": opening["capability"],
        # What every static rule in this run is validated against. Persisted so a
        # resumed campaign keeps the envelope it started with, and so a report
        # can say which chip the numbers were admissible for.
        "chip_envelope": opening["envelope"],
        # The host wrapper as it stands at the baseline. Frozen for the run:
        # see `wrapper_signature`.
        "wrapper_signature": wrapper_signature(opening["source"]),
        # The measurement's own configuration, frozen with it: see
        # `debug_options_signature`.
        "debug_options_signature": debug_options_signature(opening["source"]),
    }
    st["nodes"]["held"] = []
    _init_noise(st, a, opening, preopt_m)
    return st


def _seed_frontier(st, a, cfg, seeds):
    """Seed the frontier from the action catalog. Returns the predicate reading."""
    preds = (predicates.compute(a.op_file or "", a.op_dir)
             if (predicates and cfg.get("action_preconditions", True)) else None)
    st["predicates"] = preds
    learn = cfg.get("prior_gain_source", DEFAULT_CONFIG["prior_gain_source"]) == "learned"
    _snapshot_prior_table(st, a.op_dir)
    table = action_posterior.load_table(_prior_table_path(st)) if learn else None
    st["prior_gain_source"] = ("learned" if table else "catalog")
    for sd in seeds:
        if sd.get("conversion") and not cfg.get("conversion_actions", True):
            continue
        node = {
            "id": next_id(st, "u"), "parent": "root", "stage": sd["stage"],
            "catalog_id": sd.get("id"),   # keep the F-x/S-x/I-x id for symptom matching
            "delta": sd["delta"], "V": float(sd.get("prior_gain", 0.5)),
            "prior_gain": float(sd.get("prior_gain", 0.5)), "tried": 0,
            "requires": sd.get("requires", []),
            "prior_gain_catalog": float(sd.get("prior_gain", 0.5)),
            # Which objective this action AIMS at, from the catalogue. Read by
            # `front_demand`: the directional request is only worth making once
            # the catalogue is spent on the objective the front is short of.
            "moves": list(sd.get("moves") or []),
            # V starts as a copy of the star; `evolve` marks it the model's.
            "V_source": "catalog"}
        if table:
            learned = action_posterior.learned_prior_gain(
                node, table, preds, None,
                cfg.get("prior_gain_quantile", DEFAULT_CONFIG["prior_gain_quantile"]))
            node["prior_gain"] = learned
            # V is the MODEL's factor. Seeding it with a copy of g made every
            # untouched catalogue action score g squared, against g * V for an
            # insert whose V the model actually wrote -- so an insert needed a
            # V of ~0.86 to beat the top catalogue action. Until the model
            # speaks, its belief is neutral.
            node["V"] = SEED_V
        if node["moves"]:
            for name in node["moves"]:
                st.setdefault("objective_actions", {}).setdefault(name, []).append(
                    node["catalog_id"])
        if preds is not None and not predicates.satisfies(node["requires"], preds):
            st["nodes"]["held"].append(node)   # held, never deleted
        else:
            st["nodes"]["open"].append(node)
    return preds


def _anchoring_warning(a, cfg):
    """Whether symptom anchoring is on, and the warning it earns when it cannot fire.

    Silent no-op is the dangerous case: without metrics the two PROFILE symptoms
    cannot fire at all, so a whole campaign can run with anchoring configured on
    and never once applied to a bottleneck.

    `occupancy` is unaffected -- it is read off the source -- so this is a warning
    about two of the three symptoms rather than all of them, and it is worded that
    way. A run that logs this warning AND a non-empty seeded_symptoms is a run
    where only the source-derived symptom is live.
    """
    anchoring = float(cfg.get("symptom_boost", 1.0)) > 1.0
    warning = None
    if anchoring and not (getattr(a, "preopt_util", 0.0) or getattr(a, "preopt_bubble", 0.0)):
        warning = ("symptom_boost is enabled but no --preopt-util/--preopt-bubble "
                   "were supplied; the two PROFILE symptoms (bubble, util) cannot "
                   "fire until record is called with --util/--bubble. The "
                   "source-derived `occupancy` symptom is unaffected")
    return anchoring, warning


def _init_snapshot(a, op_file, preopt_hash):
    """Persist the baseline. Emits and raises when it cannot be written.

    Called before `save`, so a refused init leaves no state behind: a half-built
    run whose baseline is missing is worse than no run, because `select` would
    restore nothing and every candidate would stack on the last one.
    """
    if snapshot(a.op_dir, op_file, preopt_hash):
        return
    out({"initialized": False, "refused": True, "error": "baseline_snapshot_failed",
         "op_file": op_file, "snapshot": npath(a.op_dir, preopt_hash),
         "hint": "the preopt snapshot could not be written; check the "
                 "permissions on optimization/nodes/ and the free space there."})
    raise Refusal(3)


def _init_announce(a, st, opening, report):
    """Write the opening line of the trajectory log and emit init's payload."""
    warning, lint_warning = report["warning"], report["lint_root_warning"]
    n_folds, chip_note = report["n_folds"], report["chip_mismatch"]
    # The opening, as a short block rather than one line of key=value: the
    # predicates alone ran past 300 characters and pushed everything the reader
    # actually needed -- what was seeded, how many actions are live -- off the
    # right-hand edge.
    preds = report["preds"] or {}
    append_log(a.op_dir, a.op,
               f"- code {st.get('code_version')} · objectives "
               f"{', '.join(o['name'] for o in objectives_of(st))} · budget "
               f"{st['config'].get('eval_budget')} evals"
               f"\n- frontier: {len(st['nodes']['open'])} actions open, "
               f"{len(st['nodes'].get('held', []))} held · symptoms seeded: "
               f"{', '.join(report['seeded_symptoms']) or '—'} · anchoring "
               f"{report['anchoring']}"
               f"\n- predicates: "
               + ", ".join(f"{k}={v}" for k, v in sorted(preds.items()))
               + (f"\n- semantic fold: {n_folds} calls" if n_folds else
                  "\n- semantic fold: OFF (pypto not importable; the no-op check "
                  "compares ASTs only and cannot fold a restated default)")
               + (f"\n- WARNING: {warning}" if warning else "")
               + (f"\n- LINT-ROOT: {lint_warning}" if lint_warning else "")
               + "\n\n## Trajectory")
    save(a.op_dir, st)
    out({"initialized": True, "best_J": opening["j0"], "open": len(st["nodes"]["open"]),
         "held": len(st["nodes"].get("held", [])), "predicates": report["preds"],
         "preopt_snapshot": True, "preopt_hash": opening["preopt_hash"],
         "bayesian_optimization": opening["capability"],
         # So a caller never has to guess where the skill's own files are.
         "skill_root": SKILL_ROOT, "catalog": a.catalog,
         "chip_envelope": st["chip_envelope"],
         # The box every hypervolume in this run is measured in. Fixed at init,
         # so printing it once is printing it for the whole run.
         "hv_reference": _hv_reference_report(st, objectives_of(st)),
         "p_ref": a.p_ref, "p_ref_via": opening["p_ref_via"],
         "success_stop": st["config"].get("success_stop"),
         # Present only when the chip in the slot is not the chip the envelope
         # describes. The run is initialised either way; the first block is what
         # stops.
         **({"chip_mismatch": chip_note} if chip_note else {}),
         "seeded_symptoms": report["seeded_symptoms"],
         "code_version": st.get("code_version"), "anchoring": report["anchoring"],
         "semantic_fold_calls": n_folds,
         "warning": warning, "lint_root_warning": lint_warning})


def cmd_init(a):
    cfg = dict(DEFAULT_CONFIG)
    # Kept apart from the merged config: `_init_p_ref` has to tell a setting the
    # caller made from the default it inherited.
    user_cfg = json.loads(a.config) if a.config else {}
    cfg.update(user_cfg)
    try:
        objs = parse_objectives(cfg)
    except ValueError as e:
        out({"initialized": False, "refused": True, "error": "bad_objectives",
             "reason": str(e)})
        raise Refusal(4) from None
    opening = {"cv": _init_code_version(a), "capability": _init_capability(cfg),
               "envelope": _init_envelope(a)}
    op_file = a.op_file or os.path.join(a.op_dir, f"{a.op}_impl.py")
    opening["preopt_hash"] = _init_baseline_hash(a, op_file)
    # AFTER the hash check and BEFORE the baseline is snapshotted: the marker is
    # a comment, so the program the preopt measured and the program this run
    # starts from are the same one, and `--preopt-hash` still checks the bytes
    # that were actually measured.
    opening["config_key"] = _init_config_key(a, op_file)
    with open(op_file, encoding="utf-8", errors="replace") as f:
        opening["source"] = f.read()
    if opening["config_key"]:
        opening["preopt_hash"] = file_hash(op_file)
    seeds = []
    if a.catalog:
        with open(a.catalog, encoding="utf-8") as f:
            seeds = json.load(f)
    # BEFORE `j0`: without `--p-ref` the preopt is the reference, and `j0` is
    # then 100 by construction.
    opening["p_ref_via"] = _init_p_ref(a, cfg, user_cfg)
    # `--preopt-p` IS `J_OBJECTIVE`'s reading: `p` was MIX_AIC for a while and
    # is `aicore_e2e` now, which is what `J`, `--p` and `--samples` all mean.
    opening["j0"] = compute_j(a.preopt_s, a.preopt_p, a.p_ref)
    _refuse_met_target(a, cfg, opening["j0"], opening["p_ref_via"])
    st = _init_state(a, cfg, opening)
    pre_vals = _init_preopt_values(a, st, objs)
    preds = _seed_frontier(st, a, cfg, seeds)
    # The space the preopt's tile was chosen for. Every later delta is compared
    # against this, and against whatever replaced it when a candidate was kept.
    st["domain_fingerprint"] = _domain_state(st, op_file_path(st, a.op_dir))
    seeded_symptoms = _symptom_reboost(st, getattr(a, "preopt_util", 0.0),
                                       getattr(a, "preopt_bubble", 0.0),
                                       op_file_path(st, a.op_dir),
                                       cli_pipes(a, "preopt_"))  # first-move symptom bias
    _init_snapshot(a, op_file, opening["preopt_hash"])
    save(a.op_dir, st)
    evaluation_session.clear_terminal_marker(a.op_dir)
    _init_baseline_log(a, opening, objs, pre_vals)
    report = {"preds": preds, "seeded_symptoms": seeded_symptoms}
    report["anchoring"], report["warning"] = _anchoring_warning(a, cfg)
    report["lint_root_warning"] = _check_lint_root(getattr(a, "lint_root", ""))
    # Say out loud whether the semantic-no-op fold is live. It reads the real
    # `pypto` signatures, so off-device it degrades to comparing ASTs and folds
    # nothing -- and a guard that is silently off in the environment where it can
    # be tested, and on only where it cannot, is how `bubble` stayed 0.0 for an
    # entire campaign. A run that logs `folds=0` and rejects no no-ops has not
    # shown there were none.
    report["n_folds"] = len(_pypto_defaults())
    st["semantic_fold_calls"] = report["n_folds"]
    # Ask the device now, while someone is watching. `init` must not REFUSE on
    # what it hears -- it has to stay runnable with no device, and reading
    # another box's ini deliberately is a legal thing to do -- but a mismatch
    # surfaced here is one surfaced before the campaign is configured around it
    # rather than at the first block. The halt itself lives at the first
    # evaluation: see `confirm_chip`.
    report["chip_mismatch"] = confirm_chip(st, a.device)
    _init_announce(a, st, opening, report)


def _check_lint_root(lint_root):
    """Is `--lint-root` a directory `python -m pypto_op_lint` can be run from?

    A wrong one is not a small mistake. It makes the lint gate raise
    `No module named pypto_op_lint` on EVERY trial, the evaluator cannot tell
    that from a device that has gone away, and the block aborts on consecutive
    environment faults having measured nothing. A run can lose its first block
    exactly this way -- a few faults with `fault_kind: invocation`, the repo
    root passed where `cannbot-skills/plugins-official/
    pypto-op-orchestrator/hooks/pypto-op-lint` was wanted -- and an
    `os.path.isdir` at init catches it before the device is touched.

    Returns "" when there is nothing to say. This WARNS rather than refuses:
    a valid layout the check does not recognise must not stop a campaign.
    """
    if not lint_root:
        return ""
    if not os.path.isdir(lint_root):
        return f"not a directory: {lint_root!r} -- the lint gate will fault on every trial"
    if not (os.path.exists(os.path.join(lint_root, "pypto_op_lint"))
            or os.path.exists(os.path.join(lint_root, "pypto_op_lint.py"))):
        return (f"{lint_root!r} holds no importable `pypto_op_lint` -- "
                f"`python -m pypto_op_lint` will fail on every trial and the "
                f"block will abort on invocation faults. Point --lint-root at "
                f"the directory CONTAINING the package "
                f"(usually <repo>/cannbot-skills/plugins-official/"
                f"pypto-op-orchestrator/hooks/pypto-op-lint), not at the repo root")
    return ""


def _init_config_key(a, op_file):
    """Write the `panko:tune` marker for `--config-key`, or return None.

    A kernel written as a FACTORY keeps its tile values in a per-case dict, not
    in the tile call, so `apply` sees no literal to rewrite and the tile lever
    is dead -- a campaign on such a kernel runs no block at all, and every tile
    value is swept by the model one evaluation at a time.
    The marker says which case this campaign is for; `config_sites` does the
    rest. See `bayesian_optimization/config_sites.py`.

    Refused rather than ignored when the kernel is not written that way: a run
    started with `--config-key` on the wrong file would otherwise look like it
    was tuning the table and be tuning nothing.
    """
    key = getattr(a, "config_key", None)
    if key in (None, ""):
        return None
    key = int(key) if str(key).lstrip("-").isdigit() else key
    with open(op_file, encoding="utf-8", errors="replace") as f:
        src = f.read()
    new_src, action = bayesian.config_sites.mark(src, key)
    if action == "unavailable":
        out({"initialized": False, "refused": True, "error": "no_config_table",
             "config_key": key, "op_file": op_file,
             "hint": "--config-key names an entry of a per-case config dict that is "
                     "`**`-expanded into a kernel factory. This file has none, so "
                     "there is nothing to mark. Drop --config-key, or point "
                     "--op-file at the kernel that carries the table."})
        raise Refusal(3)
    with open(op_file, "w", encoding="utf-8") as f:
        f.write(new_src)
    return {"key": key, "action": action,
            "knobs": sorted(bayesian.config_sites.admitted(new_src))}


def _init_noise(st, a, opening, preopt_m):
    """Seed the noise estimate and the archive from the preopt's reading."""
    # The preopt is the first program the run measures, so it is the first
    # reading of the operator's own noise. Without samples the estimate stays
    # explicitly unmeasured -- never a transcribed constant.
    pre_samples = cli_samples(a, "preopt_")
    if pre_samples:
        st["nodes"]["closed"][0]["eval"]["samples"] = pre_samples
        st["eval_cache"][opening["preopt_hash"]]["samples"] = pre_samples
    st["noise"] = {"recent": [], "measurements": 0, "sigma_log": None,
                   "source": "unmeasured"}
    observe_noise(st, pre_samples, SEARCH_PROTOCOL)
    archive_record(st, opening["preopt_hash"], a.preopt_p, preopt_m)


def _init_p_ref(a, cfg, user_cfg):
    """Settle `P_ref` and return where it came from.

    `P_ref` is the latency the run is quoted against: `J = 100 * P_ref / p`, so
    `J >= 100` means "reached the reference". A reference kernel's measurement
    is the right value -- when there IS one.

    When there is not, there is nothing to reach. `--p-ref` being required, a
    campaign without a reference had to invent a number, and one did: it was
    given a reference its own preopt already beat, making `j0` above 100, so
    `success` -- the first halt `stop` checks -- was true before the first
    candidate. That run ended at evaluation 0 and reported it as a success.

    So `--p-ref` is optional, and leaving it out means what it says: no target.
    The preopt becomes the reference, which makes `J` read as a percentage of
    the baseline (`J = 150` is 1.5x the preopt) -- and since `J` is monotone in
    `1/p` for any fixed `P_ref`, this changes no ordering the search makes, only
    what the number means. `success_stop` goes off with it, because `j0` is then
    exactly 100 and the target is met by definition.

    A caller that asks for both -- no `--p-ref` and `success_stop: true` -- is
    asking to stop at a target it has not named, so `_refuse_met_target` fires
    and says so. That is why the explicit setting is not overridden here.
    """
    if a.p_ref is not None:
        return "explicit:cli"
    a.p_ref = float(a.preopt_p)
    if "success_stop" not in user_cfg:
        cfg["success_stop"] = False
    return "preopt"


def _refuse_met_target(a, cfg, j0, via="explicit:cli"):
    """The preopt already meets `P_ref`. Emit and stop, unless the caller said
    that is expected.

    `success` is the first halt `stop` checks, so a `P_ref` at or past the
    preopt ends the run at evaluation 0 -- and reports it as a SUCCESS, which is
    the worst way to say it: the number is right and the conclusion is wrong. A
    recorded campaign spent a device slot to be told exactly that -- a `P_ref`
    its preopt already beat, and a `j0` above 100 before anything ran.

    Two things cause it and only the caller can tell them apart, so this asks
    rather than guessing: `P_ref` was entered wrong, or the preopt genuinely
    beats the reference kernel and the target is already met. The second is a
    real situation -- it says "optimise this anyway, there is just no target to
    reach" -- and `config.success_stop: false` is how to say so.
    """
    if j0 < 100.0 or not cfg.get("success_stop", DEFAULT_CONFIG["success_stop"]):
        return
    unnamed = via == "preopt"
    out({"initialized": False, "refused": True,
         "error": "target_already_met",
         "preopt_p": a.preopt_p, "p_ref": a.p_ref, "j0": j0,
         "p_ref_via": via,
         "reason": (
             ("no --p-ref was given, so the preopt is the reference and the "
              "target is met by definition -- but `success_stop: true` asks the "
              "run to stop when the target is reached. Name the target or drop "
              "the setting; it is off by default without --p-ref.")
             if unnamed else
             (f"P_ref ({a.p_ref}) is at or past the preopt's "
              f"{a.preopt_p}, so j0 is {j0} and `success` -- the first "
              f"halt `stop` checks -- is already true. The run would end "
              f"at evaluation 0 and report it as a success.")),
         "ask_user": {
             "question": (
                 "The baseline already meets the target this run would be "
                 "measured against. Either the target is wrong, or this kernel "
                 "is already faster than the reference and there is nothing to "
                 "reach. Which is it?"),
             "options": [
                 {"id": "fix_p_ref",
                  "label": "The target is wrong",
                  "action": "re-run `init` with the reference kernel's actual "
                  "latency as --p-ref"},
                 {"id": "no_target",
                  "label": "There is no target; optimise anyway",
                  "action": "add --config '{\"success_stop\": false}'. P_ref "
                  "stays as the baseline every speedup is quoted "
                  "against, and the run ends on the budget, the wall "
                  "clock or `hv_stagnation` like any other"}]}})
    raise Refusal(3)


def _refuse_objectives(where, missing, objs, extra=None):
    """A declared objective with no value ends the run.

    Every dominance judgement, the archive, and the frontier a campaign
    delivers rest on the objective vector. If one component was never measured,
    the front is a claim about something nobody read -- so the run says which
    objective, where it should have come from, and stops. The alternative
    (skip the point and carry on) hides it: a campaign would report a frontier
    built on a single objective while its config named two.
    """
    pre = "preopt-" if where == "init" else ""
    hint = ("pass the whole measurement through (`{0}--{1}measurement '<measure_latency "
            "result as JSON>'`), which carries every reading there is; or name the one "
            "value (`--{1}metrics '{{\"<key>\": <v>}}'`, `--{1}workspace-b`/"
            "`--{1}workspace-state`). A null `p` from measure_latency carries `p_error`. "
            "Or take the objective out of config.objectives"
            ).format("init " if where == "init" else "record ", pre)
    # The one cause worth naming, because it is invisible from the reading: at
    # `runtime_debug_mode: 0` the on-device profiler writes no "AICore
    # End-to-End Time" line at all, so `aicore_e2e` is not a measurement that
    # failed -- it was never taken, and no amount of re-recording will produce
    # it until the flag is on the kernel.
    if any(m.get("objective") == "aicore_e2e" for m in missing):
        hint = ("`aicore_e2e` comes from the on-device profiler's \"AICore "
                "End-to-End Time\" line, which is written ONLY when every "
                "top-level jit kernel carries "
                "debug_options={\"runtime_debug_mode\": 1}. At mode 0 the line "
                "does not exist and the reading cannot be recovered. Check the "
                "kernel's debug_options first, then: " + hint)
    out({"refused": True, "error": "objective_unmeasured", "where": where,
         "missing": missing,
         "objectives": [o["name"] for o in objs],
         "hint": hint,
         **(extra or {})})
    raise Refusal(4)


def _init_preopt_values(a, st, objs):
    """The preopt's objective vector, refusing the run when one is missing."""
    # The cheapest place to find out: the preopt is measured before any search
    # happens, so a missing objective costs no candidate.
    pre_m = ((st.get("nodes") or {}).get("closed") or [{}])[0].get("eval", {}).get("m")
    pre_vals, missing = objective_values(objs, a.preopt_p, pre_m)
    if missing:
        _refuse_objectives("init", missing, objs,
                           {"initialized": False,
                            "note": "the preopt measurement is missing an objective, so "
                                    "no candidate can be placed either"})
    refuse_no_hv_reference(st, objs)
    return pre_vals
