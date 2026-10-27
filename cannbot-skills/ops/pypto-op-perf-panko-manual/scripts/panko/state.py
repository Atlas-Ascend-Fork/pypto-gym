#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""`search_state.json`, the node snapshots, and the paths of both.

Reading the state, writing it, and putting a recorded program back in the
working file. What the run believes lives here; what it decides does not.
"""

import ast
import hashlib
import json
import os
import shutil
import subprocess
import time

from .config import (CHIP_ENVELOPE, DEFAULT_CONFIG, OPTIONAL_ENVELOPE, REFUSAL_GLOBAL_AFTER, SCRIPTS_DIR,
                     _RENAMED_CONFIG_KEYS, _RENAMED_STATE_KEYS, bayesian, chip_profile, clamp_prior)


def read_text(path, errors="replace"):
    """The file's text, with the handle closed before the caller sees it."""
    with open(path, encoding="utf-8", errors=errors) as f:
        return f.read()


def _read_bytes(path):
    """The file's bytes, with the handle closed before the caller sees them."""
    with open(path, "rb") as f:
        return f.read()


def chip_envelope(st=None):
    """The envelope for this run. From the state when it has one, so a resumed
    campaign keeps validating against what it started with.
    """
    env = dict(CHIP_ENVELOPE)
    if st:
        env.update({k: v for k, v in (st.get("chip_envelope") or {}).items()
                    if k in env})
    return env


# --------------------------------------------------------------------------- I/O
def sdir(op_dir):
    return os.path.join(op_dir, "optimization")


def _spath(op_dir):
    return os.path.join(sdir(op_dir), "search_state.json")


def rpath(op_dir, op):
    return os.path.join(sdir(op_dir), f"{op}_optimization.md")


def migrate_keys(st):
    """Accept a state file written before the `bo` -> `bayesian_optimization` rename.

    The new spelling wins if both are present, which can only happen if a file was
    hand-edited; the old key is dropped either way so `save` cannot write it back.
    """
    for holder, table in ((st, _RENAMED_STATE_KEYS),
                          (st.get("config") or {}, _RENAMED_CONFIG_KEYS)):
        for old_key, new_key in table.items():
            if old_key in holder:
                value = holder.pop(old_key)
                holder.setdefault(new_key, value)
    _migrate_priors(st)
    return st


def load(op_dir):
    with open(_spath(op_dir), encoding="utf-8") as f:
        return migrate_keys(json.load(f))


def save(op_dir, st):
    os.makedirs(sdir(op_dir), exist_ok=True)
    accumulate_wall_clock(st["progress"], st.get("config") or {})
    # Restamp on every write. A state resumed under a different build is otherwise
    # indistinguishable from one produced entirely by the build recorded at init.
    cv = _code_version()
    if cv and st.get("code_version") and cv != st["code_version"]:
        st.setdefault("code_version_history", [])
        if cv not in st["code_version_history"]:
            st["code_version_history"].append(cv)
    if cv:
        st["code_version"] = cv
    tmp = _spath(op_dir) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=2)
    os.replace(tmp, _spath(op_dir))


def _fpath(op_dir):
    return os.path.join(sdir(op_dir), "faults.jsonl")


# ------------------------------------------------- code snapshots (harness-owned)
# Greedy accumulation is enforced HERE, not by the LLM: the harness snapshots every kept
# candidate under nodes/<code_hash>.py and deterministically restores the running global best
# into the working <op>_impl.py before each new action, rolling back non-improving candidates.
# This makes "keep the win, build the next idea on it" hold end-to-end and prevents the
# best code from ever being lost (which the agent-driven scheme could not guarantee).
def _ndir(op_dir):
    return os.path.join(sdir(op_dir), "nodes")


def npath(op_dir, code_hash):
    return os.path.join(_ndir(op_dir), f"{code_hash}.py")


def file_hash(path):
    """sha256 of the file, or None. The same digest `record` is handed."""
    try:
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except (OSError, TypeError):
        return None


_PYPTO_DEFAULTS = None


def _pypto_defaults():
    """Declared defaults of the pypto callables, or {} when pypto is absent.

    Absent is the normal case off-device (tests, the baseline arms), and it is
    handled by degrading rather than by guessing: with no defaults the semantic
    comparison still catches reformatting and comment edits and folds nothing.
    """
    global _PYPTO_DEFAULTS
    if _PYPTO_DEFAULTS is None:
        if bayesian is None:
            _PYPTO_DEFAULTS = {}
        else:
            try:                                        # pragma: no cover
                import pypto                            # noqa: PLC0415
                _PYPTO_DEFAULTS = bayesian.semantic.defaults_for(
                    getattr(pypto, "frontend", None), pypto)
            except Exception:
                _PYPTO_DEFAULTS = {}
    return _PYPTO_DEFAULTS


def _semantic_noop(st, op_dir, op_file, parent=None):
    """(True, why) when the candidate on disk IS the program it is judged against.

    The base is the one `restore` would roll back to: the action's running best
    if it has produced one, otherwise the global best. That is exactly the
    program the ratchet compares against, so it is the one the candidate has to
    differ from to be a candidate at all.

    Two guards, and the first is what makes this usable at all. EVERY rejection
    path in this harness restores the base into the working file -- `record`'s
    duplicate and non-improving branches, the static gate, `select` -- so at most
    points in a run the file IS the base, byte for byte. Comparing them then
    reports "semantic no-op" about a candidate the coder has not written yet.
    Byte-equality is a different condition with a different meaning (nothing was
    edited), and it belongs to the caller, not here.
    """
    if bayesian is None or not op_file or not os.path.exists(op_file):
        return False, ""
    r = (st["progress"].get("refine") or {})
    # The running best belongs to ONE action. Judging action B's candidate
    # against action A's local best compares it to a program it was never built
    # on top of.
    if parent is not None and r.get("action") != parent:
        r = {}
    best = _refine_base(r) or {}
    base_hash = best.get("code_hash") or ratchet_best_hash(st)
    snap = npath(op_dir, base_hash) if base_hash else None
    if not snap or not os.path.exists(snap):
        return False, ""
    try:
        cand = read_text(op_file)
        base = read_text(snap)
    except OSError:
        return False, ""
    if cand == base:
        return False, ""            # nothing was edited; not this gate's business
    return bayesian.semantic.same_program(cand, base, _pypto_defaults())


def op_file_path(st, op_dir):
    """
    The working impl file to snapshot/restore. Prefer the explicit path stored at init;
    fall back to <op_dir>/<op>_impl.py (the standard convention).
    """
    f = st.get("op_file")
    if f:
        return f
    op = st.get("op")
    return os.path.join(op_dir, f"{op}_impl.py") if op else None


def snapshot(op_dir, op_file, code_hash):
    """
    Persist current op_file bytes -> nodes/<code_hash>.py. True iff it is on disk.

    Reports failure rather than raising. Callers on the restore contract --
    `init`'s baseline above all -- must REFUSE on a False: every later candidate
    is applied on top of a snapshot, so a missing one means the search is
    accumulating onto whatever the caller happened to leave in the working file.
    """
    if not op_file or not code_hash or not os.path.exists(op_file):
        return False
    try:
        os.makedirs(_ndir(op_dir), exist_ok=True)
        shutil.copyfile(op_file, npath(op_dir, code_hash))
    except OSError:
        return False
    return True


def restore(op_dir, op_file, code_hash):
    """Restore nodes/<code_hash>.py -> op_file (deterministic roll-back to a known-good state)."""
    if not op_file or not code_hash:
        return False
    snap = npath(op_dir, code_hash)
    if not os.path.exists(snap):
        return False
    try:
        shutil.copyfile(snap, op_file)
    except OSError:
        return False
    return True


def _code_version():
    """Which harness produced this state. A run whose provenance is not
    recorded cannot be compared against another run: the same state file can
    come from a build with or without symptom anchoring, greedy accumulation,
    or the frozen evaluator, and nothing in the state says which.
    """
    here = SCRIPTS_DIR
    rev = ""
    git = shutil.which("git")
    if git:
        try:
            rev = subprocess.check_output(
                [git, "-C", here, "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL).decode().strip()
        except (OSError, subprocess.SubprocessError):
            # Not a work tree, or git cannot read it. The VERSION file below is
            # the documented fallback for a skill tree that was copied rather
            # than cloned, so an empty revision is the handled case.
            rev = ""
    if rev:
        return rev
    # The skill tree is sometimes copied outside a work tree. A VERSION file
    # written at copy time is the documented fallback.
    try:
        with open(os.path.join(here, "VERSION"), encoding="utf-8") as f:
            v = f.read().strip()
            return v or None
    except OSError:
        return None


def _gstag(st):
    """'n/limit', or 'n (limit off)' when global stagnation no longer halts."""
    lim = st["config"].get("global_stagnation_limit")
    n = st["progress"]["global_stagnation"]
    return f"{n}/{lim}" if lim else f"{n} (limit off)"


def next_id(st, prefix):
    st["_seq"] = st.get("_seq", 0) + 1
    return f"{prefix}{st['_seq']}"


# ----------------------------------------------------------------- subcommands
def _domain_state(st, op_file, live_tiles=None):
    """The derived tile domain for `op_file`, or None when BO is not in play."""
    if bayesian is None or not st["config"].get("bayesian_optimization_tile_search", False):
        return None
    if not op_file or not os.path.exists(op_file):
        return None
    return bayesian.domain.derive(op_file, live_tiles or st.get("live_tiles"))


def _structure(st, op_dir):
    """The current program's structural signature, or None if unavailable."""
    if bayesian is None:
        return None
    f = op_file_path(st, op_dir)
    try:
        return bayesian.lever.structural_signature(read_text(f, errors="strict"))
    except Exception:
        return None


def live_frontier(st, op_dir):
    """
    The frontier minus actions retired on the structure now in the file.

    A measurement disproves an action ON THE PROGRAM IT WAS TAKEN ON. A sweep
    of a loop granularity can find a sharp interior optimum and prune the other
    granularity actions -- correctly, for the program in front of it. A later
    action that merges two matmuls into one then wants a granularity that sweep
    had already measured and retired. Because the prune was permanent, every
    later structural action is explored at the old geometry, and the
    restructuring that needed the other one can never show its value. The same
    happens to a cube tile: several independent mechanisms agreeing on one value
    is still one structure's evidence, generalised past the structure that
    produced it.

    So a prune is scoped to `structural_signature` -- the program hash with tile
    literals erased, which moves when the structure moves and not when a tile
    value does. While the structure is unchanged the action stays retired; once
    something restructures the kernel its evidence is stale and it returns to the
    frontier. It returns as an ORDINARY candidate: `pypto_action_priority` ranks
    it against everything else with whatever prior_gain and `tried` count it
    carries. Nothing is promoted, nothing is ordered by fiat.

    `prune_permanent` stays permanent, for the case that is not about structure
    at all -- u12 (NZ weight format) is inexpressible without changing the
    operator's frozen layout, and no restructuring makes it expressible.
    """
    sig = _structure(st, op_dir)
    return [n for n in st["nodes"]["open"]
            if not (n.get("pruned_on") and n["pruned_on"] == sig)]


def rollback(a, st, r):
    """Roll the working file back to the incumbent, as a measured regression would."""
    restore(a.op_dir, op_file_path(st, a.op_dir),
            (_refine_base(r)["code_hash"] if _refine_base(r)
             else _base_hash(st, a.op_dir)))
    save(a.op_dir, st)


def stored_refusals(st):
    """Stored refusals, as rows that carry their own scope.

    Rows written before this field existed have no `structures` and are treated
    as evidence about no structure at all, so they match nothing and are
    re-earned. That is the safe direction: the cost is one repeated device trial,
    where trusting them is a tile deleted from the space of a program they were
    never observed on.
    """
    rows = []
    for r in st.get("bayesian_optimization_refusals") or []:
        if not r.get("params") or not r.get("sig"):
            continue
        rows.append({"params": r["params"], "sig": r["sig"],
                     "structures": list(r.get("structures") or []),
                     "scope": r.get("scope") or "structure"})
    return rows


def refusals_for_structure(rows, structure):
    """The subset a block on `structure` may act on.

    `error_signature` already refuses to sign a capacity failure or a golden
    mismatch, so what reaches here is a front-end pass rejecting the tile call.
    That was carried to every structure on the reasoning that it is a fact about
    the tile -- but the signature keeps only ErrCode + Enum + PassName, and
    `F4FFFF` is the generic FeError code that this repo's own experience table
    maps to several causes. `error_signature`'s docstring says as much: it
    establishes that the signature is stable across occurrences of the same
    refusal, NOT that two refusals sharing one share a cause. So a refusal
    earned where a tile met one structure could delete that tile from the space
    of a program that no longer has it, permanently and without a measurement.

    Structure-scoped by default therefore, and promoted only on evidence: a
    refusal seen on REFUSAL_GLOBAL_AFTER distinct structures has been shown to
    survive a restructure rather than assumed to.
    """
    return [{"params": r["params"], "sig": r["sig"]} for r in rows
            if r["scope"] == "global" or structure in r["structures"]]


def merge_refusals(rows, observed, structure, promote_after=REFUSAL_GLOBAL_AFTER):
    """Fold this block's refusals into the stored rows, stamped with `structure`."""
    index = {(json.dumps(r["params"], sort_keys=True), r["sig"]): r for r in rows}
    for obs in observed or []:
        if not obs.get("params") or not obs.get("sig"):
            continue
        key = (json.dumps(obs["params"], sort_keys=True), obs["sig"])
        row = index.get(key)
        if row is None:
            row = {"params": obs["params"], "sig": obs["sig"],
                   "structures": [], "scope": "structure"}
            index[key] = row
            rows.append(row)
        if structure and structure not in row["structures"]:
            row["structures"].append(structure)
        if len(row["structures"]) >= promote_after:
            row["scope"] = "global"
    return rows


def wrapper_signature(src):
    """Hash of everything in the file that is NOT a `@...jit` kernel.

    The host wrapper is FROZEN for the whole run, and this is what enforces it.

    `p` counts AICore time inside the jit kernel and nothing else. That is a
    sound objective exactly while the uncounted part is identical between the
    baseline and the candidate, because then it cancels in the comparison. An
    action that moves work across the boundary breaks the condition: the kernel
    gets cheaper, the wrapper gets more expensive, `p` improves, and the ratchet
    keeps a candidate that is slower to run. The upstream review put it as
    kernel 100us -> 80us with 30us added to the wrapper, scored as a 20us win.

    Two of the catalogue's strongest recorded results were that shape -- F-21's
    fusion and F-22's layout copy. Neither made the work cheaper; both moved it
    out of the region `p` measures. Removing those actions is what makes the objective
    sound, and checking the signature is what makes the removal enforceable:
    the coder is handed free-form intents, and a rule that lives only in a
    document is a rule the search cannot rely on.

    Hashed over the AST rather than the bytes, so a comment, a blank line or a
    reformat outside the kernel is invisible and only a change to what the host
    actually DOES trips it. An added module-level import trips it too: that is
    deliberate erring, since the alternative is deciding which statements are
    "work" and a module constant feeding the wrapper's padding is work.

    Top-level jit functions are what this excludes. A kernel nested inside a
    class or a factory would be hashed as wrapper, which would make every tile
    rewrite look like drift -- no kernel in this repository is written that way,
    and the check says so rather than guessing.

    Returns None when the file does not parse -- callers treat that as "cannot
    tell" and say so, rather than passing a candidate the check never ran on.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None

    def _is_kernel(n):
        return (isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                and decorated_jit(n))

    shell = [n for n in tree.body if not _is_kernel(n)]
    dumped = ast.dump(ast.Module(body=shell, type_ignores=[]))
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()


def decorated_jit(node):
    for d in getattr(node, "decorator_list", []):
        f = d.func if isinstance(d, ast.Call) else d
        while isinstance(f, ast.Attribute):
            if f.attr == "jit":
                return True
            f = f.value
        if isinstance(f, ast.Name) and f.id == "jit":
            return True
    return False


def _wrapper_drift(st, path):
    """"" if the wrapper still matches the baseline, else why it does not."""
    want = st.get("wrapper_signature")
    if not want:
        return ""                     # pre-dates the check; nothing to compare
    try:
        got = wrapper_signature(read_text(path))
    except OSError:
        return ""
    if got is None:
        return "the candidate does not parse, so the wrapper freeze could not be checked"
    if got != want:
        return ("the host wrapper differs from the baseline. PANKO optimizes the "
                "time inside the @jit kernel, and that number is only comparable "
                "while everything outside it is unchanged -- work moved into the "
                "wrapper leaves the measured region and scores as a win without "
                "being one. Restore the wrapper and put the change inside the "
                "kernel, or run this as a different experiment")
    return _debug_options_drift(st, path)


def _resolve_envelope(a):
    """(envelope, reason). None means it could not be derived and init must ask.

    Explicit beats derived: a caller who names the numbers is measuring a chip
    deliberately, and that is recorded as such so the report cannot be read as
    "these were read from the installed runtime" when they were not.
    """
    given = getattr(a, "chip_envelope", "")
    if given:
        try:
            supplied = json.loads(given)
        except ValueError as e:
            return None, f"--chip-envelope is not valid JSON: {e}"
        missing = [k for k in CHIP_ENVELOPE
                   if k not in OPTIONAL_ENVELOPE and k not in supplied]
        if missing:
            return None, ("--chip-envelope is missing " + ", ".join(sorted(missing)))
        env = {k: supplied[k] for k in CHIP_ENVELOPE if k in supplied}
        env["source"] = "explicit:cli"
        # A caller who names the numbers may name the chip too, and it is worth
        # carrying: the outcome store and the learned prior table are per chip
        # (`prior_store.store_dir`), so without it an explicitly-configured run
        # files its measurements under no chip at all.
        if supplied.get("soc"):
            env["soc"] = supplied["soc"]
            env["soc_via"] = "explicit:cli"
        return env, ""
    if chip_profile is None:
        return None, "chip_profile.py is not importable next to the harness"
    env, why = chip_profile.resolve(device=getattr(a, "device", ""))
    if env is None:
        return None, why
    out_env = {k: env[k] for k in CHIP_ENVELOPE if k != "source" and k in env}
    out_env["source"] = env.get("source", "")
    out_env["soc"] = env.get("soc", "")
    out_env["soc_via"] = env.get("soc_via", "")
    return out_env, ""


def _chip_mismatch_report(expected, live, how, device, env):
    """The halt payload for an envelope derived for another chip."""
    return {
        "error": "chip_mismatch",
        "reason": (f"the envelope was derived for {expected} but device "
                   f"{device!r} reports {live} (via {how}). The buffer "
                   "capacities and core counts PANKO validates every candidate "
                   "against are the wrong chip's."),
        "expected_soc": expected, "live_soc": live,
        "envelope_source": env.get("source", ""),
        "ask_user": {
            "question": (
                f"PANKO was initialised for {expected}, and the device it is "
                f"about to measure on is {live}. Continuing would filter the "
                "search space with another chip's limits, rejecting legal tiles "
                "before the device ever saw them. How do you want to proceed?"),
            "options": [
                {"id": "re_init",
                 "label": f"Re-initialise for {live}",
                 "action": "delete optimization/search_state.json and re-run "
                           "init on this device; any search so far is scored "
                           "against the wrong envelope and does not carry over"},
                {"id": "other_device",
                 "label": f"Point PANKO at a {expected}",
                 "action": "re-run with --device set to a device that holds the "
                           "chip this run was initialised for"},
            ]},
        "hint": "relay the question to the user; do not choose for them"}


def confirm_chip(st, device, probe=None):
    """The chip the envelope describes, re-checked against the chip in the slot.

    Two gates, because they answer at different moments. `init` resolves a NAME
    and derives the envelope from that name's ini; it must stay runnable with no
    device, or PANKO cannot be initialised offline and its own suite cannot run.
    Only a device evaluation can say whether the name was the right one, so that
    is where the name is checked -- before the first one, not after eighty.

    A mismatch is a halt and not a warning. The envelope decides which tiles are
    even legal, so the damage from a wrong one is candidates deleted before the
    device sees them, leaving nothing in the record that contradicts the result.
    A warning would be printed into a log nobody reads until the numbers look
    strange.

    "unknown" proceeds. The envelope came from a named SoC either way, and a box
    without pyACL is still a box PANKO can search; what it must not do is claim
    the check passed. The verdict is recorded, so a report can say whether the
    numbers were confirmed against silicon or only against a name.
    """
    env = st.get("chip_envelope") or {}
    expected = env.get("soc") or ""
    ask = probe or (chip_profile.confirm if chip_profile is not None else None)
    if ask is None:
        return None
    verdict, live, how = ask(expected, device)
    env["confirmed"] = {"verdict": verdict, "live_soc": live, "via": how,
                        "at": int(time.time())}
    st["chip_envelope"] = env
    if verdict != "mismatch":
        return None
    return _chip_mismatch_report(expected, live, how, device, env)


def open_node(st, action_id):
    return next((n for n in st["nodes"]["open"] if n["id"] == action_id), None)


def _refine_base(r):
    """The program an action's next candidate is built on: the last point it
    accepted (`base`), or -- for a refine record written before `base` existed,
    when the two could not differ -- its `best`. None before anything was
    accepted."""
    if not r:
        return None
    return r.get("base") or r.get("best")


def debug_options_signature(src):
    """Hash of the `debug_options` on every top-level `@...jit` kernel.

    Frozen for the run, like the wrapper, and for the same reason: it changes
    what the measurement is rather than what the kernel costs.

    `runtime_debug_mode: 1` turns on the on-device profiler's stdout record.
    `p` no longer needs it (it is read from msprof's MIX_AIC rows, which the
    flag does not affect), and the difference between the two modes is small
    and of unknown sign -- it has been seen in both directions, on different
    devices, so it is not even attributable to the flag. But "small, direction
    unknown" is exactly the size of a
    plausible win: a candidate whose only change is dropping `debug_options`
    could be recorded as an improvement while doing nothing to the kernel. The
    flag is a property of the measurement, so it is fixed at the baseline and
    every candidate is compared under the one the baseline used.

    Also keeps the diagnostics honest. `util` / `aic_util` / `aiv_util` come
    from the stdout record and `tilefwk_L1_prof_data.json`, and neither exists
    at mode 0 -- so a mid-run flip would make them appear or vanish between
    candidates the symptom index treats as comparable.

    Returns None when the file does not parse.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    flags = []
    for n in tree.body:
        if not (isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                and decorated_jit(n)):
            continue
        for dumped in _jit_debug_options(n):
            flags.append((n.name, dumped))
    flags.sort()
    return hashlib.sha256(repr(flags).encode("utf-8")).hexdigest()


def _jit_debug_options(node):
    """The dumped `debug_options` values on one node's decorators."""
    dumped = []
    for d in getattr(node, "decorator_list", []):
        for kw in (d.keywords if isinstance(d, ast.Call) else []):
            if kw.arg == "debug_options":
                dumped.append(ast.dump(kw.value))
    return dumped


def _debug_options_drift(st, path):
    """"" if the kernels' `debug_options` still match the baseline's."""
    want = st.get("debug_options_signature")
    if not want:
        return ""                     # pre-dates the check; nothing to compare
    try:
        got = debug_options_signature(read_text(path))
    except OSError:
        return ""
    if got is None or got == want:
        return ""
    return ("the kernels' `debug_options` differ from the baseline. That is the "
            "measurement's configuration, not the kernel's cost: the two modes "
            "read up to 1.8% apart on the same program, and the stdout "
            "diagnostics (util, aic_util, aiv_util) exist in one and not the "
            "other. Restore `debug_options` and put the change inside the "
            "kernel, or re-initialise the campaign under the mode you want")


def _apply_chip_memory(a, env):
    """`--chip-memory-mib` over whatever the envelope resolved to.

    Its own flag, not part of `--chip-envelope`: the buffer capacities are
    properties of the silicon and a run must not be able to vary them, while
    this is a BUDGET -- the memory a person is willing to let this kernel hold
    -- and the honest default (the chip's whole memory) assumes the kernel runs
    alone, which it does not in deployment. Stated here rather than inferred so
    the report can say which it was.
    """
    mib = getattr(a, "chip_memory_mib", None)
    if mib is None:
        return env, ""
    try:
        mib = int(mib)
    except (TypeError, ValueError):
        return None, "--chip-memory-mib is not an integer"
    if mib <= 0:
        return None, "--chip-memory-mib must be positive"
    env = dict(env)
    env["memory_mib"] = mib
    env["memory_mib_via"] = "explicit:cli"
    return env, ""


def _base_hash(st, op_dir=None):
    """The program the next action is applied on top of.

    The front point `_advance_front_base` chose at the last close, in either
    phase, falling back to the global best before the first close or when its
    source is gone. With one objective the chosen point IS the global best, so
    this is greedy accumulation.
    """
    h = st["progress"].get("base_code_hash")
    if h and (op_dir is None or os.path.exists(npath(op_dir, h))):
        return h
    return ratchet_best_hash(st)


def _migrate_priors(st):
    """Bound any `prior_gain` a state file already carries.

    `clamp_prior` bounds the value where it enters, which is the convention
    `clamp_belief` set. That does nothing for a run resumed from a state file an
    earlier build wrote, where the frontier can hold an out-of-range prior the
    selector would still multiply by. The state file is the world model and a run
    re-hydrates from it, so the bound has to hold on the way in as well.
    """
    nodes = st.get("nodes") or {}
    for bucket in ("open", "held"):
        for n in nodes.get(bucket) or ():
            if "prior_gain" in n:
                n["prior_gain"], _ = clamp_prior(n["prior_gain"])


def accumulate_wall_clock(pr, cfg):
    """Add the interval since the last save to `wall_clock_s`, capped at the grace.

    `wall_clock_s` is the time the campaign was WORKED, and it is the one figure
    `wall_clock_limit_s` bounds. Elapsed time counts every hour the orchestrator
    was not calling -- an agent session that ended, a night, a device lent out --
    and a campaign driven one invocation at a time is idle for most of its
    elapsed time. A measured run can spend a fraction of its elapsed hours
    working, with the idle time concentrated in ONE long gap between two
    consecutive successful outcomes, so halting it on elapsed time would leave
    most of the budget unspent.

    Interval-based, so several saves inside one cycle each charge their own slice
    and the total is the same. The grace is per interval, not per cycle: a cycle
    that legitimately runs past it is undercharged by the excess, which is the
    safe direction since the alternative charges idleness. A state written before
    this change carries an elapsed figure, which becomes the accumulator's
    starting point -- it overstates the work done, so a resumed run keeps a
    budget no larger than the one it was launched under.

    Elapsed time is not stored. It is `now - _start_ts` wherever it is wanted,
    and `cmd_stop` reports it beside the worked figure so a 17 h run that spanned
    three days still says so.
    """
    now = time.time()
    last = pr.get("last_seen_ts") or pr.get("_start_ts") or now
    grace = cfg.get("idle_grace_s", DEFAULT_CONFIG["idle_grace_s"])
    pr["wall_clock_s"] = int(pr.get("wall_clock_s", 0)
                             + min(max(0.0, now - last), grace))
    pr["last_seen_ts"] = now


def ratchet_best_hash(st):
    bn = st.get("best_node")
    return next((c["code_hash"] for c in st["nodes"]["closed"] if c["id"] == bn), None)


def preopt_code_hash(st):
    """The baseline program's hash: the `root` node `init` wrote.

    Not derived from `best_node`, which is `root` only until something beats it.
    """
    return next((c["code_hash"] for c in st["nodes"]["closed"]
                 if c["id"] == "root"), None)
