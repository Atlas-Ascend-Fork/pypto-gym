#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""What one measurement is, and what an objective reads from it.

`m` and its CLI spellings, the objective declarations and the value each
one takes off a measurement, the run's own noise, and the bootstrap the
front and `verify` both judge differences with.
"""

import hashlib
import json
import math
import os
import random

from .config import (DEFAULT_CONFIG, NOISE_WINDOW, OBJECTIVE_ALIASES, PIPE_METRICS, VERIFY_ALPHA, VERIFY_BOOT,
                     VERIFY_SEED, _MEASUREMENT_NOT_METRICS, feasibility)


def compute_j(s, p, p_ref):
    """`J = s * P_ref / p * 100`. `p` IS `J_OBJECTIVE`'s reading."""
    if not s or not p or p <= 0:
        return 0.0
    return round(100.0 * p_ref / p, 4)


def metrics(util, bubble, fallback=None, pipes=None):
    """
    `m` for an eval record: only the readings that exist.

    An absent key means "not measured"; 0.0 means "measured, and it was zero".
    `fallback` supplies values the caller knows but the CLI was not given --
    used for a block winner, whose program the block already measured.
    `pipes` carries the PIPE_METRICS under the same rule.
    """
    m = dict(fallback or {})
    if util is not None:
        m["util"] = util
    if bubble is not None:
        m["bubble"] = bubble
    for k in PIPE_METRICS:
        v = (pipes or {}).get(k)
        if v is not None:
            m[k] = v
    return m


def _occupancies(op_file):
    """(UB fraction, L1 fraction) for the kernel currently on disk; either may be
    None. Never raises: a symptom that can fail the run is worse than a symptom
    that stays quiet.

    None is not zero. A kernel whose tiles are all driven by symbols cannot be
    read, and a kernel with no matmul has no L1 tile to read -- neither is a
    kernel whose buffer is empty. Reporting 0.0 for those would fire the symptom
    on a fabricated number, which is what happened before `extract` learned to
    drop a partially-readable tile call instead of keeping the half it
    understood.
    """
    if not op_file or not os.path.exists(op_file):
        return None, None
    try:
        return (feasibility.ub_occupancy(op_file),
                feasibility.l1_occupancy(op_file))
    except Exception:
        return None, None


def cli_pipes(a, prefix=""):
    """PIPE_METRICS off the parsed CLI namespace, skipping what was not given."""
    return {k: getattr(a, prefix + k, None) for k in PIPE_METRICS
            if getattr(a, prefix + k, None) is not None}


def preopt_values(st, objs):
    """
    The baseline program's objective vector, from the cache `init` wrote.

    Read from the cache rather than the archive: the preopt leaves the archive
    as soon as something dominates it, and the reference point must not move
    when it does. A `--remeasure` of the preopt updates the cache entry, so the
    bound follows the pooled median like every other reader of the baseline.
    """
    h = next((c.get("code_hash") for c in ((st.get("nodes") or {}).get("closed") or [])
              if c.get("id") == "root"), None)
    entry = (st.get("eval_cache") or {}).get(h) if h else None
    if not entry:
        return None
    vals, missing = objective_values(objs, entry.get("p"), entry.get("m"))
    return None if missing else vals


def bootstrap_speedup(pre, dele, n_boot=None, seed=None, alpha=None):
    """Speedup median(pre)/median(dele) and its percentile bootstrap interval.

    The two arms are resampled independently: they are separate processes, so
    there is no pairing to preserve. Returns None below two readings per arm.
    """
    n_boot = VERIFY_BOOT if n_boot is None else n_boot
    seed = VERIFY_SEED if seed is None else seed
    alpha = VERIFY_ALPHA if alpha is None else alpha
    if len(pre) < 2 or len(dele) < 2:
        return None
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        a = [pre[rng.randrange(len(pre))] for _ in pre]
        b = [dele[rng.randrange(len(dele))] for _ in dele]
        stats.append(median_of(a) / median_of(b))
    stats.sort()
    lo = stats[int(alpha / 2 * n_boot)]
    hi = stats[int((1 - alpha / 2) * n_boot) - 1]
    return {"speedup": round(median_of(pre) / median_of(dele), 4),
            "ci_low": round(lo, 4), "ci_high": round(hi, 4),
            "level": 1 - alpha, "n_boot": n_boot}


def ci_verdict(ci):
    if ci["ci_low"] > 1.0:
        return "confirmed"
    if ci["ci_high"] < 1.0:
        return "regressed"
    return "unconfirmed"


def is_multi_objective(st):
    """Whether this run has more than one objective, so a front exists."""
    try:
        return len(objectives_of(st)) > 1
    except ValueError:
        return False


def is_bytes(o):
    """Is this objective a byte count?

    Those are the ones with a hardware ceiling and the ones measured on a log
    axis.
    """
    return str(o.get("from", "")).endswith("_b") or o.get("unit") == "B"


def _objective_from_alias(item):
    """One objective named by a bare alias."""
    if item == "latency":
        raise ValueError(
            "objective `latency` does not say which instrument it "
            "means. Name one: `aicore_e2e` (the on-device profiler's "
            "\"AICore End-to-End Time\", needs "
            "debug_options={\"runtime_debug_mode\": 1}) or `mix_aic` "
            "(msprof's per-launch MIX_AIC aicore_time). They rank "
            "programs differently, so the choice is not cosmetic.")
    if item not in OBJECTIVE_ALIASES:
        raise ValueError(f"unknown objective {item!r}; known: "
                         f"{sorted(OBJECTIVE_ALIASES)} (or give "
                         f"{{name, from, direction}})")
    return dict(OBJECTIVE_ALIASES[item], name=item)


def _objective_reference(name, raw):
    """The hypervolume reference declared on one objective, validated."""
    try:
        ref = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"objective {name}: reference must be a "
                         f"number, not {raw!r}") from exc
    if not ref > 0.0:
        raise ValueError(f"objective {name}: reference must be "
                         f"positive, not {ref}")
    return ref


def _objective_from_dict(item):
    """One objective written out as a {name, from, direction} mapping."""
    name = item.get("name")
    src = item.get("from") or (OBJECTIVE_ALIASES.get(name) or {}).get("from")
    if not name or not src:
        raise ValueError(f"objective {item!r} needs a name and a `from`")
    if src != "p" and not src.startswith("m."):
        raise ValueError(f"objective {name}: `from` must be `p` or `m.<key>`, "
                         f"not {src!r}")
    fallback = (OBJECTIVE_ALIASES.get(name) or {}).get("direction", "min")
    o = {"name": name, "from": src,
         "direction": item.get("direction", fallback)}
    # Optional, and the only thing that makes an objective "measured".
    sf = item.get("samples_from",
                  (OBJECTIVE_ALIASES.get(name) or {}).get("samples_from"))
    if sf is not None:
        if sf != "samples" and not sf.startswith("m."):
            raise ValueError(f"objective {name}: `samples_from` must be "
                             f"`samples` or `m.<key>`, not {sf!r}")
        o["samples_from"] = sf
    unit = item.get("unit", (OBJECTIVE_ALIASES.get(name) or {}).get("unit"))
    if unit:
        o["unit"] = unit
    # The hypervolume's reference on THIS axis, overriding the default
    # (the preopt's value, or the chip's memory for a byte axis). It is
    # read by `hv_reference_point` and must be fixed for the run, so it is
    # declared here with the objective rather than passed per call.
    if item.get("reference") is not None:
        o["reference"] = _objective_reference(name, item["reference"])
    return o


def _reject_unsound_objective(o, out_objs, seen):
    """Raise unless one parsed objective stands beside the ones already kept."""
    if o["direction"] not in ("min", "max"):
        raise ValueError(f"objective {o['name']}: direction must be min or max")
    if o.get("samples_from") == "samples" and o["from"] != "p":
        # `samples` is `p`'s list and only `p`'s -- `row_samples`' own
        # docstring says so. Everything else about an objective is checked
        # here (direction, duplicate names, duplicate `from`, the SHAPE of
        # `samples_from`) except the one pairing that makes the readings
        # belong to the values. `{"from": "m.mix_aic", "samples_from":
        # "samples"}` is a plausible hand-edit -- `mix_aic` used to read
        # from `p` -- and it builds a front where dominance on MIX_AIC
        # values is decided by the bootstrap of aicore_e2e readings, stored
        # on the row by `samples_by_objective` and carried through every
        # later reprune with nothing reporting it.
        raise ValueError(
            f"objective {o['name']} reads {o['from']} but takes its samples "
            f"from 'samples', which is `p`'s readings. An objective that is "
            f"not `p` needs its own 'm.<key>' sample source, or none.")
    if o["name"] in seen:
        raise ValueError(f"objective {o['name']} is listed twice")
    # Two names for one reading is not two objectives: every point would be
    # comparable on a duplicated axis and the front would be an ordering
    # dressed as a trade-off. `latency` and `mix_aic` are the same `p`.
    clash = next((x["name"] for x in out_objs if x["from"] == o["from"]), None)
    if clash:
        raise ValueError(f"objectives {clash} and {o['name']} both read "
                         f"{o['from']}; they are one objective under two names")


def parse_objectives(cfg):
    """`config.objectives` as [{name, from, direction}], validated.

    Raises ValueError on anything it cannot read: a run whose objectives are
    misspelled would otherwise optimise something else and say nothing.
    """
    spec = (cfg or {}).get("objectives") or DEFAULT_CONFIG["objectives"]
    if isinstance(spec, (str, dict)):
        spec = [spec]
    out_objs, seen = [], set()
    for item in spec:
        if isinstance(item, str):
            o = _objective_from_alias(item)
        elif isinstance(item, dict):
            o = _objective_from_dict(item)
        else:
            raise ValueError(f"objective {item!r} must be a name or a dict")
        _reject_unsound_objective(o, out_objs, seen)
        seen.add(o["name"])
        out_objs.append(o)
    if not out_objs:
        raise ValueError("config.objectives is empty")
    return out_objs


def objective_values(objs, p, m):
    """({name: value}, [missing]) for one measurement.

    A declared objective with no value is NOT skipped: `_require_objectives`
    halts the run. An objective that cannot be measured makes every dominance
    judgement in the run unsound, and a front built on it would be a claim
    about something nobody measured.
    """
    vals, missing = {}, []
    m = m or {}
    for o in objs:
        if o["from"] == "p":
            v = p if (p and float(p) > 0) else None
        else:
            key = o["from"][2:]
            v = m.get(key)
            # `workspace_b` is only a reading when the probe says so.
            if (key == "workspace_b"
                    and m.get("workspace_state") not in ("measured", "zero")):
                v = None
        if v is None:
            missing.append({"objective": o["name"], "from": o["from"]})
        else:
            vals[o["name"]] = float(v)
    return vals, missing


def objective_text(objs, vals):
    """`latency=120.5 workspace=256.0MiB` -- the objective vector for a log line.

    The trajectory is what a person reads to see why a candidate was kept or
    dropped, and under more than one objective `p` alone cannot say. Byte-valued
    objectives are printed in MiB because that is the unit the trade-off is
    discussed in.
    """
    out_parts = []
    for o in objs:
        v = vals.get(o["name"])
        if v is None:
            out_parts.append(f"{o['name']}=?")
        elif o["from"].endswith("_b"):
            out_parts.append(f"{o['name']}={v / 2 ** 20:.1f}MiB")
        else:
            out_parts.append(f"{o['name']}={round(v, 4)}")
    return " ".join(out_parts)


def objectives_of(st):
    """
    This run's objectives. Validated at `init`, so a failure here is a state
    edited by hand -- reported rather than guessed around.
    """
    return parse_objectives(st.get("config"))


def parse_samples(text):
    """`--samples "12.3,12.5"` -> [12.3, 12.5]. Unparseable entries are dropped.

    A malformed list must not fail the record: the measurement happened, `p` is
    already in hand, and the samples only sharpen what can be said about it.
    """
    rows = []
    for tok in (text or "").replace(";", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            v = float(tok)
        except ValueError:
            continue
        if v > 0:
            rows.append(v)
    return rows


def sigma_log(vals):
    """Sample standard deviation of log(latency), or None below two readings.

    Log scale, because latency is positive and multiplicative -- a 10% slowdown
    is 10% at any scale -- so that is where its noise is nearly symmetric and
    where sigma and cv coincide for small cv. The same formula lives in
    `measure_latency._sigma_log`, which runs in a different process and reports
    it for the log; this one is what the state believes.
    """
    xs = [math.log(v) for v in vals if v and v > 0]
    if len(xs) < 2:
        return None
    mu = sum(xs) / len(xs)
    var = sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)
    return round(math.sqrt(var), 6)


def observe_noise(st, samples, protocol=None):
    """Fold one measurement's spread into the run's own noise estimate.

    Carried rather than configured because the measured cv spans two orders of
    magnitude ACROSS OPERATORS, and it moves with the program as well: the two
    arms of one kernel can differ severalfold in spread.
    A single transcribed constant is wrong for almost every operator, and every
    rule that wants to know "is this difference real" needs the right one.

    The aggregate is a median over a bounded window, so one pathological
    measurement does not set the run's idea of its own noise, and the state does
    not grow without limit.
    """
    sg = sigma_log(samples)
    if sg is None:
        return st.get("noise")
    noise = st.setdefault("noise", {"recent": [], "measurements": 0})
    recent = noise.setdefault("recent", [])
    recent.append(sg)
    del recent[:-NOISE_WINDOW]
    noise["measurements"] = noise.get("measurements", 0) + 1
    noise["sigma_log"] = median_of(recent)
    noise["latest_sigma_log"] = sg
    noise["source"] = "measured"
    if protocol:
        noise["protocol"] = protocol
    return noise


def median_of(xs):
    ys = sorted(xs)
    n = len(ys)
    if not n:
        return None
    mid = n // 2
    return ys[mid] if n % 2 else round((ys[mid - 1] + ys[mid]) / 2.0, 6)


def derive_eval_timeout(cfg, preopt_compile_s):
    """(seconds, source) for `eval_timeout_s`.

    Derived from the preopt's own compile when one was measured, clamped to the
    configured floor and cap; otherwise the configured value, unchanged, and
    `source` says so. `chip_envelope` refuses when it cannot derive, and that is
    right there -- a wrong core count silently rejects legal tiles. A wrong
    timeout only spends wall clock, so refusing to start over a missing optional
    reading would cost more than it saves. What must not happen is deriving
    silently and leaving no way to tell which number a campaign ran under.
    """
    configured = int(cfg.get("eval_timeout_s", DEFAULT_CONFIG["eval_timeout_s"]))
    if cfg.get("eval_timeout_explicit"):
        return configured, "explicit:config"
    try:
        base = float(preopt_compile_s)
    except (TypeError, ValueError):
        return configured, "default"
    if base <= 0:
        return configured, "default"
    k = float(cfg.get("eval_timeout_k", DEFAULT_CONFIG["eval_timeout_k"]))
    lo = int(cfg.get("eval_timeout_floor_s", DEFAULT_CONFIG["eval_timeout_floor_s"]))
    hi = int(cfg.get("eval_timeout_cap_s", DEFAULT_CONFIG["eval_timeout_cap_s"]))
    return max(lo, min(hi, int(round(k * base)))), "derived:preopt"


def _block_reading(st, code_hash):
    """The block's own measurement of `code_hash`, or {}.

    Two places hold it, and BOTH have to be consulted. `eval_cache` is where
    every trial lands -- but it is also the duplicate index, so
    `block_apply_winner` pops the winner out of it on purpose, or `record`
    would see those bytes as already evaluated and charge nothing. That pop
    took the reading away with the entry, which is why the two recovery paths
    below silently stopped working: the promotion of a winner then carried no
    `workspace_b` and HALTED the run with `objective_unmeasured` for bytes the
    device had in fact measured, and when it did not halt it overwrote the
    archive's nine readings with `samples: []`.

    So the pop stashes what it removed under `block_winner`, keyed by hash, and
    this reads the cache first and the stash second. Nothing else looks at
    `block_winner`, so the charge is unaffected.
    """
    if not code_hash:
        return {}
    hit = (st.get("eval_cache") or {}).get(code_hash)
    if hit:
        return hit
    stash = st.get("block_winner") or {}
    return stash if stash.get("code_hash") == code_hash else {}


def block_metrics(st, code_hash):
    """Diagnostics a block recorded for `code_hash`, or {}.

    The winner is promoted by an ordinary `record` call so the ratchet lives
    in one place, but that call is made by the optimizer, which is told not to
    re-measure -- so it has no reading to pass. The block does: it measured
    this program on the device. Recovering it here keeps the promotion
    deterministic and costs no device time.
    """
    m = _block_reading(st, code_hash).get("m") or {}
    return {k: v for k, v in m.items() if v is not None}


def block_samples(st, code_hash):
    """The readings a block took of `code_hash`, or [].

    The sibling of `block_metrics`, and it exists for the same reason: the
    winner is promoted by a `record` the optimizer makes without re-measuring,
    so that call carries no `--samples`. The difference is that `m` was
    recovered and the readings were not -- the promotion overwrote the cache
    entry with one that had none, and the run's own best program ended up
    holding no measurement of itself, while the archive row for the same code
    hash still carried the readings the block had taken.

    That is not cosmetic: `verify` and `record --remeasure` read the cache, and
    the archive judges dominance on readings. The two must not disagree about
    what was measured.
    """
    return list(_block_reading(st, code_hash).get("samples") or [])


def _cli_workspace(a, prefix=""):
    """`workspace_b` / `workspace_state` off the CLI, under the probe's rules:
    a size without a state is `measured` (or `zero` at 0), a state of
    `unavailable` carries no size, and nothing given means nothing recorded --
    never a 0 standing in for a reading that was not taken."""
    b = getattr(a, prefix + "workspace_b", None)
    state = getattr(a, prefix + "workspace_state", None)
    if state == "unavailable":
        return {"workspace_state": "unavailable"}
    if b is None:
        return {"workspace_state": state} if state else {}
    return {"workspace_b": int(b),
            "workspace_state": state or ("zero" if int(b) == 0 else "measured")}


def _cli_extra_metrics(a, prefix=""):
    """`record --metrics '<json>'`: any reading the caller wants in `m`.

    This is what makes an objective the user's to choose: name the key in
    `config.objectives` (`{"name": "data_movement", "from": "m.dm_bytes"}`) and
    record its value here. No code in this harness has to know about it.
    """
    raw = getattr(a, prefix + "metrics", None)
    if not raw:
        return {}
    try:
        d = json.loads(raw)
    except ValueError:
        return {}
    return {k: v for k, v in d.items() if isinstance(d, dict) and v is not None}


def cli_samples(a, prefix=""):
    """`--samples`, falling back to the measurement's own `samples` list.

    `samples` is deliberately not a metric -- it is `p`'s readings and belongs
    on the row, not on `m` (`_MEASUREMENT_NOT_METRICS`) -- but that left it the
    one reading a caller still had to remember a flag for, which is exactly what
    `--measurement` exists to stop. And it is not a minor one: it is what the
    archive's bootstrap and the margin-based stop run on. A point recorded
    without it is compared on exact medians while every other pair uses an
    interval, and `bootstrap_speedup` returns None for it against EVERYTHING, so
    the second objective alone decides its dominance. The baseline is the worst
    case, because it is a permanent front member.

    The explicit flag still wins, on the same contract as every other reading
    `--measurement` carries.
    """
    given = parse_samples(getattr(a, prefix + "samples", None))
    if given:
        return given
    raw = getattr(a, prefix + "measurement", None)
    if not raw:
        return []
    try:
        d = json.loads(raw)
    except ValueError:
        return []
    if not isinstance(d, dict):
        return []
    vals = d.get("samples")
    if not isinstance(vals, (list, tuple)):
        return []
    out_vals = []
    for v in vals:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return []
        out_vals.append(float(v))
    return out_vals


def cli_measurement(a, prefix=""):
    """`record --measurement '<json>'`: `measure_latency`'s whole result as `m`.

    The reason this exists is that `m` used to be assembled from whichever
    flags the caller remembered. Only `--util`, `--bubble`, the per-pipe ones
    and `--workspace-*` have flags, so every other reading -- `aicore_e2e`
    among them -- reached `m` only if the caller also wrote it out by hand
    under `--metrics`, so a reading nobody wrote out was simply lost -- the
    objective among them. An objective must not depend on a caller remembering
    a flag.

    So this takes the measurement dict as it comes and keeps what is a reading:
    scalars, and lists of numbers (a `*_samples` list). Anything nested is a
    structure, not a reading, and is dropped. NOTHING IS NAMED HERE ON PURPOSE
    -- a reading `measure_latency` starts producing tomorrow is available as an
    objective source with no change to this harness, which is the same promise
    `--metrics` already makes for a value the caller computes itself.
    """
    raw = getattr(a, prefix + "measurement", None)
    if not raw:
        return {}
    try:
        d = json.loads(raw)
    except ValueError:
        return {}
    if not isinstance(d, dict):
        return {}
    out_m = {}
    for k, v in d.items():
        if k in _MEASUREMENT_NOT_METRICS or v is None:
            continue
        if isinstance(v, (int, float, str, bool)):
            out_m[k] = v
        elif (isinstance(v, list) and v
              and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                      for x in v)):
            out_m[k] = list(v)
    return out_m


def record_metrics(a, st):
    """
    `m` for one `record`, assembled in ONE place.

    It used to be assembled twice -- once to check the objectives were all
    measured and once to store -- from slightly different pieces, so a record
    that supplied a value one way could be refused for not supplying it the
    other. `J`'s own reading comes from here too, so the number the ratchet
    compares and the number the archive stores cannot disagree.

    `--measurement` is the whole reading; the block record is what a promotion
    falls back on when the promoting call carries none. A caller that passed
    both measured this program just now, so its numbers win.
    """
    fallback = block_metrics(st, a.code_hash)
    fallback.update(cli_measurement(a))
    m = metrics(getattr(a, "util", None), getattr(a, "bubble", None),
                fallback, cli_pipes(a))
    m.update(_cli_workspace(a))
    m.update(_cli_extra_metrics(a))
    return m


def incumbent_latency(st, src):
    """The measured `p` of the program this source text IS, or None.

    A tile study's `current` seed is recorded as a COMPLETED observation at the
    tile configuration the file holds (`lever._fill_seeds`), so the number
    attached to it has to belong to that program. `progress.best_aicore_e2e_us`
    is `best_node`'s, and under more than one objective `select` restores the
    front base, which can be a different and slower program -- so reading the
    global best there teaches the study one program's tiles at another
    program's latency, and every later draw is scored against it.

    None is the other half of the contract, not a failure: `_fill_seeds` queues
    a seed with no value to be measured, so an unmeasured base costs one device
    trial instead of poisoning the surrogate.
    """
    entry = (st.get("eval_cache") or {}).get(
        hashlib.sha256(src.encode("utf-8")).hexdigest()) or {}
    p = entry.get("p")
    if not entry.get("s") or not isinstance(p, (int, float)) or p <= 0:
        return None
    return float(p)


def _preopt_metrics(a):
    """`m` for the preopt, by the same rules as a record's."""
    m = metrics(getattr(a, "preopt_util", None),
                getattr(a, "preopt_bubble", None),
                cli_measurement(a, "preopt_"),
                pipes=cli_pipes(a, "preopt_"))
    m.update(_cli_workspace(a, "preopt_"))
    m.update(_cli_extra_metrics(a, "preopt_"))
    return m
