#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""The closed loop behind `prior_gain`: record outcomes, rebuild the prior table.

Every measured candidate is appended, one JSON line, to two places:

  <op_dir>/optimization/action_outcomes.jsonl   the campaign's own record
  $PANKO_HOME/outcomes/<campaign>.jsonl         the machine's shared store

After a real stop, `report` rebuilds `$PANKO_HOME/action_prior_table.json` from
the shared store, and the next campaign's `init` reads it. So `prior_gain` -- and the thompson posterior --
learn from every campaign the machine has run.

What keeps the loop from corrupting itself:

  * A run never sees its own table change. `init` copies the table it reads
    into the campaign directory and every later read uses that copy, so a
    campaign is reproducible and two concurrent campaigns cannot disturb each
    other. Its sha256 is in the state.
  * A catalogue entry whose TEXT changed is a different action. Each row
    carries a hash of the entry's delta; rows whose hash no longer matches the
    catalogue are dropped at rebuild and counted in the table's meta, so a
    store that spans several catalogue revisions cannot mix them.
  * Old evidence fades: rows are weighted 0.5 ** (age / half_life), age in
    campaigns, so a compiler or catalogue improvement is not outvoted forever
    by what used to be true.
  * Selection bias is recorded, not ignored: under thompson each row carries
    the probability the chosen action had of being selected, and `--ipw`
    reweights by it. Under the product score selection is deterministic and
    no such correction exists; that is stated in meta.
  * Experiments pin the table: an explicit `config.action_prior_table` turns
    lookup off, and `config.prior_auto_update: false` stops the rebuild.
"""

import hashlib
import json
import os
import re
import tempfile
import time
import typing

import jsonio

SCHEMA = 1
DEFAULT_HALF_LIFE = 50        # campaigns
BOOL_SKIP = {"determined"}


def panko_home():
    """$PANKO_HOME, else ~/.panko."""
    return os.environ.get("PANKO_HOME") or os.path.join(os.path.expanduser("~"), ".panko")


def soc_slug(soc):
    """A SoC name as a path component: lowercase, and nothing but [a-z0-9._-]."""
    s = re.sub(r"[^a-z0-9._-]+", "-", str(soc or "").strip().lower()).strip("-")
    return s or None


def store_dir(home=None, soc=None):
    """
    Where outcome rows go. Per CHIP, because what an action is worth is a
    property of the hardware: the same delta that wins on one chip's buffer
    geometry and core counts can be neutral on another's, and a table built
    from both describes neither. `soc=None` is the legacy flat directory, which
    holds every row written before the split.
    """
    base = os.path.join(home or panko_home(), "outcomes")
    slug = soc_slug(soc)
    return os.path.join(base, slug) if slug else base


def learned_table_path(home=None, soc=None):
    slug = soc_slug(soc)
    name = f"action_prior_table.{slug}.json" if slug else "action_prior_table.json"
    return os.path.join(home or panko_home(), name)


def delta_hash(text):
    """What identifies a catalogue action across revisions: its text."""
    return hashlib.sha256((text or "").strip().encode("utf-8")).hexdigest()[:16]


def file_sha(path):
    if not path:
        return None
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return None


def _load_json(path):
    """One JSON file's contents, read through a handle that is always closed."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _atomic_write(path, text):
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", dir=d, prefix=".tmp-", suffix=".json",
                delete=False, encoding="utf-8") as f:
            tmp = f.name
            f.write(text)
        os.replace(tmp, path)
    finally:
        # The rename leaves nothing behind; every other path must not either.
        if tmp and os.path.exists(tmp):
            os.remove(tmp)


def append_outcome(row, campaign_file, shared_file=None):
    """
    One measured candidate. Best-effort: a store that cannot be written must
    never fail the record, so errors are returned rather than raised.
    """
    line = json.dumps(dict(row, schema=SCHEMA), sort_keys=True) + "\n"
    errors = []
    for path in (campaign_file, shared_file):
        if not path:
            continue
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
        except OSError as e:
            errors.append(f"{path}: {e}")
    return errors


def _jsonl_paths(dirs):
    """Every *.jsonl file under `dirs`: each directory in turn, its files in name order."""
    paths = []
    for d in dirs:
        if not d or not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            paths.extend(os.path.join(root, name) for name in sorted(files)
                         if name.endswith(".jsonl"))
    return paths


def _parse_jsonl(lines, rows):
    """Append each non-blank line's object to `rows`; return how many lines were lost."""
    bad = 0
    try:
        for line in lines:
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except ValueError:
                bad += 1
    except OSError:
        bad += 1
    return bad


def read_rows(dirs):
    """Every outcome row under `dirs` (files named *.jsonl), malformed lines skipped."""
    rows, bad = [], 0
    for path in _jsonl_paths(dirs):
        try:
            with open(path, encoding="utf-8") as f:
                bad += _parse_jsonl(f, rows)
        except OSError:
            bad += 1
    return rows, bad


def _cell():
    return {"n_meas": 0.0, "n_kept": 0.0, "n_s0": 0.0, "sum_log_gain_kept": 0.0,
            "campaigns": set()}


def _add(cell, row, w):
    cell["n_meas"] += w
    if row.get("kept"):
        cell["n_kept"] += w
        cell["sum_log_gain_kept"] += w * max(0.0, float(row.get("log_gain") or 0.0))
    if not row.get("s"):
        cell["n_s0"] += w
    cell["campaigns"].add(row.get("campaign"))


def _add_base(cell, base, w):
    for k in ("n_meas", "n_kept", "n_s0", "sum_log_gain_kept"):
        cell[k] += w * float(base.get(k) or 0.0)


def _final(cell):
    out = {k: round(cell[k], 4) for k in ("n_meas", "n_kept", "n_s0", "sum_log_gain_kept")}
    out["n_campaigns"] = len([c for c in cell["campaigns"] if c])
    return out


class BuildOptions(typing.NamedTuple):
    """What a table build reads besides the rows: a base table to merge, the row weighting, the SoC filter."""
    base: typing.Any = None
    base_weight: typing.Any = 1.0
    half_life: typing.Any = DEFAULT_HALF_LIFE
    ipw: typing.Any = False
    min_propensity: typing.Any = 0.05
    soc: typing.Any = None
    legacy_soc: typing.Any = None


def _filter_soc(rows, soc, legacy_soc):
    """The rows recorded on `soc`, and what was left out: (rows, other chip, unlabelled)."""
    wanted, legacy = soc_slug(soc), soc_slug(legacy_soc)
    kept_soc, other, unlabelled = [], 0, 0
    for r in rows:
        got = soc_slug(r.get("soc"))
        if got == wanted:
            kept_soc.append(r)
            continue
        if got is None:
            unlabelled += 1
            if legacy == wanted:
                kept_soc.append(r)
            continue
        other += 1
    return kept_soc, other, unlabelled


def _campaign_ages(rows):
    """Each campaign's age in campaigns (newest 0), and how many campaigns the rows span."""
    order = sorted({(r.get("campaign_start") or 0, r.get("campaign")) for r in rows})
    return {c: len(order) - 1 - i for i, (_, c) in enumerate(order)}, len(order)


def _drop_drifted(rows, catalog):
    """The rows whose catalogue entry still carries the text they were measured against."""
    current = {e["id"]: delta_hash(e.get("delta")) for e in (catalog or [])}
    kept_rows, drift = [], 0
    for r in rows:
        cid = r.get("catalog_id")
        if cid and current and r.get("delta_hash") != current.get(cid):
            drift += 1
            continue
        kept_rows.append(r)
    return kept_rows, drift


def _ipw_scales(rows, min_propensity):
    """The per-class IPW renormalisers, and how many rows carry no propensity.

    Hajek-style: weights are renormalised per class so the counts keep their
    scale and only the balance between rows changes.
    """
    scales, no_prop = {}, 0
    for r in rows:
        p = r.get("propensity")
        key = r.get("catalog_id") or "insert"
        if p is None:
            no_prop += 1
            continue
        s = scales.setdefault(key, [0.0, 0])
        s[0] += 1.0 / max(p, min_propensity)
        s[1] += 1
    return scales, no_prop


def _accumulate(rows, age, half_life, ipw_scale, min_propensity):
    """Fold the rows into the global, per-action and per-action-predicate cells."""
    g, by_cat, by_pred = _cell(), {}, {}
    for r in rows:
        key = r.get("catalog_id") or "insert"
        w = 0.5 ** (age.get(r.get("campaign"), 0) / half_life) if half_life else 1.0
        if r.get("propensity") is not None and key in ipw_scale:
            tot, n = ipw_scale[key]
            w *= (1.0 / max(r["propensity"], min_propensity)) * n / tot
        _add(g, r, w)
        _add(by_cat.setdefault(key, _cell()), r, w)
        for p, v in sorted((r.get("predicates") or {}).items()):
            if isinstance(v, bool) and p not in BOOL_SKIP:
                _add(by_pred.setdefault(f"{key}|{p}={v}", _cell()), r, w)
    return g, by_cat, by_pred


def _merge_base(g, by_cat, by_pred, base, base_weight):
    """Merge a base table's cells into the ones just built."""
    _add_base(g, base.get("global") or {}, base_weight)
    for k, c in (base.get("by_catalog") or {}).items():
        _add_base(by_cat.setdefault(k, _cell()), c, base_weight)
    for k, c in (base.get("by_catalog_predicate") or {}).items():
        _add_base(by_pred.setdefault(k, _cell()), c, base_weight)


def _meta(options, counts):
    """What a table says about its own evidence: what was read, what was left out, how weighted."""
    ipw, base_used = options.ipw, bool(options.base)
    return {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "schema": SCHEMA, "generator": "prior_store.build",
        "rows_read": counts["rows_read"], "rows_used": counts["rows_used"],
        "rows_dropped_catalog_text_changed": counts["drift"],
        "soc": soc_slug(options.soc), "legacy_soc": soc_slug(options.legacy_soc),
        "rows_dropped_other_chip": counts["other"],
        "rows_unlabelled_chip": counts["unlabelled"],
        # Not evidence about `kept`; see `build`. Counted so a table can
        # say how much of the store it is deliberately not reading.
        "rows_dropped_front_phase": counts["front"],
        "campaigns": counts["campaigns"], "half_life_campaigns": options.half_life,
        "ipw": bool(ipw), "rows_without_propensity": counts["no_prop"] if ipw else None,
        "base_table_merged": base_used,
        "base_weight": options.base_weight if base_used else None,
        "WARNING_selection_bias": (
            "Rows recorded under the product score were selected deterministically, "
            "so no propensity exists for them and IPW cannot correct them."),
    }


def build(rows, catalog=None, options=BuildOptions()):
    """The prior table from outcome rows (+ an optional base table).

    `soc` keeps only the rows recorded on that chip. Rows written before the
    per-chip split carry no `soc` at all, so they are DROPPED rather than
    guessed at -- unless `legacy_soc` names the chip they were recorded on, in
    which case they count as that chip's. Both counts are in the meta, so a
    table can say how much of its evidence it declined to use.

    Phase-2 rows are dropped for a different reason. This table models one
    thing: how often an action makes the program it is applied to FASTER
    (`kept`). Rows carry `phase: "search"` since the phase split was removed;
    the `front` rows below are historical. A phase-2 candidate was applied to a
    front point and expected to be slower than the global best -- that is what
    the phase was for -- so
    every one of them would arrive as `kept: false` and drag each action's
    learned rate down in proportion to how well phase 2 worked. The rows stay
    in the store, carrying `phase` and `front_insert`, because the reward phase
    2 actually optimises is worth fitting from data rather than guessing; they
    are simply not evidence about this one.
    """
    base, base_weight, half_life = options.base, options.base_weight, options.half_life
    ipw, min_propensity = options.ipw, options.min_propensity
    soc, legacy_soc = options.soc, options.legacy_soc
    other = unlabelled = 0
    if soc:
        rows, other, unlabelled = _filter_soc(rows, soc, legacy_soc)
    front_rows = sum(1 for r in rows if r.get("phase") == "front")
    rows = [r for r in rows if r.get("phase") != "front"]
    age, campaigns = _campaign_ages(rows)
    kept_rows, drift = _drop_drifted(rows, catalog)
    ipw_scale, no_prop = _ipw_scales(kept_rows, min_propensity) if ipw else ({}, 0)
    g, by_cat, by_pred = _accumulate(kept_rows, age, half_life, ipw_scale, min_propensity)
    if base:
        _merge_base(g, by_cat, by_pred, base, base_weight)
    counts = {"rows_read": len(rows), "rows_used": len(kept_rows), "drift": drift,
              "other": other, "unlabelled": unlabelled, "front": front_rows,
              "campaigns": campaigns, "no_prop": no_prop}
    return {
        "meta": _meta(options, counts),
        "global": _final(g),
        "by_catalog": {k: _final(v) for k, v in sorted(by_cat.items())},
        "by_catalog_predicate": {k: _final(v) for k, v in sorted(by_pred.items())},
    }


def rebuild(home=None, catalog=None, options=BuildOptions(), extra_dirs=()):
    """Rebuild this machine's learned table for ONE chip. Returns (path, meta).

    The whole store is read -- the per-chip directories and the legacy flat
    files together -- and `build` decides which rows belong to this chip. The
    table is written per chip, so a campaign on one chip cannot overwrite what
    another measured.
    """
    rows, bad = read_rows([store_dir(home), *extra_dirs])
    table = build(rows, catalog, options)
    table["meta"]["malformed_lines"] = bad
    path = learned_table_path(home, options.soc)
    _atomic_write(path, json.dumps(table, indent=1, sort_keys=True))
    return path, table["meta"]


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Rebuild the PANKO action prior table.")
    ap.add_argument("--home", default=None, help="PANKO_HOME (default $PANKO_HOME or ~/.panko)")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="more directories of *.jsonl outcome files (e.g. another server's store)")
    ap.add_argument("--catalog", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                      "..", "references", "action_catalog.json"))
    ap.add_argument("--base", default="",
                    help="a historical table to merge in as a base; none by default, "
                         "since the table is built from this machine's own campaigns")
    ap.add_argument("--half-life", type=float, default=DEFAULT_HALF_LIFE)
    ap.add_argument("--ipw", action="store_true")
    ap.add_argument("--soc", default=None,
                    help="build the table for ONE chip, by the SoC name the "
                         "envelope recorded; "
                         "omitted, every row counts, which mixes chips")
    ap.add_argument("--legacy-soc", dest="legacy_soc", default=None,
                    help="the chip that rows written before the per-chip split "
                         "were recorded on; they carry no soc and are otherwise dropped")
    a = ap.parse_args()
    catalog = _load_json(a.catalog) if a.catalog else None
    base = _load_json(a.base) if a.base and os.path.exists(a.base) else None
    options = BuildOptions(base=base, half_life=a.half_life, ipw=a.ipw,
                           soc=a.soc, legacy_soc=a.legacy_soc)
    path, meta = rebuild(a.home, catalog, options, a.extra)
    jsonio.emit({"written": path, **meta})


if __name__ == "__main__":
    main()
