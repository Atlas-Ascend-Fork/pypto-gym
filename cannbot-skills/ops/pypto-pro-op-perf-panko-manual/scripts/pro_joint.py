# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Exercise the released Pro block through the compact measurement contract."""

import json
import hashlib
import math
from uuid import uuid4
from argparse import Namespace
from pathlib import Path

from panko import state
from panko.stopping import block_budget
from bayesian_optimization import block
from measure import acquire_candidate, campaign_contract
import pro_backend


def _sweep_limit(base, per_cluster, count):
    return max(base, math.ceil(per_cluster * count)) if count and per_cluster and per_cluster > 0 else base


def _trial_budget(a, st, count):
    config = st["config"]
    per = config.get("block_trials_per_cluster", 3.0)
    cap = _sweep_limit(config.get("block_min_trials", 4) or 0, per, count) if count and per and per > 0 else None
    if getattr(a, "max_trials", 0):
        cap = min(cap, a.max_trials) if cap else a.max_trials
    floor = _sweep_limit(config.get("block_share_floor", 12) or 0, config.get("block_share_sweeps", 3.0), count)
    patience = _sweep_limit(config.get("stagnation_K", 7), config.get("block_stagnation_per_cluster", 1.0), count)
    granted = block_budget(st["progress"], config, Namespace(max_trials=cap), share_floor=floor)
    if not granted:
        raise ValueError("tile budget exhausted")
    return granted, patience


def _block_checkpoint(a, st, root, count):
    request = dict(root=str(root.resolve()), parent=a.parent, code_hash=state.file_hash(a.op_file),
                   command=a.test_command, max_trials=getattr(a, "max_trials", None), live_tiles=st.get("live_tiles"))
    saved = st.get("active_tile_block")
    if saved is None:
        granted, patience = _trial_budget(a, st, count)
        saved = {"request": request, "granted": granted, "patience": patience, "receipts": [],
                 "reported": 0, "current_latency": st["eval_cache"].get(request["code_hash"], {}).get("p")}
    elif saved["request"] != request:
        raise ValueError("resume the original tile block with its original source and options")
    return saved


def _trial_reading(a, st, pending, index, confirming, dest, contract):
    """Replay the measured prefix, committing each new receipt and charge together."""
    receipts = pending["receipts"]
    replay = index < len(receipts)
    if replay:
        digest, was_confirming = receipts[index]
        if confirming != was_confirming or state.file_hash(dest) != digest:
            raise ValueError("saved tile trial or receipt changed")
    elif not confirming and index >= pending["granted"]:
        raise ValueError("tile budget exhausted")
    # A receipt may already exist if the process exited before the state commit.
    try:
        if dest.exists():
            reading = json.loads(dest.read_bytes())
        else:
            reading = acquire_candidate(a, contract, dest, st["config"]["eval_timeout_s"])
        actual = a.validate_reading(reading, a.op_file, False, allow_partial_workspace=True)
        expected = contract if reading.get("p") is not None else {
            k: v for k, v in contract.items() if k != "timed_target"}
        if actual != expected:
            raise ValueError("tile measurement contract changed")
    except ValueError:
        if not replay and dest.is_file():
            dest.rename(dest.with_suffix(f".failed-{uuid4().hex}.json"))
        raise
    if not replay:
        receipts.append([state.file_hash(dest), confirming])
        if not confirming:
            progress = st["progress"]
            progress["evals_used"] += 1
            progress["block_evals"] = progress.get("block_evals", 0) + 1
        state.save(a.op_dir, st)
    return reading


def _measure_trials(a, st, root, file, source, hw, domain, pending):
    contract = campaign_contract(st)
    from panko.block import block_recorder

    checkpoint = {"base_src": source, "apply_options": {"extra_consts": hw.extra_consts}, "meter": {"device_trials": 0},
                  "trials": [], "winners": {}, "faults": []}
    calls = []
    confirming = False
    record = block_recorder(a, st, checkpoint)

    def on_trial(rec):
        if len(calls) > pending["reported"]:
            record(rec)
            pending["reported"] = len(calls)
            state.save(a.op_dir, st)

    def confirm(*args):
        nonlocal confirming
        confirming = True
        return block.confirm_block_winner(*args)

    def evaluate(params):
        candidate, _ = block.apply_best(source, params, extra_consts=hw.extra_consts)
        if "active_tile_block" not in st:
            st["active_tile_block"] = pending
            state.save(a.op_dir, st)
        pro_backend.write_source(str(file), candidate)
        dest = root / ("tile-%03d.json" % len(calls))
        try:
            reading = _trial_reading(a, st, pending, len(calls), confirming, dest, contract)
            calls.append((reading, dest))
            evaluate.last_samples = reading.get("samples") or []
            evaluate.last_metrics = reading
            if reading.get("correctness_passed") and reading.get("p"):
                return 1, reading["p"]
            if reading.get("test_failed"):
                return 0, None, "incorrect"
            raise ValueError("incomplete tile measurement")
        finally:
            pro_backend.write_source(str(file), source)

    result = block.run(
        str(file),
        evaluate,
        hw,
        block.BlockOptions(
            stagnation_k=pending["patience"],
            max_trials=pending["granted"],
            on_trial=on_trial,
            domain=domain,
            dtype_bytes=hw.dtype_bytes,
            current_latency=pending["current_latency"],
        ),
        confirm=confirm,
    )
    if len(calls) != len(pending["receipts"]):
        raise ValueError("tile replay did not consume the saved trials")
    return result, calls


def _finish_block(st, root, result, calls, granted):
    confirm = result.get("confirm") or {}
    charged = len(calls) - confirm.get("trials", 0)
    summary = {"granted": granted, "trials": result.get("n_trials"), "device_trials": charged, "confirm": confirm,
               "best_config": result.get("best_config"), "best_objective_us": result.get("best_objective_us")}
    if result.get("aborted"):
        summary["aborted"] = result["aborted"]
    (root / "tile-block.json").write_text(json.dumps(summary, indent=2))
    st.setdefault("blocks", []).append(summary)
    if charged < 0 or charged > granted:
        raise ValueError("inconsistent tile charge")
    st.pop("active_tile_block", None)
    return confirm, charged


def _precharged_receipt(a, code_hash, path):
    return dict(parent=a.parent, code_hash=code_hash, path=str(path.resolve()),
                sha256=state.file_hash(path), charged=True)


def _save_block(a, st, root, file, result, calls, observed, changed_sites, granted):
    confirm, charged = _finish_block(st, root, result, calls, granted)
    if changed_sites:
        winner_hash = state.file_hash(str(file))
        entry = st["eval_cache"].pop(winner_hash, None)
        if entry:
            confirmed = [pair["winner"]["p"] for pair in confirm.get("pairs", [])]
            if confirmed:
                entry["p"] = sum(confirmed) / len(confirmed)
            st["block_winner"] = dict(entry, code_hash=winner_hash)
    st.pop("pending_block_receipt", None)
    reply = {"ran": True, "best_config": result.get("best_config"), "n_trials": result.get("n_trials"),
             "best_objective_us": result.get("best_objective_us"), "device_trials": charged,
             "best_was_incumbent": result.get("best_was_incumbent"), "reason": result.get("reason"),
             "code_hash": state.file_hash(str(file)) if changed_sites else None, "granted": granted,
             "charged": charged}
    if not changed_sites:
        state.save(a.op_dir, st)
        return reply
    if not observed:
        raise ValueError("no measurement of the selected source")
    if confirm.get("pairs"):
        output = save_confirmation(root, file, confirm, calls, observed[-1][0])
        reply["best_objective_us"] = sum(pair["winner"]["p"] for pair in confirm["pairs"]) / len(confirm["pairs"])
    else:
        output = observed[-1][1]
    reply.update(receipt=str(output), best_workspace_b=observed[-1][0]["workspace_b"],
                 best_workspace_state=observed[-1][0]["workspace_state"])
    st["pending_block_receipt"] = _precharged_receipt(a, winner_hash, output)
    state.save(a.op_dir, st)
    return reply


def run_block(a, st, root, *, record_rejection=None):
    """Measure tile trials and return a summary with a receipt for a changed winner.

    Each trial temporarily stages its source and restores the block input. On
    completion, stage the winner and write the block summary under root. Charge
    each completed search trial once. A first-trial precision failure can return
    the rejection recorded by the callback. Other aborts or unusable results raise ValueError.
    """
    file = Path(a.op_file)
    source = file.read_bytes().decode("utf-8")
    hw, domain = block.hw_for(str(file), st.get("live_tiles"), st.get("chip_envelope"), author_src=source)
    clusters = block.clusters_for(source, hw.extra_consts, domain=domain)
    count = len(clusters or [])
    pending = _block_checkpoint(a, st, root, count)
    granted = pending["granted"]
    result, calls = _measure_trials(a, st, root, file, source, hw, domain, pending)
    if result.get("aborted"):
        if record_rejection is not None and len(calls) == 1:
            reading, path = calls[0]
            incorrect = reading.get("test_failed") is True and reading.get("correctness_passed") is False
            if incorrect and reading.get("p") is None and reading["code_hash"] == state.file_hash(str(file)):
                st["pending_block_receipt"] = _precharged_receipt(a, reading["code_hash"], path)
                state.save(a.op_dir, st)
                reply = record_rejection(path)
                st = state.load(a.op_dir)
                st.pop("pending_block_receipt", None)
                _finish_block(st, root, result, calls, granted)
                state.save(a.op_dir, st)
                return {"rejection": reply}
        _finish_block(st, root, result, calls, granted)
        state.save(a.op_dir, st)
        raise ValueError("tile block aborted: " + str(result["aborted"]))
    if not calls and not result.get("best_was_incumbent"):
        raise ValueError("block produced no usable result: " + str(result.get("reason")))
    winner = source
    changed_sites = 0
    if result.get("best_config") and not result.get("best_was_incumbent"):
        winner, changed_sites = block.apply_best(source, result["best_config"], extra_consts=hw.extra_consts)
    observed = [(r, path) for r, path in calls if r["code_hash"] == hashlib.sha256(winner.encode()).hexdigest()]
    pro_backend.write_source(str(file), winner)
    return _save_block(a, st, root, file, result, calls, observed, changed_sites, granted)


def save_confirmation(root, file, confirm, calls, winner):
    paths = []
    offset = len(calls) - confirm["trials"]
    for pair, order in zip(confirm["pairs"], confirm["orders"]):
        reading, path = calls[offset + order.index("W")]
        if reading["code_hash"] != winner["code_hash"] or reading["p"] != pair["winner"]["p"]:
            raise ValueError("confirmation does not match winner receipt")
        paths.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        offset += 2
    output = root / "confirmed-winner.json"
    output.write_text(json.dumps({"schema": "panko-confirmed-winner-v1", "op_file": str(file), "readings": paths}))
    return output


def confirmed_receipt(envelope, validate):
    """Recompute the winner mean from distinct frozen acquisitions of the live source."""
    readings = []
    profiles = set()
    for item in envelope["readings"]:
        raw = Path(item["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != item["sha256"]:
            raise ValueError("confirmation receipt changed")
        reading = json.loads(raw)
        if "_validated_confirmed_us" in reading or not reading.get("profile_files"):
            raise ValueError("expected original acquisition")
        reading["contract"] = validate(reading, envelope["op_file"], True)
        if profiles.intersection(reading["profile_files"]):
            raise ValueError("repeated confirmation acquisition")
        profiles.update(reading["profile_files"])
        if readings and reading["contract"] != readings[0]["contract"]:
            raise ValueError("confirmation contracts differ")
        readings.append(reading)
    if not readings:
        raise ValueError("missing confirmation readings")
    value = round(sum(r["p"] for r in readings) / len(readings), 6)
    return dict(readings[-1], p=value, _validated_confirmed_us=value)


def direct(a):
    from panko.config import out
    from panko.state import confirm_chip
    from panko.stopping import stop_condition
    from pro_step import check_command

    st = state.load(a.op_dir)
    if not st["config"].get("bayesian_optimization_tile_search"):
        return out({"ran": False, "reason": "tile search is disabled"})
    if not st.get("active_tile_block") and (stop_condition(st, a) or st["progress"].get("stop_reason")):
        return out({"ran": False, "reason": "campaign stopped"})
    if not any(node["id"] == a.parent for node in st["nodes"]["open"]):
        raise ValueError("block requires an open action")
    check_command(a, st)
    bad = confirm_chip(st, st.get("device_id", ""))
    if bad is not None:
        raise ValueError(str(bad))
    a.op_file = st["op_file"]
    drift = state.wrapper_drift(st, a.op_file)
    if drift:
        raise ValueError(drift)
    if a.live_tiles:
        st["live_tiles"] = a.live_tiles
    root = Path(a.op_dir) / "optimization" / "blocks" / str(len(st.get("blocks", [])))
    if st.get("active_tile_block") and st["active_tile_block"]["request"]["root"] != str(root.resolve()):
        raise ValueError("resume the original STEP dispatch")
    root.mkdir(parents=True, exist_ok=True)
    return out(run_block(a, st, root))
