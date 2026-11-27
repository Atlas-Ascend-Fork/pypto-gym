# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Record a coder handoff through the compact campaign's normal public commands."""

import json
import re
from pathlib import Path
from uuid import uuid4
from measure import acquire_candidate, campaign_contract
from evaluation.measure_latency import campaign_command
from panko import selection, state
from panko.config import call_transition, out, Refusal
from pro_persistence import atomic_write
from panko.stopping import stop_condition
from source_analysis import program_bundle
import pro_backend


def check_command(a, st):
    c = st["measurement_contract"]
    device = str(c["device"])
    argv, cwd, env = campaign_command(a.test_command, a.op_dir, device)
    if (argv, Path(cwd).resolve(), env) != (c["argv"], Path(c["cwd"]).resolve(), c["env"]):
        raise ValueError("test command or device differs from the original acquisition")
    return device


def _measure_and_record(a, st, root, request, record):
    path = root / "measurement.json"
    try:
        reading = acquire_candidate(a, campaign_contract(st), path, st["config"]["eval_timeout_s"])
        if not reading.get("p") and not reading.get("test_failed"):
            raise ValueError(reading.get("p_error") or "incomplete candidate measurement")
    except (OSError, ValueError):
        if path.is_file():
            path.rename(path.with_suffix(f".failed-{uuid4().hex}.json"))
        retry = {"request": request, "retry_measurement": state.file_hash(a.op_file)}
        atomic_write(root / "dispatch.json", json.dumps(retry))
        raise
    atomic_write(root / "dispatch.json", json.dumps({"request": request}))
    return record(path)


def _record_candidate(a, st, root, run, request, retry_measurement):
    def record_receipt(path):
        reply = run("record", "--parent", a.parent, "--measurement-file", str(path))
        reply["measurement_file"] = str(path)
        return reply

    digest = state.file_hash(a.op_file)
    if retry_measurement:
        if retry_measurement != digest:
            raise ValueError("source changed since the interrupted measurement")
        return _measure_and_record(a, st, root, request, record_receipt)
    retune = bool(st.get("active_tile_block"))
    if not retune:
        hit = st.get("eval_cache", {}).get(digest)
        if hit:
            reply = run("record", "--parent", a.parent, "--code-hash", digest,
                        "--s", str(hit["s"]), "--p", str(hit.get("p") or 0))
            reply.update(cached=True, code_hash=digest, s=hit["s"], p=hit.get("p"))
            return reply
        gate = run("feasible", "--parent", a.parent, "--op-file", a.op_file)
        if not gate["feasible"]:
            return gate
        retune = gate.get("retune") and st["config"].get("bayesian_optimization_tile_search")
    if retune:
        from pro_joint import run_block

        block = run_block(a, st, root, record_rejection=record_receipt)
        if "rejection" in block:
            return block["rejection"]
        if block.get("receipt"):
            return record_receipt(block["receipt"])
    if stop_condition(state.load(a.op_dir), a):
        return {"next": "stop"}
    return _measure_and_record(a, st, root, request, record_receipt)


def _read_dispatch(journal, key, pending):
    if not journal.exists():
        return {}
    saved = json.loads(journal.read_text())
    resumable = "reply" in saved or pending or saved.get("retry_measurement")
    if saved["request"] != key or not resumable:
        raise ValueError("dispatch identity changed or interrupted; reconcile before retry")
    return saved


def step(a, _before, _receipt_args, configure):
    from panko_core import panko_harness as commands

    def run(name, *flags):
        argv, receipt = _receipt_args([name, "--op-dir", a.op_dir, *flags])
        return call_transition(
            lambda unused: commands.main(argv, configure=configure,
                                         before_command=lambda args: _before(args, receipt, nested=True)),
            None,
        )

    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", a.dispatch_id):
        raise ValueError("invalid dispatch ID")
    root = Path(a.op_dir) / "optimization/steps" / a.dispatch_id
    journal = root / "dispatch.json"
    key = [a.parent, str(Path(a.op_file).resolve()), a.test_command]
    st = state.load(a.op_dir)
    pending = st.get("active_tile_block")
    saved = _read_dispatch(journal, key, pending)
    if "reply" in saved:
        return out(saved["reply"])
    if pending and str(root.resolve()) != pending["request"]["root"]:
        raise ValueError("resume the original tile block or STEP dispatch")
    if key[1] != str(Path(st["op_file"]).resolve()):
        raise ValueError("candidate differs from frozen file")
    if st["progress"].get("stop_reason"):
        return out({"next": "stop"})
    if not pending and stop_condition(st, a) and run("stop")["stop"]:
        return out({"next": "stop"})
    choices = Path(selection.selects_path(a.op_dir)).read_text().splitlines()
    if (
        not choices
        or json.loads(choices[-1])["chosen"] != a.parent
        or not any(x["id"] == a.parent for x in st["nodes"]["open"])
    ):
        raise ValueError("candidate has no matching selected action")
    check_command(a, st)
    if pro_backend.ACTIVE_PROGRAM:
        program_bundle.refresh_program(pro_backend.ACTIVE_PROGRAM, preserve_current=True)
        program_bundle.require_current(pro_backend.ACTIVE_PROGRAM)
    root.mkdir(parents=True, exist_ok=True)
    if not saved:
        atomic_write(journal, json.dumps({"request": key}))
    reply = _record_candidate(a, st, root, run, key, saved.get("retry_measurement"))
    reply["stop"] = run("stop")
    if reply["stop"]["stop"]:
        reply["next"] = "stop"
    elif reply.get("stop_refine"):
        reply.update(next="close", parent=a.parent)
    elif proposal := reply.get("bayesian_optimization"):
        reply.update(next="evaluate" if proposal.get("applied") else "code", parent=a.parent)
    else:
        reply["selection"] = run("select", "--dispatch-id", a.dispatch_id)
    atomic_write(journal, json.dumps({"request": key, "reply": reply}))
    return out(reply)
