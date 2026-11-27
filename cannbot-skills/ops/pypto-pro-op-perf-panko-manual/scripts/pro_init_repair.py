# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
Persist agent fallback after an ordinary INIT/normalization refusal.

This module neither repairs source nor relaxes an acceptance gate. It keeps the
original input across coder dispatches and produces a CODE directed back to INIT,
before a search tree or an optimization action exists.
"""
import hashlib
import json
import os
from pathlib import Path


def digest(value):
    if not isinstance(value, bytes):
        value = json.dumps(value, sort_keys=True, separators=(",", ":"),
                           allow_nan=False).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def state_path(op_dir):
    return Path(op_dir).resolve() / "optimization" / "initialization_state.json"


def _save(op_dir, state):
    path = state_path(op_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load(op_dir, op_file=None):
    path = state_path(op_dir)
    if not path.exists():
        return None
    state = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(state, dict) or state.get("schema_version") != 1:
        raise ValueError("Unsupported initialization repair state.")
    if op_file and state["op_file"] != str(Path(op_file).resolve()):
        raise ValueError("Initialization repair belongs to a different op_file.")
    if digest(state["request"]) != state["request_sha256"]:
        raise ValueError("The original INIT dispatch changed.")
    if digest(Path(state["original_source"]).read_bytes()) != state["original_sha256"]:
        raise ValueError("The original initialization snapshot is missing or changed.")
    # The search-state rename commits INIT. An acceptance receipt written before
    # a failed/interrupted rename is still pending, so retry every ordinary gate.
    if state["status"] == "ready" and not (path.parent / "search_state.json").exists():
        state["status"] = "awaiting_candidate"
        state.pop("accepted_sha256", None)
        if state["attempts"]:
            state["attempts"][-1]["status"] = "awaiting_candidate"
    return state


def receipt(op_dir, state):
    return {"state_file": str(state_path(op_dir)),
            "original_source": state["original_source"],
            "original_sha256": state["original_sha256"],
            "attempt": len(state["attempts"]),
            "request_sha256": state["request_sha256"]}


def _before_search(op_dir):
    if (state_path(op_dir).parent / "search_state.json").exists():
        raise ValueError("Initialization repair cannot run after search_state.json exists.")


def _diagnosed(diagnostic):
    if not isinstance(diagnostic, dict):
        return False
    if diagnostic.get("error") in {"tile_initialization_incomplete", "invalid_case_context",
                                   "normalization_handoff_failed"}:
        return diagnostic.get("initialized") is False and diagnostic.get("refused") is True
    return (diagnostic.get("ok") is False and bool(diagnostic.get("reasons"))
            and isinstance(diagnostic.get("before"), dict)
            and isinstance(diagnostic.get("after"), dict))


def _validate_init_request(request, op_file, op_dir):
    if not isinstance(request, dict) or any(not request.get(k) for k in ('op', 'op_file', 'DEVICE', 'test_command')):
        raise ValueError("The original INIT dispatch needs op, op_file, DEVICE and test_command.")
    if request.get("candidate_file") or request.get("action_id"):
        raise ValueError("A STEP dispatch cannot be used as an original INIT request.")
    if not isinstance(request.get("config") or {}, dict):
        raise ValueError("The original INIT configuration must be a JSON object.")
    target = request.get("optimization_target")
    if target is not None:
        if not isinstance(target, dict):
            raise ValueError("The original optimization_target must be a JSON object.")
        cases = target.get("case_ids")
        if cases is not None and (not isinstance(cases, list)
                                  or not all(isinstance(case, str) and case for case in cases)):
            raise ValueError("The original target case_ids must be a list of nonempty strings.")
    if str(Path(request["op_file"]).resolve()) != op_file:
        raise ValueError("The original INIT dispatch names a different op_file.")
    if request.get("op_dir") and Path(request["op_dir"]).resolve() != Path(op_dir).resolve():
        raise ValueError("The original INIT dispatch names a different op_dir.")


def _new_repair_state(op_file, request, diagnostic, original, max_attempts):
    diagnosed_before = (diagnostic.get("normalization") or {}).get("before_sha256")
    original_path = original or request.get("normalization_before")
    if (not original_path and not diagnosed_before and _normalization_refused(diagnostic)):
        raise ValueError("A normalization refusal without source hashes needs an explicit --original.")
    origin = Path(original_path or op_file).resolve()
    source = origin.read_bytes()
    source.decode("utf-8")
    original_hash = digest(source)
    if diagnosed_before and original_hash != diagnosed_before:
        raise ValueError("The refusal identifies an earlier original; supply it with --original.")
    # Keep relative imports beside the original source, even when op_dir is
    # elsewhere. The checkpoint is data; it is never imported or executed.
    snapshot = origin.with_name(".%s.panko-init-%s.original" % (origin.name, original_hash))
    if snapshot.exists():
        if snapshot.read_bytes() != source:
            raise ValueError("Existing original initialization snapshot differs.")
    else:
        with snapshot.open("xb") as handle:
            handle.write(source)
    return {"schema_version": 1, "status": "needs_repair", "op_file": op_file,
            "original_source": str(snapshot), "original_sha256": original_hash,
            "request": request, "request_sha256": digest(request),
            "max_attempts": max_attempts, "attempts": []}


def _validate_retry_request(args, config, request):
    if str(getattr(args, "device", "")) != str(request["DEVICE"]):
        raise ValueError("The retry device differs from the original INIT dispatch.")
    if getattr(args, "op", None) != request["op"]:
        raise ValueError("The retry operator differs from the original INIT dispatch.")
    arch = request.get("PYPTOPRO_JIT_ARCH")
    if arch is not None and str(arch) != os.environ.get("PYPTOPRO_JIT_ARCH", ""):
        raise ValueError("The retry architecture differs from the original INIT dispatch.")
    if request.get("op_name") is not None and getattr(args, "op_name", None) != request["op_name"]:
        raise ValueError("The retry profile name differs from the original INIT dispatch.")
    key = getattr(args, "config_key", None)
    original_key = request.get("config_key")
    key = "" if key in (None, "") else str(key)
    original_key = "" if original_key in (None, "") else str(original_key)
    if key != original_key:
        raise ValueError("The retry config_key differs from the original INIT dispatch.")
    if getattr(args, "normalization_tolerance", 0.03) != request.get("normalization_tolerance", 0.03):
        raise ValueError("The retry changed the normalization equivalence margin.")
    for key, value in (request.get("config") or {}).items():
        if config.get(key) != value:
            raise ValueError("The retry changed the original INIT configuration: %s" % key)


def _validate_retry_cases(evidence, request):
    cases = (request.get("optimization_target") or {}).get("case_ids")
    if cases:
        for side in ("before", "after"):
            row = evidence.get(side, {})
            if not isinstance(row, dict) or not isinstance(row.get("correctness", {}), dict):
                raise ValueError("Normalization source and correctness entries must be objects.")
            covered = row.get("correctness", {}).get("case_ids", [])
            if not isinstance(covered, list) or not all(isinstance(case, str) for case in covered):
                raise ValueError("Normalization correctness case_ids must be a list of strings.")
            if not set(map(str, cases)) <= set(covered):
                raise ValueError("Normalization evidence omits a frozen target case.")


def propose(op_dir, op_file, request, diagnostic, intent, original=None,
            context=None, max_attempts=3):
    _before_search(op_dir)
    op_file = str(Path(op_file).resolve())
    if not _diagnosed(diagnostic):
        raise ValueError("Run ordinary INIT/normalize first; this is not a repairable declaration refusal.")
    if not isinstance(intent, str) or not intent.strip():
        raise ValueError("A concrete repair intent is required.")
    _validate_init_request(request, op_file, op_dir)
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer.")
    state = load(op_dir, op_file)
    normalization = diagnostic.get("normalization") or {}
    if not isinstance(normalization, dict):
        raise ValueError("The normalization refusal must contain a JSON object.")
    diagnosed_before = normalization.get("before_sha256")
    diagnosed_after = normalization.get("after_sha256")
    if diagnosed_after and digest(Path(op_file).read_bytes()) != diagnosed_after:
        raise ValueError("The refusal describes different candidate bytes; rerun ordinary INIT.")
    if state is None:
        state = _new_repair_state(op_file, request, diagnostic, original, max_attempts)
    else:
        _validate_existing_repair(state, diagnosed_before, request, original)
    if len(state["attempts"]) >= state["max_attempts"]:
        raise ValueError("Initialization repair attempts exhausted; "
                         "report the unresolved dependency and required input.")
    attempt_key = digest({"intent": intent, "context": context or {},
                          "source": digest(Path(op_file).read_bytes())})
    if any(row["proposal_sha256"] == attempt_key for row in state["attempts"]):
        raise ValueError("This repair was already attempted; provide new evidence or a different concrete repair.")
    state["attempts"].append({"status": "awaiting_candidate", "intent": intent,
                              "proposal_sha256": attempt_key, "diagnostic": diagnostic,
                              "context_sha256": digest(context or {}),
                              "input_sha256": digest(Path(op_file).read_bytes())})
    state["status"] = "awaiting_candidate"
    _save(op_dir, state)
    return {"directive": "CODE", "phase": 'init_repair', "op_file": op_file,
            "intent": intent, 'init_repair': receipt(op_dir, state)}


def _validate_existing_repair(state, diagnosed_before, request, original):
    if diagnosed_before and state["original_sha256"] != diagnosed_before:
        raise ValueError("The refusal compares a different original initialization source.")
    if digest(request) != state["request_sha256"]:
        raise ValueError("Keep the original INIT dispatch unchanged during repair.")
    if original and digest(Path(original).read_bytes()) != state["original_sha256"]:
        raise ValueError("A repaired candidate cannot replace the original initialization source.")
    if state["status"] != "needs_repair":
        raise ValueError("A repair is pending or already accepted; inspect status before proposing again.")


def reject(op_dir, op_file, reason):
    _before_search(op_dir)
    state = load(op_dir, op_file)
    if not state or state["status"] != "awaiting_candidate":
        raise ValueError("No initialization repair candidate is pending.")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Record the failed check or missing evidence before restoring the original.")
    source = Path(op_file).read_bytes()
    candidate_hash = digest(source)
    target = state_path(op_dir).parent / "initialization_candidates" / (candidate_hash + ".py")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source)
    state["attempts"][-1].update(status="rejected", reason=reason,
                                 candidate_sha256=candidate_hash, candidate_path=str(target))
    Path(op_file).write_bytes(Path(state["original_source"]).read_bytes())
    state["status"] = "needs_repair"
    _save(op_dir, state)
    return {"restored": True, "phase": 'init_repair', 'init_repair': receipt(op_dir, state)}


def pending(op_dir, op_file):
    state = load(op_dir, op_file)
    if state and state["status"] != "ready":
        _before_search(op_dir)
        return state
    return None


def bind_retry(op_dir, op_file, args, config):
    """Keep all ordinary INIT gates; only supply the immutable comparison input."""
    state = pending(op_dir, op_file)
    if state is None:
        return None
    if state["status"] != "awaiting_candidate":
        raise ValueError("Propose a new initialization repair before retrying a rejected candidate.")
    if not config.get("bayesian_optimization_tile_search", False):
        raise ValueError("Initialization repair cannot bypass readiness by disabling tile search.")
    request = state["request"]
    _validate_retry_request(args, config, request)
    before = getattr(args, "normalization_before", "")
    if before and digest(Path(before).read_bytes()) != state["original_sha256"]:
        raise ValueError("--normalization-before must identify the immutable original, not a repair candidate.")
    args.normalization_before = state["original_source"]
    changed = digest(Path(op_file).read_bytes()) != state["original_sha256"]
    if changed:
        if not getattr(args, "preopt_hash", ""):
            raise ValueError("A changed initialization repair needs --preopt-hash and fresh candidate measurements.")
        evidence_path = getattr(args, "normalization_evidence", "")
        if evidence_path:
            evidence = json.loads(Path(evidence_path).read_text(encoding="utf-8"))
            if not isinstance(evidence, dict):
                raise ValueError("Normalization evidence must be an object.")
            if (str(evidence.get("device", "")) != str(request["DEVICE"])
                    or evidence.get("test_command") != request["test_command"]):
                raise ValueError("Normalization evidence uses a different device or test command.")
            _validate_retry_cases(evidence, request)
    return receipt(op_dir, state)


def check_context(op_dir, op_file, context):
    """Keep case discovery inside the workload named by the original request."""
    state = load(op_dir, op_file)
    cases = (state["request"].get("optimization_target") or {}).get("case_ids")
    if cases and str(context.get("__panko_case_id__", "")) not in set(map(str, cases)):
        raise ValueError("The retry case context must identify a frozen target case.")


def accepted(op_dir, op_file, search):
    """Validate and persist repair acceptance before publishing the search state."""
    state = load(op_dir, op_file)
    if state is None or state["status"] == "ready":
        return
    if state["status"] != "awaiting_candidate":
        raise ValueError("No proposed initialization repair is awaiting acceptance.")
    handoff = search.get("initialization_handoff") or {}
    if not handoff.get("ok") or not (handoff.get("tile_readiness") or {}).get("ok"):
        raise ValueError("Only ordinary INIT acceptance can finish initialization repair.")
    if handoff.get("before_sha256") != state["original_sha256"]:
        raise ValueError("The accepted INIT did not compare the immutable original.")
    if (handoff.get('init_repair') or {}).get("request_sha256") != state["request_sha256"]:
        raise ValueError("The accepted INIT belongs to a different repair request.")
    current_hash = digest(Path(op_file).read_bytes())
    baseline = next((node for node in search.get("nodes", {}).get("closed", [])
                     if node.get("id") == "root"), {})
    if baseline.get("code_hash") != current_hash:
        raise ValueError("The accepted search baseline differs from the current repair candidate.")
    state.update(status="ready", accepted_sha256=current_hash)
    if state["attempts"]:
        state["attempts"][-1].update(status="accepted", candidate_sha256=state["accepted_sha256"])
    _save(op_dir, state)


def _normalization_refused(diagnostic):
    return diagnostic.get("error") == "normalization_handoff_failed" or diagnostic.get("ok") is False
