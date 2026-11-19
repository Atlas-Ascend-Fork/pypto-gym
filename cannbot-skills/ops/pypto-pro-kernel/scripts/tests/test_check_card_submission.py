# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Regression tests for objective evidence checks and non-blocking excerpt hints."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "check_card_submission.py"
with pytest.MonkeyPatch.context() as imports:
    imports.syspath_prepend(str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("card_checker", SCRIPT)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)


def store(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    data = value if isinstance(value, bytes) else json.dumps(value).encode()
    path.write_bytes(data)
    return {"path": name, "sha256": hashlib.sha256(data).hexdigest()}


@pytest.fixture
def bundle(tmp_path):
    card = tmp_path / "card.md"
    card.write_text("## After\n```python\nreturn compute(8)\n```\n"
                    "## Before\n```python\nreturn compute(4)\n```\n", encoding="utf8")
    sources = {side: store(tmp_path, f"{folder}/operator.py", f"def kernel():\r\n    return compute({n})\r\n".encode())
               for side, folder, n in (("before", "baseline", 4), ("after", "optimized", 8))}
    oracle = store(tmp_path, "reference.py", b"def reference(x): return x + x\n")
    roles = {"main": "benefit", "large": "benefit", "tail": "correctness", "empty": "exclusion"}
    plan = store(tmp_path, "plan.json", {
        "metrics": ["synchronized_wall_us"],
        "cases": {case_id: {"role": role} for case_id, role in roles.items()},
    })
    binding = {"source_sha256": {side: ref["sha256"] for side, ref in sources.items()},
               "plan_sha256": plan["sha256"]}
    raw = {"metric": "synchronized_wall_us", **binding,
           "orders": {"AB": [[100, 97]] * 9 + [[100, 100]],
                      "BA": [[97, 100]] * 9 + [[100, 100]]}}
    manifest = {"schema_version": 2, "sources": sources, "oracle": oracle, "plan": plan,
                "correctness": store(tmp_path, "correctness.json", {
                    "oracle_sha256": oracle["sha256"],
                    "cases": {case_id: {"before": "passed", "after": "passed"}
                              for case_id in ("main", "large", "tail")}, **binding}),
                "timings": [store(tmp_path, "timing.json", {"case_id": "main", **raw}),
                            store(tmp_path, "large.json", {"case_id": "large", **raw})],
                "artifacts": [store(tmp_path, "run.log", b"raw output\n")]}
    return card, manifest


def check(root, bundle):
    card, manifest = bundle
    store(root, "manifest.json", manifest)
    return checker.check_submission(card, root, root / "manifest.json")


def test_valid_arbitrary_paths_display_indentation_and_repeated_reference(tmp_path, bundle):
    bundle[1]["artifacts"].append(bundle[1]["sources"]["before"])
    raw = json.loads((tmp_path / "timing.json").read_bytes())
    raw["orders"]["warmup"] = [[100, 97]]
    bundle[1]["timings"][0] = store(tmp_path, "timing.json", raw)
    result = check(tmp_path, bundle)
    assert any("before: indentation-normalized text located" in row for row in result)
    assert any("after: indentation-normalized text located" in row for row in result)


@pytest.mark.parametrize("ref", ["source", "attachment"])
def test_tampered_bytes_are_rejected(tmp_path, bundle, ref):
    item = bundle[1]["sources"]["after"] if ref == "source" else bundle[1]["artifacts"][0]
    (tmp_path / item["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        check(tmp_path, bundle)


def test_artifact_cannot_escape_the_evidence_root(tmp_path, bundle):
    bundle[1]["sources"]["before"]["path"] = "../outside.py"
    with pytest.raises(ValueError, match="escapes"):
        check(tmp_path, bundle)


@pytest.mark.parametrize("edit", ["missing", "failed", "oracle"])
def test_incomplete_or_inconsistent_correctness_is_rejected(tmp_path, bundle, edit):
    manifest = bundle[1]
    report = json.loads((tmp_path / "correctness.json").read_bytes())
    if edit == "missing":
        del report["cases"]["tail"]
    elif edit == "failed":
        report["cases"]["main"]["after"] = "failed"
    else:
        report["oracle_sha256"] = "0" * 64
    manifest["correctness"] = store(tmp_path, "correctness.json", report)
    with pytest.raises(ValueError, match="missing.*tail|both correctness|oracle"):
        check(tmp_path, bundle)


@pytest.mark.parametrize("order,pairs,error", [
    ("BA", [[100, 97]] * 10, "strict wins"),
    ("AB", [[100, 97]] * 9, "10 pairs"),
    ("AB", [[100, 90]] * 8 + [[100, 100]] * 2, "strict wins"),
    ("AB", [[1, .99]] * 4 + [[100, 97]] * 2 + [[10000, 9900]] * 4, "median"),
    ("AB", [[100, 0]] * 10, "positive and finite"),
    ("AB", [[100, float("nan")]] * 10, "nonfinite"),
    ("AB", [[100, True]] * 10, "numeric duration"),
])
def test_timing_order_arithmetic_and_invalid_values(tmp_path, bundle, order, pairs, error):
    raw = json.loads((tmp_path / "timing.json").read_bytes())
    raw["orders"][order] = pairs
    bundle[1]["timings"][0] = store(tmp_path, "timing.json", raw)
    with pytest.raises(ValueError, match=error):
        check(tmp_path, bundle)


def test_wall_collector_output_is_accepted_after_explicit_binding(tmp_path, bundle, monkeypatch):
    from types import SimpleNamespace

    profile_spec = importlib.util.spec_from_file_location(
        "profile_for_card_test", SCRIPT.with_name("profile_kernel.py")
    )
    profile = importlib.util.module_from_spec(profile_spec)
    profile_spec.loader.exec_module(profile)
    torch = SimpleNamespace(__version__="test", npu=SimpleNamespace(synchronize=lambda: None))
    torch_npu = SimpleNamespace(__version__="test")
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu)
    clock = [0]
    monkeypatch.setattr(profile.time, "perf_counter_ns", lambda: clock[0])

    def advance(ns):
        clock[0] += ns

    raw = profile.compare_wall(lambda: advance(100000), lambda: advance(90000),
                               warmup=1, repeats=10, include_pairs=True)
    original = json.loads((tmp_path / "timing.json").read_bytes())
    raw.update({key: original[key] for key in ("case_id", "source_sha256", "plan_sha256")})
    bundle[1]["timings"][0] = store(tmp_path, "timing.json", raw)
    assert sum("strict wins" in row for row in check(tmp_path, bundle)) == 4


def test_current_manifest_cannot_attach_old_results_to_changed_source(tmp_path, bundle):
    manifest = bundle[1]
    manifest["sources"]["after"] = store(
        tmp_path, "optimized/operator.py",
        b"def compute(n): return n + 1\n\ndef kernel():\n    return compute(8)\n")
    with pytest.raises(ValueError, match="correctness: source_sha256"):
        check(tmp_path, bundle)
    report = json.loads((tmp_path / "correctness.json").read_bytes())
    report["source_sha256"]["after"] = manifest["sources"]["after"]["sha256"]
    manifest["correctness"] = store(tmp_path, "correctness.json", report)
    with pytest.raises(ValueError, match="main/synchronized_wall_us: source_sha256"):
        check(tmp_path, bundle)


def test_changed_case_definition_cannot_reuse_old_results(tmp_path, bundle):
    plan = json.loads((tmp_path / "plan.json").read_bytes())
    plan["cases"]["main"]["seed"] = 7
    bundle[1]["plan"] = store(tmp_path, "plan.json", plan)
    with pytest.raises(ValueError, match="correctness: plan_sha256"):
        check(tmp_path, bundle)


def test_missing_planned_benefit_timing_fails(tmp_path, bundle):
    bundle[1]["timings"].pop()
    with pytest.raises(ValueError, match="missing planned timings.*large"):
        check(tmp_path, bundle)


def test_single_benefit_case_fails(tmp_path, bundle):
    plan = json.loads((tmp_path / "plan.json").read_bytes())
    del plan["cases"]["large"]
    bundle[1]["plan"] = store(tmp_path, "plan.json", plan)
    with pytest.raises(ValueError, match="at least two benefit cases"):
        check(tmp_path, bundle)


def test_native_capture_matches_raw_sums_and_each_case_needs_every_metric(tmp_path, bundle):
    manifest = bundle[1]
    plan = json.loads((tmp_path / "plan.json").read_bytes())
    plan["metrics"].append("native_task_sum_us")
    manifest["plan"] = store(tmp_path, "plan.json", plan)
    digest = manifest["plan"]["sha256"]
    for index, file in enumerate(("correctness.json", "timing.json", "large.json")):
        record = json.loads((tmp_path / file).read_bytes())
        record["plan_sha256"] = digest
        ref = store(tmp_path, file, record)
        if index == 0:
            manifest["correctness"] = ref
        else:
            manifest["timings"][index - 1] = ref

    def native_capture(case_id, before_us, after_us):
        before = [("old", before_us)]
        after = [("new1", after_us // 2), ("new2", after_us - after_us // 2)]
        refs = {}
        for label, tasks in {"before": before, "after": after,
                             "AB": (before + after) * 10, "BA": (after + before) * 10}.items():
            csv = "Start Time(us),Duration(us),Name\n" + "".join(
                f"{i},{us},{name}\n" for i, (name, us) in enumerate(tasks))
            refs[label] = store(tmp_path, f"{case_id}-{label}.csv", csv.encode("utf8"))
        raw = json.loads((tmp_path / "timing.json").read_bytes())
        return dict(raw, case_id=case_id, metric="native_task_sum_us", native_captures=refs,
                    orders={"AB": [[before_us, after_us]] * 10, "BA": [[after_us, before_us]] * 10})

    raw = native_capture("main", 10, 8)
    manifest["timings"].append(store(tmp_path, "native-main.json", raw))
    # Case main supplies both metrics but case large has wall time only.
    with pytest.raises(ValueError, match="missing planned timings.*large.*native_task_sum_us"):
        check(tmp_path, bundle)
    manifest["timings"].append(store(tmp_path, "native-large.json", native_capture("large", 20, 16)))
    assert sum("strict wins" in row for row in check(tmp_path, bundle)) == 8
    raw["orders"] = {"AB": [[18, 4]] * 10, "BA": [[4, 18]] * 10}
    manifest["timings"][-2] = store(tmp_path, "native-main.json", raw)
    with pytest.raises(ValueError, match="raw sums differ from native capture"):
        check(tmp_path, bundle)


@pytest.mark.parametrize("problem,error", [
    ("duplicate", "duplicate collector order"),
    ("missing_raw", "include_pairs=True"),
    ("missing_BA", "AB and BA"),
    ("missing_case_id", "requires case_id"),
])
def test_incomplete_collector_output_is_not_guessed(tmp_path, bundle, problem, error):
    raw = json.loads((tmp_path / "timing.json").read_bytes())
    raw["orders"] = [{"order": order, "raw_pairs_us": pairs} for order, pairs in raw["orders"].items()]
    if problem == "duplicate":
        raw["orders"].append(raw["orders"][0])
    elif problem == "missing_raw":
        raw["orders"][0].pop("raw_pairs_us")
    elif problem == "missing_BA":
        raw["orders"].pop()
    else:
        raw.pop("case_id")
    bundle[1]["timings"][0] = store(tmp_path, "timing.json", raw)
    with pytest.raises(ValueError, match=error):
        check(tmp_path, bundle)


def test_unlocated_fragment_is_reported_for_review_not_rejected(tmp_path, bundle):
    bundle[0].write_text("## Before\n```python\ny = generic(x)\n```\n"
                         "## After\n```python\ny = specialized(x)\n```\n", encoding="utf8")
    assert sum(row.startswith("Review required:") for row in check(tmp_path, bundle)) == 2


def test_cli_does_not_guess_launcher_syntax_or_claim_card_acceptance(tmp_path, bundle):
    launcher = store(tmp_path, "driver.py", b"print('ok')\n")
    ran = subprocess.run([sys.executable, "-X", "utf8", str(tmp_path / "driver.py")],
                         capture_output=True, text=True, check=True)
    assert ran.stdout.strip() == "ok"
    bundle[1]["artifacts"].extend([launcher, store(tmp_path, "commands.json", {
        "argv": ["python", "-X", "utf8", "driver.py"]})])
    check(tmp_path, bundle)
    ran = subprocess.run([sys.executable, str(SCRIPT), str(bundle[0]), str(tmp_path),
                          str(tmp_path / "manifest.json")], capture_output=True, text=True, check=True)
    assert "Independent card assessment required" in ran.stdout
    assert not ran.stderr
    failed = subprocess.run([sys.executable, str(SCRIPT), str(bundle[0]), str(tmp_path),
                             str(tmp_path / "missing.json")], capture_output=True, text=True)
    assert failed.returncode == 1
    assert not failed.stdout
    assert failed.stderr.startswith("Evidence check failed: ")
