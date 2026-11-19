# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Verify evidence-file integrity and paired timing statistics."""

import argparse
import hashlib
import json
import logging
import math
from decimal import Decimal
from fractions import Fraction
from pathlib import Path, PureWindowsPath
import re
from statistics import median
import sys
from textwrap import dedent

from profile_kernel import verified_native_pairs

TIMING_METRICS = ("native_task_sum_us", "synchronized_wall_us")
CAPTURE_LABELS = ("before", "after", "AB", "BA")


def require(condition, message):
    if not condition:
        raise ValueError(message)



def read_object(contents, label):
    def unique_keys(items):
        result = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON field {key!r}")
            result[key] = value
        return result

    def invalid_number(value):
        raise ValueError(f"nonfinite JSON number {value}")

    try:
        obj = json.loads(
            contents, object_pairs_hook=unique_keys, parse_float=Decimal,
            parse_constant=invalid_number,
        )
    except ValueError as exc:
        raise ValueError(f"{label}: {exc}") from exc
    require(isinstance(obj, dict), f"{label}: expected a JSON object")
    return obj



def read_artifact(root, ref):
    require(isinstance(ref, dict), "each file reference requires path and sha256")
    name, digest = ref.get("path"), ref.get("sha256")
    require(isinstance(name, str) and name.strip(), "missing artifact path")
    rel = Path(name.replace("\\", "/"))
    require(not PureWindowsPath(name).anchor and not rel.is_absolute(),
            f"{name}: expected a path relative to the example root")
    path = (root / rel).resolve()
    require(path.is_relative_to(root), f"{name}: path escapes the example root")
    require(isinstance(digest, str) and re.fullmatch(r"[0-9A-Fa-f]{64}", digest),
            f"{name}: expected a SHA-256 hex digest")
    require(path.is_file(), f"{name}: expected a regular file")
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == digest.lower(), f"{name}: SHA-256 mismatch")
    return data


def card_excerpts(lines):
    excerpts = {}
    section = fence = snippet = None
    for line in lines:
        if fence is not None:
            if re.fullmatch(rf" {{0,3}}{fence[0]}{{{len(fence)},}}[ \t]*", line):
                fence = None
            elif snippet is not None:
                snippet.append(line)
            continue
        heading = re.fullmatch(r" {0,3}(#{1,2})[ \t]+(.+?)[ \t]*", line)
        if heading:
            name = heading[2]
            section = name if heading[1] == "##" and name in ("Before", "After") else None
            if section:
                require(section not in excerpts, f"duplicate ## {section}")
                excerpts[section] = []
            continue
        opening = re.fullmatch(r" {0,3}(`{3,}|~{3,})(.*)", line)
        if opening:
            fence = opening[1]
            snippet = None
            if section and opening[2].strip().lower() == "python":
                snippet = []
                excerpts[section].append(snippet)
    require(fence is None, "card contains an unclosed code fence")
    require(set(excerpts) == {"Before", "After"}, "card requires independent ## Before and ## After sections")
    result = {}
    for name in ("Before", "After"):
        blocks = excerpts.get(name, [])
        require(len(blocks) == 1 and any(line.strip() for line in blocks[0]),
                f"## {name} requires exactly one nonempty Python code block")
        result[name.lower()] = "\n" + "\n".join(blocks[0]) + "\n"
    return result



def duration(value, label):
    require(type(value) in (int, Decimal), f"{label}: expected a numeric duration")
    # The collectors record positive, finite, machine-representable microseconds.
    # Keep the JSON number exact for comparisons at the admission boundary.
    try:
        us = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label}: duration exceeds finite float range") from exc
    require(math.isfinite(us) and us > 0, f"{label}: duration must be positive and finite")
    return Fraction(value)



def timing_orders(raw, case_id):
    orders = raw.get("orders")
    if isinstance(orders, list):
        by_order = {}
        for item in orders:
            require(isinstance(item, dict) and item.get("order") in ("AB", "BA"),
                    f"{case_id}: each collector order must identify AB or BA")
            order = item["order"]
            require(order not in by_order, f"{case_id}: duplicate collector order {order}")
            require("raw_pairs_us" in item,
                    f"{case_id}/{order}: raw_pairs_us required; collect with include_pairs=True")
            by_order[order] = item["raw_pairs_us"]
        orders = by_order
    require(isinstance(orders, dict) and {"AB", "BA"}.issubset(orders),
            f"{case_id}: orders must contain AB and BA raw pair arrays, "
            "as a mapping or collector order entries with raw_pairs_us")
    return orders


def check_timing(raw, case_id, metric, root):
    orders = timing_orders(raw, case_id)
    verified = None
    if metric == "native_task_sum_us":
        captures = raw.get("native_captures")
        require(isinstance(captures, dict) and set(captures) == set(CAPTURE_LABELS),
                f"{case_id}: native timing requires before, after, AB and BA capture files")
        try:
            verified = verified_native_pairs({
                label: read_artifact(root, captures[label]) for label in CAPTURE_LABELS
            })
        except ValueError as exc:
            raise ValueError(f"{case_id}: {exc}") from exc
    results = []
    for order in ("AB", "BA"):
        pairs = orders[order]
        label = f"{case_id}/{order}"
        require(isinstance(pairs, list) and len(pairs) >= 10, f"{label}: need at least 10 pairs")
        if verified is not None:
            require(len(pairs) == len(verified[order]),
                    f"{label}: pair count differs from native capture")
        ratios = []
        for index, pair in enumerate(pairs, 1):
            pair_label = f"{label} pair {index}"
            require(isinstance(pair, list) and len(pair) == 2,
                    f"{pair_label}: expected [first_us, second_us]")
            first, second = (duration(value, pair_label) for value in pair)
            if verified is not None:
                # Allow float serialization and summation roundoff. Use the
                # decimal CSV sums for the actual performance decision.
                measured = verified[order][index - 1]
                require(all(math.isclose(reported, captured, rel_tol=1e-12)
                            for reported, captured in zip((first, second), measured)),
                        f"{pair_label}: raw sums differ from native capture")
                first, second = (duration(value, pair_label) for value in measured)
            ratios.append(second / first if order == "AB" else first / second)
        wins = sum(ratio < 1 for ratio in ratios)
        require(wins * 10 >= len(ratios) * 9, f"{label}: only {wins}/{len(ratios)} strict wins")
        middle = median(ratios)
        require(middle <= Fraction(97, 100), f"{label}: median after/before exceeds 0.97")
        results.append(f"{label} ({metric}): {wins}/{len(ratios)} strict wins, "
                       f"median after/before {float(middle):.6g}")
    return results



def check_binding(record, source_hashes, plan_hash, label):
    hashes = record.get("source_sha256")
    require(isinstance(hashes, dict) and set(hashes) == set(source_hashes)
            and all(isinstance(hashes[side], str) and hashes[side].lower() == value
                    for side, value in source_hashes.items()),
            f"{label}: source_sha256 must match both verified implementations")
    digest = record.get("plan_sha256")
    require(isinstance(digest, str) and digest.lower() == plan_hash,
            f"{label}: plan_sha256 must match the verified validation plan")


def plan_coverage(plan):
    metrics = plan.get("metrics")
    require(isinstance(metrics, list) and metrics
            and all(metric in TIMING_METRICS for metric in metrics)
            and len(set(metrics)) == len(metrics),
            "validation plan requires distinct supported metrics")
    cases = plan.get("cases")
    require(isinstance(cases, dict) and cases, "validation plan requires cases")
    required, benefit = set(), set()
    for case_id, case in cases.items():
        require(case_id and case_id == case_id.strip() and isinstance(case, dict)
                and case.get("role") in ("benefit", "correctness", "exclusion"),
                "each planned case requires a nonempty id and a supported role")
        if case["role"] != "exclusion":
            required.add(case_id)
        if case["role"] == "benefit":
            benefit.add(case_id)
    require(len(benefit) >= 2, "validation plan requires at least two benefit cases")
    return required, {(case_id, metric) for case_id in benefit for metric in metrics}


def check_correctness(report, oracle_sha256, required_ids):
    digest = report.get("oracle_sha256")
    require(isinstance(digest, str) and digest.lower() == oracle_sha256,
            "correctness report must identify the verified oracle SHA-256")
    cases = report.get("cases")
    require(isinstance(cases, dict) and cases, "correctness report requires case results")
    missing, unexpected = required_ids - cases.keys(), cases.keys() - required_ids
    require(not missing and not unexpected,
            f"correctness case coverage differs from plan: missing {sorted(missing)}, "
            f"unexpected {sorted(unexpected)}")
    for case_id, result in cases.items():
        require(case_id.strip() and isinstance(result, dict)
                and result.get("before") == result.get("after") == "passed",
                f"{case_id}: both correctness results must be passed")



def locate_excerpt(snippet, source):
    """Locate an indentation-normalized excerpt for source-level review."""
    wanted = dedent(snippet).strip("\n").splitlines()
    lines = source.splitlines()
    for start in range(len(lines) - len(wanted) + 1):
        candidate = lines[start:start + len(wanted)]
        if wanted == dedent("\n".join(candidate)).splitlines():
            return start + 1
    return None


def excerpt_messages(card, sources):
    # Markdown and excerpt matching are aids to review, never an acceptance verdict.
    try:
        snippets = card_excerpts(card.read_text(encoding="utf-8-sig").splitlines())
    except ValueError as exc:
        yield f"Review required: {exc}"
        return
    for side, snippet in snippets.items():
        try:
            source = sources[side].decode("utf-8-sig")
        except UnicodeError:
            yield f"Review required: {side} source is not UTF-8 text"
            continue
        line = locate_excerpt(snippet, source)
        if line:
            yield f"{side}: indentation-normalized text located at source line {line}"
        else:
            yield f"Review required: {side} excerpt not automatically located"


def check_submission(card, root, manifest):
    root = root.resolve()
    require(root.is_dir(), f"{root}: expected an example directory")
    evidence = read_object(manifest.read_bytes(), manifest)
    require(type(evidence.get("schema_version")) is int and evidence["schema_version"] == 2,
            "expected schema_version 2")
    refs = evidence.get("sources")
    require(isinstance(refs, dict), "missing sources")
    sources = {side: read_artifact(root, refs.get(side)) for side in ("before", "after")}
    source_hashes = {side: hashlib.sha256(data).hexdigest() for side, data in sources.items()}
    oracle = read_artifact(root, evidence.get("oracle"))
    plan_bytes = read_artifact(root, evidence.get("plan"))
    plan_hash = hashlib.sha256(plan_bytes).hexdigest()
    plan = read_object(plan_bytes, "validation plan")
    required_ids, expected = plan_coverage(plan)
    dependencies = plan.get("dependencies", [])
    require(isinstance(dependencies, list), "plan dependencies must be an array of file references")
    for ref in dependencies:
        read_artifact(root, ref)
    report = read_object(read_artifact(root, evidence.get("correctness")), "correctness")
    check_binding(report, source_hashes, plan_hash, "correctness")
    check_correctness(report, hashlib.sha256(oracle).hexdigest(), required_ids)
    artifacts = evidence.get("artifacts", [])
    require(isinstance(artifacts, list), "artifacts must be an array of file references")
    for ref in artifacts:
        read_artifact(root, ref)
    timings = evidence.get("timings")
    require(isinstance(timings, list) and timings, "missing timing files")
    seen, results = set(), []
    for timing in timings:
        raw = read_object(read_artifact(root, timing), "timing")
        case_id, metric = raw.get("case_id"), raw.get("metric")
        require(isinstance(case_id, str) and metric in TIMING_METRICS,
                "timing requires case_id and a supported metric")
        key = (case_id, metric)
        require(key in expected, f"{case_id}/{metric}: timing is outside the validation plan")
        require(key not in seen, f"{case_id}/{metric}: duplicate timing record")
        seen.add(key)
        check_binding(raw, source_hashes, plan_hash, f"{case_id}/{metric}")
        results.extend(check_timing(raw, case_id, metric, root))
    require(seen == expected, f"missing planned timings: {sorted(expected - seen)}")
    results.extend(excerpt_messages(card, sources))
    return results


def _cli_logger(stream):
    logger = logging.Logger(f"{__name__}.cli", level=logging.INFO)
    logger.addHandler(logging.StreamHandler(stream))
    return logger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("card", type=Path)
    parser.add_argument("example_root", type=Path)
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args()
    try:
        results = check_submission(args.card, args.example_root, args.evidence)
    except (OSError, ValueError) as exc:
        _cli_logger(sys.stderr).error("Evidence check failed: %s", exc)
        return 1
    logger = _cli_logger(sys.stdout)
    for result in results:
        logger.info("%s", result)
    logger.info("File hashes, record bindings, declared case/metric coverage and paired timing checks passed. "
                "Independent card assessment required: verify plan adequacy, "
                "source execution, hardware provenance, mechanism and language meaning.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
