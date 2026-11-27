# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Pro services supplied to the native PANKO command core."""
import ast
import hashlib
import statistics
from pathlib import Path

from bootstrap import KERNEL_PROFILER, core_path
from source_analysis.kernel_boundary import discover_owned_bindings, measurement_dump, shell_dump
from source_analysis.runtime_files import write_runtime_file
import pro_predicates as predicates
from evaluation import workspace_probe as feasibility
catalog = str(Path(__file__).resolve().parents[1] / "references" / "action_catalog.json")
defaults = {"prior_gain_source": "catalog", "symptom_boost": 1.0, "action_preconditions": True,
            "static_feasibility": True, "objectives": ["aicore_e2e", "workspace"], "prior_auto_update": False,
            "action_prior_table": ""}
default_reference = (1.0, "no_target:1us")
EXPORT_OUTCOMES = False


def bayesian():
    import bayesian_optimization
    return bayesian_optimization


def revision():
    digest = hashlib.sha256(KERNEL_PROFILER.read_bytes())
    for label, root in (("core", core_path()), ("pro", Path(__file__).resolve().parent)):
        for path in sorted(root.rglob("*.py")):
            relative = path.relative_to(root)
            if any(part in ("tests", "_core", "__pycache__") for part in relative.parts):
                continue
            digest.update((label + "/" + relative.as_posix()).encode() + b"\0" + path.read_bytes())
    return "runtime-sha256:" + digest.hexdigest()


def state_metadata(args):
    source = Path(args.op_file).read_text()
    return {"wrapper_owned_bindings": discover_owned_bindings(source, args.pro_context),
            "wrapper_signature": wrapper_signature(source, args.pro_context), "case_context": args.pro_context,
            "backend": "pypto-pro-base-v2", "measurement_contract": args.pro_measurement_contract,
            "program": ACTIVE_PROGRAM}


ACTIVE_PROGRAM = None


def write_source(path, source):
    from source_analysis import program_bundle
    if ACTIVE_PROGRAM:
        if Path(path).resolve() != Path(ACTIVE_PROGRAM['view']).resolve():
            raise ValueError('working view differs from the campaign')
        program_bundle.write_candidate(ACTIVE_PROGRAM, source)
    else:
        write_runtime_file(Path(path), source.encode("utf-8"))


def export_source(source, destination):
    from source_analysis import program_bundle
    from source_analysis.program_view import HEADER
    data = Path(source).read_bytes()
    if data.startswith(HEADER.encode("utf-8")):
        original = ACTIVE_PROGRAM or program_bundle.attached(source)
        root = Path(str(destination) + ".program")
        if root.is_symlink():
            raise ValueError("export directory cannot be a symlink")
        namespace = original.get("package") or program_bundle.package_namespace(Path(original["root"]))
        if namespace:
            parts = namespace.split(".")
            if not all(part.isidentifier() for part in parts):
                raise ValueError("invalid export package name")
            root = program_bundle.member(root, Path(*parts) / "__init__.py").parent
        target = dict(original, root=str(root), view=str(destination))
        program_bundle.write_candidate(target, data.decode("utf-8"), exporting=True)
    else:
        write_runtime_file(Path(destination), data)


def verification_policy():
    from pro_verify import policy
    return policy()


def wrapper_signature(src, context=None, owned=()):
    return _signature(src, shell_dump, context, owned)


def debug_options_signature(src):
    return _signature(src, measurement_dump)


def _signature(src, dump, *args):
    try:
        signature = dump(ast.parse(src), *args)
    except SyntaxError:
        return None
    return hashlib.sha256(signature.encode()).hexdigest()


def entry_reads(row):
    reads = row.get('reads') or []
    samples = row.get('samples') or []
    # Legacy Pro search receipts each contributed nine launches. The cache
    # retained these batches even when the archive kept only the first median.
    legacy = row.get("remeasured") or "samples_by_objective" in row
    complete_batches = len(samples) > 9 and len(samples) % 9 == 0
    if legacy and len(reads) <= 1 and complete_batches:
        return [statistics.median(samples[i:i + 9]) for i in range(0, len(samples), 9)]
    if reads:
        return list(reads)
    return [statistics.median(samples)] if samples else []


def reading_interval(xs, ys, alpha):
    from pro_verify import reading_interval as interval
    return interval(xs, ys, alpha=alpha)


def tile_constants(st, path, src):
    from bayesian_optimization.domain import resolve_extra_consts
    return {"extra_consts": resolve_extra_consts(src, path, st.get("case_context"))}


def verify_initialization(args, st):
    from bayesian_optimization import readiness
    import pro_init_repair as init_repair
    from panko.config import out, Refusal
    if st["config"].get("bayesian_optimization_tile_search"):
        report = readiness.check(args.op_file, Path(args.op_file).read_text(), st["chip_envelope"],
                                 args.pro_context, st.get("live_tiles"))
        if not report["ok"]:
            out({"initialized": False, "refused": True,
                 "error": "tile_initialization_incomplete", "tile_readiness": report})
            raise Refusal(4)
        if args.pro_repair:
            st["initialization_handoff"] = {
                "ok": True, "tile_readiness": report, "init_repair": args.pro_repair,
                "before_sha256": init_repair.digest(Path(args.normalization_before).read_bytes())}
            init_repair.accepted(args.op_dir, args.op_file, st)
