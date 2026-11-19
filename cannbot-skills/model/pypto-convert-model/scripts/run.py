# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Main runner: model -> all target formats -> tested.

Usage:
    python run.py [model_id ...]   # default: all models in registry

Paths:
    Frozen test evidence (committed to repo):
        ROOT/references/matrix.md
        ROOT/references/results/<id>.json
    Runtime artifacts (gitignored; default ~/.cache/pypto-convert-model,
    override with $CONVERT_MODEL_WORKDIR):
        WORKDIR/models/   HF/timm download cache
        WORKDIR/outputs/  converted artifacts
        WORKDIR/logs/     master + per-conversion logs
"""
import argparse
import gc
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import compare as cmp  # noqa: E402
from converters import CONVERTERS  # noqa: E402
from device_util import onnxruntime_providers, pick_device  # noqa: E402
from loaders import load  # noqa: E402
from log_util import conversion_logger, master_logger, timed  # noqa: E402
from registry import REGISTRY, TARGET_FORMATS  # noqa: E402

WORKDIR = Path(os.environ.get(
    "CONVERT_MODEL_WORKDIR",
    str(Path.home() / ".cache" / "pypto-convert-model"),
))
CACHE = WORKDIR / "models"
OUT = WORKDIR / "outputs"
RESULTS = ROOT / "references" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
MATRIX = ROOT / "references" / "matrix.md"


@dataclass(frozen=True)
class ConversionContext:
    """Model state shared by each target-format conversion."""

    model_id: str
    model: object
    sample: object
    reference: object
    metadata: dict
    output_root: Path


def run_target(target: Path, fmt: str, sample, model_template, meta=None):
    """Run one converted target; ``meta`` is retained for compatibility."""
    if fmt == "onnx":
        return cmp.run_onnx(target, sample)
    if fmt == "pt":
        return cmp.run_pt(target, sample)
    if fmt == "safetensors":
        return cmp.run_safetensors(target, model_template, sample)
    raise ValueError(fmt)


def _artifact_size_mb(path: Path) -> float:
    if path.is_file():
        return path.stat().st_size / 1e6
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) / 1e6


def _artifact_display_path(path: Path) -> str:
    try:
        return str(path.relative_to(WORKDIR))
    except ValueError:
        return str(path)


def _test_converted_artifact(context: ConversionContext, fmt: str,
                             path: Path, clog) -> dict:
    try:
        with timed(clog, f"test {context.model_id} <- {fmt}"):
            if fmt == "safetensors":
                template, _, _ = load(context.model_id, CACHE)
                out = run_target(path, fmt, context.sample, template)
                del template
            else:
                out = run_target(path, fmt, context.sample, None)
            result_diff = cmp.diff(context.reference, out)
        return {
            "diff": result_diff,
            "status": "PASS" if result_diff["allclose"] else "DIFF",
        }
    except NotImplementedError as error:
        clog.warning(f"test skipped: {error}")
        return {"status": "TEST_SKIP", "error": str(error)}
    except Exception as error:
        clog.exception(f"test failed: {error}")
        return {"status": "TEST_FAIL", "error": repr(error)}


def _convert_and_test_single_format(context: ConversionContext, fmt: str) -> dict:
    clog = conversion_logger(context.model_id, fmt)
    out_dir = context.output_root / fmt
    out_dir.mkdir(exist_ok=True)
    t0 = time.time()
    try:
        with timed(clog, f"convert {context.model_id} -> {fmt}"):
            path = CONVERTERS[fmt](
                context.model, context.sample, out_dir, context.metadata,
            )
        result = {
            "status": "PENDING",
            "convert_seconds": round(time.time() - t0, 2),
            "artifact": _artifact_display_path(path),
            "size_mb": round(_artifact_size_mb(path), 2),
        }
        result.update(_test_converted_artifact(context, fmt, path, clog))
    except NotImplementedError as error:
        result = {"status": "SKIP", "error": str(error)}
        clog.warning(f"skipped: {error}")
    except Exception as error:
        result = {"status": "CONVERT_FAIL", "error": repr(error)}
        clog.exception(f"convert failed: {error}")
    return result


def process_model(model_id: str) -> dict:
    mlog = master_logger()
    mlog.info(f"=== {model_id} ===")
    out_root = OUT / model_id
    out_root.mkdir(parents=True, exist_ok=True)
    results = {fmt: {"status": "PENDING"} for fmt in TARGET_FORMATS}

    # Load source once
    try:
        with timed(mlog, f"load {model_id}"):
            model, sample, meta = load(model_id, CACHE)
        ref = cmp.reference_output(model, sample)
    except Exception as e:
        mlog.exception(f"LOAD FAILED for {model_id}: {e}")
        for fmt in TARGET_FORMATS:
            results[fmt] = {"status": "LOAD_FAIL", "error": str(e)}
        (RESULTS / f"{model_id}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        return results

    context = ConversionContext(
        model_id=model_id,
        model=model,
        sample=sample,
        reference=ref,
        metadata=meta,
        output_root=out_root,
    )
    for fmt in TARGET_FORMATS:
        results[fmt] = _convert_and_test_single_format(context, fmt)
        gc.collect()
    # Write per-model JSON
    (RESULTS / f"{model_id}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    # Free the loaded model
    del context
    del model
    gc.collect()
    return results


STATUS_EMOJI = {
    "PASS": "OK",
    "DIFF": "DIFF",
    "SKIP": "SKIP",
    "CONVERT_FAIL": "C-FAIL",
    "TEST_FAIL": "T-FAIL",
    "TEST_SKIP": "T-SKIP",
    "LOAD_FAIL": "L-FAIL",
    "PENDING": "?",
}


MATRIX_FORMATS = TARGET_FORMATS


def write_matrix(all_results: dict):
    _, dev_label = pick_device()
    lines = ["# Conversion Matrix",
             "",
             f"Test device (auto-detected): **{dev_label}**",
             "",
             "| model | " + " | ".join(MATRIX_FORMATS) + " |",
             "|---" + "|---" * len(MATRIX_FORMATS) + "|"]
    for mid in REGISTRY:
        if mid not in all_results:
            continue
        row = [mid]
        for fmt in MATRIX_FORMATS:
            r = all_results[mid].get(fmt, {})
            tag = STATUS_EMOJI.get(r.get("status", "PENDING"), "?")
            extra = ""
            d = r.get("diff")
            if d:
                extra = f"<br/>max_abs={d['max_abs']:.2e}"
            row.append(f"{tag}{extra}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("Legend: OK=pass, DIFF=converted but output differs, "
                 "SKIP=not applicable, C-FAIL=convert failed, T-FAIL=test failed, "
                 "T-SKIP=test skipped, L-FAIL=load failed.")
    MATRIX.write_text("\n".join(lines), encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="*", help="model ids (default: all)")
    args = p.parse_args()
    targets = args.models or list(REGISTRY.keys())

    dev, label = pick_device()
    master_logger().info(f"test device: {label} ({dev}); ORT providers: {onnxruntime_providers()}")

    all_results = {}
    # Load any existing per-model results so partial progress accumulates
    for jp in RESULTS.glob("*.json"):
        try:
            all_results[jp.stem] = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            pass

    for mid in targets:
        try:
            all_results[mid] = process_model(mid)
        except KeyboardInterrupt:
            master_logger().warning("Interrupted by user")
            break
        except Exception as e:
            master_logger().exception(f"unexpected error for {mid}: {e}")
            all_results[mid] = {fmt: {"status": "LOAD_FAIL", "error": traceback.format_exc()}
                                for fmt in TARGET_FORMATS}
        write_matrix(all_results)

    write_matrix(all_results)
    master_logger().info(f"Matrix written to {MATRIX}")


if __name__ == "__main__":
    main()
