# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run a converted artifact and compare its output against the original."""
import logging
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

import numpy as np
import torch
from device_util import onnxruntime_providers, pick_device, to_device

LOGGER = logging.getLogger(__name__)


def _as_args(sample):
    if isinstance(sample, tuple):
        return sample
    if isinstance(sample, dict):
        return tuple(sample.values())
    return (sample,)


def _to_numpy_args(sample) -> list:
    return [a.cpu().numpy() for a in _as_args(sample)]


def _to_numpy_safe(t: torch.Tensor) -> np.ndarray:
    """Numpy doesn't grok bfloat16 / float8; cast to float32 before exporting."""
    t = t.detach()
    if t.dtype in (torch.bfloat16, torch.float16) or str(t.dtype).startswith("torch.float8_"):
        t = t.to(torch.float32)
    return t.cpu().numpy()


def _flatten_output(out) -> np.ndarray:
    """Reduce model output to one numpy array while preserving its shape."""
    if isinstance(out, np.ndarray):
        x = out
    elif isinstance(out, torch.Tensor):
        x = _to_numpy_safe(out)
    elif isinstance(out, (list, tuple)):
        x = _flatten_output(out[0])
    elif hasattr(out, "logits"):
        x = _to_numpy_safe(out.logits)
    elif hasattr(out, "last_hidden_state"):
        x = _to_numpy_safe(out.last_hidden_state)
    elif isinstance(out, dict) and out:
        x = _flatten_output(next(iter(out.values())))
    else:
        raise TypeError(f"unknown output type {type(out)}")
    return np.asarray(x)


def _model_device(model) -> torch.device:
    for p in model.parameters():
        return p.device
    for b in model.buffers():
        return b.device
    return torch.device("cpu")


def _is_oom(err: BaseException) -> bool:
    msg = str(err).lower()
    oom_markers = (
        "out of memory", "cuda out of memory", "no_memory", "nv_err_no_memory",
        "cudnn_status_alloc", "cublas_status_alloc",
    )
    return any(marker in msg for marker in oom_markers)


def reference_output(model, sample) -> np.ndarray:
    """Run the model on the best available accelerator and restore its device.

    Restoring matters because the same Module instance is later handed to the
    converters (ONNX / TorchScript), which assume CPU placement.
    On accelerator OOM (common for large models on unified-memory hosts) fall
    back to CPU so the conversion run can continue.
    """
    device, _ = pick_device()
    orig = _model_device(model)
    if device == orig:
        with torch.no_grad():
            return _flatten_output(model(*_as_args(sample)))

    try:
        model_on_device = to_device(model, device)
        args = to_device(_as_args(sample), device)
        with torch.no_grad():
            out = model_on_device(*args)
        result = _flatten_output(out)
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:  # type: ignore[attr-defined]
        if not _is_oom(e):
            raise
        # Restore to CPU and run there.
        model.to(orig)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        with torch.no_grad():
            return _flatten_output(model(*_as_args(sample)))
    finally:
        try:
            model.to(orig)
        except Exception as error:
            LOGGER.warning("Failed to restore model to %s: %s", orig, error)
    return result


def run_onnx(path: Path, sample, meta=None) -> np.ndarray:
    """Run an ONNX artifact; ``meta`` is retained for API compatibility."""
    ort = import_module("onnxruntime")
    sess = ort.InferenceSession(str(path), providers=onnxruntime_providers())
    arrs = _to_numpy_args(sample)
    inames = [i.name for i in sess.get_inputs()]
    feed = {}
    for i, name in enumerate(inames):
        if i < len(arrs):
            feed[name] = arrs[i]
    out = sess.run(None, feed)
    return _flatten_output(out[0])


def run_pt(path: Path, sample, meta=None) -> np.ndarray:
    """Run a TorchScript artifact; ``meta`` is retained for compatibility."""
    device, _ = pick_device()
    try:
        m = torch.jit.load(str(path), map_location=device).eval()
        args = to_device(_as_args(sample), device)
    except Exception:
        # If the TorchScript artifact can't be moved to the device, fall back to CPU
        m = torch.jit.load(str(path), map_location="cpu").eval()
        args = _as_args(sample)
    with torch.no_grad():
        out = m(*args)
    return _flatten_output(out)


def run_safetensors(path: Path, model_template, sample, meta=None) -> np.ndarray:
    """Run safetensors weights; ``meta`` is retained for compatibility."""
    load_model = import_module("safetensors.torch").load_model
    load_model(model_template, str(path), strict=False)
    model_template.eval()
    return reference_output(model_template, sample)


@dataclass(frozen=True)
class _DiffLayout:
    reference: np.ndarray
    output: np.ndarray
    ref_shape: tuple[int, ...]
    out_shape: tuple[int, ...]

    @property
    def ref_size(self) -> int:
        return int(self.reference.size)

    @property
    def out_size(self) -> int:
        return int(self.output.size)

    @property
    def size(self) -> int:
        return min(self.ref_size, self.out_size)


@dataclass(frozen=True)
class _DiffErrors:
    reference: np.ndarray
    output: np.ndarray
    finite: np.ndarray
    invalid: np.ndarray
    absolute: np.ndarray
    relative: np.ndarray


def _prepare_diff_layout(a: np.ndarray, b: np.ndarray) -> _DiffLayout:
    reference = np.asarray(a)
    output = np.asarray(b)
    if np.iscomplexobj(reference) or np.iscomplexobj(output):
        raise TypeError("complex model outputs are not supported by real-valued diff")
    return _DiffLayout(
        reference=reference.ravel(),
        output=output.ravel(),
        ref_shape=tuple(int(dimension) for dimension in reference.shape),
        out_shape=tuple(int(dimension) for dimension in output.shape),
    )


def _empty_diff(layout: _DiffLayout, atol: float, rtol: float) -> dict:
    size_match = layout.ref_size == layout.out_size
    empty_value = 0.0 if size_match else float("inf")
    return {
        "max_abs": empty_value, "mean_abs": empty_value,
        "max_rel": empty_value, "size": 0,
        "ref_size": layout.ref_size, "out_size": layout.out_size,
        "size_match": size_match, "ref_shape": layout.ref_shape,
        "out_shape": layout.out_shape,
        "shape_match": layout.ref_shape == layout.out_shape,
        "allclose": size_match and layout.ref_shape == layout.out_shape,
        "atol": atol, "rtol": rtol, "p50": empty_value,
        "p95": empty_value, "p99": empty_value, "top_mismatch": [],
    }


def _calculate_errors(layout: _DiffLayout) -> _DiffErrors:
    reference = layout.reference[:layout.size].astype(np.float64, copy=False)
    output = layout.output[:layout.size].astype(np.float64, copy=False)
    finite = np.isfinite(reference) & np.isfinite(output)
    same_inf = (
        np.isinf(reference)
        & np.isinf(output)
        & (np.signbit(reference) == np.signbit(output))
    )
    invalid = ~(finite | same_inf)
    absolute = np.zeros(layout.size, dtype=np.float64)
    absolute[finite] = np.abs(reference[finite] - output[finite])
    absolute[invalid] = np.inf
    relative = np.zeros(layout.size, dtype=np.float64)
    relative[finite] = absolute[finite] / (np.abs(reference[finite]) + 1e-9)
    relative[invalid] = np.inf
    return _DiffErrors(
        reference=reference,
        output=output,
        finite=finite,
        invalid=invalid,
        absolute=absolute,
        relative=relative,
    )


def _percentiles(absolute: np.ndarray, invalid: np.ndarray) -> tuple[float, float, float]:
    if bool(invalid.any()):
        return float("inf"), float("inf"), float("inf")
    return (
        float(np.median(absolute)),
        float(np.percentile(absolute, 95)),
        float(np.percentile(absolute, 99)),
    )


def _top_mismatches(absolute: np.ndarray) -> list[tuple[int, float]]:
    top_count = min(5, absolute.size)
    indices = np.argpartition(-absolute, top_count - 1)[:top_count]
    indices = indices[np.argsort(-absolute[indices])]
    return [(int(index), float(absolute[index])) for index in indices]


def diff(a: np.ndarray, b: np.ndarray, atol=1e-4, rtol=1e-3) -> dict:
    layout = _prepare_diff_layout(a, b)
    if layout.size == 0:
        return _empty_diff(layout, atol, rtol)
    errors = _calculate_errors(layout)
    p50, p95, p99 = _percentiles(errors.absolute, errors.invalid)
    size_match = layout.ref_size == layout.out_size
    shape_match = layout.ref_shape == layout.out_shape
    return {
        "max_abs": float(errors.absolute.max()),
        "mean_abs": float(errors.absolute.mean()),
        "max_rel": float(errors.relative.max()),
        "size": layout.size,
        "ref_size": layout.ref_size,
        "out_size": layout.out_size,
        "size_match": size_match,
        "ref_shape": layout.ref_shape,
        "out_shape": layout.out_shape,
        "shape_match": shape_match,
        "allclose": (
            size_match
            and shape_match
            and not bool(errors.invalid.any())
            and bool(
                np.isclose(
                    errors.reference[errors.finite],
                    errors.output[errors.finite],
                    atol=atol,
                    rtol=rtol,
                ).all()
            )
        ),
        "atol": atol,
        "rtol": rtol,
        "p50": p50,
        "p95": p95,
        "p99": p99,
        "top_mismatch": _top_mismatches(errors.absolute),
    }
