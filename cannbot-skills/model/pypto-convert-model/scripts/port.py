# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Any-to-any model porter for {onnx, pt, safetensors}.

Usage examples:
    # TorchScript -> ONNX, just give an input shape
    python scripts/port.py model.pt out.onnx --input-shape 1,3,224,224

    # safetensors needs an architecture; hand it an HF repo
    python scripts/port.py model.safetensors out.onnx \\
        --hf-repo google/mobilenet_v2_1.0_224

    # ONNX -> TorchScript
    python scripts/port.py model.onnx out.pt --input-shape 1,3,224,224

The output of the converted artifact is compared against the source
artifact's output on the same dummy input and a `max_abs` diff is printed.
The verification stage runs on the best available accelerator
(NPU > CUDA > XPU > MPS > CPU); override with CONVERT_EXP_DEVICE=...
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

import numpy as np
import torch

# cudnn picks heuristically and can return slightly different fp32 results on
# repeated forwards even with identical weights. Pin it so the verification
# diff actually reflects conversion quality, not kernel-selection noise.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

compare_arrays = import_module("compare").diff
_device_util = import_module("device_util")
onnxruntime_providers = _device_util.onnxruntime_providers
pick_device = _device_util.pick_device
to_device = _device_util.to_device
TensorOutputAdapter = import_module("loaders").TensorOutputAdapter

LOGGER = logging.getLogger(__name__)

VALID_FORMATS = ("onnx", "pt", "safetensors")
EXT_TO_FORMAT = {".onnx": "onnx", ".pt": "pt", ".pth": "pt", ".safetensors": "safetensors"}


class _BelowErrorFilter(logging.Filter):
    def filter(self, record):
        return record.levelno < logging.ERROR


def _configure_logging():
    LOGGER.handlers.clear()
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.addFilter(_BelowErrorFilter())
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.ERROR)
    formatter = logging.Formatter("%(message)s")
    stdout_handler.setFormatter(formatter)
    stderr_handler.setFormatter(formatter)
    LOGGER.addHandler(stdout_handler)
    LOGGER.addHandler(stderr_handler)


@dataclass(frozen=True)
class PortOptions:
    """Conversion options shared across all stages of one port operation."""

    in_fmt: str
    out_fmt: str
    input_shape: tuple[int, ...] | None
    hf_repo: str | None
    skip_verify: bool = False


# ---------- helpers ----------

def _detect_format(path: Path, override: str | None) -> str:
    if override:
        if override not in VALID_FORMATS:
            raise ValueError(f"format must be one of {VALID_FORMATS}, got {override!r}")
        return override
    fmt = EXT_TO_FORMAT.get(path.suffix.lower())
    if fmt is None:
        raise ValueError(
            f"cannot infer format from {path.name}. Use --input-format/--output-format."
        )
    return fmt


def _parse_shape(shape: str | None) -> tuple[int, ...] | None:
    if not shape:
        return None
    parts = [p.strip() for p in shape.replace("x", ",").split(",") if p.strip()]
    return tuple(int(p) for p in parts)


def _to_numpy_safe(t: torch.Tensor) -> np.ndarray:
    t = t.detach()
    if t.dtype in (torch.bfloat16, torch.float16) or str(t.dtype).startswith("torch.float8_"):
        t = t.to(torch.float32)
    return t.cpu().numpy()


def _flatten(out) -> np.ndarray:
    if isinstance(out, np.ndarray):
        return out
    if isinstance(out, torch.Tensor):
        return _to_numpy_safe(out)
    if isinstance(out, (list, tuple)):
        return _flatten(out[0])
    if hasattr(out, "logits"):
        return _to_numpy_safe(out.logits)
    if hasattr(out, "last_hidden_state"):
        return _to_numpy_safe(out.last_hidden_state)
    if isinstance(out, dict) and out:
        return _flatten(next(iter(out.values())))
    raise TypeError(f"unknown output type {type(out)}")


def _diff(a: np.ndarray, b: np.ndarray, atol: float, rtol: float) -> dict:
    return compare_arrays(a, b, atol=atol, rtol=rtol)


def _tensor_stats(arr: np.ndarray) -> dict:
    """Compact summary stats for raw-log output."""
    if arr.size == 0:
        return {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "first5": [],
            "empty": True,
        }
    return {
        "shape": tuple(int(x) for x in arr.shape) if arr.ndim else (),
        "dtype": str(arr.dtype),
        "size": int(arr.size),
        "min": float(arr.min()) if arr.size else 0.0,
        "max": float(arr.max()) if arr.size else 0.0,
        "mean": float(arr.mean()) if arr.size else 0.0,
        "std": float(arr.std()) if arr.size else 0.0,
        "first5": [float(x) for x in arr.ravel()[:5].tolist()],
    }


def _tolerances_for(device_label: str) -> tuple[float, float]:
    """CUDA fp32 forwards have ~5e-4 cudnn accumulation noise even with the
    same weights and input, so a CPU-tight tolerance gives misleading DIFFs.
    Tighten when the diff is measured fully on CPU.
    """
    if device_label == "cpu":
        return 1e-4, 1e-3
    return 5e-3, 1e-2


# ---------- input source loaders ----------

class Source:
    """Wraps the source model as something we can: forward, jit-trace, get state_dict."""

    def __init__(self, module: torch.nn.Module, sample: tuple, device: torch.device):
        self.module = module.eval()
        self.sample = sample
        self.device = device

    def forward(self) -> np.ndarray:
        m = to_device(self.module, self.device)
        args = to_device(self.sample, self.device)
        with torch.no_grad():
            return _flatten(m(*args))


def _build_sample(input_shape: tuple[int, ...] | None,
                  hf_repo: str | None) -> tuple[torch.Tensor, ...]:
    if input_shape is not None:
        return (torch.rand(*input_shape, dtype=torch.float32),)
    if hf_repo is not None:
        # Fall back to a 1×3×224×224 image — works for most CV HF models in the registry.
        return (torch.rand(1, 3, 224, 224, dtype=torch.float32),)
    raise ValueError("either --input-shape or --hf-repo must be provided")


def _load_pt_source(path: Path, sample, device: torch.device) -> Source:
    m = torch.jit.load(str(path), map_location="cpu").eval()
    return Source(m, sample, device)


def _load_onnx_as_torch(path: Path) -> torch.nn.Module:
    """Convert ONNX → torch.nn.Module via onnx2torch so we can re-export elsewhere."""
    try:
        convert = import_module("onnx2torch").convert
    except ImportError as e:
        raise RuntimeError(
            "onnx2torch is required for ONNX-as-source conversions. "
            "pip install onnx2torch"
        ) from e
    return convert(str(path)).eval()


def _load_onnx_source(path: Path, sample, device: torch.device) -> Source:
    m = _load_onnx_as_torch(path)
    return Source(m, sample, device)


def _load_safetensors_source(path: Path, sample, hf_repo: str | None,
                             device: torch.device) -> Source:
    if not hf_repo:
        raise ValueError("--hf-repo is required when the input is safetensors")
    load_model = import_module("safetensors.torch").load_model

    inner, sample = _instantiate_hf_model(hf_repo, sample)
    load_model(inner, str(path), strict=False)
    inner.eval()
    return Source(inner, sample, device)


def _instantiate_hf_model(hf_repo: str, sample: tuple) -> tuple[torch.nn.Module, tuple]:
    """Try image-classification first; fall back to AutoModel for general nets.

    Returns the bare nn.Module wrapped to emit a tensor (not a HF dataclass).
    """
    transformers = import_module("transformers")

    try:
        inner = transformers.AutoModelForImageClassification.from_pretrained(
            hf_repo, torch_dtype=torch.float32,
        )
    except Exception:
        inner = transformers.AutoModel.from_pretrained(
            hf_repo, torch_dtype=torch.float32,
        )

    return TensorOutputAdapter(inner).eval(), sample


# ---------- target writers ----------

def _write_pt(source: Source, out: Path) -> Path:
    try:
        traced = torch.jit.trace(source.module, source.sample, strict=False)
    except Exception:
        traced = torch.jit.script(source.module)
    out.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(out))
    return out


def _write_onnx(source: Source, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    names = [f"input{i}" for i in range(len(source.sample))]
    output_names = ["output"]
    dynamic_axes = {n: {0: "batch"} for n in names + output_names}
    with torch.no_grad():
        try:
            torch.onnx.export(
                source.module, source.sample, str(out),
                input_names=names, output_names=output_names,
                opset_version=17, dynamic_axes=dynamic_axes,
                do_constant_folding=True, dynamo=False,
            )
        except TypeError:
            torch.onnx.export(
                source.module, source.sample, str(out),
                input_names=names, output_names=output_names,
                opset_version=17, dynamic_axes=dynamic_axes,
                do_constant_folding=True,
            )
    return out


def _write_safetensors(source: Source, out: Path) -> Path:
    save_model = import_module("safetensors.torch").save_model
    out.parent.mkdir(parents=True, exist_ok=True)
    save_model(source.module, str(out))
    return out


# ---------- target reload + forward (for verification) ----------

def _run_pt(path: Path, sample, device: torch.device) -> np.ndarray:
    try:
        m = torch.jit.load(str(path), map_location=device).eval()
        args = to_device(sample, device)
    except Exception:
        m = torch.jit.load(str(path), map_location="cpu").eval()
        args = sample
    with torch.no_grad():
        return _flatten(m(*args))


def _run_onnx(path: Path, sample) -> np.ndarray:
    ort = import_module("onnxruntime")
    sess = ort.InferenceSession(str(path), providers=onnxruntime_providers())
    arrs = [a.detach().cpu().numpy() for a in sample]
    inames = [i.name for i in sess.get_inputs()]
    feed = {n: arrs[i] for i, n in enumerate(inames) if i < len(arrs)}
    out = sess.run(None, feed)
    return _flatten(out[0])


def _run_safetensors(path: Path, source_module: torch.nn.Module,
                     sample, device: torch.device) -> np.ndarray:
    load_model = import_module("safetensors.torch").load_model
    load_model(source_module, str(path), strict=False)
    source_module.eval()
    m = to_device(source_module, device)
    args = to_device(sample, device)
    with torch.no_grad():
        return _flatten(m(*args))


# ---------- port() helpers ----------

def _load_source(in_path: Path, in_fmt: str,
                 input_shape: tuple[int, ...] | None,
                 hf_repo: str | None) -> tuple[Source, str, float]:
    """Load the source model. Returns (source, device_label, load_time_sec)."""
    t = time.time()
    sample = _build_sample(input_shape, hf_repo)
    device, label = pick_device()
    if in_fmt == "pt":
        src = _load_pt_source(in_path, sample, device)
    elif in_fmt == "onnx":
        src = _load_onnx_source(in_path, sample, device)
    elif in_fmt == "safetensors":
        src = _load_safetensors_source(in_path, sample, hf_repo, device)
    else:
        raise ValueError(in_fmt)
    return src, label, time.time() - t


def _convert_target(src: Source, out_path: Path, out_fmt: str) -> float:
    """Write the source model in the target format. Returns convert_time_sec."""
    t = time.time()
    if out_fmt == "pt":
        _write_pt(src, out_path)
    elif out_fmt == "onnx":
        _write_onnx(src, out_path)
    elif out_fmt == "safetensors":
        _write_safetensors(src, out_path)
    else:
        raise ValueError(out_fmt)
    return time.time() - t


def _verify_roundtrip(src: Source, out_path: Path,
                      out_fmt: str) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Forward on source and converted target. Returns (ref, out, timing_dict)."""
    timings: dict[str, float] = {}
    t = time.time()
    ref = src.forward()
    timings["ref_forward"] = time.time() - t

    t = time.time()
    if out_fmt == "pt":
        out = _run_pt(out_path, src.sample, src.device)
    elif out_fmt == "onnx":
        out = _run_onnx(out_path, src.sample)
    elif out_fmt == "safetensors":
        out = _run_safetensors(out_path, src.module, src.sample, src.device)
    else:
        raise ValueError(out_fmt)
    timings["target_forward"] = time.time() - t

    return ref, out, timings


# ---------- orchestration ----------

def _port(in_path: Path, out_path: Path, options: PortOptions) -> dict:
    in_fmt = options.in_fmt
    out_fmt = options.out_fmt
    if in_fmt == out_fmt:
        raise ValueError(f"input and output format are both {in_fmt}; nothing to do")
    if in_fmt not in VALID_FORMATS or out_fmt not in VALID_FORMATS:
        raise ValueError(f"formats must be one of {VALID_FORMATS}")

    if in_path.resolve() == out_path.resolve() or (out_path.exists() and in_path.samefile(out_path)):
        raise ValueError("input and output paths refer to the same file")

    timings: dict[str, float] = {}

    # 1. Load source model
    src, label, timings["load"] = _load_source(
        in_path, in_fmt, options.input_shape, options.hf_repo,
    )

    # 2. Convert to target format
    timings["convert"] = _convert_target(src, out_path, out_fmt)

    sample_shapes = [tuple(int(x) for x in a.shape) for a in src.sample]
    sample_dtypes = [str(a.dtype) for a in src.sample]

    info: dict = {
        "input": str(in_path),
        "output": str(out_path),
        "in_fmt": in_fmt,
        "out_fmt": out_fmt,
        "size_mb": round(out_path.stat().st_size / 1e6, 2),
        "timings": timings,
        "sample_shapes": sample_shapes,
        "sample_dtypes": sample_dtypes,
    }

    if options.skip_verify:
        info["verified"] = False
        return info

    # 3. Verify round-trip
    info["device"] = label
    info["ort_providers"] = onnxruntime_providers() if "onnx" in (in_fmt, out_fmt) else None

    ref, out, vt = _verify_roundtrip(src, out_path, out_fmt)
    timings.update(vt)

    info["ref_stats"] = _tensor_stats(ref)
    info["out_stats"] = _tensor_stats(out)

    # ONNX Runtime here runs CPU on aarch64, so an ONNX leg compared against a
    # CUDA reference will always show CPU/CUDA fp32 drift; widen the tolerance
    # to match. Pure CPU comparisons keep the tighter tolerance.
    if label != "cpu":
        atol, rtol = _tolerances_for(label)
    else:
        atol, rtol = _tolerances_for("cpu")
    info["diff"] = _diff(ref, out, atol=atol, rtol=rtol)
    info["status"] = "PASS" if info["diff"]["allclose"] else "DIFF"
    return info


_PORT_OPTION_NAMES = (
    "in_fmt", "out_fmt", "input_shape", "hf_repo", "skip_verify",
)


def _resolve_port_options(options, legacy_values, legacy_keywords):
    if isinstance(options, PortOptions):
        if legacy_values or legacy_keywords:
            raise TypeError("PortOptions cannot be combined with legacy arguments")
        return options

    positional_values = legacy_values
    if options is not None:
        positional_values = (options, *legacy_values)
    if len(positional_values) > len(_PORT_OPTION_NAMES):
        raise TypeError("too many positional arguments")

    values = dict(legacy_keywords)
    for name, value in zip(_PORT_OPTION_NAMES, positional_values):
        if name in values:
            raise TypeError(f"multiple values for argument {name!r}")
        values[name] = value
    return PortOptions(**values)


def port(in_path: Path, out_path: Path, options=None, *legacy_values, **legacy_keywords) -> dict:
    """Port a model while preserving the original seven-argument Python API."""
    resolved_options = _resolve_port_options(options, legacy_values, legacy_keywords)
    return _port(in_path, out_path, resolved_options)


def _print_verbose(info: dict) -> None:
    """Dump every numerical detail useful for debugging a port run."""
    def log(*args) -> None:
        LOGGER.info("[INFO] %s", " ".join(str(arg) for arg in args))

    log(f"input        : {info['input']} ({info['in_fmt']})")
    log(f"output       : {info['output']} ({info['out_fmt']})  size={info['size_mb']} MB")
    log(f"sample shapes: {info['sample_shapes']}  dtypes={info['sample_dtypes']}")
    if "device" in info:
        log(f"device       : {info['device']}")
        if info.get("ort_providers"):
            log(f"ORT providers: {info['ort_providers']}")
    t = info.get("timings", {})
    if t:
        log("timings (s)  : " + "  ".join(f"{k}={v:.3f}" for k, v in t.items()))
    if info.get("verified") is False:
        log("verification skipped")
        return
    rs, os_ = info.get("ref_stats"), info.get("out_stats")
    if rs and os_:
        log(f"ref forward  : shape={rs['shape']} dtype={rs['dtype']} "
            f"min={rs['min']:.4g} max={rs['max']:.4g} mean={rs['mean']:.4g} std={rs['std']:.4g}")
        log(f"               first5={['%.4g' % x for x in rs['first5']]}")
        log(f"out forward  : shape={os_['shape']} dtype={os_['dtype']} "
            f"min={os_['min']:.4g} max={os_['max']:.4g} mean={os_['mean']:.4g} std={os_['std']:.4g}")
        log(f"               first5={['%.4g' % x for x in os_['first5']]}")
    d = info.get("diff", {})
    if d:
        log(f"diff abs     : max={d['max_abs']:.3e} mean={d['mean_abs']:.3e} "
            f"p50={d['p50']:.3e} p95={d['p95']:.3e} p99={d['p99']:.3e}")
        log(f"diff rel max : {d['max_rel']:.3e}")
        log(f"tolerance    : atol={d['atol']:.0e} rtol={d['rtol']:.0e} "
            f"(allclose={d['allclose']})")
        if d.get("top_mismatch"):
            top = "  ".join(f"[{i}]={v:.3e}" for i, v in d["top_mismatch"])
            log(f"top mismatch : {top}")


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="input model path")
    parser.add_argument("output", type=Path, help="output model path")
    parser.add_argument("--input-format", choices=VALID_FORMATS,
                   help="override format inferred from extension")
    parser.add_argument("--output-format", choices=VALID_FORMATS,
                   help="override format inferred from extension")
    parser.add_argument("--input-shape",
                   help="comma-separated input shape, e.g. 1,3,224,224. "
                        "Used to build the dummy input.")
    parser.add_argument("--hf-repo",
                   help="HF repo id used to instantiate the architecture when "
                        "safetensors is the input or output. Required for "
                        "safetensors as input.")
    parser.add_argument("--skip-verify", action="store_true",
                   help="skip the round-trip diff check")
    parser.add_argument("-v", "--verbose", action="store_true",
                   help="dump per-stage timings, tensor stats, diff distribution, "
                        "and top mismatch indices as raw log lines")
    return parser


def main(argv: list[str] | None = None) -> int:
    _configure_logging()
    args = _build_cli_parser().parse_args(argv)

    try:
        in_fmt = _detect_format(args.input, args.input_format)
        out_fmt = _detect_format(args.output, args.output_format)
        shape = _parse_shape(args.input_shape)
        options = PortOptions(
            in_fmt=in_fmt,
            out_fmt=out_fmt,
            input_shape=shape,
            hf_repo=args.hf_repo,
            skip_verify=args.skip_verify,
        )
        info = port(args.input, args.output, options)
    except Exception as error:
        LOGGER.error("FAIL (%s): %s", type(error).__name__, error)
        return 2

    if args.verbose:
        _print_verbose(info)
        LOGGER.info("")

    LOGGER.info(
        "%s -> %s: wrote %s (%s MB)",
        info["in_fmt"], info["out_fmt"], info["output"], info["size_mb"],
    )
    if info.get("verified") is False:
        LOGGER.info("verification skipped (--skip-verify)")
        return 0
    d = info["diff"]
    LOGGER.info(
        "device=%s  status=%s  max_abs=%.2e  mean_abs=%.2e",
        info.get("device", "?"), info["status"], d["max_abs"], d["mean_abs"],
    )
    return 0 if info["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
