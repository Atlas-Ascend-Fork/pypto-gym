#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
"""Deterministic PyPTO-Pro Stage-2 golden scaffold generator (CPU-only).

Reads the validated JSON machine contract in SPEC.md and the golden template,
then emits `<op>_golden.py` with the *deterministic* parts filled in:

  - file/function names (`{op}` placeholders)
  - `def <op>_golden(<tensor args...>, <scalar kwargs from default_params>)`
  - the canonical SPEC formula as an exact, reviewable source hint
  - `_make_inputs(device)` constructing every tensor arg for every P0 case
  - `_validate()` harness iterating single- and multi-case factory formats

The op-specific math body, value/structure-constrained inputs, and op-specific
property checks stay as `# TODO:` markers for the implementing agent to fill.

Pure standard library. No torch, no NPU — runs anywhere.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

_LOGGER = logging.getLogger("gen_golden_scaffold")


_DTYPE_MAP = {
    "bfloat16": "torch.bfloat16", "bf16": "torch.bfloat16",
    "float16": "torch.float16", "fp16": "torch.float16", "half": "torch.float16",
    "float32": "torch.float32", "fp32": "torch.float32", "float": "torch.float32",
    "float64": "torch.float64", "fp64": "torch.float64",
    "int8": "torch.int8", "uint8": "torch.uint8",
    "int16": "torch.int16", "int32": "torch.int32", "int64": "torch.int64",
    "bool": "torch.bool",
}
_INT_DTYPES = {"torch.int8", "torch.uint8", "torch.int16", "torch.int32",
               "torch.int64", "torch.bool"}


def _norm_dtype(cell: str) -> str:
    key = cell.replace("`", "").strip().lower().split()[0] if cell.strip() else ""
    key = key.replace("torch.", "")           # torch.float32 -> float32
    return _DTYPE_MAP.get(key, "torch.float32")

# ---------------------------------------------------------------------------
# Codegen helpers
# ---------------------------------------------------------------------------


def _ctor(shape: list[int], dtype: str) -> tuple[str, bool]:
    """Return (constructor_expr, needs_constraint_todo)."""
    shp = ", ".join(str(s) for s in shape)
    size = f"({shp},)" if shape else "()"
    if dtype in _INT_DTYPES:
        if dtype == "torch.bool":
            expr = f"torch.randint(0, 2, {size}, dtype={dtype}, device=device)"
        elif dtype == "torch.uint8":
            expr = f"torch.randint(0, 256, {size}, dtype={dtype}, device=device)"
        else:
            expr = f"torch.randint(0, 8, {size}, dtype={dtype}, device=device)"
        return expr, True  # integer tensors are usually indices/quantized -> constrain
    return f"torch.randn({size}, dtype={dtype}, device=device)", False


def _build_signature(op: str, tensors: list[dict], kwargs: dict) -> str:
    lines = [f"def {op}_golden("]
    for t in tensors:
        lines.append(f"    {t['name']}: torch.Tensor,")
    for k, v in kwargs.items():
        ann = "int" if isinstance(v, int) and not isinstance(v, bool) else \
              "bool" if isinstance(v, bool) else \
              "float" if isinstance(v, float) else "object"
        lines.append(f"    {k}: {ann} = {v!r},")
    lines.append("):")
    return "\n".join(lines)


def _case_input_lines(tensors: list[dict], case: dict) -> list[str]:
    lines: list[str] = []
    names: list[str] = []
    for tensor in tensors:
        name = tensor["name"]
        dtype = case.get("input_dtypes", {}).get(name, tensor["dtype"])
        expr, constrain = _ctor(case["input_shapes"][name], _norm_dtype(dtype))
        tag = "  # TODO: constrain" if constrain else ""
        lines.append(f"    {name} = {expr}{tag}")
        names.append(name)
    kwargs = ", ".join(f'"{key}": {value!r}' for key, value in case["params"].items())
    lines.append(f"    args = [{', '.join(names)}]")
    lines.append(f"    kwargs = {{{kwargs}}}")
    return lines


def _build_make_inputs(tensors: list[dict], cases: list[dict]) -> str:
    body = ["def _make_inputs(device):",
            '    """Construct every P0 case from the validated machine contract.',
            "",
            "    Integer tensors are emitted as generic randint and flagged",
            "    `# TODO: constrain`; replace those values with legal inputs.",
            '    """']
    if len(cases) == 1:
        body.append(f"    # P0 case: {cases[0]['name']}")
        body.extend(_case_input_lines(tensors, cases[0]))
        body.append("    return args, kwargs")
        return "\n".join(body)

    body.append("    cases = []")
    for case in cases:
        body.append(f"    # P0 case: {case['name']}")
        body.extend(_case_input_lines(tensors, case))
        body.append(f'    cases.append(("{case["name"]}", args, kwargs))')
    body.append("    return cases")
    return "\n".join(body)


def _build_validate(op: str, case_names: list[str]) -> str:
    expected = repr(tuple(case_names))
    return "\n".join([
        "def _validate():",
        '    """Run every contract P0 case and check output finiteness.',
        "    Fill op-specific property/range checks at the TODO markers below.",
        '    """',
        "    device = _get_device()",
        "    print('=' * 60)",
        f"    print('{op}_golden validation report')",
        "    print('=' * 60)",
        "    print(f'Device: {device}')",
        "",
        "    raw_cases = _make_inputs(device)",
        "    if (isinstance(raw_cases, tuple) and len(raw_cases) == 2",
        "            and isinstance(raw_cases[0], list) and isinstance(raw_cases[1], dict)):",
        f"        cases = [({expected}[0], raw_cases[0], raw_cases[1])]",
        "    else:",
        "        cases = raw_cases",
        "    assert isinstance(cases, list) and cases, '_make_inputs returned no cases'",
        "    observed_names = [case[0] for case in cases]",
        "    assert len(observed_names) == len(set(observed_names)), 'duplicate case names'",
        f"    missing = [name for name in {expected} if name not in observed_names]",
        "    assert not missing, f'missing contract P0 cases: {missing}'",
        f"    unexpected = [name for name in observed_names if name not in {expected}]",
        "    assert not unexpected, f'unexpected P0 cases: {unexpected}'",
        "",
        "    for case_name, args, kwargs in cases:",
        "        assert isinstance(case_name, str) and case_name",
        "        assert isinstance(args, list) and isinstance(kwargs, dict)",
        "        print(f'\\n[case: {case_name}]')",
        f"        outs = {op}_golden(*args, **kwargs)",
        "        outs = outs if isinstance(outs, (tuple, list)) else (outs,)",
        "",
        "        print('[finiteness]')",
        "        for i, o in enumerate(outs):",
        "            ok = torch.isfinite(o).all().item()",
        "            print(f'  out[{i}] shape={tuple(o.shape)} finite={ok} '",
        "                  f'... {\"PASS\" if ok else \"FAIL\"}')",
        "            assert ok, f'{case_name} out[{i}] contains NaN/Inf'",
        "",
        "        # TODO: output shape equality vs the matching SPEC P0 case",
        "        # TODO: value-range checks derived from _SPEC_FORMULA",
        "        # TODO: math properties (monotonicity / symmetry / conservation)",
        "        # TODO: generalization sampling over dynamic axes",
        "",
        "    print('\\n' + '=' * 60)",
        "    print('validation complete')",
        "    print('=' * 60)",
        "    return True",
    ])


def _load_spec_contract(path: Path) -> dict[str, Any]:
    """Load Stage 1's canonical parser without copying its schema rules."""
    parser_path = (
        Path(__file__).resolve().parents[2]
        / "pypto-pro-intent-understand" / "scripts" / "validate_spec.py"
    )
    module_spec = importlib.util.spec_from_file_location("pypto_spec_contract", parser_path)
    if module_spec is None or module_spec.loader is None:
        raise ValueError(f"cannot load canonical SPEC parser: {parser_path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module.load_spec_contract(path)


def generate(spec_path: str, template_path: str) -> str:
    contract = _load_spec_contract(Path(spec_path))
    op = contract["op_name"]
    kwargs = contract.get("default_params") or {}
    tensors = [
        {
            "name": item["name"],
            "shape": json.dumps(item["shape"], ensure_ascii=False),
            "dtype": _norm_dtype(item["dtype"]),
        }
        for item in contract["inputs"]
    ]
    cases = contract["p0_cases"]

    with open(template_path, encoding="utf-8") as f:
        template = f.read()
    out = template.replace("{op}", op).replace(
        "{formula_literal}", repr(contract["formula"]),
    )

    cpu = f"def {op}_golden_cpu(" in out
    function = f"{op}_golden_cpu" if cpu else f"{op}_golden"
    signature = _build_signature(op, tensors, kwargs).replace(f"{op}_golden(", f"{function}(", 1)
    out = out.replace(f"def {function}(x: torch.Tensor) -> torch.Tensor:", signature, 1)
    if cpu:
        # The CPU template has no NPU device helper or case factory. Install the same
        # contract-driven factory/harness, with an explicitly CPU-only device.
        if "_SPEC_FORMULA =" not in out:
            out = out.replace("import torch", f"import torch\n\n_SPEC_FORMULA = {contract['formula']!r}", 1)
        if "def _get_device(" not in out:
            out = out.replace("def _validate():", "def _get_device():\n    return torch.device('cpu')\n\n\ndef _validate():", 1)
        if "def _make_inputs(" not in out:
            out = out.replace("def _validate():", _build_make_inputs(tensors, cases) + "\n\n\ndef _validate():", 1)

    # Swap the default _make_inputs (lines from `def _make_inputs` to its
    # `return [x], {}`) with the generated one.
    out = _replace_block(out, "def _make_inputs(device):",
                         _build_make_inputs(tensors, cases))
    out = _replace_block(out, "def _validate():",
                         _build_validate(op, [case["name"] for case in cases]).replace(
                             f"{op}_golden(*args", f"{function}(*args"))
    return out


def _replace_block(src: str, header: str, new_block: str) -> str:
    """Replace from `header` line up to (excluding) the next top-level `def `,
    `if __name__`, or the `# ===` banner that follows."""
    lines = src.splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith(header)), None)
    if start is None:
        return src
    end = len(lines)
    _stops = ("def ", "if __name__", "# ====", "# ----")
    for j in range(start + 1, len(lines)):
        if any(lines[j].startswith(s) for s in _stops):
            end = j
            break
    return "\n".join(lines[:start] + new_block.splitlines() + [""] + lines[end:])


# ---------------------------------------------------------------------------

_SELF_TEST_SPEC = '''# demo_add specification

```json machine-contract
{
  "schema_version": 1,
  "op_name": "demo_add",
  "formula": "y = a + b",
  "supported_dtypes": ["float32"],
  "inputs": [
    {"name": "a", "shape": ["M", 1024], "dtype": "float32", "value_range": [-1, 1]},
    {"name": "b", "shape": ["M", 1024], "dtype": "float32", "value_range": [-1, 1]}
  ],
  "outputs": [
    {"name": "y", "shape": ["M", 1024], "dtype": "float32", "value_range": [-2, 2]}
  ],
  "default_params": {"eps": 0.000001},
  "tolerance": {"rtol": 0.004, "atol": 0.004},
  "dynamic_axes_ranges": {"M": [1, 128]},
  "shape_constraints": ["a and b have the same shape"],
  "p0_cases": [
    {
      "name": "typical",
      "params": {"eps": 0.000001},
      "input_shapes": {"a": [8, 1024], "b": [8, 1024]},
      "output_shapes": {"y": [8, 1024]}
    },
    {
      "name": "large",
      "params": {"eps": 0.000002},
      "input_shapes": {"a": [32, 1024], "b": [32, 1024]},
      "output_shapes": {"y": [32, 1024]}
    }
  ],
  "perf_target": null
}
```
'''


def _self_test(template_path: str) -> int:
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False,
                                     encoding="utf-8") as f:
        f.write(_SELF_TEST_SPEC)
        spec = f.name
    code = generate(spec, template_path)
    os.unlink(spec)
    single_factory = _build_make_inputs(
        [{"name": "x", "dtype": "torch.float32"}],
        [{"name": "only", "params": {}, "input_shapes": {"x": [4]}}],
    )
    p0_case_tokens = (
        'cases.append(("typical", args, kwargs))',
        'cases.append(("large", args, kwargs))',
        "torch.randn((8, 1024,), dtype=torch.float32",
        "torch.randn((32, 1024,), dtype=torch.float32",
        'kwargs = {"eps": 2e-06}',
    )
    checks = {
        "signature": "def demo_add_golden(" in code and "a: torch.Tensor" in code
        and "eps: float = 1e-06" in code,
        "formula_hint": "_SPEC_FORMULA = 'y = a + b'" in code,
        "all_p0_cases": all(token in code for token in p0_case_tokens),
        "validate_wired": "raw_cases = _make_inputs(device)" in code
        and "for case_name, args, kwargs in cases:" in code
        and "missing contract P0 cases" in code,
        "single_p0_compat": "return args, kwargs" in single_factory
        and "return cases" not in single_factory,
        "compiles": _compiles(code),
        "no_placeholder": "{op}" not in code and "{formula_literal}" not in code,
    }
    for k, v in checks.items():
        (_LOGGER.info if v else _LOGGER.error)("  [%s] %s", "PASS" if v else "FAIL", k)
    return 0 if all(checks.values()) else 1


def _compiles(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError as e:
        _LOGGER.warning("    SyntaxError: %s", e)
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Deterministic golden scaffold generator")
    ap.add_argument("--spec", help="path to SPEC.md")
    ap.add_argument("--template", required=True, help="path to golden-template.py.tmpl")
    ap.add_argument("--out", help="output path for <op>_golden.py (default: stdout)")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    if a.self_test:
        return _self_test(a.template)
    if not a.spec:
        ap.error("--spec is required unless --self-test")
    try:
        code = generate(a.spec, a.template)
    except ValueError as e:
        _LOGGER.error("[FAIL] %s", e)
        return 1
    if not _compiles(code):
        _LOGGER.error("[FAIL] generated scaffold does not parse")
        return 1
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(code)
        _LOGGER.info("[OK] wrote %s", a.out)
    else:
        _LOGGER.info(code)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
