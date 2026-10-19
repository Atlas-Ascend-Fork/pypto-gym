#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import argparse
import importlib
import logging
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch
import torch_npu  # noqa: F401

_p = os.path.dirname(__file__)
while not os.path.isdir(os.path.join(_p, 'src')):
    _p = os.path.dirname(_p)
sys.path.insert(0, _p)
sys.path.insert(0, os.path.join(_p, 'ops', 'pypto_tensor'))


from experimental.vector.ApplyRMSProp.apply_rms_prop_impl import RMSPropConfig, apply_rms_prop_kernel

LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG = RMSPropConfig()


@dataclass(frozen=True)
class ApplyRMSPropTestCase:
    name: str
    description: str
    shape: tuple[int, int]


class InvalidTestIdError(ValueError):
    """Raised when the requested test id is not registered."""


class DevicePreparationError(RuntimeError):
    """Raised when the NPU device environment is invalid."""


TEST_CASES = {
    "level0": ApplyRMSPropTestCase(
        name="Level 0 (8x8)",
        description="Small input test",
        shape=(8, 8),
    ),
    "level1": ApplyRMSPropTestCase(
        name="Level 1 (1024x1024)",
        description="Typical input test",
        shape=(1024, 1024),
    ),
    "level2": ApplyRMSPropTestCase(
        name="Level 2 (boundary)",
        description="Boundary values test",
        shape=(16, 16),
    ),
}


def apply_rms_prop_golden(
    var: np.ndarray,
    ms: np.ndarray,
    mom: np.ndarray,
    grad: np.ndarray,
    config: RMSPropConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NumPy reference implementation of ApplyRMSProp."""
    var_new = var.copy()
    ms_new = ms.copy()
    mom_new = mom.copy()

    grad_sq = grad**2
    ms_new = ms_new + (grad_sq - ms_new) * (1.0 - config.rho)
    mom_new = mom_new * config.momentum + (grad * config.lr) / np.sqrt(ms_new + config.epsilon)
    var_new = var_new - mom_new
    return var_new, ms_new, mom_new


def resolve_device(device_id: int | None, run_mode: str) -> str:
    if run_mode == "npu" and device_id is not None:
        return f"npu:{device_id}"
    return "cpu"


def create_test_tensors(shape: tuple[int, int], device: str) -> tuple[torch.Tensor, ...]:
    var_torch = torch.rand(shape, dtype=torch.float32, device=device)
    ms_torch = torch.rand(shape, dtype=torch.float32, device=device)
    mom_torch = torch.randn(shape, dtype=torch.float32, device=device) * 2.0 - 1.0
    grad_torch = torch.randn(shape, dtype=torch.float32, device=device) * 2.0 - 1.0
    return var_torch, ms_torch, mom_torch, grad_torch


def ensure_within_tolerance(name: str, diff: float, expected: np.ndarray) -> None:
    atol = 1e-5
    rtol = 1e-5
    threshold = atol + rtol * np.abs(expected).max()
    if diff >= threshold:
        raise ValueError(f"{name} mismatch: {diff}")


def run_apply_rms_prop_case(
    case: ApplyRMSPropTestCase,
    device_id: int | None = None,
    run_mode: str = "npu",
    config: RMSPropConfig = DEFAULT_CONFIG,
) -> None:
    device = resolve_device(device_id, run_mode)
    var_torch, ms_torch, mom_torch, grad_torch = create_test_tensors(case.shape, device)

    var_np = var_torch.cpu().numpy().copy()
    ms_np = ms_torch.cpu().numpy().copy()
    mom_np = mom_torch.cpu().numpy().copy()
    grad_np = grad_torch.cpu().numpy().copy()

    apply_rms_prop_kernel(var_torch, ms_torch, mom_torch, grad_torch, config)

    expected_var, expected_ms, expected_mom = apply_rms_prop_golden(var_np, ms_np, mom_np, grad_np, config)
    var_diff = np.abs(var_torch.cpu().numpy() - expected_var).max()
    ms_diff = np.abs(ms_torch.cpu().numpy() - expected_ms).max()
    mom_diff = np.abs(mom_torch.cpu().numpy() - expected_mom).max()

    ensure_within_tolerance("var", var_diff, expected_var)
    ensure_within_tolerance("ms", ms_diff, expected_ms)
    ensure_within_tolerance("mom", mom_diff, expected_mom)

    LOGGER.info(
        "%s: shape=%s, var_diff=%.8f, ms_diff=%.8f, mom_diff=%.8f",
        case.name.lower().split()[0],
        case.shape,
        var_diff,
        ms_diff,
        mom_diff,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PyPTO ApplyRMSProp Tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s              Run all tests
  %(prog)s level0       Run level 0 test only
  %(prog)s --list       List all available tests
        """,
    )
    parser.add_argument("test_id", type=str, nargs="?", help="Test ID to run. If not specified, all tests will run.")
    parser.add_argument("--list", action="store_true", help="List all available tests and exit")
    parser.add_argument(
        "--run_mode",
        type=str,
        nargs="?",
        default="npu",
        choices=["npu", "sim"],
        help="Run mode, such as npu/sim etc.",
    )
    return parser


def log_available_tests() -> None:
    LOGGER.info("")
    LOGGER.info("%s", "=" * 60)
    LOGGER.info("Available Tests")
    LOGGER.info("%s", "=" * 60)
    LOGGER.info("")
    for test_id, test_case in sorted(TEST_CASES.items()):
        LOGGER.info("  ID: %s", test_id)
        LOGGER.info("    name: %s", test_case.name)
        LOGGER.info("    description: %s", test_case.description)
        LOGGER.info("")


def get_selected_tests(test_id: str | None) -> list[tuple[str, ApplyRMSPropTestCase]]:
    if test_id is None:
        return list(TEST_CASES.items())
    if test_id not in TEST_CASES:
        raise InvalidTestIdError(f"Invalid test ID: {test_id}")
    return [(test_id, TEST_CASES[test_id])]


def prepare_device_id(run_mode: str) -> int | None:
    if run_mode != "npu":
        return None
    if importlib.util.find_spec("torch_npu") is None:
        raise DevicePreparationError("torch_npu is required when run_mode is npu.")
    importlib.import_module("torch_npu")
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    return device_id


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args()
    if args.list:
        log_available_tests()
        return 0
    try:
        selected_tests = get_selected_tests(args.test_id)
        device_id = prepare_device_id(args.run_mode)
        for _, test_case in selected_tests:
            run_apply_rms_prop_case(test_case, device_id, args.run_mode)
        if len(selected_tests) > 1:
            LOGGER.info("All ApplyRMSProp tests passed!")
        return 0
    except DevicePreparationError as exc:
        LOGGER.error("ERROR: %s", exc)
        return 1
    except InvalidTestIdError as exc:
        LOGGER.error("ERROR: %s", exc)
        LOGGER.error("Valid test IDs are: %s", ", ".join(sorted(TEST_CASES)))
        LOGGER.error("")
        LOGGER.error("Use --list to see all available tests.")
        return 1
    except Exception as exc:
        LOGGER.error("")
        LOGGER.error("Error: %s", exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
