#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Pytest 配置控制
"""
import os
import sys
from pathlib import Path

import torch
import pytest
from typing import Optional  # 必须加在 conftest 顶部！
import logging  # 顶部加


# 算子测试资产路径注入(集中在此, 各用例文件无需 sys.path 操作):
#   <op>/tests/         用例间互相 import / 算子级共享 helper
#   <op>/tests/assets/  golden/参考实现(模块名以算子名作前缀, 避免跨算子撞名)
#   <op>/op_kernel/     jit 模式的 kernel 宿主入口
for _tests_dir in sorted((Path(__file__).resolve().parent / "ops").rglob("tests")):
    for _path_dir in (_tests_dir, _tests_dir / "assets", _tests_dir.parent / "op_kernel"):
        _path_str = str(_path_dir)
        if _path_dir.is_dir() and _path_str not in sys.path:
            sys.path.insert(0, _path_str)


def duration_estimate(seconds: float):
    """
    Decorator: annotate a test case with estimated duration (seconds).

    This decorator marks test cases with their expected execution time,
    allowing pytest to reorder tests for optimal parallel execution.

    Args:
        seconds: Estimated execution time in seconds

    Example:
        @duration_estimate(120)
        def test_something():
            ...
    """
    def decorator(func):
        func.duration_estimate = seconds
        return func
    return decorator


def _set_process_desc(desc: str):
    try:
        import setproctitle
        setproctitle.setproctitle(desc)
    except ModuleNotFoundError:
        pass


@pytest.fixture
def device():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    return f"npu:{device_id}"


@pytest.fixture
def device_id():
    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)
    return device_id


def pytest_configure(config):
    """按 --mode 注入运行时环境(sys.path / 环境变量), 供 op_caller 消费."""
    mode = config.getoption("--mode", "jit")
    os.environ["GYM_TEST_MODE"] = mode

    if mode == "aclnn":
        lib_dir = config.getoption("--aclnn-lib-dir", "")
        if not lib_dir:
            soc = os.environ.get("GYM_SOC", "ascend950")
            repo_root = os.path.dirname(os.path.abspath(__file__))
            lib_dir = os.path.join(repo_root, "build", soc, "tests")
        if os.path.isdir(lib_dir):
            sys.path.insert(0, lib_dir)
        else:
            pytest.exit(
                f"[conftest] pybind11 bridge dir not found: {lib_dir}\n"
                f"  - build with ENABLE_TESTS=ON, or\n"
                f"  - pass --aclnn-lib-dir=<path>, or\n"
                f"  - run with --mode jit (default)",
                returncode=1)


def pytest_addoption(parser: pytest.Parser):
    parser.addoption("--device", nargs="+", type=int,
                     help="Device ID, default 0")
    parser.addoption(
        "--test_case_info", action="store", default="", help="Test case info."
    )
    parser.addoption(
        "--cards-per-case", type=int, default=1,
        help="Number of cards required for each test case. Default is 1 (single-card cases)."
    )
    parser.addoption(
        "--mode", choices=["jit", "aclnn"], default="jit",
        help="Test execution mode: jit (Python host wrapper) / aclnn (C++ aclnn via pybind11 bridge)"
    )
    parser.addoption(
        "--aclnn-lib-dir", default="",
        help="Directory containing pybind11 bridge .so (default: <repo>/build/<soc>/tests/)"
    )


def _is_case_match_cards(item, target_cards) -> bool:
    cards_marker = item.get_closest_marker("world_size")
    if cards_marker is None:
        return True
    required_cards = cards_marker.args
    if not required_cards:
        return True
    if isinstance(required_cards[0], int):
        return target_cards == required_cards[0]
    return True


@pytest.hookimpl(optionalhook=True)
def pytest_configure_node(node):
    """Configure devices for each worker node"""
    print(f"========== pytest_configure_node called for node {node.gateway.id} ==========", flush=True)

    # 从环境变量获取设备列表
    available_devices_env = os.environ.get("PYTEST_AVAILABLE_DEVICES", "")
    if not available_devices_env:
        return

    device_id_lst = [int(d.strip()) for d in available_devices_env.split(",") if d.strip()]
    cards_per_case: int = node.config.getoption("--cards-per-case", 1)

    worker_id = str(node.gateway.id)
    worker_idx = int(worker_id.lstrip("gw"))

    if cards_per_case > 1:
        num_groups = len(device_id_lst) // cards_per_case
        if worker_idx >= num_groups:
            return
        start_idx = worker_idx * cards_per_case
        end_idx = start_idx + cards_per_case
        device_group = device_id_lst[start_idx:end_idx]
        device_group_str = ",".join(map(str, device_group))

        # 使用 remote_exec 设置环境变量
        node.gateway.remote_exec(
            f'import os; os.environ["ASCEND_VISIBLE_DEVICES"] = "{device_group_str}"'
        )
        node.gateway.remote_exec(
            f'import os; os.environ["TILE_FWK_DEVICE_ID_LIST"] = "{device_group_str}"'
        )
    else:
        if worker_idx >= len(device_id_lst):
            return
        device_id = device_id_lst[worker_idx]

        node.gateway.remote_exec(
            f'import os; os.environ["ASCEND_VISIBLE_DEVICES"] = "{device_id}"'
        )
        node.gateway.remote_exec(
            f'import os; os.environ["TILE_FWK_DEVICE_ID"] = "{device_id}"'
        )


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(item, nextitem):
    device_list_str: Optional[str] = os.environ.get("TILE_FWK_DEVICE_ID_LIST", None)
    if device_list_str is not None:
        device_list = device_list_str.split(",")
        _set_process_desc(f"Devices[{','.join(device_list)}]")
    else:
        device_id: Optional[str] = os.environ.get("TILE_FWK_DEVICE_ID", None)
        if device_id is not None:
            _set_process_desc(f"Device[{device_id}]")
    return None


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    device_list_str: Optional[str] = os.environ.get("TILE_FWK_DEVICE_ID_LIST", None)
    case_name: str = str(item.name)
    if device_list_str is not None:
        device_list = device_list_str.split(",")
        _set_process_desc(f"Case(Devices[{','.join(device_list)}]::{case_name})")
    else:
        device_id: Optional[str] = os.environ.get("TILE_FWK_DEVICE_ID", None)
        if device_id is not None:
            _set_process_desc(f"Case(Device[{device_id}]::{case_name})")
    return None


def _get_test_time_cost(item):
    if hasattr(item.function, 'duration_estimate'):
        return item.function.duration_estimate
    if hasattr(item, 'cls') and item.cls and hasattr(item.cls, 'duration_estimate'):
        return item.cls.duration_estimate
    time_marker = item.get_closest_marker("duration_estimate")
    if time_marker and time_marker.args:
        return time_marker.args[0]
    return None


def _get_soc_version():
    try:
        import torch_npu
        soc_version = torch_npu.npu.get_soc_version()
        return soc_version
    except Exception as e:
        pytest.exit(f"Error: Failed to get soc version, error info: {str(e)}", returncode=1)
        return None


def _is_case_match_soc(item, target_soc):
    soc_marker = item.get_closest_marker("soc")
    if soc_marker is None:
        supported_socs = ["910"]
    else:
        supported_socs = soc_marker.args
        if isinstance(supported_socs[0], str):
            supported_socs = [soc.strip() for soc in supported_socs]
        elif isinstance(supported_socs[0], list):
            supported_socs = [soc.strip() for soc in supported_socs[0]]
    if target_soc == 260:
        target_tag = "950"
    else:
        target_tag = "910"
    return target_tag in supported_socs


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    if not items:
        return

    target_soc = _get_soc_version()
    filtered_items = [item for item in items if _is_case_match_soc(item, target_soc)]

    cards_per_case = config.getoption("--cards-per-case", 1)
    card_filtered_items = [item for item in filtered_items
                          if _is_case_match_cards(item, cards_per_case)]

    timed_tests = []
    untimed_tests = []
    for item in card_filtered_items:
        time_cost = _get_test_time_cost(item)
        if time_cost is not None:
            timed_tests.append((item, time_cost))
        else:
            untimed_tests.append(item)

    timed_tests.sort(key=lambda x: x[1], reverse=True)
    reordered_items = [item for item, _ in timed_tests] + untimed_tests

    items[:] = reordered_items