# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""测试脚本直跑自举: 定位仓根/tests 公共层/构建桥/安装态 vendor, 兜底注入 sys.path 与环境变量。

pytest 会由 pytest.ini(pythonpath=tests/common) 与根 conftest 注入这些路径;
直接 ``python test_xxx.py`` 执行时由本模块兜底。已存在的环境/路径不覆盖,
vendor 优先取 ASCEND_CUSTOM_OPP_PATH 已配置值, 其次仓内 build_out 安装态。
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_repo_root():
    root = _THIS_DIR
    for _ in range(8):
        if os.path.isfile(os.path.join(root, "pytest.ini")):
            return root
        root = os.path.dirname(root)
    return None


def _find_vendor(repo):
    # 1) 环境已配置(stest/用户手工) -> 不动
    for part in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(":"):
        if part and os.path.isdir(os.path.join(part, "op_impl")):
            return None
    # 2) 仓内安装态
    cand = os.path.join(repo, "build_out", "run", "ascend950", "vendors", "pypto_gym")
    if os.path.isdir(os.path.join(cand, "op_impl")):
        return cand
    return None


def _ensure_env():
    repo = _find_repo_root()
    if repo is None:
        return
    # tests/common: op_caller 等仓级公共层
    common = os.path.join(repo, "tests", "common")
    if os.path.isdir(common) and common not in sys.path:
        sys.path.insert(0, common)
    # aclnn 桥(pybind11 产物目录, 含 aclnn_bridge_<op>.so 的加载器)
    bridge = os.path.join(repo, "build", "ascend950", "tests")
    if os.path.isdir(bridge) and bridge not in sys.path:
        sys.path.append(bridge)
    # 安装态 vendor(aclnn 二进制 + AICPU .so + torch 绑定)
    vendor = _find_vendor(repo)
    if vendor is not None:
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = vendor
        lib = os.path.join(vendor, "op_api", "lib")
        os.environ["LD_LIBRARY_PATH"] = (
            lib if not os.environ.get("LD_LIBRARY_PATH")
            else lib + ":" + os.environ["LD_LIBRARY_PATH"]
        )
        py = os.path.join(vendor, "python")
        if os.path.isdir(py) and py not in sys.path:
            sys.path.append(py)
    # 直跑默认 aclnn 模式(op_caller 按 GYM_TEST_MODE 选入口; 可用环境变量覆盖)
    os.environ.setdefault("GYM_TEST_MODE", "aclnn")


_ensure_env()
