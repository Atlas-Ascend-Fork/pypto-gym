#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""算子调用入口适配(仓级测试公共层).

按 pytest --mode 选择 jit / aclnn 路径:
    jit   -> import <op_kernel>/<op>.py 的宿主入口(默认与算子同名)
    aclnn -> import 预编译桥 aclnn_bridge_<op> 的 <op> 接口(签名与 jit 入口一致)

路径由根 conftest.py / pytest.ini 统一注入, 本模块不做 sys.path 操作.
历史算子入口名不合约定时, 用 jit_module/jit_attr 显式覆盖.
"""

import importlib
import json
import os


def get_mode() -> str:
    """返回当前测试模式(jit/aclnn), 由根 conftest 的 --mode 写入 GYM_TEST_MODE."""
    return os.environ.get("GYM_TEST_MODE", "jit")


def get_op_caller(op: str, *, jit_module: str = None, jit_attr: str = None,
                  aclnn_module: str = None, aclnn_attr: str = None):
    """返回算子在当前模式下的调用入口(torch.Tensor 接口)."""
    mode = get_mode()
    if mode == "jit":
        module_name = jit_module
        if not module_name:
            registry = os.environ.get("GYM_OPS_REGISTRY")
            if registry:
                with open(registry, "r", encoding="utf-8") as fh:
                    mapping = json.load(fh)
                module_name = mapping.get(op)
                if not module_name:
                    raise RuntimeError(f"op {op!r} not found in GYM_OPS_REGISTRY={registry}")
            else:
                module_name = op
        module = importlib.import_module(module_name)
        return getattr(module, jit_attr or op)
    if mode == "aclnn":
        module = importlib.import_module(aclnn_module or f"aclnn_bridge_{op}")
        return getattr(module, aclnn_attr or op)
    raise ValueError(f"unsupported test mode: {mode} (expected jit/aclnn)")
