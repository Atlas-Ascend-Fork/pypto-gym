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
"""AICPU kernel 交付物契约测试（无需板卡）。

对安装态 vendor 包断言本算子 AICPU kernel 的交付契约：
  1. libpypto_gym_aicpu_kernels.so 存在且为 AArch64 ELF64（AICPU 侧运行架构）；
  2. .so 内含本算子各 OpType 的注册名字符串（REGISTER_CPU_KERNEL / codegen 入口
     已链入的证据；按 NUL 结尾精确匹配，避免与同 .so 内其他算子的同前缀名误命中）；
  3. cust_aicpu_kernel.json 含本算子条目，且 kernelSo 指向实际 .so 名（防 codegen
     CLI 硬编码的源仓 .so 名未被合并脚本重写——历史真实缺陷）、engine/functionName/
     opKernelLib 符合 AICPU 自定义算子契约。

vendor 根定位：ASCEND_CUSTOM_OPP_PATH（stest 流程注入，逐段找含 op_impl/ 者）→
ASCEND_HOME_PATH/opp/vendors/pypto_gym；均未找到时优雅跳过（交付物测试依赖安装态，
用 fixture 级 skip 保证用例仍被 pytest 收集，不触发退出码 5）。
"""

import json
import os
import sys

import pytest

# 直跑自举: 借主算子 tests/_standalone_env.py 注入 vendor 环境(ASCEND_CUSTOM_OPP_PATH)
_PEER_TESTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "mixed_quant_sparse_flash_mla_pto_bnn", "tests")
sys.path.insert(0, _PEER_TESTS)
import _standalone_env  # noqa: E402,F401

pytestmark = pytest.mark.soc("950")

SO_REL = os.path.join("op_impl", "cpu", "aicpu_kernel", "impl", "libpypto_gym_aicpu_kernels.so")
JSON_REL = os.path.join("op_impl", "cpu", "config", "cust_aicpu_kernel.json")
SO_NAME = "libpypto_gym_aicpu_kernels.so"

#: 本算子 AICPU kernel 注册的 OpType（与 op_kernel_aicpu 的 json / codegen --op-type 对齐）
EXPECTED_OP_TYPES = ("MixedQuantSparseFlashMlaMetadataPtoBnn",)

EM_AARCH64 = 0xB7


def _vendor_root():
    for part in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(":"):
        if part and os.path.isdir(os.path.join(part, "op_impl")):
            return part
    home = os.environ.get("ASCEND_HOME_PATH", "")
    cand = os.path.join(home, "opp", "vendors", "pypto_gym") if home else ""
    return cand if cand and os.path.isdir(cand) else None


@pytest.fixture(scope="module")
def vendor():
    root = _vendor_root()
    if root is None:
        pytest.skip("no installed pypto_gym vendor; set ASCEND_CUSTOM_OPP_PATH (stest flow does)")
    return root


def test_aicpu_so_is_aarch64_elf(vendor):
    path = os.path.join(vendor, SO_REL)
    assert os.path.isfile(path), f"missing {path}"
    with open(path, "rb") as fh:
        head = fh.read(20)
    assert head[:4] == b"\x7fELF", "not an ELF file"
    assert head[4] == 2, "not ELF64"
    e_machine = int.from_bytes(head[18:20], "little")
    assert e_machine == EM_AARCH64, f"e_machine={e_machine:#x}, expected EM_AARCH64({EM_AARCH64:#x})"


def test_aicpu_so_registers_expected_op_types(vendor):
    path = os.path.join(vendor, SO_REL)
    with open(path, "rb") as fh:
        blob = fh.read()
    for op_type in EXPECTED_OP_TYPES:
        # NUL 结尾精确匹配注册名 C 字符串（同 .so 内存在同前缀 OpType 时不误命中）
        assert op_type.encode() + b"\x00" in blob, \
            f"{op_type} registration string not found in {os.path.basename(path)}"


def test_cust_aicpu_json_contract(vendor):
    path = os.path.join(vendor, JSON_REL)
    assert os.path.isfile(path), f"missing {path}"
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    for op_type in EXPECTED_OP_TYPES:
        assert op_type in doc, f"{op_type} missing from cust_aicpu_kernel.json"
        info = doc[op_type]["opInfo"]
        assert info["kernelSo"] == SO_NAME, (
            f"{op_type}: kernelSo={info['kernelSo']!r} != {SO_NAME!r} "
            "(merge must rewrite the codegen default)")
        assert info["engine"] == "DNN_VM_AICPU"
        assert info["functionName"] == "RunCpuKernel"
        assert info["opKernelLib"] == "CUSTAICPUKernel"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
