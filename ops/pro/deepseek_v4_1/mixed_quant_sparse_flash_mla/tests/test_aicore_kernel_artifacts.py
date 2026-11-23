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
"""AI Core kernel 交付物契约测试（无需板卡）。

对安装态 vendor 包断言主算子 AI Core kernel 二进制的交付契约：
  1. kernel/<soc>/<op_file>/ 下 .o 与 .json 成对存在，.o 非空且为 ELF；
  2. 二进制 json 契约：kernelName 前缀 = OpType、coreType=MIX（Cube+Vector 融合）、
     intercoreSync=1、kernelList 非空且逐项含 tilingKey（多分支编译完整）、
     supportSuperKernel == 1 —— SK 资格字段（由 @pl.jit(sk_cap=...) 驱动产出，
     ACLGraph superkernel 融合读取；丢失即 SK 配置回退）；
  3. json.sha256 与 .o 实际 SHA-256 一致（交付物完整性；捕获"源变更后二进制
     未重编/陈旧产物入包"一类缺陷——本仓历史上真实发生过 .so 不重链事故）；
  4. binary_info_config.json 收录本算子（runtime 二进制索引）。

vendor 根定位与跳过语义同 test_aicpu_kernel_artifacts.py。
"""

import glob
import hashlib
import json
import os
import sys

import pytest

import _standalone_env  # noqa: F401  直跑自举: 注入 vendor 环境(ASCEND_CUSTOM_OPP_PATH)

pytestmark = pytest.mark.soc("950")

OP_FILE = "mixed_quant_sparse_flash_mla"
OP_TYPE = "MixedQuantSparseFlashMla"
KERNEL_GLOB = os.path.join("op_impl", "ai_core", "tbe", "kernel", "*", OP_FILE)
CONFIG_GLOB = os.path.join("op_impl", "ai_core", "tbe", "kernel", "config", "*", "binary_info_config.json")


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


def _bin_pairs(vendor):
    pairs = []
    for json_path in sorted(glob.glob(os.path.join(vendor, KERNEL_GLOB, "*.json"))):
        pairs.append((json_path[: -len(".json")] + ".o", json_path))
    return pairs


def test_kernel_binaries_exist(vendor):
    pairs = _bin_pairs(vendor)
    assert pairs, f"no kernel binary json under {vendor}/{KERNEL_GLOB}"
    for o_path, _ in pairs:
        assert os.path.isfile(o_path) and os.path.getsize(o_path) > 0, f"missing/empty {o_path}"
        with open(o_path, "rb") as fh:
            assert fh.read(4) == b"\x7fELF", f"{o_path} is not ELF"


def test_binary_json_contract(vendor):
    for _, json_path in _bin_pairs(vendor):
        with open(json_path, encoding="utf-8") as fh:
            doc = json.load(fh)
        name = os.path.basename(json_path)
        assert doc["kernelName"].startswith(OP_TYPE), f"{name}: kernelName={doc['kernelName']!r}"
        assert doc["coreType"] == "MIX", f"{name}: coreType={doc['coreType']!r}"
        assert int(doc["intercoreSync"]) == 1, f"{name}: intercoreSync != 1"
        assert int(doc.get("supportSuperKernel", 0)) == 1, \
            f"{name}: supportSuperKernel != 1 (sk_cap not in effect)"
        kernel_list = doc.get("kernelList") or []
        assert kernel_list, f"{name}: empty kernelList"
        for entry in kernel_list:
            assert entry["kernelName"].startswith(OP_TYPE), f"{name}: bad kernelList entry {entry}"
            assert "tilingKey" in entry, f"{name}: kernelList entry without tilingKey"


def test_binary_sha256_matches_object(vendor):
    for o_path, json_path in _bin_pairs(vendor):
        with open(json_path, encoding="utf-8") as fh:
            expected = json.load(fh).get("sha256")
        if not expected:
            continue
        with open(o_path, "rb") as fh:
            actual = hashlib.sha256(fh.read()).hexdigest()
        assert actual == expected, \
            f"{os.path.basename(o_path)}: sha256 mismatch (stale binary packaged?)"


def test_binary_info_config_indexes_op(vendor):
    cfgs = glob.glob(os.path.join(vendor, CONFIG_GLOB))
    assert cfgs, "no binary_info_config.json in vendor"
    indexed = False
    for cfg in cfgs:
        with open(cfg, encoding="utf-8") as fh:
            if OP_TYPE in json.load(fh):
                indexed = True
                break
    assert indexed, f"{OP_TYPE} not indexed in binary_info_config.json"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
