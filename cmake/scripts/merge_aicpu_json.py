#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Merge per-kernel AICPU opInfo json files into the vendor cust_aicpu_kernel.json.

Every PyPTO AICPU codegen artifact carries the shared kernel library name, so the
merge rewrites ``kernelSo`` to the vendor library built by this repository.
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--kernel-so", required=True)
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()

    merged = {}
    for path in args.inputs:
        data = json.loads(path.read_text())
        for op_type, info in data.items():
            if op_type in merged:
                raise SystemExit(f"duplicate AICPU op type {op_type} in {path}")
            info["opInfo"]["kernelSo"] = args.kernel_so
            merged[op_type] = info
    args.output.write_text(json.dumps(merged, indent=4) + "\n")


if __name__ == "__main__":
    main()
