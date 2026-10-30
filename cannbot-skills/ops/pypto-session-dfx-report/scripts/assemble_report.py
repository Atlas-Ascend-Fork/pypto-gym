#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Assemble the final DFX report from LLM-authored body + script-rendered sections.

Usage:
    python3 assemble_report.py <op_dir> [--out <report.md>]

Reads:
  dfx/report_body.md      LLM-authored sections 0/1/4 (template-based)
  dfx/sections/2_time.md  script-rendered (deterministic)
  dfx/sections/3_token.md
  dfx/sections/5_exec.md
  dfx/sections/6_data.md

Ordering: body(0,1) → 2 → 3 → body(4) → 5 → 6
"""
import argparse
import logging
import os
import re
import sys

LOGGER = logging.getLogger(__name__)


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("op_dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    op_dir = os.path.abspath(args.op_dir)
    dfx = os.path.join(op_dir, "dfx")
    body = os.path.join(dfx, "report_body.md")
    if not os.path.isfile(body):
        LOGGER.error("%s not found — author it first (see templates/report-body-template.md)", body)
        return 1

    body_text = read_text(body)

    # split body into part_a (sections 0,1) and part_b (section 4)
    m = re.search(r"(?m)^## 4\.", body_text)
    if not m:
        LOGGER.error("report_body.md must contain '## 4. ...' heading")
        return 1
    part_a = body_text[:m.start()].rstrip() + "\n"
    part_b = body_text[m.start():].rstrip() + "\n"

    sections = {}
    missing = []
    for sec in ("2_time", "3_token", "5_exec", "6_data"):
        p = os.path.join(dfx, "sections", f"{sec}.md")
        if os.path.isfile(p):
            sections[sec] = read_text(p).rstrip() + "\n"
        else:
            missing.append(p)
    if missing:
        LOGGER.error("missing rendered sections (run extract_dfx.py first): %s", ", ".join(missing))
        return 1

    parts = [
        part_a,
        sections.get("2_time", ""),
        sections.get("3_token", ""),
        "\n---\n\n",
        part_b,
    ]
    for sec in ("5_exec", "6_data"):
        parts.append("\n---\n\n" + sections.get(sec, ""))

    out = args.out or os.path.join(dfx, os.path.basename(op_dir.rstrip("/")) + "_dfx_report.md")
    content = "".join(parts)
    with open(out, "w", encoding="utf-8") as f:
        f.write(content)
    LOGGER.info("OK: %s (%d bytes, %d lines)", out, os.path.getsize(out), len(content.splitlines()))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.exit(main())
