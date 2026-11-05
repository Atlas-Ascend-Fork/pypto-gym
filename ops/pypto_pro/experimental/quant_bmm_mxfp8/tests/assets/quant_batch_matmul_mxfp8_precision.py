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
"""All-elements comparison with this operator's numerical criteria."""

import logging
import sys
import numpy as np

_LOG = logging.getLogger("quant_bmm_mxfp8")
# Own handler only: never configure the root logger. pypto_pro logs through
# `logging.info(...)` on the root, so a basicConfig here would surface the
# framework's internal INFO lines (kernel generation, env dumps) that were
# previously dropped for lack of a root handler. The formatter is message-only
# and the stream is stdout, so the emitted text is byte-identical to print().
_LOG.setLevel(logging.INFO)
if not _LOG.handlers:
    _stream = logging.StreamHandler(sys.stdout)
    _stream.setFormatter(logging.Formatter("%(message)s"))
    _LOG.addHandler(_stream)
_LOG.propagate = False


def compare(actual, expected, rtol, atol):
    if actual.shape != expected.shape:
        raise AssertionError(f"Output shape {actual.shape}, expected {expected.shape}")
    a, e = actual.astype(np.float64), expected.astype(np.float64)
    diff = np.abs(a - e)
    finite = np.isfinite(a) & np.isfinite(e)
    bad = (~finite) | (diff > atol + rtol * np.abs(e))
    finite_diff = diff[finite]
    max_abs = float(finite_diff.max()) if finite_diff.size else float("inf")
    max_rel = float((diff[finite] / np.maximum(np.abs(e[finite]), 1e-30)).max()) if finite_diff.size else float("inf")
    rmse = float(np.sqrt(np.mean(finite_diff ** 2))) if finite_diff.size else float("inf")
    count = int(bad.sum())
    _LOG.info(f"  max_abs={max_abs:.8g} max_rel={max_rel:.8g} RMSE={rmse:.8g} "
              f"bad={count}/{a.size} nonfinite={int((~finite).sum())}")
    if count:
        for index in np.argwhere(bad)[:5]:
            ix = tuple(int(i) for i in index)
            _LOG.info(f"  mismatch {ix}: actual={a[ix]:.9g}, golden={e[ix]:.9g}, "
                      f"abs={diff[ix]:.9g}, limit={atol + rtol * abs(e[ix]):.9g}")
    return count == 0
