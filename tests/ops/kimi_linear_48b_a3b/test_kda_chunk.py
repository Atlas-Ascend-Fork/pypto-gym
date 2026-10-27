# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Numerical test for the kimi_linear_48b_a3b KDA chunk/prefill PyPTO kernel."""

import logging
import os

import numpy as np
import torch

import pypto

from _kda_test_common import _HAS_NPU, torch_npu, _naive_recurrent_kda, make_kda_inputs, chunk_grid

from kda_chunk_impl import kda_chunk_wrapper


def run_case(case, dev):
    """Run one chunk-kernel case and assert it matches the golden recurrent op."""
    q, k, v, g, beta, h0, scale = make_kda_inputs(case, dev)

    og, sg = _naive_recurrent_kda(q, k, v, g, beta, initial_state=h0,
                                  output_final_state=True,
                                  use_qk_l2norm_in_kernel=True, scale=scale)
    op, sp = kda_chunk_wrapper(q, k, v, g, beta, h0, scale=scale)

    name = f"B{case.batch}H{case.num_heads}T{case.seq_len}_state{case.with_state}_g{case.gate}"
    out_diff = (op.cpu().float() - og.cpu().float()).abs().max().item()
    st_diff = (sp.cpu().float() - sg.cpu().float()).abs().max().item()
    logging.info(f"[{name}] out.max_diff={out_diff:.3e} state.max_diff={st_diff:.3e}")

    np.testing.assert_allclose(op.cpu().float().numpy(), og.cpu().float().numpy(),
                               rtol=6e-3, atol=6e-3, err_msg=f"{name}: out mismatch")
    np.testing.assert_allclose(sp.cpu().float().numpy(), sg.cpu().float().numpy(),
                               rtol=6e-3, atol=6e-3, err_msg=f"{name}: state mismatch")
    return max(out_diff, st_diff)


def test_kda_chunk():
    if not _HAS_NPU:
        return
    torch_npu.npu.config.allow_internal_format = True
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    torch.npu.set_device(device_id)
    dev = f"npu:{device_id}"

    max_err = 0.0
    # chunk_grid() is the single source of truth (enforced by test_cases_sync.py).
    for case in chunk_grid():
        max_err = max(max_err, run_case(case, dev))

    logging.info(f"\n[summary] max abs error across all cases = {max_err:.3e}")


def main():
    pypto.set_host_options(compile_monitor_enable=1, compile_timeout=10,
                           compile_timeout_stage=5,
                           compile_monitor_print_interval=2)
    test_kda_chunk()
    logging.info("[PRECISION_PASS]")
    logging.info("All tests passed!")


if __name__ == "__main__":
    if _HAS_NPU:
        main()
    else:
        logging.info("skip: no NPU")
