# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""CompressorV2 vector compression and cache update entry."""
import hashlib
import importlib.util
import sys
from pathlib import Path

import pypto_pro.language as pl

# OPC stages these two files together. Resolve the computation file explicitly
# so both staged builds and repository tests use this operator's own helpers.
_SHARED_PATH = Path(__file__).with_name("compressor_v2.py")
_SHARED_NAME = "_compressor_v2_shared_" + hashlib.sha256(str(_SHARED_PATH).encode()).hexdigest()[:16]
if _SHARED_NAME not in sys.modules:
    _spec = importlib.util.spec_from_file_location(_SHARED_NAME, _SHARED_PATH)
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[_SHARED_NAME] = _module
    _spec.loader.exec_module(_module)
_shared = sys.modules[_SHARED_NAME]
CompressorV2Tiling = _shared.CompressorV2Tiling
CompressorVectorKey = _shared.CompressorVectorKey
compress_cache = _shared.compress_cache


# CANN ABI: ten inputs, four outputs, workspace and tiling must remain separate parameters.
@pl.jit(arch="3510", auto_mutex=True, tiling_key=CompressorVectorKey,
        datatype={"x": "io_dtype", "wkv": "io_dtype", "wgate": "io_dtype", "out": "io_dtype"})
def compressor_v2_vector(
    x: pl.Ptr[pl.DT_UINT8], wkv: pl.Ptr[pl.DT_UINT8], wgate: pl.Ptr[pl.DT_UINT8],
    cache_in_ptr: pl.Ptr[pl.DT_FP32],
    table_ptr: pl.Ptr[pl.DT_INT32], cu_ptr: pl.Ptr[pl.DT_INT32],
    used_ptr: pl.Ptr[pl.DT_INT32], start_ptr: pl.Ptr[pl.DT_INT32],
    projected_in_ptr: pl.Ptr[pl.DT_FP32], history_in_ptr: pl.Ptr[pl.DT_FP32],
    out: pl.Ptr[pl.DT_UINT8], cache_out_ptr: pl.Ptr[pl.DT_FP32],
    projected_out_ptr: pl.Ptr[pl.DT_FP32], history_out_ptr: pl.Ptr[pl.DT_FP32],
    workspace: pl.Ptr[pl.DT_UINT8], tiling: CompressorV2Tiling,
):
    compress_cache(cache_out_ptr, table_ptr, cu_ptr, used_ptr, start_ptr,
                   projected_in_ptr, history_in_ptr, out, tiling)
