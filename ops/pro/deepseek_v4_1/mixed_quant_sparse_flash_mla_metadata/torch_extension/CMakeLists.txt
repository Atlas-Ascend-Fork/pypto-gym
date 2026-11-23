# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
if(ENABLE_TORCH_BINDINGS AND NOT PREPARE_BUILD)
    gym_add_torch_binding(mixed_quant_sparse_flash_mla_metadata_pto_bnn)
endif()
