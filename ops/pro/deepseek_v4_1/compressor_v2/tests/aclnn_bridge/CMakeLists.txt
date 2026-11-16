# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
gym_add_aclnn_bridge(compressor_v2)
target_include_directories(aclnn_bridge_compressor_v2 PRIVATE
    "${ASCEND_AUTOGEN_DIR}"
    "${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/include")
add_dependencies(aclnn_bridge_compressor_v2 opbuild_gen_default)
target_link_libraries(aclnn_bridge_compressor_v2 PRIVATE
    "${ASCEND_CANN_PACKAGE_PATH}/lib64/libascendcl.so")
