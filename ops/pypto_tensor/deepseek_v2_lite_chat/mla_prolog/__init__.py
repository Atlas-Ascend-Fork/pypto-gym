# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""
DeepSeek-V2-Lite MLA KV Prolog - PyPTO算子库适配层

使用sys.modules注入方式集成到整网

替换位置：DeepseekV2MLA.ml_a (modeling_deepseek.py)

替换代码片段：
    # 原始实现
    compressed_kv = self.kv_a_proj(hidden_states)
    k_pe = compressed_kv[:, :, self.kv_lora_rank:]
    compressed_kv = compressed_kv[:, :, :self.kv_lora_rank]
    compressed_kv_norm = self.kv_a_layernorm(compressed_kv)
    
    # PyPTO替换（移除第一个matmul版本）
    if deepseek_v2_lite_pto_kernels.USE_PTO_MLA_PROLOG:
        return deepseek_v2_lite_pto_kernels.mla_prolog_hybrid(...)
"""

import logging
import sys

# 全局开关（按算子粒度）
USE_PTO_MLA_PROLOG = False

# 导入算子实现
try:
    from .mla_prolog import mla_prolog_hybrid_optimized
    MLA_PROLOG_AVAILABLE = True
except Exception as e:
    logging.warning(f"[WARNING] MLA Prolog PyPTO operator import failed: {e}")
    MLA_PROLOG_AVAILABLE = False

# 导入golden（fallback）
try:
    from .mla_prolog_baseline_torch import mla_prolog_torch_baseline
    MLA_PROLOG_GOLDEN_AVAILABLE = True
except Exception as e:
    logging.warning(f"[WARNING] MLA Prolog Golden import failed: {e}")
    MLA_PROLOG_GOLDEN_AVAILABLE = False

# 导入动态选择策略
try:
    from .mla_prolog_dynamic_selection import mla_prolog_dynamic_selection, SEQ_LEN_THRESHOLD
    DYNAMIC_SELECTION_AVAILABLE = True
except Exception as e:
    logging.warning(f"[WARNING] MLA Prolog dynamic selection import failed: {e}")
    DYNAMIC_SELECTION_AVAILABLE = False


def mla_prolog_wrapper(hidden_states, kv_a_weight, kv_b_weight, ln_weight, eps, cos, sin, pos_ids):
    """
    MLA Prolog wrapper（支持动态选择）

    根据序列长度自动选择最优方案：
    - 短序列（≤20 tokens）: PyPTO融合算子（+5.9%优势）
    - 长序列（>20 tokens）: Baseline（+10.6%优势）

    参数：
        hidden_states: [bsz, seq_len, hidden_size]
        kv_a_weight: [kv_lora_rank + rope_dim, hidden_size]
        kv_b_weight: [num_heads * 256, kv_lora_rank]
        ln_weight: [kv_lora_rank]
        eps: float
        cos: [seq_len, rope_dim]
        sin: [seq_len, rope_dim]
        pos_ids: [bsz, seq_len]

    返回：
        k_nope: [bsz, num_heads, seq_len, qk_nope_head_dim]
        value: [bsz, num_heads, seq_len, v_head_dim]
        k_pe: [bsz, 1, seq_len, rope_dim]
    """
    if USE_PTO_MLA_PROLOG and DYNAMIC_SELECTION_AVAILABLE:
        try:
            # 使用动态选择策略（最优性能）
            return mla_prolog_dynamic_selection(
                hidden_states, kv_a_weight, kv_b_weight, ln_weight, eps, cos, sin, pos_ids,
                use_pypto=True
            )
        except Exception as exc:
            logging.error(f"[ERROR] MLA Prolog dynamic selection failed, fallback to golden: {exc}")
            if MLA_PROLOG_GOLDEN_AVAILABLE:
                return mla_prolog_torch_baseline(
                    hidden_states, kv_a_weight, kv_b_weight, ln_weight, eps, cos, sin, pos_ids
                )
            else:
                raise
    else:
        # Fallback到golden实现
        if MLA_PROLOG_GOLDEN_AVAILABLE:
            return mla_prolog_torch_baseline(
                hidden_states, kv_a_weight, kv_b_weight, ln_weight, eps, cos, sin, pos_ids
            )
        else:
            raise RuntimeError("MLA Prolog算子不可用（PyPTO和Golden均失败）")

# 兼容旧命名
mla_prolog_wrapper = mla_prolog_wrapper

__all__ = [
    'USE_PTO_MLA_PROLOG',
    'MLA_PROLOG_AVAILABLE',
    'MLA_PROLOG_GOLDEN_AVAILABLE',
    'mla_prolog_wrapper',
    'mla_prolog_wrapper',
]
