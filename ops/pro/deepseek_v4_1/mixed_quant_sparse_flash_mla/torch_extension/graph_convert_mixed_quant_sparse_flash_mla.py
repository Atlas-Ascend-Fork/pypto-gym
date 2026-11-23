# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software and you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
# -----------------------------------------------------------------------------------------------------------
"""torchair GE 图/aclgraph 转换器注册（参考 ops-transformer graph_convert_* 模式）。

作用（与 cann_ops_transformer 的 quant_flash_attn_grad / flash_mla 一致）:

1. 在 torchair 的 fx2ge 转换表里注册本算子(``torch.ops.pypto_gym.mixed_quant_sparse_flash_mla``),
   使 torch.compile(torchair backend) 遇到该节点时不再报 "unsupported op", 而是得到
   明确的 "GE not supported!" —— 本算子只支持 aclgraph(NPUGraph 捕获)路径, 不提供 GE
   算子下沉;
2. ``auto_convert_to_tensor`` 声明算子签名中张量/可选张量形参, 供 torchair 假值追踪使用。

torchair 未安装时本模块为纯 no-op(``_TORCHAIR_AVAILABLE = False``), 不影响 eager /
NPUGraph 两条路径。
"""

try:
    import torch  # noqa: F401
    import torch_npu  # noqa: F401
    import torchair
    from torchair._ge_concrete_graph.fx2ge_converter import (
        register_fx_node_ge_converter,
    )
    from torchair.ge._ge_graph import Tensor, auto_convert_to_tensor
    from typing import Optional

    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False

if _TORCHAIR_AVAILABLE:

    # 两个转换函数的签名/命名是 torchair 转换器 ABI: auto_convert_to_tensor 按签名
    # 注解定位张量形参, 函数名镜像算子类型名(与 ops-transformer graph_convert_*
    # 同款 PascalCase) —— G.FNM.03 参数数/G.NAM.01 命名为算子签名固有, 不可重构。
    @auto_convert_to_tensor(
        # 18 个张量形参(签名顺序): 8 个必选(含 tiling_device) + 8 个可选输入 + out/lse 两个可选输出;
        # 第一组标记张量形参位置, 第二组标记可选(None-able)张量。
        [False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False,
         False, False, False],
        [False, False, False, False, False, False, False,
         False, True, True, True, True, True, True, True,
         True, True, True],
    )
    def MixedQuantSparseFlashMla(
        q: Tensor,
        ori_kv: Tensor,
        ori_sparse_indices: Tensor,
        ori_block_table: Tensor,
        cu_seqlens_q: Tensor,
        sinks: Tensor,
        metadata: Tensor,
        tiling_device: Tensor,
        ori_topk_length: Optional[Tensor] = None,
        cmp_kv: Optional[Tensor] = None,
        cmp_sparse_indices: Optional[Tensor] = None,
        cmp_block_table: Optional[Tensor] = None,
        cmp_topk_length: Optional[Tensor] = None,
        seqused_q: Optional[Tensor] = None,
        seqused_ori_kv: Optional[Tensor] = None,
        seqused_cmp_kv: Optional[Tensor] = None,
        out: Optional[Tensor] = None,
        lse: Optional[Tensor] = None,
        quant_mode: int = 1,
        softmax_scale: float = 0.0,
        layout_q: str = "TND",
        layout_kv: str = "PA_BBND",
        return_softmax_lse: bool = False,
        block_dim: Optional[int] = None,
        fd_enable: Optional[int] = None,
    ):
        result = q.new_empty(q.size())
        lse_result = q.new_empty((1, q.size(0), q.size(1)))
        return result, lse_result

    @register_fx_node_ge_converter(
        torch.ops.pypto_gym.mixed_quant_sparse_flash_mla.default
    )
    def convert_mixed_quant_sparse_flash_mla(
        q: Tensor,
        ori_kv: Tensor,
        ori_sparse_indices: Tensor,
        ori_block_table: Tensor,
        cu_seqlens_q: Tensor,
        sinks: Tensor,
        metadata: Tensor,
        tiling_device: Tensor,
        ori_topk_length: Optional[Tensor] = None,
        cmp_kv: Optional[Tensor] = None,
        cmp_sparse_indices: Optional[Tensor] = None,
        cmp_block_table: Optional[Tensor] = None,
        cmp_topk_length: Optional[Tensor] = None,
        seqused_q: Optional[Tensor] = None,
        seqused_ori_kv: Optional[Tensor] = None,
        seqused_cmp_kv: Optional[Tensor] = None,
        quant_mode: int = 1,
        softmax_scale: float = 0.0,
        layout_q: str = "TND",
        layout_kv: str = "PA_BBND",
        return_softmax_lse: bool = False,
        out: Optional[Tensor] = None,
        lse: Optional[Tensor] = None,
        block_dim: Optional[int] = None,
        fd_enable: Optional[int] = None,
    ):
        raise AssertionError("GE not supported!")
