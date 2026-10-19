## GLM-4.5 模型 PyPTO 算子替换指南

本文选取 `ops/pypto_tensor/glm_v4_5/` 目录下 `glm_gate_impl.py` 文件中的 `gate` 算子作为典型案例（其它算子替换逻辑完全相同），重点介绍基于 vllm 工程的 PyPTO 算子便捷替换方案，旨在为 GLM-4.5 模型整网融合后的算子适配工作提供实践支撑。

### 1. 调整目录结构

1. 新建 `glm_pto_kernels` 目录，该目录与现有 `vllm`、`vllm_ascend` 目录同级；
2. 将 `ops/pypto_tensor/glm_v4_5/` 目录下的 `glm_gate_impl.py` 文件复制至新建的 `glm_pto_kernels` 目录中，并重命名为 `glm_gate.py`；
3. 在 `glm_pto_kernels` 目录下新建空的 `__init__.py` 文件。

- 调整后的完整目录结构如下：

```
glm-net/
├── glm_pto_kernels/
│   ├── __init__.py
│   └── glm_gate.py
├── vllm/
│   └── ......
└── vllm_ascend/
    └── ......
```

### 2. 配置算子适配层接口

在 `glm_pto_kernels` 目录下新增的 `__init__.py` 文件中，定义算子适配层接口，作为原生代码与 PyPTO 算子之间的衔接层。

示例：`gate` 算子适配层定义，在 `glm_pto_kernels/__init__.py` 中添加如下函数：

```
def gate(gate_layer, hidden_states):
    # 导入 PyPTO 实现的 gate 算子
    from glm_pto_kernels.glm_gate import gate as gate_pto

    # 获取输入张量的批次大小（bs）和门控层权重的维度（ne）
    bs = hidden_states.shape[0]
    ne = gate_layer.weight.shape[0]

    # 初始化输出张量：维度为(bs, ne)，数据类型/设备与门控层权重保持一致
    router_logits_res = torch.empty(
        (bs, ne),
        dtype=gate_layer.weight.dtype,
        device=hidden_states.device
    )

    # 调用 PyPTO 实现的 gate 算子，完成核心计算
    gate_pto(
        gate_layer.weight,
        hidden_states,
        router_logits_res
    )

    # 返回计算结果
    return router_logits_res
```

### 3. 修改算子调用逻辑

打开目标文件 `vllm/model_executor/models/glm4_moe.py`，在文件顶部的导入区域新增如下代码，引入 PyPTO 算子适配层库：

```
import glm_pto_kernels
```

在该文件中定位到 `Glm4MoE.forward()` 方法，找到 `gate` 算子的原生调用代码，替换 PyPTO 实现的 `gate` 算子的调用代码：

- 原调用代码

```
router_logits = self.gate(hidden_states.to(dtype=torch.float32))
```

- 替换后代码

```
router_logits = glm_pto_kernels.gate(self.gate, hidden_states.to(dtype=torch.float32))
```

完成以上修改并保存文件后，即可实现 `gate` 算子从原生实现到 PyPTO 算子实现的切换，模型运行时会调用 PyPTO 实现的 `gate` 算子逻辑。


### 4. 其它算子适配层接口
以下提供 `ops/pypto_tensor/glm_v4_5/` 目录下各类算子的适配层接口定义（`glm_pto_kernels/` 内对应的文件名沿用注释中目标工程的命名约定），将对应函数直接复制到 `glm_pto_kernels/__init__.py` 文件中即可完成适配层配置；每个函数注释内均标注了目标替换文件、函数及具体替换方法，按说明操作即可完成算子切换。

- glm_attention_impl.py 相关的算子适配层函数
```
# 配置算子开关，可灵活切换 PyPTO 或是原始方案
USE_PTO_FA              = True

# 函数定义以及适配说明
def paged_attention(query, key_cache, value_cache, block_tables, actual_seqs_cpu, output):
    '''
    [适配说明]
    目标文件: vllm_ascend/attention/attention_v1.py
    目标函数: _forward_decode_only()
    替换方法: 将torch_npu._npu_paged_attention()调用替换为当前函数, 替换后代码如下, 缩进已调整为与源码一致，可直接复制:
                 # <----复制开始---->
                 if glm_pto_kernels.USE_PTO_FA:
                     # PTO内核实现
                     glm_pto_kernels.paged_attention(
                         query, self.key_cache, self.value_cache,
                         attn_metadata.block_tables, attn_metadata.seq_lens, output
                     )
                 else:
                     # 原始方案
                     torch_npu._npu_paged_attention(
                         query=query,
                         key_cache=self.key_cache,
                         value_cache=self.value_cache,
                         num_kv_heads=self.num_kv_heads,
                         num_heads=self.num_heads,
                         scale_value=self.scale,
                         block_table=attn_metadata.block_tables,
                         context_lens=attn_metadata.seq_lens,
                         out=output)
                 # <----复制结束---->
    '''
    from glm_pto_kernels.glm_attention import attention
    attention(
        query, key_cache, value_cache, block_tables, actual_seqs_cpu, output
    )
    return output
```

- glm_attention_pre_quant_impl.py 相关的算子适配层函数
```
# 配置算子开关，可灵活切换 PyPTO 或是原始方案
USE_PTO_FA_PRE          = True

# 函数定义以及适配说明
def attention_pre(hidden_states, residual, layer, attention, positions):
    '''
    [适配说明]
    目标文件: vllm/model_executor/models/glm4_moe.py
    目标函数: Glm4MoeDecoderLayer.forward()
    替换方法:
    1. 修改Glm4MoeAttention.forward()，替换为如下代码, 缩进已调整为与源码一致，可直接复制:
    # <----复制开始---->
    def forward(
        self,
        layer,  # 增加layer输入
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,  # 增加residual输入
    ) -> torch.Tensor:
        if glm_pto_kernels.USE_PTO_FA_PRE:
            # PTO内核实现
            q, k, v, residual = glm_pto_kernels.attention_pre(hidden_states, residual, layer, self)
        else:
            # 原始方案
            qkv, _ = self.qkv_proj(hidden_states)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            if self.use_qk_norm:
                q = self.q_norm(q.reshape(-1, self.num_heads,
                                        self.head_dim)).reshape(q.shape)
                k = self.k_norm(k.reshape(-1, self.num_kv_heads,
                                        self.head_dim)).reshape(k.shape)

            q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output, residual  # 增加residual输出
    # <----复制结束---->

    2. 将input_layernorm()与self_attn()合并，替换为如下代码, 缩进已调整为与源码一致，可直接复制:
        # <----复制开始---->
        if glm_pto_kernels.USE_PTO_FA_PRE:
            # PTO内核实现
            hidden_states, residual = self.self_attn(self, positions=positions,
                                        hidden_states=hidden_states, residual=residual)
        else:
            # 原始方案
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(
                    hidden_states, residual)
            hidden_states = self.self_attn(self, positions=positions,
                                        hidden_states=hidden_states, residual=residual)
        # <----复制结束---->
    '''
    from glm_pto_kernels.glm_attention_pre_quant import attention_pre_quant as attention_pre_quant_pto
    cos, sin = attention.rotary_emb.cos_sin_cache.index_select(0, positions).chunk(2, dim=-1)
    cos = cos.unsqueeze(1).contiguous()
    sin = sin.unsqueeze(1).contiguous()
    bs = hidden_states.shape[0]
    q = torch.empty((bs, attention.q_size), dtype=hidden_states.dtype, device=f'{hidden_states.device}')
    k = torch.empty((bs, attention.kv_size), dtype=hidden_states.dtype, device=f'{hidden_states.device}')
    v = torch.empty((bs, attention.kv_size), dtype=hidden_states.dtype, device=f'{hidden_states.device}')
    residual_res = torch.empty((bs, hidden_states.shape[1]),
                               dtype=hidden_states.dtype, device=f'{hidden_states.device}')
    if residual is None:
        residual = torch.zeros((bs, hidden_states.shape[1]),
                               dtype=hidden_states.dtype, device=f'{hidden_states.device}')
    attention_pre_quant_pto(
        hidden_states, residual,
        layer.input_layernorm.weight.data,
        layer.input_layernorm.bias.data,
        attention.qkv_proj.aclnn_input_scale_reciprocal.data,
        attention.qkv_proj.aclnn_input_offset.data,
        attention.qkv_proj.weight.data,
        attention.qkv_proj.quant_bias.data,
        attention.qkv_proj.deq_scale.data,
        attention.q_norm.weight.data,
        attention.q_norm.bias.data,
        attention.k_norm.weight.data,
        attention.k_norm.bias.data,
        cos, sin, q, k, v, residual_res
    )

    return q, k, v, residual_res
```

- glm_ffn_shared_expert_quant_impl.py 相关的算子适配层函数
```
# 配置算子开关，可灵活切换 PyPTO 或是原始方案
USE_PTO_SHARE_EXEPERTS  = True

# 函数定义以及适配说明
def ffn_share_expert_quant(layer, hidden_states):
    '''
    [适配说明]
    目标文件: glm_pto_kernels/glm_ffn_shared_expert_quant.py
    目标函数: ffn_share_expert_quant()
    替换方法: 将_native_select_experts()调用替换为当前函数, 替换后代码如下, 缩进已调整为与源码一致，可直接复制:
        # <----复制开始---->
        if glm_pto_kernels.USE_PTO_SHARE_EXEPERTS and self.is_down_proj_quant:
            return glm_pto_kernels.ffn_share_expert_quant(self, x)
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x
        # <----复制结束---->
    '''
    from glm_pto_kernels.glm_ffn_shared_expert_quant import ffn_shared_expert_quant as ffn_shared_expert_quant_pto
    ffn_res = torch.empty_like(hidden_states, device=hidden_states.device)
    w13_int8 = layer.gate_up_proj.weight
    w13_scale = layer.gate_up_proj.weight_scale
    w2_int8 = layer.down_proj.weight
    w2_scale = layer.down_proj.weight_scale
    ffn_shared_expert_quant_pto(hidden_states, w13_int8, w13_scale, w2_int8, w2_scale, ffn_res)
    return ffn_res
```

- glm_select_experts_impl.py 相关的算子适配层函数
```
# 配置算子开关，可灵活切换 PyPTO 或是原始方案
USE_PTO_SELECT_EXEPERTS = True

# 函数定义以及适配说明
def select_experts(router_logits, top_k, renormalize,
                   topk_group=None, num_expert_group=None, e_score_correction_bias=None):
    '''
    [适配说明]
    目标文件: vllm_ascend/ops/fused_moe/experts_selector.py
    目标函数: select_experts()
    替换方法: 将_native_select_experts()调用替换为当前函数, 替换后代码如下, 缩进已调整为与源码一致，可直接复制:
        # <----复制开始---->
        if glm_pto_kernels.USE_PTO_SELECT_EXEPERTS:
            # PTO内核实现
            topk_weights, topk_ids = glm_pto_kernels.select_experts(
                router_logits, top_k, renormalize, topk_group, num_expert_group, e_score_correction_bias
            )
        else:
            # Original Fallback
            topk_weights, topk_ids = _native_select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                top_k=top_k,
                use_grouped_topk=use_grouped_topk,
                renormalize=renormalize,
                topk_group=topk_group,
                num_expert_group=num_expert_group,
                custom_routing_function=custom_routing_function,
                scoring_func=scoring_func,
                e_score_correction_bias=e_score_correction_bias,
                global_num_experts=global_num_experts,
            )
        # <----复制结束---->
    '''
    from glm_pto_kernels.glm_select_experts import select_experts
    bs = router_logits.shape[0]
    device_info = router_logits.device
    topk_weights = torch.empty(
    (bs, top_k), dtype=router_logits.dtype, device=device_info)
    topk_ids = torch.empty((bs, top_k), dtype=torch.int32, device=device_info)
    select_experts(
        router_logits, top_k, renormalize, topk_group, num_expert_group, e_score_correction_bias, topk_weights, topk_ids
    )
    return topk_weights, topk_ids
```

替换时需严格匹配注释中的目标文件和目标函数，直接复制注释内的代码片段即可（缩进已适配源码），通过开关变量判断调用 PyPTO 适配层函数，未开启则执行原始逻辑，保证切换无侵入性。
