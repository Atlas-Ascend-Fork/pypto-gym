# attention_pre_quant


## 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 功能说明

`attention_pre_quant` 算子对应 GLM-4.5 网络中 Attention 模块的前序计算逻辑，融合了以下关键操作：

- 输入 LayerNorm（Input Layer Normalization）
- 输入量化（Input Quantization）
- 量化矩阵乘法（Quantized MatMul）
- Q/K 的 LayerNorm（Query/Key Layer Normalization）
- Q/K 的 RoPE（Rotary Position Embedding）

该算子通过融合多个小算子，显著提升在 NPU 上的执行效率和内存带宽利用率。

## 数学公式

$
\text{qkv} = \text{x} @ \text{weight}
$

$
\begin{aligned}
\text{q}, \text{k}, \text{v} &= \text{Split}(\text{qkv}, [\text{q\_head\_size}, \text{k\_head\_size}, \text{v\_head\_size}] )
\end{aligned}
$

$
\text{RMSNorm}(\text{q}) = \frac{\text{q}}{\sqrt{\frac{1}{d} \sum_{i=1}^{d} q_i^2} + \epsilon} \odot \text{q\_gamma}
$

$
\text{RMSNorm}(\text{k}) = \frac{\text{k}}{\sqrt{\frac{1}{d} \sum_{i=1}^{d} k_i^2} + \epsilon} \odot \text{k\_gamma}
$

$
\text{RoPE}(\text{q}, m) = \bigoplus_{i=1}^{d/2} \begin{bmatrix}
\cos m\theta_i & -\sin m\theta_i \\
\sin m\theta_i & \cos m\theta_i
\end{bmatrix} \begin{bmatrix} q_{2i-1} \\ q_{2i} \end{bmatrix}
$

$
\text{RoPE}(\text{k}, m) = \bigoplus_{i=1}^{d/2} \begin{bmatrix}
\cos m\theta_i & -\sin m\theta_i \\
\sin m\theta_i & \cos m\theta_i
\end{bmatrix} \begin{bmatrix} k_{2i-1} \\ k_{2i} \end{bmatrix}
$



## 函数原型

```
def attention_pre_quant(
    hidden_states : torch.Tensor,
    residual : Optional[torch.Tensor],
    input_layernorm_weight : torch.Tensor,
    input_layernorm_bias : torch.Tensor,
    atten_qkv_input_scale_reciprocal : torch.Tensor,
    atten_qkv_input_offset : torch.Tensor,
    atten_qkv_weight : torch.Tensor,
    atten_qkv_quant_bias : torch.Tensor,
    atten_qkv_deq_scale : torch.Tensor,
    atten_q_norm_weight : torch.Tensor,
    atten_q_norm_bias : torch.Tensor,
    atten_k_norm_weight : torch.Tensor,
    atten_k_norm_bias : torch.Tensor,
    cos : torch.Tensor,
    sin : torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    residual_res: torch.Tensor
) -> None:
```

## 参数说明

>**说明：**<br>
>
>- batch_size表示输入样本批量大小（当前支持范围1至32）、seq_len表示输入样本序列长度（当前支持为1）、num_tokens表示batch_size和seq_len合轴的大小、hidden_size表示模型隐藏层维度（当前支持5120）、total_head_size表示QKV投影后的总输出维度（当前支持1792）、head_size表示每个注意力头的维度大小（当前支持128）、half_rotary_dim表示旋转位置编码维度（当前支持32）、q_size表示query投影后的输出维度（当前支持1536）、kv_size表示key/value投影后的输出维度（当前支持128）。

-   **hidden_states**（`Tensor`）：输入tensor。不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，shape为[num_tokens, hidden_size]。

-   **residual**（`Tensor`）：残差连接输入张量。不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，shape为[num_tokens, hidden_size]。

-   **input_layernorm_weight**（`Tensor`）：输入层归一化的权重参数，不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，shape为[hidden_size]。

-   **input_layernorm_bias**（`Tensor`）：输入层归一化的偏置参数，不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，shape为[hidden_size]。

-   **atten_qkv_input_scale_reciprocal**（`Tensor`）：注意力QKV输入量化的缩放系数倒数，不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，shape为[hidden_size]。

-   **atten_qkv_input_offset**（`Tensor`）：注意力QKV输入量化的偏移量，不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，shape为[hidden_size]。

-   **atten_qkv_weight**（`Tensor`）：注意力QKV的权重矩阵，不支持非连续，数据格式支持NZ，数据类型支持`int8`，shape为[hidden_size, total_head_size]。

-   **atten_qkv_quant_bias**（`Tensor`）：注意力QKV权重量化的偏置，不支持非连续，数据格式支持ND，数据类型支持`int32`，shape为[total_head_size]。

-   **atten_qkv_deq_scale**（`Tensor`）：注意力QKV权重量化的反量化缩放系数，不支持非连续，数据格式支持ND，数据类型支持`float32`，shape为[total_head_size]。

-   **atten_q_norm_weight**（`Tensor`）：query层归一化的权重参数，不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，shape为[head_size]。

-   **atten_q_norm_bias**（`Tensor`）：query层归一化的偏置参数，不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，shape为[head_size]。

-   **atten_k_norm_weight**（`Tensor`）：key层归一化的权重参数，不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，shape为[head_size]。

-   **atten_k_norm_bias**（`Tensor`）：key层归一化的偏置参数，不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，shape为[head_size]。

-   **cos**（`Tensor`）：旋转位置编码的余弦值，不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，shape为[num_tokens, 1, half_rotary_dim]。

-   **sin**（`Tensor`）：旋转位置编码的正弦值，不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，shape为[num_tokens, 1, half_rotary_dim]。

-   **query**（`Tensor`）：计算得到的query向量，不支持非连续，数据格式支持ND，数据类型为`bfloat16`，shape为[num_tokens, q_size]。

-   **key**（`Tensor`）：计算得到的key向量，不支持非连续，数据格式支持ND，数据类型为`bfloat16`，shape为[num_tokens, kv_size]。

-   **value**（`Tensor`）：计算得到的value向量，不支持非连续，数据格式支持ND，数据类型为`bfloat16`，shape为[num_tokens, kv_size]。

-   **residual_res**（`Tensor`）：更新后的残差输出，不支持非连续，数据格式支持ND，数据类型为`bfloat16`，shape为[num_tokens, hidden_size]。

## 调用示例

- 详见算子实现 [glm_attention_pre_quant_impl.py](./glm_attention_pre_quant_impl.py)，测试用例见 [tests/ops/glm_v4_5/test_glm_attention_pre_quant.py](../../../../../tests/ops/glm_v4_5/test_glm_attention_pre_quant.py)



# attention


## 产品支持情况

- Ascend 950PR：不支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 功能说明

`attention` 算子是基于先进的分页思想设计的注意力机制优化技术，专为大模型推理场景而生。它有效解决了传统注意力机制在处理长序列和动态批处理时面临的三大核心挑战：

- 内存碎片化：频繁的序列增长/收缩导致缓存分配不连续。
- 显存利用率低：固定大小的KV缓存块造成大量闲置空间。
- 推理吞吐受限：序列对齐与冗余计算拖慢整体处理速度。

通过引入类似操作系统“分页管理”的机制，实现了非连续缓存块的灵活管理，显著提升显存使用效率与推理吞吐能力。

## 数学公式

$
\text{atten} = \text{Softmax}\left(\text{Zoom}(Q \cdot K^T)\right ) \cdot V
$

其中：
-  Q, K, V ：分别为查询（Query）、键（Key）、值（Value），由输入变量线性变换得到。
- Zoom ：表示缩放操作，目的是防止点积过大导致 Softmax 后梯度消失。
- Softmax ：对每个 Q 对应的 K 权重进行归一化，输出注意力权重。
- atten ：注意力权重与 V 的加权求和，融合了上下文信息。


## 函数原型

```
def attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    actual_seqs: torch.Tensor,
    attn_res: torch.Tensor
) -> None:
```

## 参数说明

>**说明：**<br>
>
>- batch_size表示输入样本批量大小（当前支持范围1至32）、seq_len表示输入样本序列长度（当前支持为1）、num_tokens表示batch_size和seq_len合轴的大小、num_head表示查询端的多头数量（当前支持为12）、head_size表示每个注意力头的维度（当前支持为128）、num_blocks表示总共可用的缓存块数量、block_size表示每个缓存块能容纳的词元数量（当前支持为128）、kv_head_num表示键/值端的多头数量（当前支持为1）、max_num_blocks_per_query表示单个请求最多可以占用的缓存块数。


-   **query**（`Tensor`）：数据格式支持ND，数据类型支持`bfloat16` ，shape为 [num_tokens, num_head, head_size]。

-   **key_cache**（`Tensor`）：数据格式支持ND，数据类型支持`bfloat16` ，shape为 [num_blocks, block_size, kv_head_num, head_size]。

-   **value_cache**（`Tensor`）：数据格式支持ND，数据类型支持`bfloat16` ，shape为 [num_blocks, block_size, kv_head_num, head_size]。

-   **block_tables**（`Tensor`）：数据格式支持ND，数据类型支持`int32` ，shape为 [batch_size, max_num_blocks_per_query]。

-   **actual_seqs**（`Tensor`）：数据格式支持ND，数据类型支持`int32` ，shape为 [batch_size]。

-   **attn_res**（`Tensor`）：数据格式支持ND，数据类型支持`bfloat16` ，shape为 [num_tokens, num_head, head_size]。

## 调用示例

- 详见算子实现 [glm_attention_impl.py](./glm_attention_impl.py)，测试用例见 [tests/ops/glm_v4_5/test_glm_attention.py](../../../../../tests/ops/glm_v4_5/test_glm_attention.py)




# gate


## 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 功能说明

`gate` 算子对应GLM4.5网络中进入专家选择前的matmul操作，将模型主维度由d_model投影到路由器专用维度d_router


## 数学公式
$
\text{gateOut} = \text{hiddenStates} @ \text{weight}
$


## 函数原型

```
def gate(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    router_logits_out: torch.Tensor
):
```

## 参数说明

>
>- batch_size表示输入样本批量大小（当前支持范围1至32）、seq_len表示输入样本序列长度（当前支持为1）、num_tokens表示batch_size和seq_len合轴的大小、num_router_experts表示路由专家数（当前支持为160）。

-   **gate_weight**（`Tensor`）：表示投影权重矩阵，将通用特征转化为路由专用特征, 不支持非连续的Tensor, 数据格式支持ND，数据类型支持 `FP32`, shape为[num_router_experts, hidden_size]。

-   **hidden_states**（`Tensor`）：当前层输入特征矩阵, 不支持非连续的Tensor, 数据格式支持ND，数据类型支持 `FP32`, shape为[num_tokens, hidden_size]。

-   **router_logits_res**（`Tensor`）：表示经过投影权重优化后的路由特征矩阵, 数据格式支持ND，数据类型支持 `FP32`, shape为[num_tokens ,num_router_experts]。

## 调用示例

- 详见算子实现 [glm_gate_impl.py](./glm_gate_impl.py)，测试用例见 [tests/ops/glm_v4_5/test_glm_gate.py](../../../../../tests/ops/glm_v4_5/test_glm_gate.py)




# select_experts


## 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 功能说明

`select_experts` 算子对应GLM4.5网络中的select_experts（专家选择器）是MoE架构的核心组件。他负责智能地将输入的token分配给不同的专家网络进行处理。

## 数学公式

$
\text{topk\_weights} = \text{topk\_weights} + \text{e\_score\_bias}
$

$
\text{topk\_weights} = \text{group\_top\_k}(\text{topk\_weights, num\_expert\_group, topk\_group})
$

$
\text{topk\_ids} = \text{topk}(\text{topk\_weights, top\_k})\text{topk\_weights}
$

$
\text{topk\_weights} = \text{renormalize}(\text{topk\_weights})
$


## 函数原型

```
def select_experts(
    router_logits: torch.Tensor,
    top_k: int,
    renormalize: bool,
    topk_group: int,
    num_expert_group: int,
    e_score_correction_bias: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
```

## 参数说明

>
>-  batch_size表示输入样本批量大小（当前支持范围1至32）、seq_len表示输入样本序列长度（当前支持为1）、num_tokens表示batch_size和seq_len合轴的大小、num_router_experts表示路由专家数（当前支持为160）、num_experts_per_topk表示每个词元分配的专家数量（当前支持为8）。

-   **router_logits**（`Tensor`）：表示路由的分数，表示专家的数量, 数据格式支持ND，数据类型支持 `float32`, shape为[num_tokens, num_router_experts]。

-   **top_k**（`int`）：表示专家的数量, 数据类型支持`int8`。

-   **renormalize**（`bool`）：表示是否归一化路由权重, 数据类型支持`bool`。

-   **topk_group**（`int`）：表示可选择专家的数量,数据类型支持`int32`。

-   **num_expert_group**（`int`）：表示每组专家的数量, 数据类型支持`int32`。

-   **e_score_correction_bias**（`Tensor`）：表示校正专家的偏差值, 数据格式支持ND，数据类型支持 `bfloat16`, shape为[num_router_experts]。

-   **topk_weights**（`Tensor`）：每个专家的责任权重, 数据格式支持ND，数据类型支持 `FP32`, shape为[num_tokens, num_experts_per_topk]。

-   **topk_ids**（`Tensor`）：表示为每个token选择的专家编号, 数据格式支持ND，数据类型支持 `int32`, shape为[num_tokens, num_experts_per_topk]。

## 调用示例

- 详见算子实现 [glm_select_experts_impl.py](./glm_select_experts_impl.py)，测试用例见 [tests/ops/glm_v4_5/test_glm_select_experts.py](../../../../../tests/ops/glm_v4_5/test_glm_select_experts.py)




# ffn_shared_expert_quant


## 产品支持情况

- Ascend 950PR：不支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 功能说明

`ffn_shared_expert_quant` 算子对应GLM4.5网络中MoE共享专家的计算逻辑，包含`symmetric_quantization_per_token`、`matmul`、`dequant_dynamic`和`swiglu`，用于进行单个共享专家的量化前向传播计算，通过在不同任务或数据流之间复用同一组权重参数，以学习通用的特征表示，同时减少模型的参数总量。

## 数学公式

$
\text{hiddenStatesQuant}, \text{hiddenStatesScale} = \text{Quant}(\text{hiddenStates})
$

$
\text{swigluOut} = \text{Swiglu}((\text{hiddenStatesQuant} @ \text{w13}) \odot \text{hiddenStatesScale} \odot \text{w13Scale})
$

$
\text{downProjQuant}, \text{downProjScale} = \text{Quant}(\text{swigluOut})
$

$
\text{ffnRes} = (\text{downProjQuant} @ \text{w2}) \odot \text{downProjScale} \odot \text{w2Scale}
$


## 函数原型

```
def ffn_shared_expert_quant(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    ffn_res: torch.Tensor
) -> None:
```

## 参数说明
>**说明：**<br>
>
>- batch_size表示输入样本批量大小（当前支持范围1至32）、seq_len表示输入样本序列长度（当前支持为1）、num_tokens表示batch_size和seq_len合轴的大小、hidden_size表示隐藏层大小（当前支持5120）、intermediate_size表示中间层的维度（当前支持1536）。

-   **hidden_states**（`Tensor`）：当前共享专家的输入特征向量。当前仅支持连续，数据格式ND，数据类型支持`bfloat16`，shape为[num_tokens, hidden_size]。

-   **w13**（`Tensor`）：gate_proj & up_proj量化权重。当前仅支持连续，数据格式NZ，数据类型支持`int8`，shape为[hidden_size, intermediate_size * 2]。

-   **w13_scale**（`Tensor`）：w13权重的缩放因子。当前仅支持连续，数据格式ND，数据类型支持`bfloat16`，shape为[intermediate_size * 2]。

-   **w2**（`Tensor`）：down_proj量化权重。当前仅支持连续，数据格式NZ，数据类型支持`int8`，shape为[intermediate_size, hidden_size]。

-   **w2_scale**（`Tensor`）：w2权重的缩放因子。当前仅支持连续，数据格式ND，数据类型支持`bfloat16`，shape为[hidden_size]。

-   **ffn_res**（`Tensor`）：输出tensor。当前仅支持连续，数据格式ND，数据类型支持`bfloat16`，shape为[num_tokens, hidden_size]。


## 调用示例

- 详见算子实现 [glm_ffn_shared_expert_quant_impl.py](./glm_ffn_shared_expert_quant_impl.py)，测试用例见 [tests/ops/glm_v4_5/test_glm_ffn_shared_expert_quant.py](../../../../../tests/ops/glm_v4_5/test_glm_ffn_shared_expert_quant.py)


# attention_fusion


## 产品支持情况

- Ascend 950PR：不支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 功能说明

`attention_fusion` 算子是 `attention_fusion` 和 `attention_fusion` 的深度协同融合版本。不仅继承了前序算子的所有优势，更通过端到端的算子级融合，打破模块间的数据搬运壁垒，实现从输入到输出的全链路高效执行。

## 数学公式

$
\text{qkv} = \text{x} @ \text{weight}
$

$
\begin{aligned}
\text{q}, \text{k}, \text{v} &= \text{Split}(\text{qkv}, [\text{q\_head\_size}, \text{k\_head\_size}, \text{v\_head\_size}] )
\end{aligned}
$

$
\text{RMSNorm}(\text{q}) = \frac{\text{q}}{\sqrt{\frac{1}{d} \sum_{i=1}^{d} q_i^2} + \epsilon} \odot \text{q\_gamma}
$

$
\text{RMSNorm}(\text{k}) = \frac{\text{k}}{\sqrt{\frac{1}{d} \sum_{i=1}^{d} k_i^2} + \epsilon} \odot \text{k\_gamma}
$

$
\text{RoPE}(\text{q}, m) = \bigoplus_{i=1}^{d/2} \begin{bmatrix}
\cos m\theta_i & -\sin m\theta_i \\
\sin m\theta_i & \cos m\theta_i
\end{bmatrix} \begin{bmatrix} q_{2i-1} \\ q_{2i} \end{bmatrix}
$

$
\text{RoPE}(\text{k}, m) = \bigoplus_{i=1}^{d/2} \begin{bmatrix}
\cos m\theta_i & -\sin m\theta_i \\
\sin m\theta_i & \cos m\theta_i
\end{bmatrix} \begin{bmatrix} k_{2i-1} \\ k_{2i} \end{bmatrix}
$

$
\text{atten} = \text{Softmax}\left(\text{Zoom}(Q \cdot K^T)\right ) \cdot V
$

## 函数原型

```
def attention(
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        input_layernorm_weight: torch.Tensor,
        input_layernorm_bias: torch.Tensor,
        qkv_proj_scale: torch.Tensor,
        qkv_proj_offset: torch.Tensor,
        qkv_proj_weight: torch.Tensor,
        qkv_proj_quant_bias: torch.Tensor,
        qkv_proj_deq_scale: torch.Tensor,
        q_norm_weight: torch.Tensor,
        q_norm_bias: torch.Tensor,
        k_norm_weight: torch.Tensor,
        k_norm_bias: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        actual_seq_lens: torch.Tensor,
        slot_mapping: torch.Tensor,
        enable_residual: bool,
        eps: float,
        num_decode_tokens: int
) -> tuple[torch.Tensor, torch.Tensor]:
```

## 参数说明

>**说明：**<br>
>
>- batch_size表示输入样本批量大小（当前支持范围1至32）、seq_len表示输入样本序列长度（当前支持为1）、num_tokens表示batch_size和seq_len合轴的大小、hidden_size表示模型隐藏层维度（当前支持5120）、total_head_size表示QKV投影后的总输出维度（当前支持1792）、head_size表示每个注意力头的维度大小（当前支持128）、half_rotary_dim表示旋转位置编码维度（当前支持32）、num_blocks表示总共可用的缓存块数量、block_size表示每个缓存块能容纳的词元数量（当前支持为128）、kv_head_num表示键/值端的多头数量（当前支持为1）、max_num_blocks_per_query表示单个请求最多可以占用的缓存块数。

-   **hidden_states**（`Tensor`）：输入tensor。不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，shape为[num_tokens, hidden_size]。

-   **residual**（`Tensor`）：残差连接输入张量。不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，shape为[num_tokens, hidden_size]。

-   **input_layernorm_weight**（`Tensor`）：输入层归一化的权重参数，不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，shape为[hidden_size]。

-   **input_layernorm_bias**（`Tensor`）：输入层归一化的偏置参数，不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，shape为[hidden_size]。

-   **qkv_proj_scale**（`Tensor`）：注意力QKV输入量化的缩放系数倒数，不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，shape为[hidden_size]。

-   **qkv_proj_offset**（`Tensor`）：注意力QKV输入量化的偏移量，不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，shape为[hidden_size]。

-   **qkv_proj_weight**（`Tensor`）：注意力QKV的权重矩阵，不支持非连续，数据格式支持NZ，数据类型支持`int8`，shape为[hidden_size, total_head_size]。

-   **qkv_proj_quant_bias**（`Tensor`）：注意力QKV权重量化的偏置，不支持非连续，数据格式支持ND，数据类型支持`int32`，shape为[total_head_size]。

-   **qkv_proj_deq_scale**（`Tensor`）：注意力QKV权重量化的反量化缩放系数，不支持非连续，数据格式支持ND，数据类型支持`float32`，shape为[total_head_size]。

-   **q_norm_weight**（`Tensor`）：query层归一化的权重参数，不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，shape为[head_size]。

-   **q_norm_bias**（`Tensor`）：query层归一化的偏置参数，不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，shape为[head_size]。

-   **k_norm_weight**（`Tensor`）：key层归一化的权重参数，不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，shape为[head_size]。

-   **k_norm_bias**（`Tensor`）：key层归一化的偏置参数，不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，shape为[head_size]。

-   **cos**（`Tensor`）：旋转位置编码的余弦值，不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，shape为[num_tokens, 1, half_rotary_dim]。

-   **sin**（`Tensor`）：旋转位置编码的正弦值，不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，shape为[num_tokens, 1, half_rotary_dim]。

-   **key_cache**（`Tensor`）：数据格式支持ND，数据类型支持`bfloat16` ，shape为 [num_blocks, block_size, kv_head_num, head_size]。

-   **value_cache**（`Tensor`）：数据格式支持ND，数据类型支持`bfloat16` ，shape为 [num_blocks, block_size, kv_head_num, head_size]。

-   **block_tables**（`Tensor`）：数据格式支持ND，数据类型支持`int32` ，shape为 [batch_size, max_num_blocks_per_query]。

-   **actual_seq_lens**（`Tensor`）：数据格式支持ND，数据类型支持`int32` ，shape为 [batch_size]。

-   **slot_mapping**（`Tensor`）：数据格式支持ND，数据类型支持`int32` ，shape为 [batch_size]。

-   **enable_residual**（`bool`）：表示是否使用残差, 数据类型支持`bool`。

-   **eps**（`float`）：表示精度值, 数据类型支持`float`。

-   **num_decode_tokens**（`int`）：数据类型支持`int`。

## 调用示例

- 详见算子实现 [glm_attention_fusion_impl.py](./glm_attention_fusion_impl.py)，测试用例见 [tests/ops/glm_v4_5/test_glm_attention_fusion.py](../../../../../tests/ops/glm_v4_5/test_glm_attention_fusion.py)


# moe_fusion


## 产品支持情况

- Ascend 950PR：不支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 功能说明

`moe_fusion` 算子是 `gate` 、 `select_experts` 和 `ffn_shared_expert_quant` 的深度协同融合版本，目的是实现低延迟的MoE推理。

## 数学公式

$
\text{topk\_weights} = \text{topk\_weights} + \text{e\_score\_bias}
$

$
\text{topk\_weights} = \text{group\_top\_k}(\text{topk\_weights, num\_expert\_group, topk\_group})
$

$
\text{topk\_ids} = \text{topk}(\text{topk\_weights, top\_k})\text{topk\_weights}
$

$
\text{topk\_weights} = \text{renormalize}(\text{topk\_weights})
$

$
\text{hidden\_states\_quant}, \text{hidden\_states\_scale} = \text{quant}(\text{hidden\_states})
$

$
\text{swiglu\_out} = \text{swiglu}((\text{hidden\_states\_quant} @ \text{w13}) \odot \text{hidden\_states\_scale} \odot \text{w13\_scale})
$

$
\text{down\_proj\_quant}, \text{down\_proj\_scale} = \text{quant}(\text{swiglu\_out})
$

$
\text{ffn\_res} = (\text{down\_proj\_quant} @ \text{w2}) \odot \text{down\_proj\_scale} \odot \text{w2\_scale}
$

$
\text{gate\_out} = \text{hidden\_states} @ \text{weight}
$

## 函数原型

```
@dataclass
class MoeFusionInputs:
    gate_weight: torch.Tensor
    hidden_states: torch.Tensor
    top_k: int
    renormalize: bool
    topk_group: int
    num_expert_group: int
    e_score_bias: torch.Tensor
    w13: torch.Tensor
    w13_scale: torch.Tensor
    w2: torch.Tensor
    w2_scale: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    ffn_res: torch.Tensor

def moe_fusion(inputs: MoeFusionInputs):
```

## 参数说明
>**说明：**<br>
>
>- batch_size表示输入样本批量大小（当前支持范围1至32）、seq_len表示输入样本序列长度（当前支持为1）、num_tokens表示batch_size和seq_len合轴的大小、num_router_experts表示路由专家数（当前支持为160）、hidden_size表示隐藏层大小（当前支持5120）、intermediate_size表示中间层的维度（当前支持1536）、num_experts_per_topk表示每个词元分配的专家数量（当前支持为8）。

-   **gate_weight**（`Tensor`）：表示投影权重矩阵，将通用特征转化为路由专用特征, 不支持非连续的Tensor, 数据格式支持ND，数据类型支持 `FP32`, shape为[num_router_experts, hidden_size]。

-   **hidden_states**（`Tensor`）：当前共享专家的输入特征向量。当前仅支持连续，数据格式ND，数据类型支持`bfloat16`，shape为[num_tokens, hidden_size]。

-   **top_k**（`int`）：表示专家的数量, 数据类型支持`int8`。

-   **renormalize**（`bool`）：表示是否归一化路由权重, 数据类型支持`bool`。

-   **topk_group**（`int`）：表示可选择专家的数量,数据类型支持`int32`。

-   **num_expert_group**（`int`）：表示每组专家的数量, 数据类型支持`int32`。

-   **e_score_bias**（`Tensor`）：表示校正专家的偏差值, 数据格式支持ND，数据类型支持 `bfloat16`, shape为[num_router_experts]。

-   **w13**（`Tensor`）：gate_proj & up_proj量化权重。当前仅支持连续，数据格式NZ，数据类型支持`int8`，shape为[hidden_size, intermediate_size * 2]。

-   **w13_scale**（`Tensor`）：w13权重的缩放因子。当前仅支持连续，数据格式ND，数据类型支持`bfloat16`，shape为[intermediate_size * 2]。

-   **w2**（`Tensor`）：down_proj量化权重。当前仅支持连续，数据格式NZ，数据类型支持`int8`，shape为[intermediate_size, hidden_size]。

-   **w2_scale**（`Tensor`）：w2权重的缩放因子。当前仅支持连续，数据格式ND，数据类型支持`bfloat16`，shape为[hidden_size]。

-   **topk_weights**（`Tensor`）：每个专家的责任权重, 数据格式支持ND，数据类型支持 `FP32`, shape为[num_tokens, num_experts_per_topk]。

-   **topk_ids**（`Tensor`）：表示为每个token选择的专家编号, 数据格式支持ND，数据类型支持 `int32`, shape为[num_tokens, num_experts_per_topk]。

-   **ffn_res**（`Tensor`）：输出tensor。当前仅支持连续，数据格式ND，数据类型支持`bfloat16`，shape为[num_tokens, hidden_size]。

## 性能测试结果

> 测试环境: Ascend 910B, CANN 8.5.0

| 算子 | 输入 shape | 输入 dtype | kernel 耗时 (μs) | 数据来源 |
|------|-----------|-----------|-----------------|---------|
| ffn_shared_expert_quant | hidden_states [num_tokens, 5120], w13 [5120, 3072] | bfloat16/int8 | 61.2 | Swimlane |
| gate                    | hidden_states [num_tokens, 5120], weight [160, 5120] | fp32          | 11.9 | Swimlane |
| select_experts          | router_logits [num_tokens, 160]                        | fp32/bf16     | 7.3  | Swimlane |
| attention               | b=16, s1=1, s2=16384, n1=12, d=128                     | bfloat16      | 273.2 | Swimlane |
| attention_pre_quant     | num_tokens=8, hidden_size=5120, q_size=1536              | bfloat16/int8 | 76.4 | Swimlane |
| attention_fusion        | b=16, s1=1, s2=8192, hidden=5120, n1=12, d=128          | bfloat16/int8 | 201.5 | Swimlane |
| moe_fusion              | num_tokens=32, hidden=5120, ne=160, top_k=8              | bfloat16/int8 | 66.6 | Swimlane |