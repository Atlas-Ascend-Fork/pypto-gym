# deepseek V3.2 样例 (Examples)

本目录包含了一系列 PyPTO deepseek V3.2 EXP 的开发样例代码，我们对 DeepSeek-V3.2-Exp 进行了拆解，交付了五个算子：mla prolog, lightning indexer prolog, sparese flash attention, mla_indexer_prolog和lightning indexer。

## 参数说明/约束
-  shape 格式字段含义说明
    | 字段名       | 英文全称/含义                  | 取值规则与说明                                                                 |
    |--------------|--------------------------------|------------------------------------------------------------------------------|
    | b            | Batch（输入样本批量大小）      | 取值范围：decode场景1~128 prefill场景固定为1                                                          |
    | s1            | query Seq-Length | 取值范围：decode场景 1~4  prefill场景1~1K                                                            |
    | s2            | key Seq-Length | 取值范围：1~128K                                                             |
    | h           | Head-Size（隐藏层大小）        | 取值固定为：7168                                                            |
    | n_q(n1)            | query 的 Head-Num（多头数）             | 取值范围：128                                       |
    | n_kv(n2)           | kv 的 head 数             | 取值范围：1                                      |
    | kv_lora_rank            | kv 低秩矩阵维度             | 取值范围：512                                      |
    | rope_dim            |   qk 位置编码维度           | 取值范围：64                                      |
    | v_head_dim            |  value 的头维度            | 取值范围：128                                      |
    | q_head_dim            |   query 的头维度              | 取值范围：192                                      |
    | q_lora_rank            |query 低秩矩阵维度              | 取值范围：1536                                      |
    | idx_n_heads         | indexer里query的head num                | 取值固定为：64                                                             |
    | idx_head_dim         | indexer里query的头维度                | 取值固定为：128                                                             |
    | selected_count         | topk选择的个数                 | 取值固定为：2048                                                             |
    | block_num     | PagedAttention 场景下per-tile量的块数    | 取值为计算 `B*Skv/BlockSize` 的结果后向上取整（Skv 表示 kv 的序列长度，允许取 0） |
    | block_size    | PagedAttention 场景下的块大小  | 取值范围：128                                                           |
    | t            | BS 合轴后的大小                | 取值范围：b * s1|
# mla_polog_quant


## 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 功能说明

MLA Prolog 模块将hidden状态 $\bold{X}$ 转换为查询投影 $\bold{q}$、键投影 $\bold{k}$ 和值投影 $\bold{v}$，其结构与 DeepSeek V3 的架构一致。在解码阶段，采用了权重吸收技术。
## 计算公式

**RmsNorm公式**
$$
\text{RmsNorm}(x) = \gamma \cdot \frac{x_i}{\text{RMS}(x)}
$$
$$
\text{RMS}(x) = \sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2 + \epsilon}
$$

 **路径1：标准Query计算**

包括下采样、RmsNorm和两次上采样：
$$
c^Q = RmsNorm(x \cdot W^{DQ})
$$
$$
q^C = c^Q \cdot W^{UQ}
$$
$$
q^N = q^C \cdot W^{UK}
$$

**路径2：位置编码Query计算**

对Query进行ROPE旋转位置编码：
$$
q^R = ROPE(c^Q \cdot W^{QR})
$$

**路径3：标准Key计算**

包括下采样、RmsNorm，将计算结果存入Cache：
$$
c^{KV} = RmsNorm(x \cdot W^{DKV})
$$
$$
k^C = Cache(c^{KV})
$$

**路径4：位置编码Key计算**

对Key进行ROPE旋转位置编码，并将结果存入Cache：
$$
k^R = Cache(ROPE(x \cdot W^{KR}))
$$

## 函数原型
```
def mla_prolog_quant_compute(token_x, w_dq, w_uq_qr, dequant_scale, w_uk, w_dkv_kr, gamma_cq, gamma_ckv, cos,
		sin, cache_index, kv_cache, kr_cache, k_scale_cache, q_norm_out, q_norm_scale_out, query_nope_out,
   		query_rope_out, kv_cache_out, kr_cache_out, k_scale_cache_out, epsilon_cq, epsilon_ckv, cache_mode,
    	tile_config, rope_cfg):
```

## 参数说明

-   **token_x**（`Tensor`）：公式中用于计算Query和Key的输入tensor，不支持非连续的 Tensor，数据格式支持ND，数据类型支持`bfloat16`，shape为[t, h]。
-   **w_dq**（`Tensor`）：公式中用于计算Query的下采样权重矩阵$W^{DQ}$，不支持非连续的 Tensor。数据格式支持NZ，数据类型支持`bfloat16`，shape为[h, q_lora_rank]。
-   **w_uq_qr**（`Tensor`）：公式中用于计算Query的上采样权重矩阵$W^{UQ}$和位置编码权重矩阵$W^{QR}$，不支持非连续的 Tensor，数据格式支持NZ，数据类型支持`int8`，shape为[q_lora_rank, n_q * q_head_dim]。
-   **dequant_scale**（`Tensor`）：用于MatmulQcQr矩阵乘后w_uq_qr反量化操作的per-channel参数，不支持非连续的 Tensor。数据格式支持ND，数据类型支持`float`，shape为[n_q*q_head_dim, 1]。
-   **w_uk**（`Tensor`）：公式中用于计算Key的上采样权重$W^{UK}$，不支持非连续的 Tensor，数据格式支持ND，数据类型支持`bfloat16`，shape为[n_q, qk_nope_head_dim, kv_lora_rank]。
-   **w_dkv_kr**（`Tensor`）：公式中用于计算Key的下采样权重矩阵$W^{DKV}$和位置编码权重矩阵$W^{KR}$，不支持非连续的 Tensor，数据格式支持NZ，数据类型支持`bfloat16`，shape为[h, kv_lora_rank+rope_dim]。
-   **gamma_cq**（`Tensor`）：计算$c^Q$的RmsNorm公式中的$\gamma$参数，不支持非连续的 Tensor，数据格式支持ND，数据类型支持`bfloat16`，shape为[q_lora_rank]。
-   **gamma_ckv**（`Tensor`）：计算$c^{KV}$的RmsNorm公式中的$\gamma$参数，不支持非连续的 Tensor，数据格式支持ND，数据类型支持`bfloat16`，shape为[kv_lora_rank]。
-   **cos**（`Tensor`）：用于计算旋转位置编码的余弦参数矩阵，不支持非连续的 Tensor，数据格式支持ND，数据类型支持`bfloat16`，shape为[t, rope_dim]。
-   **sin**（`Tensor`）：用于计算旋转位置编码的正弦参数矩阵，不支持非连续的 Tensor，数据格式支持ND，数据类型支持`bfloat16`，shape为[t, rope_dim]。
-   **cache_index**（`Tensor`）：用于存储kv_cache和kr_cache的索引，不支持非连续的 Tensor，数据格式支持ND，数据类型支持`int64`，shape为[t]。
-   **kv_cache**（`Tensor`）：用于cache索引的aclTensor，计算结果原地更新（对应公式中的$k^C$），不支持非连续的 Tensor。数据格式支持ND，数据类型支持`int8`，cache_mode为"PA_BSND"，shape为[block_num, block_size, n_kv, kv_lora_rank]。
-   **kr_cache**（`Tensor`）：用于key位置编码的cache，计算结果原地更新（对应公式中的$k^R$），不支持非连续的 Tensor。数据格式支持ND，cache_mode为"PA_BSND"，数据类型支持`bfloat16`，cache_mode为"PA_BSND"、shape为[block_num, block_size, n_kv, rope_dim]。
-   **k_scale_cache**（`Tensor`）：表示 key 反量化因子的缓存，必选参数，不支持非连续的 Tensor，数据格式支持 ND，cache_mode为"PA_BSND"，数据类型支持`float`，shape为[block_num, block_size, n_kv, 4]。
-   **epsilon_cq**（`float`）：计算$c^Q$的RmsNorm公式中的$\epsilon$参数。用户未特意指定时，建议传入1e-05，仅支持double类型，默认值为1e-05。
-   **epsilon_ckv**（`float`）：计算$c^{KV}$的RmsNorm公式中的$\epsilon$参数。用户未特意指定时，建议传入1e-05，仅支持double类型，默认值为1e-05。
-   **cache_mode**（`str`）：表示kv_cache的模式，支持"PA_BSND"。
-   **tile_config**（`class MlaTileConfig`）：表示tile切分配置。
-   **rope_cfg**（`class RopeTileShapeConfig`）：表示rope tile切分配置。

## 返回值说明
-   **q_norm_out**（`Tensor`）：Query做RmsNorm_cq后的输出tensor（对应$q^C$），不支持非连续的 Tensor。数据格式支持ND，数据类型支持`int8`，shape为[t, q_lora_rank]。
-   **q_norm_scale_out**（`Tensor`）：Query做RmsNorm_cq后的反量化参数，不支持非连续的 Tensor。数据格式支持ND，数据类型支持`float`，shape为[t, 1]。
-   **q_nope_out**（`Tensor`）：公式中Query的输出tensor（对应$q^N$），不支持非连续的 Tensor。数据格式支持ND，数据类型支持`bfloat16`，shape为[t, n_q, kv_lora_rank]。
-   **q_rope_out**（`Tensor`）：公式中Query位置编码的输出tensor（对应$q^R$），不支持非连续的 Tensor。数据格式支持ND，数据类型支持`bfloat16`，shape为[t, n_q, rope_dim]。
-   **kv_cache_out**（`Tensor`）：Key输出到`kv_cache`中的tensor（对应$k^C$），不支持非连续的 Tensor。数据格式支持ND，cache_mode为"PA_BSND"，数据类型支持`int8`，shape为[block_num, block_size, n_kv, kv_lora_rank]。
-   **kr_cache_out**（`Tensor`）：Key的位置编码输出到`kr_cache`中的tensor（对应$k^R$），不支持非连续的 Tensor。数据格式支持ND，cache_mode为"PA_BSND"，数据类型支持`bfloat16`，shape为[block_num, block_size, n_kv, qk_rope_dim]。
-   **k_scale_cache_out**（`Tensor`）：Key做反量化后输出的反量化参数，不支持非连续的 Tensor。数据格式支持ND，数据类型支持`float`，cache_mode为"PA_BSND"，shape为[block_num, block_size, n_kv, 4]。

## 调用示例

- 详见算子实现 [mla_prolog_quant_impl.py](mla_prolog_quant_impl.py)，测试用例见 [tests/ops/deepseek_v32_exp/test_mla_prolog_quant.py](../../../../../tests/ops/deepseek_v32_exp/test_mla_prolog_quant.py)

# lightning_indexer_prolog

## 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 功能说明

用于 Deepseek IndexerAttention 中，计算 Lightning Indexer 所需要的 query，key 和 weights。
Indexer Prolog 的量化策略如下：Q_b_proj 使用 W8A8 量化，其他 Linear 均不量化；query 使用 A8 量化，key(cache) 使用 C8 量化；反量化因子以 FP16 存储；weights 以 FP16 存储；

## 计算公式
 **Query 的计算公式如下：**

 Q 的计算采用了动态的 Per-Token-Head 量化，其中 Hadamard 变换通过矩阵右乘 hadamard_q 实现。而 $\bold{q}, \bold{w}_{qb}$ 均是 Int8 类型。

$$
\bold{q}, \bold{q}_{scale} = \text{DynamicQuant}(\text{Hadamard}(\text{RoPE}(\text{DeQuant}(\bold{q} \cdot \bold{w}_{qb}))))
$$

**Key(cache) 的计算公式如下：**

Cache 的计算同样采用了动态的 Per-Token-Head 量化，其中 Hadamard 变换通过矩阵右乘 hadamard_k 实现。

$$
\bold{k}, \bold{k}_{scale} = \text{DynamicQuant}(\text{Hadamard}(\text{RoPE}(\text{LayerNorm}(\bold{x} \cdot \bold{w}_k))))
$$

**Weights 的计算公式如下：**

Weights 的计算没有采用量化，同时需要最后转化为 FP16 数据类型，供后续的 Lightning Indexer 计算使用。

$$
\bold{weight} = (\bold{x} \cdot \bold{w}_{proj}) * \text{scale}
$$

## 函数原型

```
def lightning_indexer_prolog_quant_compute(x_in, q_norm_in, q_norm_scale_in, w_qb_in, w_qb_scale_in, wk_in, w_proj_in,
				ln_gamma_k_in, ln_beta_k_in, cos_idx_rope_in, sin_idx_rope_in, hadamard_q_in, hadamard_k_in, k_int8_in, k_scale_in,
                k_cache_index_in, q_int8_out, q_scale_out, k_int8_out,k_scale_out, weights_out, attrs, configs):
```

## 参数说明


-   **x_in**（`Tensor`）：表示 hidden 状态token\_x，必选参数，不支持非连续的Tensor，数据格式支持ND，数据类型支持`bfloat16`,shape为[t, h]。
-   **q_norm_in**（`Tensor`）：表示经过 rmsnorm 后量化的 query，必选参数，不支持非连续的Tensor，数据格式支持ND，数据类型支持`int8`,shape为[t, q_lora_rank]。
-   **q_norm_scale_in**（`Tensor`）：表示 query 的反量化因子，必选参数，不支持非连续的Tensor，数据格式支持ND，数据类型支持`float32`,shape为[t, 1]。
-   **wq_b_in**（`Tensor`）：表示 query 的权重，必选参数，不支持非连续的Tensor，数据格式支持NZ，数据类型支持`int8`,shape为[q_lora_rank, idx_n_heads*idx_head_dim]。
-   **wq_qb_scale_in**（`Tensor`）：表示 query 的权重反量化因子，必选参数，不支持非连续的Tensor，数据格式支持ND，数据类型支持`float32`,shape为[idx_n_heads*idx_head_dim, 1]。
-   **wk_in**（`Tensor`）：表示 key 的权重，必选参数，不支持非连续的Tensor，数据格式支持NZ，数据类型支持`bfloat16`,shape为[h, idx_head_dim]。
-   **w_proj_in**（`Tensor`）：表示 weights 的权重，必选参数，不支持非连续的Tensor，数据格式支持NZ，数据类型支持`bfloat16`，shape为[h, idx_n_heads]。
-   **ln_gamma_k_in**（`Tensor`）：表示 key 的 layernorm 缩放，必选参数，不支持非连续的Tensor，数据格式支持ND，数据类型支持`bfloat16`，shape为[idx_head_dim]。
-   **ln_beta_k_in**（`Tensor`）：表示 key 的 layernorm 偏移，必选参数，不支持非连续的Tensor，数据格式支持ND，数据类型支持`bfloat16`,shape为[idx_head_dim]。
-   **cos_idx_rope_in**（`Tensor`）：表示用于 RoPE 的 cos，不支持非连续的 Tensor，数据格式支持 ND，数据类型支持`bfloat16`，shape为[t, rope_dim]。
-   **sin_idx_rope_in**（`Tensor`）：表示用于 RoPE 的 sin，不支持非连续的 Tensor，数据格式支持 ND，数据类型支持`bfloat16`，shape为[t, rope_dim]。
-   **hadamard_q_in**（`Tensor`）：表示用于 query Hadamard 变换的权重矩阵，不支持非连续的 Tensor，数据格式支持 ND，数据类型支持`bfloat16`，shape为[idx_head_dim, idx_head_dim]。
-   **hadamard_k_in**（`Tensor`）：表示用于 key Hadamard 变换的权重矩阵，不支持非连续的 Tensor，数据格式支持 ND，数据类型支持`bfloat16`，shape为[idx_head_dim, idx_head_dim]。
-   **k_int8_in**（`Tensor`）：表示 key 的缓存（k_cache），必选参数，不支持非连续的 Tensor，数据格式支持 ND，cache_mode为"PA_BSND"，数据类型支持`int8`，shape为[block_num, block_size, n_kv, idx_head_dim]。
-   **k_scale_in**（`Tensor`）：表示 key 反量化因子的缓存，必选参数，不支持非连续的 Tensor，数据格式支持 ND，cache_mode为"PA_BSND"，数据类型支持`float16`，shape为[block_num, block_size, n_kv, 1]。
-   **k_cache_index_in**（`Tensor`）：表示更新 key 缓存的位置，必选参数，不支持非连续的 Tensor，数据格式支持 ND，数据类型支持`int64`，shape为[t]。
-   **attrs.layernorm_epsilon_k**（`float`）：表示 key layernorm 防除 0 系数，必选参数，数据类型支持`float32`。
-   **attrs.layout_query**（`str`）：可选参数，用于标识输入`query`的数据排布格式，默认值"TND"。当前仅支持 "TND"。
-   **attrs.layout_key**（`str`）：可选参数，用于标识输入`key`的数据排布格式，默认值"PA_BSND"。当前仅支持 "PA_BSND"。
-   **configs**（`class IndexerPrologQuantConfigs`）：表示tile切分配置。

## 返回值说明

-   **q_int8_out**（`Tensor`）：公式中 query 的输出 tensor，不支持非连续的 Tensor，数据格式支持 ND，数据类型支持`int8`，shape为[t, idx_n_heads, idx_head_dim]。
-   **q_scale_out**（`Tensor`）：公式中 query 反量化因子的输出 tensor，不支持非连续的 Tensor，数据格式支持 ND，数据类型支持`float16`，shape为[t, idx_n_heads, 1]。
-   **k_int8_out**（`Tensor`）：表示 key 的缓存（k_cache）的输出 tensor，不支持非连续的 Tensor，数据格式支持 ND，cache_mode为"PA_BSND"，数据类型支持`int8`，shape为[block_num, block_size, n_kv, idx_head_dim]。
-   **k_scale_out**（`Tensor`）：表示 key 反量化因子的缓存的输出 tensor，不支持非连续的 Tensor，cache_mode为"PA_BSND"，数据格式支持 ND，数据类型支持`float16`，shape为[block_num, block_size, n_kv, 1]。
-   **weights_out**（`Tensor`）：公式中 weights 的输出 tensor，不支持非连续的 Tensor，数据格式支持 ND，数据类型支持`float16`，shape为[t, idx_n_heads]。

## 调用示例
-   算子实现见 [lightning_indexer_prolog_quant_impl.py](lightning_indexer_prolog_quant_impl.py)，测试用例见 [tests/ops/deepseek_v32_exp/test_lightning_indexer_prolog_quant.py](../../../../../tests/ops/deepseek_v32_exp/test_lightning_indexer_prolog_quant.py)


# sparse_flash_attention_quant


## 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 功能说明

对于每个查询 token $\bold{x}_i$，索引模块会为每个键值缓存项（表示键值对或 MLA 潜在表示）计算一个相关性得分 $I_{i,j}$。然后，通过将注意力机制应用于查询 token $\bold{x}_i$ 以及得分最高的前 $k$ 个缓存项，来计算输出 $\bold{o}_i$：

## 计算公式

$$
\bold{o}_i = \text{Attn}(\bold{x}_i, \{\bold{c}_j | j \in \text{Top-k}(\bold{I}_{i, :})\})
$$

## 函数原型

```
def sparse_flash_attention_quant_compute(query_nope, query_rope, key_nope_2d, key_rope_2d, k_nope_scales,
		topk_indices, block_table, kv_act_seqs, attention_out, nq, n_kv, softmax_scale, topk, block_size,
        max_blocknum_perbatch, tile_config):
```

## 参数说明

-   **query_nope**（`Tensor`）：必选参数，表示MLA结构中的query的rope信息，不支持非连续的 Tensor，数据格式支持ND，数据类型支持`bfloat16`，shape为[t * n_q, kv_lora_rank]。
-   **query_rope**（`Tensor`）：必选参数，表示MLA结构中的query的nope信息，不支持非连续的 Tensor，数据格式支持ND，数据类型支持`bfloat16`，shape为[t * n_q, rope_dim]。
-   **key_nope_2d**（`Tensor`）：必选参数，表示MLA结构中的key的rope信息，不支持非连续的 Tensor，数据格式支持ND，数据类型支持`int8`，shape为[block_num * block_size, kv_lora_rank]。
-   **key_rope_2d**（`Tensor`）：必选参数，表示MLA结构中的key的nope信息，不支持非连续的 Tensor，数据格式支持ND，数据类型支持`bfloat16`，shape为[block_num * block_size, rope_dim]。
-   **k_nope_scales**（`Tensor`）：必选参数，表示k_nope的反量化缩放因子，必选参数，不支持非连续的 Tensor，数据格式支持ND，数据类型支持`float`，shape为[block_num * block_size, 4]。
-   **topk_indices**（`Tensor`）：必选参数，表示每个token选出的topk索引，必选参数，不支持非连续的 Tensor，数据格式支持ND，数据类型支持`int32`，shape为[t, n_kv * selected_count]。
-   **block_table**（`Tensor`）：必选参数，表示PageAttention中KV存储使用的block映射表，不支持非连续的 Tensor，数据格式支持ND，数据类型支持`int32`，shape为[b, s2_max/block_size]，其中第二维表示长度不小于所有batch中最大的s2对应的block数量，即s2_max / block_size向上取整。
-   **kv_act_seqs**（`Tensor`）：必选参数，数据格式支持ND,表示不同Batch中`key`和`value`的有效token数，数据类型支持`int32`,shape为[b]。
-   **nq**（`int`）：必选参数，代表缩放系数，作为query和key矩阵乘后Muls的scalar值，数据类型支持float。
-   **n_kv**（`int`）：必选参数，代表缩放系数，作为query和key矩阵乘后Muls的scalar值，数据类型支持float。
-   **softmax_scale**（`float`）：必选参数，代表缩放系数，作为query和key矩阵乘后Muls的scalar值，数据类型支持float。
-   **topk**（`int`）：必选参数，代表选取的token个数，数据类型支持int。
-   **block_size**（`int`）：必选参数，代表sparse阶段的block大小，数据类型支持int。
-   **max_blocknum_perbatch**（`int`）：必选参数，每个batch最大的blocksize数量，数据类型支持int。
-   **tile_config**（`class SaTileShapeConfig`）：TileShapeConfig配置结构体，表示tile切分配置，配置项数据类型支持int。


## 返回值说明

-   **attention_out**（`Tensor`）：公式中的输出。数据格式支持ND，数据类型支持`bfloat16`，输出shape[b, s1, n_q, kv_lora_rank]。

## 调用示例

-   详见算子实现 [sparse_flash_attention_quant_impl.py](sparse_flash_attention_quant_impl.py)，测试用例见 [tests/ops/deepseek_v32_exp/test_sparse_flash_attention_quant.py](../../../../../tests/ops/deepseek_v32_exp/test_sparse_flash_attention_quant.py)


# sparse_attention_antiquant


## 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 功能说明

sa_antiquant是在sfa_quant基础上做的 存8算16 优化。在sfa_quant场景中，key_nope_2d，key_rope_2d 和 k_nope_scales 分别是 int8，bf16 和 fp32 类型；在后续 attention 的计算上，会离散地存储这三个 tensor，需要调三次离散访存指令去分别调用进行反量化和 concat；而 sa_antiquant 会将同一个 token 的 nope，rope 和 nope_scale 按尾轴合并在一起，仅需一条离散访存指令，总计可以节省 b * s * topk 次离散访存命令，节省搬运指令，提升搬运效率。

## 函数原型

```
def sparse_attention_antiquant_compute(query_nope, query_rope, nope_cache, topk_indices, block_table,
		kv_act_seqs, attention_out, nq, n_kv, softmax_scale, topk, block_size, max_blocknum_perbatch,
        tile_config):
```

## 参数说明

-   **query_nope**（`Tensor`）：必选参数，表示MLA结构中的query的rope信息，不支持非连续的 Tensor，数据格式支持ND，数据类型支持`bfloat16`，shape为[t * n_q, kv_lora_rank]。
-   **query_rope**（`Tensor`）：必选参数，表示MLA结构中的query的nope信息，不支持非连续的 Tensor，数据格式支持ND，数据类型支持`bfloat16`，shape为[t * n_q, rope_dim]。
-   **nope_cache**（`Tensor`）：必选参数，表示MLA结构中的key的反量化缩放因子，不支持非连续的 Tensor，数据格式支持ND，数据类型支持`int8`，shape为[block_num * block_size, kv_lora_rank + rope_dim * 2 + 4 * scale_size]，其中scale_size=4。
-   **topk_indices**（`Tensor`）：必选参数，表示每个token选出的topk索引，必选参数，不支持非连续的 Tensor，数据格式支持ND，数据类型支持`int32`，shape为[t, n_kv * selected_count]。
-   **block_table**（`Tensor`）：必选参数，表示PageAttention中KV存储使用的block映射表，不支持非连续的 Tensor，数据格式支持ND，数据类型支持`int32`，shape为[b, s2_max/block_size]，其中第二维表示长度不小于所有batch中最大的s2对应的block数量，即s2_max / block_size向上取整。
-   **kv_act_seqs**（`Tensor`）：必选参数，数据格式支持ND,表示不同Batch中`key`和`value`的有效token数，数据类型支持`int32`,shape为[b]。
-   **nq**（`int`）：必选参数，代表缩放系数，作为query和key矩阵乘后Muls的scalar值，数据类型支持float。
-   **n_kv**（`int`）：必选参数，代表缩放系数，作为query和key矩阵乘后Muls的scalar值，数据类型支持float。
-   **softmax_scale**（`float`）：必选参数，代表缩放系数，作为query和key矩阵乘后Muls的scalar值，数据类型支持float。
-   **topk**（`int`）：必选参数，代表选取的token个数，数据类型支持int。
-   **block_size**（`int`）：必选参数，代表sparse阶段的block大小，数据类型支持int。
-   **max_blocknum_perbatch**（`int`）：必选参数，每个batch最大的blocksize数量，数据类型支持int。
-   **tile_config**（`class SaTileShapeConfig`）：TileShapeConfig配置结构体，表示tile切分配置，配置项数据类型支持int。


## 返回值说明

-   **attention_out**（`Tensor`）：公式中的输出。数据格式支持ND，数据类型支持`bfloat16`，输出shape[b * s1 * n_q, kv_lora_rank]。

## 调用示例

-   详见算子实现 [sparse_attention_antiquant_impl.py](sparse_attention_antiquant_impl.py)，测试用例见 [tests/ops/deepseek_v32_exp/test_sparse_attention_antiquant.py](../../../../../tests/ops/deepseek_v32_exp/test_sparse_attention_antiquant.py)


# mla_indexer_polog_quant


## 产品支持情况

- Ascend 950PR：不支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 功能说明

MLA Indexer Prolog 模块将MLA Prolog和Lightning Indexer Prolog两个算子进行了更大范围的融合，实现了算子间的流水并行，提升了算子的性能。

## 函数原型
```
def mla_indexer_prolog_quant_compute(
    token_x, mla_w_dq, mla_w_uq_qr, mla_dequant_scale, mla_w_uk, mla_w_dkv_kr, mla_gamma_cq,
    mla_gamma_ckv, cos, sin, cache_index, mla_kv_cache, mla_kr_cache,
    mla_k_scale_cache, ip_w_qb_in, ip_w_qb_scale_in, ip_wk_in, ip_w_proj_in,
    ip_ln_gamma_k_in, ip_ln_beta_k_in, ip_hadamard_q_in, ip_hadamard_k_in,
    ip_k_cache, ip_k_cache_scale, mla_query_nope_out, mla_query_rope_out,
    mla_q_norm_out, mla_q_norm_scale_out, mla_kv_cache_out, mla_kr_cache_out,
    mla_k_scale_cache_out, ip_q_int8_out, ip_q_scale_out, ip_k_int8_out,
    ip_k_scale_out, ip_weights_out, mla_epsilon_cq, mla_epsilon_ckv,
    mla_cache_mode, mla_tile_config,
    ip_attrs, ip_configs, rope_cfg
):
```

## 参数说明

-   **token_x**（`Tensor`）：公式中用于计算Query和Key的输入tensor，不支持非连续的 Tensor，数据格式支持ND，数据类型支持`bfloat16`，shape为[t, h]。
-   **mla_w_dq**（`Tensor`）：公式中用于计算Query的下采样权重矩阵$W^{DQ}$，不支持非连续的 Tensor。数据格式支持NZ，数据类型支持`bfloat16`，shape为[h, q_lora_rank]。
-   **mla_w_uq_qr**（`Tensor`）：公式中用于计算Query的上采样权重矩阵$W^{UQ}$和位置编码权重矩阵$W^{QR}$。不支持非连续，数据格式支持NZ，数据类型支持`int8`，shape为[q_lora_rank, n_q*q_head_dim]。
-   **mla_dequant_scale**（`Tensor`）：用于MatmulQcQr矩阵乘后w_uq_qr反量化操作的per-channel参数，不支持非连续的 Tensor。数据格式支持ND，数据类型支持`float`，shape为[n_q*q_head_dim, 1]。
-   **mla_w_uk**（`Tensor`）：公式中用于计算Key的上采样权重$W^{UK}$。不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，shape为[n_q, qk_nope_head_dim, kv_lora_rank]。
-   **mla_w_dkv_kr**（`Tensor`）：公式中用于计算Key的下采样权重矩阵$W^{DKV}$和位置编码权重矩阵$W^{KR}$。不支持非连续，数据格式支持NZ，数据类型支持`bfloat16`，shape为[h, kv_lora_rank+rope_dim]。
-   **mla_gamma_cq**（`Tensor`）：计算$c^Q$的RmsNorm公式中的$\gamma$参数。不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，shape为[q_lora_rank]。
-   **mla_gamma_ckv**（`Tensor`）：计算$c^{KV}$的RmsNorm公式中的$\gamma$参数。不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，shape为[kv_lora_rank]。
-   **cos**（`Tensor`）：用于计算旋转位置编码的余弦参数矩阵。不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，shape为[t, rope_dim]。
-   **sin**（`Tensor`）：用于计算旋转位置编码的正弦参数矩阵。不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，shape为[t, rope_dim]。
-   **cache_index**（`Tensor`）：用于存储kv_cache和kr_cache的索引。不支持非连续，数据格式支持ND，数据类型支持`int64`，shape为[T]。
-   **mla_kv_cache**（`Tensor`）：用于cache索引的aclTensor，计算结果原地更新（对应公式中的$k^C$），不支持非连续的 Tensor。数据格式支持ND，cache_mode为"PA_BSND"，数据类型支持`int8`，cache_mode为"PA_BSND"、shape为[block_num, block_size, n_kv, kv_lora_rank]。
-   **mla_kr_cache**（`Tensor`）：用于key位置编码的cache，计算结果原地更新（对应公式中的$k^R$），不支持非连续的 Tensor。数据格式支持ND，cache_mode为"PA_BSND"，数据类型支持`bfloat16`，cache_mode为"PA_BSND"、shape为[block_num, block_size, n_kv, rope_dim]。
-   **mla_k_scale_cache**（`Tensor`）：表示 key 反量化因子的缓存，必选参数，不支持非连续的 Tensor，数据格式支持 ND，cache_mode为"PA_BSND"，数据类型支持`float`，shape为[block_num, block_size, n_kv, 4]。
-   **ip_w_qb_in**（`Tensor`）：表示 query 的权重，必选参数，不支持非连续的Tensor，数据格式支持NZ，数据类型支持`int8`,shape为[q_lora_rank, idx_n_heads*idx_head_dim]。
-   **ip_w_qb_scale_in**（`Tensor`）：表示 query 的权重反量化因子，必选参数，不支持非连续的Tensor，数据格式支持ND，数据类型支持`float32`,shape为[idx_n_heads*idx_head_dim, 1]。
-   **ip_wk_in**（`Tensor`）：表示 key 的权重，必选参数，不支持非连续的Tensor，数据格式支持NZ，数据类型支持`bfloat16`,shape为[h, idx_head_dim]。
-   **ip_w_proj_in**（`Tensor`）：表示 weights 的权重，必选参数，不支持非连续的Tensor，数据格式支持NZ，数据类型支持`bfloat16`，shape为[h, idx_n_heads]。
-   **ip_ln_gamma_k_in**（`Tensor`）：表示 key 的 layernorm 缩放，必选参数，不支持非连续的Tensor，数据格式支持ND，数据类型支持`bfloat16`，shape为[idx_head_dim]。
-   **ip_ln_beta_k_in**（`Tensor`）：表示 key 的 layernorm 偏移，必选参数，不支持非连续的Tensor，数据格式支持ND，数据类型支持`bfloat16`,shape为[idx_head_dim]。
-   **ip_hadamard_q_in**（`Tensor`）：表示用于 query Hadamard 变换的权重矩阵，不支持非连续的 Tensor，数据格式支持 ND，数据类型支持`bfloat16`，shape为[idx_head_dim, idx_head_dim]。
-   **ip_hadamard_k_in**（`Tensor`）：表示用于 key Hadamard 变换的权重矩阵，不支持非连续的 Tensor，数据格式支持 ND，数据类型支持`bfloat16`，shape为[idx_head_dim, idx_head_dim]。
-   **ip_k_cache**（`Tensor`）：表示 key 的缓存，必选参数，不支持非连续的 Tensor，数据格式支持 ND，数据类型支持`int8`，cache_mode为"PA_BSND"，shape为[block_num, block_size, n_kv, idx_head_dim]。
-   **ip_k_cache_scale**（`Tensor`）：表示 key 反量化因子的缓存，必选参数，不支持非连续的 Tensor，数据格式支持 ND，cache_mode为"PA_BSND"，数据类型支持`float16`，shape为[block_num, block_size, n_kv, 1]。
-   **mla_epsilon_cq**（`float`）：计算$c^Q$的RmsNorm公式中的$\epsilon$参数。用户未特意指定时，建议传入1e-05，仅支持double类型，默认值为1e-05。
-   **mla_epsilon_ckv**（`float`）：计算$c^{KV}$的RmsNorm公式中的$\epsilon$参数。用户未特意指定时，建议传入1e-05，仅支持double类型，默认值为1e-05。
-   **mla_cache_mode**（`str`）：表示kv_cache的模式，支持"PA_BSND"
-   **mla_tile_config**（`class MlaTileConfig`）：表示mla子图的tile切分配置。
-   **ip_attrs**（`class IndexerPrologQuantAttr`）：lightning indexer prolog子图计算所需的属性值，包括layernorm_epsilon_k，layout\_query，layout\_key
-   **ip.layernorm_epsilon_k**（`float`）：表示 key layernorm 防除 0 系数，必选参数，数据类型支持`float32`。
-   **ip.layout_query**（`str`）：可选参数，用于标识输入`query`的数据排布格式，默认值"TND"。当前仅支持 "TND"。
-   **ip.layout_key**（`str`）：可选参数，用于标识输入`key`的数据排布格式，默认值"PA_BSND"。当前仅支持 "PA_BSND"。
-   **ip_config**（`class IndexerPrologQuantConfigs`）：表示ip子图的tile切分配置及动态分档配置。
-   **rope_cfg**（`class RopeTileShapeConfig`）：表示rope子图的tile切分配置及动态分档配置。

## 返回值说明
-   **mla_query_nope_out**（`Tensor`）：公式中Query的输出tensor（对应$q^N$），不支持非连续的 Tensor。数据格式支持ND，数据类型支持`bfloat16`，shape为[t, n_q, kv_lora_rank]。
-   **mla_query_rope_out**（`Tensor`）：公式中Query位置编码的输出tensor（对应$q^R$），不支持非连续的 Tensor。数据格式支持ND，数据类型支持`bfloat16`，shape为[t, n_q, rope_dim]。
-   **mla_q_norm_out**（`Tensor`）：对公式中Query位置编码的输出tensor做rmsnorm转换并量化后的输出，不支持非连续的 Tensor。数据格式支持ND，数据类型支持`int8`，shape为[t, q_lora_rank]。
-   **mla_q_norm_scale_out**（`Tensor`）：对公式中Query位置编码的输出tensor做rmsnorm转换并量化后的反量化系数输出，不支持非连续的 Tensor。数据格式支持ND，数据类型支持`float32`，shape为[t, 1]。
-   **mla_kv_cache_out**（`Tensor`）：Key输出到`kv_cache`中的tensor（对应$k^C$），不支持非连续的 Tensor。数据格式支持ND，cache_mode为"PA_BSND"，数据类型支持`int8`，shape为[block_num, block_size, n_kv, kv_lora_rank]。
-   **mla_kr_cache_out**（`Tensor`）：Key的位置编码输出到`kr_cache`中的tensor（对应$k^R$），不支持非连续的 Tensor。数据格式支持ND，cache_mode为"PA_BSND"，数据类型支持`bfloat16`，shape为[block_num, block_size, n_kv, qk_rope_dim]。
-   **mla_k_scale_cache_out**（`Tensor`）：Key做反量化后输出的反量化参数，不支持非连续的 Tensor。数据格式支持ND，cache_mode为"PA_BSND"，数据类型支持`float`，shape为[block_num, block_size, n_kv, 4]。
-   **ip_q_int8_out**（`Tensor`）：公式中 query 的输出 tensor，数据格式支持 ND，数据类型支持`int8`，shape为[t, idx_n_heads, idx_head_dim]。
-   **ip_q_scale_out**（`Tensor`）：公式中 query 反量化因子的输出 tensor，不支持非连续的 Tensor，数据格式支持 ND，数据类型支持`float16`，shape为[t, idx_n_heads, 1]。
-   **ip_k_int8_out**（`Tensor`）：表示 key 的缓存（k_cache）的输出 tensor，不支持非连续的 Tensor，数据格式支持 ND，cache_mode为"PA_BSND"，数据类型支持`int8`，shape为[block_num, block_size, n_kv, idx_head_dim]。
-   **ip_k_scale_out**（`Tensor`）：表示 key 反量化因子的缓存的输出 tensor，不支持非连续的 Tensor，数据格式支持 ND，cache_mode为"PA_BSND"，数据类型支持`float16`，shape为[block_num, block_size, n_kv, 1]。
-   **ip_weights_out**（`Tensor`）：公式中 weights 的输出 tensor，不支持非连续的 Tensor，数据格式支持 ND，数据类型支持`float16`，shape为[t, idx_n_heads]。

## 调用示例

- 详见算子实现 [mla_indexer_prolog_quant_impl.py](mla_indexer_prolog_quant_impl.py)，测试用例见 [tests/ops/deepseek_v32_exp/test_mla_indexer_prolog_quant.py](../../../../../tests/ops/deepseek_v32_exp/test_mla_indexer_prolog_quant.py)

# lightning indexer


## 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 功能说明

LightningIndexer基于一系列操作得到每一个 token 对应的 Top-$k$ 个位置。对于某个 token 对应的 Index Query $Q_{index}\in\R^{g\times d}$，给定上下文 Index Key $K_{index}\in\R^{S_{k}\times d},W\in\R^{g\times 1}$，其中 $g$ 为 GQA 对应的 group size，$d$ 为每一个头的维度，$S_{k}$ 是上下文的长度，LightningIndexer的具体计算公式如下：
$$
\text{Top-}k\left\{[1]_{1\times g}@\left[(W@[1]_{1\times S_{k}})\odot\text{ReLU}\left(Q_{index}@K_{index}^T\right)\right]\right\}
$$

## 函数原型

```
def lightning_indexer_decode_compute(
    idx_query, idx_query_scale, idx_key_cache, idx_key_scale, idx_weight, act_seq_key, block_table, topk_res,
    unroll_list, configs, selected_count):
```

## 参数说明

-   **idx_query**（`Tensor`）：必选参数，不支持非连续，数据格式支持ND，数据类型支持`int8`，shape为[t, n_q, idx_head_dim]。
-   **idx_query_scale**（`Tensor`）：必选参数，表示idx_query的缩放系数，数据格式支持ND，数据类型支持`float16`，shape为[t, n_q, idx_head_dim]。
-   **idx_key_cache**（`Tensor`）：必选参数，不支持非连续，数据格式支持ND，数据类型支持`int8`，shape为[t, n_kv, idx_head_dim]。
-   **idx_key_scale**（`Tensor`）：必选参数，表示idx_key_cache的缩放系数，数据格式支持ND，数据类型支持`float16`，shape为[t, n_kv, idx_head_dim]
-   **idx_weight**（`Tensor`）：必选参数，不支持非连续，数据格式支持ND，数据类型支持`float16`，支持输入shape[t, n_q]。
-   **act_seq_key**（`Tensor`）：必选参数，表示不同Batch中`key`的有效token数，数据类型支持`int32`, shape为[b]。
-   **block_table**（`Tensor`）：必选参数，表示PageAttention中KV存储使用的block映射表，数据格式支持ND，数据类型支持`int32`，shape为[b, ceilDiv(max(s2), block_size)], 其中max(s2)为s2中最大值, ceilDiv表示向上取整。
-   **unroll_list**（`List`）：非必选参数，表示多档位切分配置。
-   **configs**（`class LightningIndexerConfigs`）：非必选参数，LightningIndexerConfigs配置结构体，表示tile切分配置和优化选项。
-   **selected_count**（`int`）：必选参数，topk选择数量，默认为2048。

## 返回值说明

-   **topk_res**（`Tensor`）：公式中的输出，数据类型支持`int32`。数据格式支持ND，输出shape[t, n_kv, selected_count]。

## 性能测试结果

> 测试环境: Ascend 910B, CANN 8.5.0
>
> 测试方法: Swimlane (泳道图) — `debug_options={"runtime_debug_mode": 1}`, 解析 `merged_swimlane.json` X 事件 span

| 算子 | 输入 shape | 输入 dtype | kernel 耗时 (μs) | 数据来源 |
|------|-----------|-----------|-----------------|---------|
| mla_prolog_quant     | b=4, s=2, h=7168, n1=128, s2=1024 | bfloat16/int8 | 117.7 | Swimlane |
| lightning_indexer_decode | b=4, s1=2, s2=64k, n1=64, d=128 | fp16/int8    | 296.8 | Swimlane |
| lightning_indexer_prolog | b=4, s1=2, s2=64k, h=7168, n=64, d=128 | bfloat16/int8 | 84.4  | Swimlane |
| mla_indexer_prolog    | b=4, s1=2, s2=1k, h=7168, n1=128/64, d=128 | bfloat16/int8 | 159.2 | Swimlane |
| sparse_flash_attention_quant | b=4, s=2, s2~64k, n1=128, n2=1, d=576 | bfloat16/int8 | 111.0 | Swimlane |
| sparse_attention_antiquant   | b=4, s=2, s2~64k, n1=128, n2=1, d=576 | bfloat16/int8 | 105.0 | Swimlane |
