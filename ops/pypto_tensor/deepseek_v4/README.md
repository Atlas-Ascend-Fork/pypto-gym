# Compressed_Flash_Attention


## 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 功能说明

- API功能：`CompressedFlashAttention`算子旨在完成以下公式描述的Attention计算，支持Compressed Attention。
- 计算公式：

  $$
  O = \text{softmax}(Q@\tilde{K}^T \cdot \text{softmax\_scale})@\tilde{V}
  $$

  其中$\tilde{K}=\tilde{V}$为基于kv_cache、kv_win等入参控制的实际参与计算的 $KV$。

## 函数原型

```
torch.ops.pypto.compress_flash_attention(
    q,
    cmp_kv,
    sinks,
    cmp_block_table,
    seqused_kv,
    ori_kv,
    ori_block_table,
    cmp_ratio
) -> (Tensor)
```

## 参数说明

- **q**（`Tensor`）：必选参数，对应公式中的$Q$，不支持非连续，数据格式支持ND，数据类型支持`bfloat16`。`layout_query`为BSND时shape为[B*S1,N1,D]，其中N1仅支持64。
- **cmp_kv**（`Tensor`）：必选参数，对应公式中的$\tilde{K}和\tilde{V}$的一部分，为经过压缩的KV，不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，`layout_kv`为PA_ND时shape为[block\_num, cmp\_block\_size, KV\_N, D]，其中block\_num2为PageAttention时block总数，cmp\_block\_size为一个block的token数，cmp\_block\_size取值为16的倍数，最大支持1024。
- **sinks**（`Tensor`）：必选参数，注意力下沉tensor，数据格式支持ND，数据类型支持`float32`，shape为[N1]。
- **cmp_block_table**（`Tensor`）：必选参数，表示PageAttention中cmpKvCache存储使用的block映射表。数据格式支持ND，数据类型支持`int32`，shape为2维，其中第一维长度为B，第二维长度不小于所有batch中最大的S3对应的block数量，即S3\_max / block\_size向上取整。
- **seqused_kv**（`Tensor`）：必选参数，表示不同Batch中`ori_kv`实际参与运算的token数，维度为B，数据格式支持ND，数据类型支持`int32`，不输入则所有token均参与运算。
- **ori_kv**（`Tensor`）：必选参数，对应公式中的$\tilde{K}和\tilde{V}$的一部分，为原始不经压缩的KV，不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，`layout_kv`为PA_ND时shape为[block\_num1, ori\_block\_size, KV\_N, D]，其中block\_num1为PageAttention时block总数，ori\_block\_size为一个block的token数，ori\_block\_size取值为16的倍数，最大支持1024，KV_N仅支持1。
- **ori_block_table**（`Tensor`）：必选参数，表示PageAttention中oriKvCache存储使用的block映射表。数据格式支持ND，数据类型支持`int32`，shape为2维，其中第一维长度为B，第二维长度不小于所有batch中最大的S2对应的block数量，即S2\_max / block\_size向上取整。
- **cmp_ratio**（`int`）：必选参数，表示对ori_kv的压缩率，数据类型支持`int`，数据支持128。

## 返回值说明

- **attention\_out**（`Tensor`）：公式中的输出。数据格式支持ND，数据类型支持`bfloat16`，shape为[B,S1,N1,D]。

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持aclgraph模式。
- 参数q中的D和seqused_kv、kv_cache的D值相等为512。
- 参数seqused_kv、kv_cache的数据类型必须保持一致。
- 本接口仅支持decode场景，不支持prefill场景。
- block_size支持128。

## 调用方法

```
python3  tests/ops/deepseek_v4/test_compress_flash_attention.py
```

# Compressor

## 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 功能说明

- API功能：Compressor将每4或128个token的 KV cache 压缩成一个，然后每个token与这些压缩的 KV cache进行 DSA 计算。在长序列的情况下，Compressor可以有效地减少计算开销。
- 主要计算过程为：

  1. 将输入$X$与$W^{KV}$做Matmul运算得到$kv\_state$，将输入$X$与$W^{Gate}$做Matmul运算后再与$Ape$做Add运算得到$score\_state$，$kv\_state$与$score\_state$根据输入的start_pos完成更新。
  2. 对$kv\_state$和$score\_state$进行数据重排，再对$score\_state$进行softmax运算将softmax结果与$kv\_state$做Mul计算，后进行Reducesum运算。
  3. 根据输入数据norm_weight、rope_sin、rope_cos，进行 RmsNorm 和 ROPE 运算，根据 rotate 决定是否需要额外进行 Hadamard Transform，得到$cmp\_kv$结果输出。

## 函数原型

```
torch.ops.pypto.compressor(
    x,
    kv_state,
    score_state,
    kv_block_table,
    state_block_table,
    sin,
    cos,
    wkv,
    wgate,
    ape,
    weight,
    hadamard,
    start_pos,
    ratio,
    rope_head_dim,
    rotate
) -> (Tensor)
```

## 参数说明

- **x**（`Tensor`）：必选参数，表示原始不经压缩的数据，对应公式中的$X$。不支持非连续，数据格式支持$ND$，数据类型支持`bfloat16`。支持输入shape[B,S,H]。
- **kv\_state**（`Tensor`）：必选参数，表示kv\_state的历史数据，对应公式中的$kv\_state$。不支持非连续，数据格式支持$ND$，数据类型支持`float32`。支持输入shape[block_num,block_size,coff*D]。
- **score\_state**（`Tensor`）：必选参数，表示score\_state中的历史数据, 对应公式中的$score\_state$。不支持非连续，数据格式支持$ND$，数据类型支持`float32`。支持输入shape[block_num,block_size,coff*D]。
- **kv\_block\_table**（`Tensor`）：必选参数，表示kv\_state中的历史数据的page table。不支持非连续，数据格式支持$ND$，数据类型支持`int32`。支持输入shape[B, ceil(max_S/block_size)]。
- **score\_block\_table**（`Tensor`）：必选参数，表示score\_state中的历史数据的page table。不支持非连续，数据格式支持$ND$，数据类型支持`int32`。支持输入shape[B, ceil(max_S/block_size)]。
- **sin**（`Tensor`）：必选参数，表示Rope计算的权重系数。数据类型支持`bfloat16`。支持输入shape[min(T,T//ratio+B),rope_head_dim]。
- **cos**（`Tensor`）：必选参数，表示Rope计算的权重系数。数据类型支持`bfloat16`。支持输入shape[min(T,T//ratio+B),rope_head_dim]。
- **wkv**（`Tensor`）：必选参数，表示KV和压缩权重的权重参数，对应公式中的$W^{KV}$。不支持非连续，数据格式支持$ND$，数据类型支持`bfloat16`。支持输入shape[coff*D,H]。
- **wgate**（`Tensor`）：必选参数，表示KV和压缩权重的权重参数，对应公式中的$W^{Gate}$。不支持非连续，数据格式支持$ND$，数据类型支持`bfloat16`。支持输入shape[coff*D,H]。
- **ape**（`Tensor`）：必选参数，表示输入的positional biases，对应公式中的$Ape$。不支持非连续，数据格式支持$ND$，数据类型支持`float32`。支持输入shape[ratio,coff*D]。
- **weight**（`Tensor`）：必选参数，表示计算RmsNorm时的权重系数。数据类型支持`bfloat16`。支持输入shape[D,]。
- **start\_pos**（`Tensor`）：可选参数，表示计算起始位置。不支持非连续，数据格式支持$ND$，数据类型支持`int32`。支持输入shape[B,]。当输入为None时，表示从0开始进行计算。
- **hadamard**（`Tensor`）：可选参数，表示 Hadamard Transform 的权重矩阵。不支持非连续，数据格式支持$ND$，数据类型支持`bfloat16`。支持输入shape[D, D]。
- **ratio**（`int`）：必选参数，表示数据压缩率。支持4/128。
- **rope\_head\_dim**（`int`）：必选参数，表示rope_cos和rope_sin的hidden层最小单元。目前仅支持64。
- **rotate**（`bool`）：必选参数，表示是否需要额外进行 Hadamard Transform。

## 返回值说明

- **out**（`Tensor`）：必选输出，表示压缩后的数据。不支持非连续，数据格式支持$ND$。数据类型支持`bfloat16`。支持输出shape[min(T, T // ratio + B), D]。不压缩的条目的输出数据值是零。

## 约束说明

- 该接口支持 B 泛化。
- S 支持 1/2/3/4。
- D 支持128/512。
- H 支持4096。
- block_size 支持 128。

## 调用方法

```
python3  tests/ops/deepseek_v4/test_compressor.py
```

# Quant_Lightning_Indexer_Prolog


## 产品支持情况

- Ascend 950PR：不支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 功能说明

- API功能：`QuantLightningIndexerProlog`算子旨在完成以下公式描述的Prolog计算，主要为后续LightningIndexer计算提供输入q、weight及q_scale。
- 计算公式：

  q, q_scale的计算公式为：

  $$
  q\_tmp = \text{qr}@{idx\_wq\_b} \cdot \text{qr\_scale} \cdot \text{idx\_wq\_b\_scale}
  $$

  $$
  q\_hadamard = \text{Cat}(\{q\_tmp[:, :nope\_dim], Rope(q\_tmp[:, nope\_dim:])\}, -1)@hadamard
  $$

  $$
  q, q\_scale = Quant(q\_hadamard)
  $$

  其中，Rope表示旋转位置编码计算，Quant表示量化计算。
Weights的计算公式为：

$$
weights = x@\text{weights\_proj} \cdot {\frac{1}{\sqrt{\text{idx\_nq} \cdot \text{head\_dim}}}}
$$

## 函数原型

```
torch.ops.pypto.quant_lightning_indexer_prolog(
    qr,
    idx_wq_b,
    x,
    weights_proj,
    cos,
    sin,
    hadamard,
    qr_scale,
    idx_wq_b_scale
) -> (Tensor, Tensor, Tensor)
```

## 参数说明

- **qr**（`Tensor`）：必选参数，进行q矩阵计算的左输入，不支持非连续，数据格式支持ND，数据类型支持`int8`。`layout_query`为TND时shape为[t, q_lora_rank]。
- **idx_wq_b**（`Tensor`）：必选参数，进行q矩阵计算的右输入，不支持非连续，数据格式支持ND，数据类型支持`int8`。`layout_query`为TND时shape为[q_lora_rank, idx_nq*head_dim]。
- **x**（`Tensor`）：必选参数，进行weights矩阵计算的左输入，不支持非连续，数据格式支持ND，数据类型支持`bfloat16`。`layout_query`为TND时shape为[t， h]。
- **weights_proj**（`Tensor`）：必选参数，进行weights矩阵计算的右输入，不支持非连续，数据格式支持ND，数据类型支持`bfloat16`。`layout_query`为TND时shape为[h, idx_nq]。
- **cos**（`Tensor`）：必选参数， 用于q的位置编码计算，不支持非连续，数据格式支持ND，数据类型支持`bfloat16`。`layout_query`为TND时shape为[t， rope_dim]。
- **sin**（`Tensor`）：必选参数，用于q的位置编码计算，不支持非连续，数据格式支持ND，数据类型支持`bfloat16`。`layout_query`为TND时shape为[t， rope_dim]。
- **hadamard**（`Tensor`）：必选参数， 进行q的hadamard矩阵计算时的右输入，不支持非连续，数据格式支持ND，数据类型支持`bfloat16`。`layout_query`为TND时shape为[head_dim, head_dim]。
- **qr_scale**（`Tensor`）：必选参数，qr矩阵计算后的反量化系数输入，不支持非连续，数据格式支持ND，数据类型支持`float32`。`layout_query`为TND时shape为[t, 1]。
- **idx_wq_b_scale**（`Tensor`）：必选参数，用于qr矩阵计算后的乘法输入，不支持非连续，数据格式支持ND，数据类型支持`float32`。`layout_query`为TND时shape为[idx_nq * head_dim, 1]。

## 返回值说明

- **q**（`Tensor`）：必选输出，公式中的输出q。数据格式支持ND，数据类型支持`int8`。当layout\_query为TND时shape为[t, idx_nq * head_dim]。
- **weights**（`Tensor`）：必选输出，公式中的输出weights。数据格式支持ND，数据类型支持`float16`。当layout\_query为TND时shape为[t, idx_nq]。
- **q_scale**（`Tensor`）：必选输出，公式中的输出q_scale。数据格式支持ND，数据类型支持`float16`。当layout\_query为TND时shape为[t, idx_nq]。

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持aclgraph模式。
- q_lora_rank, idx_nq, head_dim, h, rope_dim仅支持默认值，t支持[1-64k]。
- 所有输入输出数据排布仅支持TND。
- 所有输入输出的数据类型仅支持所列场景，不支持额外类型。

## 调用方法

```
python3  tests/ops/deepseek_v4/test_lightning_indexer_prolog_quant_v4.py

```

# Mla_Prolog


## 产品支持情况

- Ascend 950PR：不支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 功能说明

MLA Prolog 模块将hidden states $x$ 转换为 $Query$和 ${Key-Value}$。

## 计算公式

1. $Query(q)$ 的计算
   Query 的计算，包括两次采样和 RmsNorm（其中第二次 RmsNorm 权重恒为 1），最后对 -1 轴的后 rope\_dim 维度进行 inplace interleaved rope 计算：

$$
c^Q = RmsNorm(x @ wq\_a)
$$

$$
q = RmsNorm(c^Q @ wq\_b)
$$

$$
q[..., -rope\_dim:] = ROPE(q[..., -rope\_dim:])
$$

2. $Key-Value(kv)$ 的计算
   kv 的计算，包括一次下采样和 RmsNorm，最后对 -1 轴的后 rope\_dim 维度进行 inplace interleaved rope 计算：

$$
kv = RmsNorm(x @ wkv)
$$

$$
kv[..., -rope\_dim:] = ROPE(kv[..., -rope\_dim:])
$$

## 函数原型

```
torch.ops.pypto.mla_prolog_quant(
    token_x,
    wq_a,
    wq_b,
    wkv,
    rope_cos,
    rope_sin,
    gamma_cq,
    gamma_ckv,
    wq_b_scale
) -> (Tensor, Tensor, Tensor, Tensor)
```

## 参数说明

- **token_x**（`Tensor`）：公式中用于计算Query和Key-Value的输入tensor，不支持非连续的 Tensor，数据格式支持ND，数据类型支持`bfloat16`，shape为[t, h]。
- **wq_a**（`Tensor`）：公式中用于计算Query的下采样权重矩阵$wq_a$，数据格式支持NZ/ND，数据类型支持`bfloat16`，shape为[h, q_lora_rank]。
- **wq_b**（`Tensor`）：公式中用于计算Query的上采样权重矩阵$wq_b$，数据格式支持NZ/ND，数据类型支持`int8`，shape为[q_lora_rank, num_heads*head_dim]。
- **wkv**（`Tensor`）：公式中用于计算Key-Value的下采样权重矩阵$wkv$，数据格式支持NZ/ND，数据类型支持`bfloat16`，shape为[h, head_dim]。
- **rope_cos**（`Tensor`）：用于计算旋转位置编码的余弦参数矩阵，不支持非连续的 Tensor，数据格式支持ND，数据类型支持`bfloat16`，shape为[t, rope_dim]。
- **rope_sin**（`Tensor`）：用于计算旋转位置编码的正弦参数矩阵，不支持非连续的 Tensor，数据格式支持ND，数据类型支持`bfloat16`，shape为[t, rope_dim]。
- **gamma_cq**（`Tensor`）：计算$c^Q$的RmsNorm公式中的$\gamma$参数，不支持非连续的 Tensor，数据格式支持ND，数据类型支持`bfloat16`，shape为[q_lora_rank]。
- **gamma_ckv**（`Tensor`）：计算$c^{KV}$的RmsNorm公式中的$\gamma$参数，不支持非连续的 Tensor，数据格式支持ND，数据类型支持`bfloat16`，shape为[head_dim]。
- **wq_b_scale**（`Tensor`）：用于矩阵乘wq_b后反量化操作的per-channel参数，不支持非连续的 Tensor。数据格式支持ND，数据类型支持`float`，shape为[num_heads*head_dim, 1]。


## 返回值说明

- **q_out**（`Tensor`）：公式中Query的输出tensor（对应公式中的$q$），不支持非连续的 Tensor。数据格式支持ND，数据类型支持`bfloat16`，shape为[t, num_heads, head_dim]。
- **kv_out**（`Tensor`）：公式中Key-Value的输出tensor（对应公式中的$kv$），不支持非连续的 Tensor。数据格式支持ND，数据类型支持`bfloat16`，shape为[t, head_dim]。
- **qr_out**（`Tensor`）：公式中Query做完第一次rmsnorm和quant后的输出tensor（对应公式中的$c^Q$，不支持非连续的 Tensor，数据格式支持ND，数据类型支持`int8`, shape为[t, q_lora_rank]。
- **qr_scale_out**（`Tensor`）：公式中Query做完第一次rmsnorm后的输出tensor（对应公式中的$c^Q$，不支持非连续的 Tensor，数据格式支持ND，数据类型支持`float32`, shape为[t, 1]。

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持aclgraph模式。
- head_dim支持512，h支持4096，q_lora_rank支持1024，num_heads支持64，rope_dim支持64。
- t值域范围支持[1, 64k]。
- A5暂不支持int8量化版本。
- 非量化实现可以参考example。


## 调用方法

```
量化：
python3  tests/ops/deepseek_v4/test_mla_prolog_quant_v4.py

非量化：
python3  tests/ops/deepseek_v4/test_mla_prolog_v4.py

```

# Sliding_Window_Attention


## 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 功能说明

- API功能：`SlidingWindowAttention`算子旨在完成以下公式描述的Attention计算，支持Sliding Window Attention。
- 计算公式：

  $$
  O = \text{softmax}(Q@\tilde{K}^T \cdot \text{softmax\_scale})@\tilde{V}
  $$

  其中$\tilde{K}=\tilde{V}$为基于kv_cache、kv_win等入参控制的实际参与计算的 $KV$。

## 函数原型

```
torch.ops.pypto.sliding_window_attention(
    q,
    ori_block_table,
    ori_kv,
    seqused_kv,
    sinks,
    win_size,
    mask,
    cu_seqlens_q
) -> (Tensor)
```

## 参数说明

- **q**（`Tensor`）：必选参数，对应公式中的$Q$，不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，shape为[T1, N1,D]，其中N1仅支持64，D仅支持512。
- **ori_block_table**（`Tensor`）：必选参数，表示PageAttention中oriKvCache存储使用的block映射表。数据格式支持ND，数据类型支持`int32`，shape为2维，其中第一维长度为B，第二维长度不小于所有batch中最大的S2对应的block数量，即S2\_max / block\_size向上取整， block\_size仅支持128。
- **ori_kv**（`Tensor`）：必选参数，为原始的KV，不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，shape为[block\_num1, block\_size, N2, D]，其中block\_num1为PageAttention时block总数，block\_size为一个block的token数，仅支持128，N2仅支持1。
- **seqused_kv**（`Tensor`）：必选参数，表示不同Batch中`ori_kv`的输入样本序列长度S2，维度为B，数据格式支持ND，数据类型支持`int32`。
- **sinks**（`Tensor`）：必选参数，注意力下沉tensor，数据格式支持ND，数据类型支持`float32`，shape为[N1]。
- **win_size**（`Int`）：必选参数，窗口大小，数据类型支持`int32`，仅支持128。
- **mask**（`Tensor`）：必选参数，计算过程中使用到的掩码，数据类型支持`bool`，生成方式固定，调用get_mask方法，shape为[4 * N1, 4 * block\_size]，其中N1仅支持64，block\_size仅支持128。
- **cu_seqlens_q**（`Tensor`）：必选参数，表示不同Batch中`q`的有效token数，维度为B+1，大小为参数中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须>=前一个元素的值，数据类型支持`int32`。

## 返回值说明

- **atten\_out**（`Tensor`）：注意力计算结果。数据格式支持ND，数据类型支持`bfloat16`，shape为[T1, N1, D]。

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持aclgraph模式。
- 参数q中的D和ori_kv的D值相等为512。
- 参数q、ori_kv的数据类型必须保持一致。
- block_size支持128。

## 调用方法

```
python3  tests/ops/deepseek_v4/test_win_attention.py
```

# Sparse_Compress_Flash_Attention


## 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 功能说明

- API功能：`SparseCompressFlashAttention`算子旨在完成以下公式描述的Attention计算，支持Sparse Compressed Attention。
- 计算公式：

  $$
  O = \text{softmax}(Q@\tilde{K}^T \cdot \text{softmax\_scale})@\tilde{V}
  $$

  其中$\tilde{K}=\tilde{V}$为基于ori_kv、cmp_kv以及cmp_kv等入参控制的实际参与计算的 $KV$。

## 函数原型

```
torch.ops.pypto.sparse_compress_flash_attention(
    query,
    q_act_seqs,
    ori_kv,
    cmp_kv,
    ori_block_table,
    cmp_block_table,
    atten_sink,
    seqused_kv,
    cmp_sparse_indices,
    softmax_scale,
    win_size,
    cmp_ratio
) -> (Tensor)
```

## 参数说明

- **query**（`Tensor`）：必选参数，对应公式中的$Q$，不支持非连续，数据格式支持ND，数据类型支持`bfloat16`。shape为[T1*N1,D]，其中，N1仅支持64。
- **q_act_seqs**（`Tensor`）：必选参数，在`layout_query`为TND时生效。表示不同Batch中`q`的有效token数，维度为B+1，大小为参数中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须>=前一个元素的值，数据类型支持`int32`。
- **ori_kv**（`Tensor`）：必选参数，对应公式中的$\tilde{K}和\tilde{V}$的一部分，为原始不经压缩的KV，不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，`layout_kv`为PA_ND时shape为[block\_num1* ori\_block\_size, KV\_N*D]，其中block\_num1为PageAttention时block总数，ori\_block\_size为一个block的token数，ori\_block\_size取值为128，KV_N仅支持1。
- **cmp_kv**（`Tensor`）：必选参数，对应公式中的$\tilde{K}和\tilde{V}$的一部分，为经过压缩的KV，不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，`layout_kv`为PA_ND时shape为[block\_num2* cmp\_block\_size, KV\_N*D]，其中block\_num2为PageAttention时block总数，cmp\_block\_size为一个block的token数，cmp\_block\_size取值为128。
- **ori_block_table**（`Tensor`）：必选参数，表示PageAttention中oriKvCache存储使用的block映射表。数据格式支持ND，数据类型支持`int32`，shape为2维，其中第一维长度为B，第二维长度不小于所有batch中最大的S2对应的block数量，即S2\_max / block\_size向上取整。
- **cmp_block_table**（`Tensor`）：必选参数，表示PageAttention中cmpKvCache存储使用的block映射表。数据格式支持ND，数据类型支持`int32`，shape为2维，其中第一维长度为B，第二维长度不小于所有batch中最大的S3对应的block数量，即S3\_max / block\_size向上取整。
- **atten_sink**（`Tensor`）：必选参数，注意力下沉tensor，数据格式支持ND，数据类型支持`float32`，shape为[N1]。
- **seqused_kv**（`Tensor`）：必选参数，表示不同Batch中`ori_kv`实际参与运算的token数，维度为B，数据格式支持ND，数据类型支持`int32`，不输入则所有token均参与运算。
- **cmp_sparse_indices**（`Tensor`）：必选参数，代表离散取cmpKvCache的索引，不支持非连续，数据格式支持ND，数据类型支持`int32`。当`layout_query`为TND时，shape需要传入[Q\_T * KV\_N, K2]，其中K2为对`cmp_kv`一次离散选取的token数，K2仅支持512。
- **softmax_scale**（`double`）：必选参数，代表缩放系数，作为q与ori_kv和cmp_kv矩阵乘后Muls的scalar值，数据类型支持`float`。
- **win_size**（`int`）：必选参数，窗口大小，数据类型支持int32，仅支持128。
- **cmp_ratio**（`int`）：必选参数，表示对ori_kv的压缩率，数据类型支持`int`，数据支持4。

## 返回值说明

- **attention\_out**（`Tensor`）：公式中的输出。数据格式支持ND，数据类型支持`bfloat16`。shape为[T1*N1,D]。

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持aclgraph模式。
- 参数q中的D和ori_kv、cmp_kv的D值相等为512。
- 参数q、ori_kv、cmp_kv的数据类型必须保持一致。
- 为了提高算子性能，当前q、ori_kv、cmp_kv、attention_out进行了高维合轴处理。
- 仅支持TND格式。
- block_size支持128。

## 调用方法

```
python3  tests/ops/deepseek_v4/test_sparse_compress_flash_attention.py
```

# hc_pre


## 产品支持情况

- Ascend 950PR：不支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 功能说明

- API功能：hc_pre算子旨在完成以下计算过程。
- 计算过程：

1. 计算 RMSNorm 的分母

$$
rsqrt = \sqrt{\frac{1}{\frac{1}{n}\sum_{i=1}^n x_i^2 + \epsilon}}
$$

2. 计算 mixes

$$
mixes = (x @ hc\_fn) \odot rsqrt
$$

3. Sinkhorn-Knopp 算法

$$
pre, post, comb = sinkhorn(mixes, hc\_scale, hc\_base, hc\_mult, hc\_sinkhorn\_iters)
$$

Sinkhorn-Knopp 算法每次迭代会进行逐行归一化，再做逐列归一化，$hc\_sinkhorn\_iters$ 控制迭代次数。

4. 利用 pre 和 x 计算 y

$$
y = rowsum(pre \odot x)
$$

## 函数原型

```
torch.ops.pypto.hc_pre(
    x,
    hc_fn,
    hc_scale,
    hc_base,
    hc_mult: int=4,
    hc_split_sinkhorn_iters: int=20,
    hc_eps: float=1e-6
) -> (Tensor, Tensor, Tensor)
```

## 参数说明

- **x**（`Tensor`）：必选参数，对应公式中的$x$，不支持非连续，数据格式支持ND，数据类型支持`bfloat16`。`layout_x`为TND时shape为[t, hc_mult, h]。
- **hc_fn**（`Tensor`）：必选参数，对应公式中的$hc\_fn$，不支持非连续，数据格式支持ND，数据类型支持`float32`，`layout_x`为TND时shape为[mix_hc, hc_mult*h]，其中mix_hc = (2+hc_mult)*hc_mult。
- **hc_scale**（`Tensor`）：必选参数，对应公式中的$hc\_scale$，不支持非连续，数据格式支持ND，数据类型支持`float32`，shape为[3, ]。
- **hc_base**（`Tensor`）：对应公式中的$hc\_base$，不支持非连续，数据格式支持ND，数据类型支持`float32`，shape为[mix_hc, ]。
- **hc_mult**（`int`）：可选参数，表示mHC中的expansion rate，数据类型支持`int`，默认值为`4`。
- **hc_split_sinkhorn_iters**（`int`）：可选参数，表示sinkhornde 迭代次数，数据类型支持`int`， 默认值`20`。
- **hc_eps**（`float`）：可选参数，表示RMSNorm分母计算与Sinkhorn-Knopp计算中用于数值稳定的加法值，数据类型支持`float`， 默认值为`1e-6`。

## 返回值说明

- **y**（`Tensor`）：公式中的输出。数据格式为ND，数据类型为`bfloat16`。当layout\_x为TND时shape为[t, h]。
- **post**（`Tensor`）：公式中sinkhorn的输出post，数据格式为ND，数据类型为`float`。当layout\_x为TND时shape为[t, hc_mult]。
- **comb**（`Tensor`）：公式中sinkhorn的输出comb，数据格式为ND，数据类型为`float`。当layout\_x为TND时shape为[t, hc_mult, hc_mult]。

## 约束说明

- 该接口支持推理场景下使用。
- 入参x中的shape [t, hc_mult, h]中，h仅支持`4096`。
- 入参的shape、dtype等需与参数说明保持一致。
- t的值域范围为[1, 64k]

## 调用方法

```
python3  tests/ops/deepseek_v4/test_hc_pre.py

```