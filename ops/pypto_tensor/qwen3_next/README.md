# Qwen3Next 样例 (Examples)

本目录包含了 PyPTO Qwen3Next 模型的开发样例代码。我们对 Qwen3Next 的核心注意力机制进行了实现，交付了 **Chunk Gated Delta Rule** 算子，该算子是一种高效的线性注意力机制，专为长序列建模场景设计。


## 产品支持情况

- Ascend 950PR：不支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 参数说明/约束

- shape 格式字段含义说明

| 字段名 | 英文全称/含义 | 取值规则与说明 |
|--------|---------------|----------------|
| T | Total Tokens（总词元数） | 取值范围：所有 batch 的序列长度之和 |
| B | Batch（输入样本批量大小） | 取值范围：根据 act_seq_len 推断 |
| L | Chunk Length（分块长度） | 取值固定为：128 |
| Nqk | Query/Key Head Num（QK多头数） | 取值范围：支持 2、4、16 等 |
| Nv | Value Head Num（V多头数） | 取值范围：支持 4、8、32 等，需满足 Nv // Nqk 为整数（GQA分组） |
| D | Head Dimension（头维度） | 取值固定为：128 |
| S | Sequence Length（序列长度） | 取值范围：支持动态序列长度 |

---

# chunk_gated_delta_rule


## 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 功能说明

`chunk_gated_delta_rule` 算子对应 Qwen3Next 网络中的核心注意力计算模块，实现了基于 **Gated Delta Rule** 的分块线性注意力机制。该算子融合了以下关键操作：

- **L2 归一化**（L2 Normalization）：对 Query 和 Key 进行归一化处理
- **门控累积计算**（Gate Cumulative Sum）：计算门控信号的累积和
- **衰减掩码生成**（Decay Mask Generation）：生成时序衰减掩码矩阵
- **预注意力计算**（Pre-Attention）：计算 KKT 矩阵与衰减掩码的乘积
- **矩阵求逆**（Matrix Inversion）：对注意力矩阵进行高效求逆
- **循环状态注意力**（Recurrent State Attention）：结合历史状态计算最终注意力输出

该算子通过分块处理和算子融合，显著降低了传统 Attention 的 O(n²) 复杂度，实现了线性时间复杂度的注意力计算，特别适用于超长序列的高效推理场景。

## 数学公式

### L2 归一化

$$
\text{L2Norm}(\mathbf{x}) = \frac{\mathbf{x}}{\sqrt{\sum_{i=1}^{d} x_i^2 + \epsilon}}
$$

$$
\mathbf{q}_{norm} = \text{L2Norm}(\mathbf{q}), \quad \mathbf{k}_{norm} = \text{L2Norm}(\mathbf{k})
$$

### 门控累积和

$$
\mathbf{g}_{cum} = \text{cumsum}(\mathbf{g}) = \mathbf{T}_{tril} \cdot \mathbf{g}
$$

其中 $\mathbf{T}_{tril}$ 为下三角全1矩阵。

### 衰减掩码

$$
\mathbf{D}_{decay} = \exp\left((\mathbf{g}_{cum} - \mathbf{g}_{cum}^T) \odot \mathbf{T}_{tril}\right)
$$

### 预注意力矩阵

$$
\mathbf{k}_{\beta} = \mathbf{k} \odot \boldsymbol{\beta}
$$

$$
\mathbf{A} = (\mathbf{k}_{\beta} \cdot \mathbf{k}^T) \odot \mathbf{D}_{decay} \odot \mathbf{M}_{mask}
$$

### 矩阵求逆（迭代法）

$$
\mathbf{A}^{-1} = (\mathbf{I} - \mathbf{A})^{-1} \approx \mathbf{I} + \mathbf{A} + \mathbf{A}^2 + \cdots
$$

采用分块递推方式高效计算：

$$
\mathbf{A}^{-1}_{i,:i} = \mathbf{A}^{-1}_{i-1,:i-1} + \mathbf{A}_{i,:i} \cdot \mathbf{A}^{-1}_{i-1,:i-1}
$$

### Value 和 Key 累积衰减

$$
\mathbf{v}_{out} = \mathbf{A}^{-1} \cdot (\mathbf{v} \odot \boldsymbol{\beta})
$$

$$
\mathbf{k}_{cumdecay} = \mathbf{A}^{-1} \cdot (\mathbf{k}_{\beta} \odot \exp(\mathbf{g}_{cum}))
$$

### 循环状态注意力

$$
\mathbf{v}' = \mathbf{k}_{cumdecay} \cdot \mathbf{S}^T
$$

$$
\mathbf{o}_{inter} = (\mathbf{q} \odot \exp(\mathbf{g}_{cum})) \cdot \mathbf{S}^T
$$

$$
\mathbf{o}_{chunk} = \mathbf{o}_{inter} + (\mathbf{q} \cdot \mathbf{k}^T \odot \mathbf{D}_{decay} \odot \mathbf{T}_{tril}) \cdot (\mathbf{v}_{out} - \mathbf{v}')
$$

### 状态更新

$$
\mathbf{S}_{new} = \mathbf{S} \cdot \exp(g_{last}) + \mathbf{v}_{new}^T \cdot \mathbf{k}_{gexp}
$$

其中：
- $\mathbf{k}_{gexp} = \mathbf{k} \odot \exp(g_{last} - \mathbf{g})$
- $g_{last}$ 为当前块最后一个位置的门控值

## 函数原型

```python
def chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    gate: torch.Tensor,
    states: torch.Tensor,
    mask: torch.Tensor,
    tril_mask: torch.Tensor,
    eye: torch.Tensor,
    act_seq_len: torch.Tensor,
    core_attn_out: torch.Tensor,
    last_state_data: torch.Tensor
):
```

## 参数说明

>**说明：**<br>
>
>- T 表示所有 batch 序列长度的总和、B 表示输入样本批量大小、L 表示分块长度（固定为128）、Nqk 表示 Query/Key 的多头数量、Nv 表示 Value 的多头数量（支持 GQA，需满足 Nv // Nqk 为整数）、D 表示每个注意力头的维度（固定为128）。

-   **query**（`Tensor`）：查询向量。不支持非连续，数据格式支持ND，数据类型支持`float32`，shape为[T, Nqk, D]。

-   **key**（`Tensor`）：键向量。不支持非连续，数据格式支持ND，数据类型支持`float32`，shape为[T, Nqk, D]。

-   **value**（`Tensor`）：值向量。不支持非连续，数据格式支持ND，数据类型支持`float32`，shape为[T, Nv, D]。

-   **beta**（`Tensor`）：Beta 缩放因子，用于控制 Key 和 Value 的加权。不支持非连续，数据格式支持ND，数据类型支持`float32`，shape为[T, Nv]。

-   **gate**（`Tensor`）：门控信号，用于控制时序衰减。不支持非连续，数据格式支持ND，数据类型支持`float32`，shape为[T, Nv]。

-   **states**（`Tensor`）：初始循环状态矩阵。不支持非连续，数据格式支持ND，数据类型支持`float32`，shape为[B, Nv, D, D]。

-   **mask**（`Tensor`）：注意力掩码矩阵（下三角负值掩码）。不支持非连续，数据格式支持ND，数据类型支持`float32`，shape为[L, L]。

-   **tril_mask**（`Tensor`）：下三角掩码矩阵。不支持非连续，数据格式支持ND，数据类型支持`float32`，shape为[L, L]。

-   **eye**（`Tensor`）：用于矩阵求逆的单位矩阵（经过特殊处理）。不支持非连续，数据格式支持ND，数据类型支持`float32`，shape为[16, 128]。

-   **act_seq_len**（`Tensor`）：各 batch 的累积序列长度索引。不支持非连续，数据格式支持ND，数据类型支持`int32`，shape为[B+1]。例如 [0, 4096, 8192] 表示 batch 0 的序列长度为 4096，batch 1 的序列长度为 4096。

-   **core_attn_out**（`Tensor`）：注意力计算的输出结果。数据格式支持ND，数据类型支持`float32`，shape为[T, Nv, D]。

-   **last_state_data**（`Tensor`）：更新后的循环状态矩阵，可用于下一个序列块的计算。数据格式支持ND，数据类型支持`float32`，shape为[B, Nv, D, D]。

## 子模块说明

### l2norm

L2 归一化模块，对 Query 和 Key 进行归一化处理。

```python
def l2norm(
    query: pypto.Tensor,
    key: pypto.Tensor,
    eps: float = 1e-6
) -> tuple[pypto.Tensor, pypto.Tensor]:
```

**参数：**

- **query**（`Tensor`）：输入查询张量，不支持非连续，数据格式支持 ND，数据类型支持 `float32`，shape 为 [L, D]。

- **key**（`Tensor`）：输入键张量，不支持非连续，数据格式支持 ND，数据类型支持 `float32`，shape 为 [L, D]。

- **eps**（`float`）：防止除零的小常数，数据类型支持 `float`，默认值为 1e-6。

**返回值：**

- **query_after_l2norm**（`Tensor`）：归一化后的查询张量，数据格式支持 ND，数据类型支持 `float32`，shape 为 [L, D]。

- **key_after_l2norm**（`Tensor`）：归一化后的键张量，数据格式支持 ND，数据类型支持 `float32`，shape 为 [L, D]。

---

### pre_attn

预注意力计算模块，计算门控累积、衰减掩码、预注意力矩阵和加权键。

```python
def pre_attn(
    gate_view: pypto.Tensor,
    key_view_2d: pypto.Tensor,
    beta_view: pypto.Tensor,
    tril: pypto.Tensor,
    mask: pypto.Tensor,
) -> tuple[pypto.Tensor, pypto.Tensor, pypto.Tensor, pypto.Tensor]:
```

**参数：**

- **gate_view**（`Tensor`）：门控信号，不支持非连续，数据格式支持 ND，数据类型支持 `float32`，shape 为 [L, 1]。

- **key_view_2d**（`Tensor`）：键张量，不支持非连续，数据格式支持 ND，数据类型支持 `float32`，shape 为 [L, D]。

- **beta_view**（`Tensor`）：Beta 缩放因子，不支持非连续，数据格式支持 ND，数据类型支持 `float32`，shape 为 [L, 1]。

- **tril**（`Tensor`）：下三角矩阵，不支持非连续，数据格式支持 ND，数据类型支持 `float32`，shape 为 [L, L]。

- **mask**（`Tensor`）：掩码矩阵，不支持非连续，数据格式支持 ND，数据类型支持 `float32`，shape 为 [L, L]。

**返回值：**

- **gate_cum**（`Tensor`）：门控累积和，数据格式支持 ND，数据类型支持 `float32`，shape 为 [L, 1]。

- **decay_mask**（`Tensor`）：衰减掩码，数据格式支持 ND，数据类型支持 `float32`，shape 为 [L, L]。

- **A**（`Tensor`）：预注意力矩阵，数据格式支持 ND，数据类型支持 `float32`，shape 为 [L, L]。

- **key_beta**（`Tensor`）：加权键，数据格式支持 ND，数据类型支持 `float32`，shape 为 [L, D]。

---

### inverse_pto

矩阵求逆模块，采用分块递推算法高效计算大矩阵的逆。

```python
def inverse_pto(
    attn: pypto.Tensor,
    eye: pypto.Tensor,
    size: int
) -> pypto.Tensor:
```

**参数：**

- **attn**（`Tensor`）：待求逆的注意力矩阵，不支持非连续，数据格式支持 ND，数据类型支持 `float32`，shape 为 [L, L]。

- **eye**（`Tensor`）：单位矩阵（经过特殊处理），不支持非连续，数据格式支持 ND，数据类型支持 `float32`，shape 为 [L, L]。

- **size**（`int`）：矩阵大小，数据类型支持 `int`。

**返回值：**

- **attn_inv**（`Tensor`）：逆矩阵，数据格式支持 ND，数据类型支持 `float32`，shape 为 [L, L]。

---

### inverse_pto_min_length

尾轴拼接优化的矩阵求逆模块。

```python
def inverse_pto_min_length(
    attn_dim0: pypto.Tensor,
    attn_dim1: pypto.Tensor,
    eye: pypto.Tensor,
    row_num: int,
    col_num: int,
) -> pypto.Tensor:
```

**参数：**

- **attn_dim0**（`Tensor`）：沿第0维拼接的注意力矩阵块，不支持非连续，数据格式支持 ND，数据类型支持 `float32`，shape 为 [L, L // 8]。

- **attn_dim1**（`Tensor`）：沿第1维拼接的注意力矩阵块，不支持非连续，数据格式支持 ND，数据类型支持 `float32`，shape 为 [L // 8, L]。

- **eye**（`Tensor`）：单位矩阵，不支持非连续，数据格式支持 ND，数据类型支持 `float32`，shape 为 [L, L]。

- **row_num**（`int`）：行数，数据类型支持 `int`，取值为 L // 8。

- **col_num**（`int`）：列数，数据类型支持 `int`，取值为 L。

**返回值：**

- **res**（`Tensor`）：求逆结果矩阵，数据格式支持 ND，数据类型支持 `float32`，shape 为 [L, L]。

---

### inverse_matmul

小矩阵求逆模块，用于分块矩阵求逆的子计算。

```python
def inverse_matmul(
    attn: pypto.Tensor,
    attn_1_1_inv: pypto.Tensor,
    attn_2_2_inv: pypto.Tensor,
    x_ofs: int,
    y_ofs: int,
    len: int,
) -> pypto.Tensor:
```

**参数：**

- **attn**（`Tensor`）：原始注意力矩阵，不支持非连续，数据格式支持 ND，数据类型支持 `float32`，shape 为 [L, L]。

- **attn_1_1_inv**（`Tensor`）：左上角子矩阵的逆，不支持非连续，数据格式支持 ND，数据类型支持 `float32`，shape 为 [len, len]。

- **attn_2_2_inv**（`Tensor`）：右下角子矩阵的逆，不支持非连续，数据格式支持 ND，数据类型支持 `float32`，shape 为 [len, len]。

- **x_ofs**（`int`）：行偏移量，数据类型支持 `int`。

- **y_ofs**（`int`）：列偏移量，数据类型支持 `int`。

- **len**（`int`）：子矩阵长度，数据类型支持 `int`。

**返回值：**

- **attn_inv**（`Tensor`）：合并后的逆矩阵，数据格式支持 ND，数据类型支持 `float32`，shape 为 [len * 2, len * 2]。

---

### cal_value_and_key_cumdecay

计算加权值和键累积衰减。

```python
def cal_value_and_key_cumdecay(
    attn: pypto.Tensor,
    value_view: pypto.Tensor,
    beta_view: pypto.Tensor,
    key_beta: pypto.Tensor,
    gate_cum: pypto.Tensor,
) -> tuple[pypto.Tensor, pypto.Tensor]:
```

**参数：**

- **attn**（`Tensor`）：逆注意力矩阵，不支持非连续，数据格式支持 ND，数据类型支持 `float32`，shape 为 [L, L]。

- **value_view**（`Tensor`）：值张量，不支持非连续，数据格式支持 ND，数据类型支持 `float32`，shape 为 [L, D]。

- **beta_view**（`Tensor`）：Beta 缩放因子，不支持非连续，数据格式支持 ND，数据类型支持 `float32`，shape 为 [L, D]。

- **key_beta**（`Tensor`）：加权键，不支持非连续，数据格式支持 ND，数据类型支持 `float32`，shape 为 [L, D]。

- **gate_cum**（`Tensor`）：门控累积和，不支持非连续，数据格式支持 ND，数据类型支持 `float32`，shape 为 [L, 1]。

**返回值：**

- **value_out**（`Tensor`）：加权值输出，数据格式支持 ND，数据类型支持 `float32`，shape 为 [L, D]。

- **key_cum_out**（`Tensor`）：键累积衰减输出，数据格式支持 ND，数据类型支持 `float32`，shape 为 [L, D]。

---

### recurrent_state_attn_all

循环状态注意力计算，结合历史状态计算最终注意力输出并更新状态。

```python
def recurrent_state_attn_all(
    query: pypto.Tensor,
    key: pypto.Tensor,
    value: pypto.Tensor,
    k_cumdecay: pypto.Tensor,
    gate: pypto.Tensor,
    state: pypto.Tensor,
    decay_mask: pypto.Tensor,
    tril: pypto.Tensor,
) -> tuple[pypto.Tensor, pypto.Tensor]:
```

**参数：**

- **query**（`Tensor`）：查询张量，不支持非连续，数据格式支持 ND，数据类型支持 `float32`，shape 为 [L, D]。

- **key**（`Tensor`）：键张量，不支持非连续，数据格式支持 ND，数据类型支持 `float32`，shape 为 [L, D]。

- **value**（`Tensor`）：值张量，不支持非连续，数据格式支持 ND，数据类型支持 `float32`，shape 为 [L, Dv]。

- **k_cumdecay**（`Tensor`）：键累积衰减，不支持非连续，数据格式支持 ND，数据类型支持 `float32`，shape 为 [L, Dk]。

- **gate**（`Tensor`）：门控累积和，不支持非连续，数据格式支持 ND，数据类型支持 `float32`，shape 为 [L, 1]。

- **state**（`Tensor`）：当前循环状态，不支持非连续，数据格式支持 ND，数据类型支持 `float32`，shape 为 [D, D]。

- **decay_mask**（`Tensor`）：衰减掩码，不支持非连续，数据格式支持 ND，数据类型支持 `float32`，shape 为 [L, L]。

- **tril**（`Tensor`）：下三角矩阵，不支持非连续，数据格式支持 ND，数据类型支持 `float32`，shape 为 [L, L]。

**返回值：**

- **chunk_attn_out**（`Tensor`）：当前块的注意力输出，数据格式支持 ND，数据类型支持 `float32`，shape 为 [L, D]。

- **state_new**（`Tensor`）：更新后的状态，数据格式支持 ND，数据类型支持 `float32`，shape 为 [Dv, Dk]。

---

## 支持的配置

| 配置项 | 支持范围 | 说明 |
|--------|----------|------|
| T（总序列长度） | 1K ~ 1M+ | 支持超长序列 |
| B（批量大小） | 1 ~ 128 | 根据显存调整 |
| Nqk（QK头数） | 2, 4, 16 | Query/Key 多头数 |
| Nv（V头数） | 4, 8, 32 | Value 多头数（GQA） |
| D（头维度） | 128 | 固定值 |
| L（块大小） | 128 | 固定值 |

## 运行时配置

```python
@pypto.frontend.jit(
    runtime_options={
        "stitch_function_max_num": 128,
        "ready_on_host_tensors": ["tensor_a", "tensor_b"]
    },
    debug_options={"runtime_debug_mode": 1},
)
def test_kernel(
    tensor_a: pypto.Tensor(),
    tensor_b: pypto.Tensor(),
    tensor_c: pypto.Tensor(),
):
    ******
```

## 性能特点

1. **线性时间复杂度**：相比传统 O(n²) 的 Attention，Gated Delta Rule 实现了 O(n) 的线性复杂度
2. **分块处理**：采用固定大小（128）的分块处理，有效利用 NPU 的并行计算能力
3. **算子融合**：多个计算步骤在单个 kernel 中完成，减少内存访问开销
4. **状态复用**：循环状态可跨序列块复用，支持流式推理
5. **GQA 支持**：支持 Grouped Query Attention，减少 KV Cache 内存占用

## 调用示例

- 算子实现详见 [gated_delta_rule_impl.py](./gated_delta_rule_impl.py)
- 测试用例详见 [tests/ops/qwen3_next/test_gated_delta_rule_qwen3_next.py](../../../../../tests/ops/qwen3_next/test_gated_delta_rule_qwen3_next.py)

## 性能测试结果

> 测试环境: Ascend 910B, CANN 8.5.0
>
> 测试方法: Swimlane (泳道图) — `debug_options={"runtime_debug_mode": 1}`, 解析 `merged_swimlane.json` X 事件 span
>
> `test_b2_nqk2_nv4_s1k` 测试触发 aicpu 异常 (`rtDeviceSynchronizeWithTimeout` / exception invalid error)，未能采集到 Swimlane 性能数据。其余测试用例均为大 shape，标记为 skip。

## 参考文献

- [Gated Delta Networks](https://arxiv.org/abs/2412.06464) - Gated Delta Rule 的理论基础
- [Linear Attention](https://arxiv.org/abs/2006.16236) - 线性注意力机制
- [GQA: Grouped Query Attention](https://arxiv.org/abs/2305.13245) - 分组查询注意力
