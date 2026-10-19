# sum_lstm Operator

## 产品支持情况

- Ascend 950PR：不支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 功能说明

sum_lstm算子是Arctic-Inference框架中基于LSTM的speculator的核心机制。它具有以下特点：

- **输入融合**: 将主要状态输入与额外输入信号进行加权融合
- **RMS归一化**: 使用RMSNorm代替LayerNorm，提高计算效率
- **GELU激活**: 使用近似GELU激活函数
- **门控机制**: 标准的LSTM门控逻辑，控制信息的遗忘、输入和输出
- **状态更新**: 通过遗忘门和输入门更新细胞状态
- **输出生成**: 通过输出门控制最终的隐藏状态输出

该算子主要用于序列预测和生成任务中的speculative decoding，加速大语言模型的推理过程。

## 数学公式

Sum LSTM算子的计算过程如下：

1. **输入融合**:
   ```
   fused = states_4d + alpha * z4_4d
   ```

2. **门控分割**:
   将fused按最后一维分割为4个部分，每个部分大小为 `D_GATE`:
   ```
   pre_f, pre_i, pre_o, pre_c = split(fused, 4, dim=-1)
   ```

3. **门控激活**:
   ```
   f_gate = sigmoid(pre_f)    # 遗忘门
   i_gate = sigmoid(pre_i)    # 输入门
   o_gate = sigmoid(pre_o)    # 输出门
   ```

4. **细胞候选处理**:
   ```
   c_cand_norm = RMSNorm(pre_c, eps_cell)
   c_cand_norm = c_cand_norm * w_cell + b_cell  (如果权重存在)
   c_act = GELU(c_cand_norm)
   ```

5. **细胞状态更新**:
   ```
   c_new = prev_cell * f_gate + c_act * i_gate
   ```

6. **隐藏状态处理**:
   ```
   h_temp = RMSNorm(c_new, eps_state)
   h_temp = h_temp * w_state + b_state  (如果权重存在)
   h_act = GELU(h_temp)
   ```

7. **最终输出**:
   ```
   h_new = h_act * o_gate
   ```

其中：
- **RMSNorm(x, eps)**: `x * rsqrt(mean(x^2) + eps)`
- **GELU(x)**: `x * sigmoid(1.702 * x)` (近似实现)

## 函数原型

```python
def sum_lstm_kernel(
    states_4d: pypto.Tensor((BATCH_SIZE, D_GATE_4), pypto.DT_FP16),
    z4_4d: pypto.Tensor((BATCH_SIZE, D_GATE_4), pypto.DT_FP16),
    prev_cell: pypto.Tensor((BATCH_SIZE, D_GATE), pypto.DT_FP16),
    w_cell: pypto.Tensor((D_GATE,), pypto.DT_FP16),
    b_cell: pypto.Tensor((D_GATE,), pypto.DT_FP16),
    w_state: pypto.Tensor((D_GATE,), pypto.DT_FP16),
    b_state: pypto.Tensor((D_GATE,), pypto.DT_FP16),
    config: LstmConfig,
    h_out: pypto.Tensor((BATCH_SIZE, D_GATE), pypto.DT_FP16),
    c_out: pypto.Tensor((BATCH_SIZE, D_GATE), pypto.DT_FP16)
) -> None
```

## 参数说明

- **states_4d**: 张量，形状为 `(BATCH_SIZE, D_GATE_4)`，数据类型为 FP16。表示LSTM的状态输入，包含4个门控信号（遗忘门、输入门、输出门、细胞候选）。
- **z4_4d**: 张量，形状为 `(BATCH_SIZE, D_GATE_4)`，数据类型为 FP16。表示额外的输入信号，与states_4d融合。
- **prev_cell**: 张量，形状为 `(BATCH_SIZE, D_GATE)`，数据类型为 FP16。表示前一时刻的细胞状态。
- **w_cell**: 张量，形状为 `(D_GATE,)`，数据类型为 FP16。细胞路径的权重参数。
- **b_cell**: 张量，形状为 `(D_GATE,)`，数据类型为 FP16。细胞路径的偏置参数。
- **w_state**: 张量，形状为 `(D_GATE,)`，数据类型为 FP16。状态路径的权重参数。
- **b_state**: 张量，形状为 `(D_GATE,)`，数据类型为 FP16。状态路径的偏置参数。
- **config**: LstmConfig 对象，包含超参数：
  - `alpha`: 融合系数，默认 0.1
  - `eps_cell`: 细胞RMSNorm的epsilon，默认 1e-6
  - `eps_state`: 状态RMSNorm的epsilon，默认 1e-6

## 返回值说明

- **h_out**: 张量，形状为 `(BATCH_SIZE, D_GATE)`，数据类型为 FP16。表示新的隐藏状态输出。
- **c_out**: 张量，形状为 `(BATCH_SIZE, D_GATE)`，数据类型为 FP16。表示新的细胞状态输出。

## 性能测试结果

> 测试环境: Ascend 910B, CANN 8.5.0
>
> 测试方法: Swimlane (泳道图) — `debug_options={"runtime_debug_mode": 1}`, 解析 `merged_swimlane.json` X 事件 span

| 算子 | 输入 shape | 输入 dtype | kernel 耗时 (μs) | 数据来源 |
|--------|-----------|-----------|-----------------|---------|
| sum_lstm | states=[32,16384], z4=[32,16384], cell=[32,4096] | float16 | 51.5 | Swimlane |

## 调用示例
- 算子入口与对外签名见 [sum_lstm.py](sum_lstm.py)
- 测试与 golden 实现见 [tests/ops/arctic/test_sum_lstm.py](../../../../../tests/ops/arctic/test_sum_lstm.py)
