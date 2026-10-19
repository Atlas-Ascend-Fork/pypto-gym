# chunk_kda — Kimi Delta Attention (chunked gated delta-rule linear attention)

带每维遗忘门 `g` 的 delta-rule 线性注意力，前向算子。按 `chunk_size=128` 切分序列：
chunk 内构造下三角 `A` 矩阵并用 8×8 分块前向代入求逆完成 delta 修正，
chunk 间以状态 `S` 递推。内部全程 fp32（cumsum/exp/求逆/递推），
bf16 输入上转、`o` 下转 bf16，`S` 保持 fp32。


## 产品支持情况

- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 算子签名

```python
chunk_kda_wrapper(q, k, v, g, beta, scale=None, initial_state=None,
                  output_final_state=False, use_qk_l2norm_in_kernel=False,
                  cu_seqlens=None, **kwargs) -> (o, S)
```

| I/O | 变量 | Shape | Dtype | 说明 |
|-----|------|-------|-------|------|
| 入 | q, k, g | [B, T, H, K] | bf16 | query / key / 每维遗忘门(log 域) |
| 入 | v | [B, T, H, V] | bf16 | value |
| 入 | beta | [B, T, H] | fp32 | delta 更新权重 |
| 入 | scale | scalar | float | None → K**-0.5 |
| 入 | initial_state | [N, H, V, K] / None | fp32 | 初始状态 S，None 清零 |
| 出 | o | [B, T, H, V] | bf16 | 注意力输出 |
| 出 | S | [N, H, V, K] / None | fp32 | output_final_state=True 返回 |

## 支持的 dtype / 形状约束

- dtype：bf16 输入（内部 fp32），o bf16，S fp32。
- **T % 128 == 0**（cu_seqlens=None 时）；varlen 模式支持非对齐尾部 chunk。
- **K == V == 128**（P0）。
- bf16 o 容差 rtol/atol 1e-2；fp32 状态 S 容差 1e-3。

## 已知约束

- **输入稳定性**：内部精度测试需缩放稳定输入（q/k/v ~*0.1，g=logsigmoid(randn)≤0，beta=sigmoid）。
- **fp32 求逆硬约束**：(I+A) 与 8×8 前向代入求逆全程 fp32，bf16 会 NaN。S 累加器禁止下窄于 fp32。

## Testing

Tests under [`tests/ops/ling_3_0_flash/chunk_kda/`](../../../../../../tests/ops/ling_3_0_flash/chunk_kda/).
