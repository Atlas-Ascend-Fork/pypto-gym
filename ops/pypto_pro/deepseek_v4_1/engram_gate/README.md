# EngramGate（PyPTO-Pro）

`engram_gate` 使用 PyPTO-Pro 在昇腾 NPU 上融合实现 Engram residual gate 的双路
RMS、加权点积、signed-sqrt sigmoid 门控和 residual update。

## 接口

```python
y = engram_gate_wrapper(x, key, value, weight, image_mask, eps=1e-6, clamp_value=1e-6)
```

| 参数 | dtype | shape | 说明 |
| :--- | :--- | :--- | :--- |
| `x`, `key` | BF16 | `[t, hc, dim]` | query / key 激活 |
| `value` | BF16 | `[t, dim]` | value，按 token 共享（不随 hc 变化） |
| `weight` | FP32 | `[hc, dim]` | 逐 `(hc, dim)` 组合权重 |
| `image_mask` | BOOL | `[t]` | `True` 的 token 原样返回 `x` |
| `y` | BF16 | `[t, hc, dim]` | 输出，与 `x` 同 shape |

## 算子介绍

计算公式（沿最后一维 `dim`，逐 `(token, hc)` 行）：

```text
rstd = rsqrt(mean(x²) + eps) * rsqrt(mean(key²) + eps)
dot  = sum(x * weight * key) * rstd / sqrt(dim)
gate = sigmoid(copysign(sqrt(max(|dot|, clamp_value)), dot))
y    = bf16(x + gate * value)
```

`image_mask` 为 `True` 的 token 直接返回 `x`（逐位相等）。`dim` 支持到单次行扫描上限
（`CHUNK = 8512`，覆盖全部算例；`dim > 8512` 暂不支持）。

## 实现结构

- **单个** `@pl.jit(auto_mutex=True)` kernel（`engram_gate_kernel`）服务全部算例：
  `t / hc / dim` 都是运行时动态轴；寄存器级计算在 `engram_gate_vf` 内完成。
- 每个 `(token, hc)` 行整行驻留 UB（固定行宽 `CHUNK`）：pass1 在寄存器里累加
  `Σx²`、`Σkey²`、`Σ(x·w·key)` 并算出 gate 标量；pass2 读同一块 UB 行写回
  `x + gate·value`，不做 GM 重读。
- 满段用融合乘加 `mul_add_dst` 的无掩码快循环，尾块单独用 `update_mask` 处理；
  3 槽轮转 buffer 配 `auto_mutex`；`weight` 按核驻留。
- host 侧 `engram_gate_wrapper` 只做形状校验 + `torch.empty` 分配 + 单次 launch；
  启动核数由 PyPTO-Pro 平台信息 `get_platform_info().vector_core_num` 动态取
  （不写死）。全部数值计算都在 kernel 内。

## 支持算例

| case | t | hc | dim | image_mask | 说明 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `partial_image_mask` | 5 | 4 | 5120 | `idx%2==1` | 部分 mask 直通 |
| `model_token_count_nomask` | 72 | 4 | 5120 | 全 False | 多 token |
| `chunked_one_chunk_tail_one` | 3 | 4 | 8449 | 全 False | 超 8192、非 64 倍数尾块 |
| `model_dim_plus_one_partial_mask` | 5 | 4 | 5121 | `idx%2==1` | `dim` 非 64 倍数 + 部分 mask |
| `compile_default_shape` | 2 | 4 | 5120 | 全 False | 小 token 默认形状 |

## 精度

与独立 CPU 参考实现（`engram_gate_golden_cpu.py`，FP32 计算、BF16 输出）按官方
方案 A（Scheme A）精度标准（`precision_compare.py`）比对：**5/5 PASS**，
`matched_ratio = 1.0`；`image_mask` 为 `True` 的行与 `x` 逐位相等。

## 性能

> 下表为**参考数值**，实际耗时每次运行可能有所波动。

Ascend 950PR，msprof `Task Duration` 取 min（固定采集协议：预热后 32 次调用、每次后
`torch.npu.synchronize()`，丢前 5 次对剩余 27 次取 min）：

| case | Task Duration (µs) |
| :--- | ---: |
| `partial_image_mask` | 2.381 |
| `model_token_count_nomask` | 4.449 |
| `chunked_one_chunk_tail_one` | 2.992 |
| `model_dim_plus_one_partial_mask` | 2.409 |
| `compile_default_shape` | 2.224 |

## 运行测试

```bash
python test_engram_gate.py
```

逐个构建输入、上板运行公开 wrapper，并与独立 CPU golden 用官方精度标准比较。

## 文件结构

```
ops/pypto_pro/deepseek_v4_1/engram_gate/
├── engram_gate.py            # kernel（engram_gate_vf / engram_gate_kernel）+ host wrapper
└── README.md

tests/ops/pypto_pro/deepseek_v4_1/engram_gate/
├── engram_gate_golden_cpu.py # 独立 CPU 参考实现
├── precision_compare.py      # 官方方案 A 精度标准
└── test_engram_gate.py       # 逐 case 测试 + main 入口
```
