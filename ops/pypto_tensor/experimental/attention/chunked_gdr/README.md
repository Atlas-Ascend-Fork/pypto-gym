# chunked_gated_delta_rule

分块门控 Delta Rule 线性注意力算子，将传统 O(n²) softmax attention 降低到 O(n) 复杂度。

---


## 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 算子概述

本算子实现 Chunked Gated Delta Rule Linear Attention 机制，用于 Qwen3-Next 等模型的注意力层。通过按可配置 chunk_size (L=32/64/128) 分块处理序列，在每个 chunk 内执行 L2 归一化、预注意力、矩阵求逆、累积衰减和循环状态注意力，实现 O(n) 复杂度的线性注意力。

支持 GQA（Grouped Query Attention）模式：Nv 必须是 Nqk 的整数倍。

提供两个版本：
- **aligned 版本**: 序列长度整除 L，直接切片写回
- **unaligned 版本**: 序列长度不整除 L，使用 fillpad + assemble 处理尾部不满 chunk

## 数学公式

$$
\hat{q} = q / \sqrt{\sum q_i^2 + \epsilon}, \quad \hat{k} = k / \sqrt{\sum k_i^2 + \epsilon}
$$

$$
g_{cum} = \text{tril} \cdot g, \quad D_{decay} = \exp((g_{cum} - g_{cum}^T) \cdot \text{tril})
$$

$$
A = (\hat{k} \cdot \beta) @ \hat{k}^T \cdot D_{decay} \cdot \text{mask}
$$

$$
A_{inv} = (I - A)^{-1}, \quad v_{out} = A_{inv} \cdot (v \cdot \beta)
$$

$$
\text{chunk\_out} = \hat{q} \cdot e^{g_{cum}} @ S + \text{attn} @ (v_{out} - v_{prime})
$$

## 接口

```python
def chunked_gated_delta_rule_wrapper(
    query: torch.Tensor,       # [T, Nqk, D] float32
    key: torch.Tensor,         # [T, Nqk, D] float32
    value: torch.Tensor,       # [T, Nv, D] float32
    beta: torch.Tensor,        # [T, Nv] float32
    gate: torch.Tensor,        # [T, Nv] float32
    states: torch.Tensor,      # [B, Nv, D, D] float32
    act_seq_len: torch.Tensor, # [B+1] int32
    chunk_size = "auto",       # "auto" | 32 | 64 | 128
    mask: torch.Tensor = None, # [L, L] float32 (auto时可选)
    tril_mask: torch.Tensor = None, # [L, L] float32 (auto时可选)
    eye: torch.Tensor = None,  # [L//8, L] float32 (auto时可选)
    enable_perf_debug: bool = False,
    stitch_function_max_num: int = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    ...
```

### Auto chunk_size 策略

当 `chunk_size="auto"` (默认) 时，wrapper 自动选择最优 L：

| max_seq_len | 选择 L | 原因 |
|-------------|--------|------|
| ≤ 32 | L=32 | 单chunk最快 (~84us) |
| ≤ 64 | L=64 | 单chunk次快 (~90us) |
| > 64 | L=128 | 多chunk串行依赖使小L更慢 |

**关键规律**: aligned 多chunk场景(T>L)，L=128 是唯一最优选择。L=64比L=128慢28%，L=32慢98%。

### 使用示例

```python
import torch
from chunked_gated_delta_rule_impl import chunked_gated_delta_rule_wrapper

# Auto模式 — 只需核心输入，自动选择L和生成mask/tril/eye
B, Nqk, Nv, D, T = 2, 2, 8, 128, 512
states = torch.zeros(B, Nv, D, D)
act_seq_len = torch.tensor([0, 256, 512], dtype=torch.int32)

attn_out, last_state = chunked_gated_delta_rule_wrapper(
    query, key, value, beta, gate, states,
    act_seq_len,  # chunk_size="auto" (默认)
)

# 手动模式 — 需传入匹配L的mask/tril/eye
from chunked_gated_delta_rule_impl import prepare_chunk_helpers
helpers = prepare_chunk_helpers(128)  # L=128
attn_out, last_state = chunked_gated_delta_rule_wrapper(
    query, key, value, beta, gate, states,
    act_seq_len, chunk_size=128,
    mask=helpers['mask'], tril_mask=helpers['tril_mask'], eye=helpers['eye'],
)
```

## 参数说明

| 参数 | dtype | shape | 说明 |
|------|-------|-------|------|
| `query` | float32 | [T, Nqk, D=128] | 查询向量，T 和 B 为动态轴 |
| `key` | float32 | [T, Nqk, D=128] | 键向量 |
| `value` | float32 | [T, Nv, D=128] | 值向量，Nv % Nqk == 0 |
| `beta` | float32 | [T, Nv] | Beta 缩放因子 |
| `gate` | float32 | [T, Nv] | 门控衰减信号 |
| `states` | float32 | [B, Nv, D, D] | 初始循环状态矩阵 |
| `act_seq_len` | int32 | [B+1] | 各 batch 累积序列长度索引 |
| `chunk_size` | — | "auto"/32/64/128 | chunk大小，"auto"自动选择 |
| `mask` | float32 | [L, L] | 预注意力掩码 (auto时自动生成) |
| `tril_mask` | float32 | [L, L] | 下三角掩码 (auto时自动生成) |
| `eye` | float32 | [L//8, L] | 求逆用单位矩阵 (auto时自动生成) |
| **返回值1** | float32 | [T, Nv, D=128] | 注意力计算输出 |
| **返回值2** | float32 | [B, Nv, D, D] | 更新后的循环状态 |

## 约束条件

- **head_dim (D) = 128**: 固定值，不可更改
- **chunk_size (L) = 32/64/128**: 可配置，必须是 8 的倍数
- **GQA**: Nv 必须是 Nqk 的整数倍（Nv % Nqk == 0）
- **Nv ≥ 4**: 确保 parallel=True 利用 ≥4 核
- **dtype**: 仅支持 float32 输入输出
- **动态轴**: T（总序列长度）和 B（batch 数）为动态轴

## 性能数据

### 当前基线 (NPU 910B3, FP32, B1+B2+B3 优化)

| Case | Nqk | Nv | T | L | Task Time | AICore Util | Δ vs pre-B |
|------|-----|-----|-----|-----|-----------|-------------|-----------|
| aligned_gqa | 2 | 8 | 128 | 128 | 134.9 us | 19.8% | **-12.1%** |
| aligned_multi_batch_gqa | 2 | 8 | 512 | 128 | 345.0 us | 23.5% | **-11.2%** |
| unaligned_gqa | 2 | 8 | 130 | 128 | 193.5 us | 23.4% | **-14.0%** |
| aligned_large_gqa | 4 | 4 | 512 | 128 | 192.5 us | 23.5% | **-3.5%** |
| unaligned_single | 2 | 4 | 130 | 128 | 141.3 us | 17.4% | **-6.5%** |
| aligned_single_chunk_L64 | 2 | 4 | 64 | 64 | 85.7 us | 14.2% | **-4.5%** |
| aligned_multi_chunk_L64 | 2 | 4 | 128 | 64 | 105.5 us | 15.8% | **-1.9%** |
| aligned_single_chunk_L32 | 2 | 4 | 32 | 32 | 81.9 us | 14.1% | **-2.9%** |

**8/8 用例全部提升（3~14%），主生产负载 Nv=8 一致改善 11~14%**

### 新增优化项 (B1+B2+B3)

| 优化 | 描述 | 收益 |
|------|------|------|
| B1: g_exp 跨Phase传递 | Phase4 g_exp 传入 Phase5, 消除重复 gate.exp() | 1 exp() + 512B temp |
| B2: final_state_1 内联 | 合并到 state_new 表达式, 消除 64KB temp | 64KB workspace |
| B3: state UB prefetch | `state_ub = state + 0.0` 强制早期 COPY_IN | 隐藏 GM→UB 延迟 |

### chunk_size 对大case影响

| Case (aligned, T>L) | L=128 | L=64 | L=32 |
|----------------------|-------|------|------|
| B=2,Nqk=2,Nv=8,T=512 | **345us**† | 498us (+28%)‡ | 771us (+98%)‡ |
| B=2,Nqk=4,Nv=4,T=512 | **192us**† | 232us (+16.5%)‡ | — |

† B1+B2+B3 优化后; ‡ L=64/L=32 来自 pre-B 基线 (相对 delta 仍有效)

**结论**: 大case(T>L)用小L不会更快，S loop串行依赖是根本瓶颈。

## 已否决优化项

| 优化项 | 否决原因 |
|--------|---------|
| cumsum 替代 matmul(tril,gate) | 瓶颈是S loop串行依赖，不在gate_cum；unaligned编译失败 |
| inplace=True on reshape | 精度灾难: max_diff从1e-05跳到1e-01 |
| A_inv 双 matmul 合并 | concat开销+[L,2D]不fit cube pipeline |
| stitch=128 | L=64内存失败; L=128性能更差 |
| 0switch tile模式 | 大case+20% task_time退化 |
| B loop parallel=True | PyPTO禁止嵌套parallel loops |
| L=64/L=32 用于大case | S loop串行依赖使小L更慢(+28~98%) |
| device_sched_mode=1/3 | +5~8% slower; L2亲和无效(state已在L2) |
| submit_before_loop=True | 灾难性+183~357%退化(pipeline完全破坏) |
| cube_nbuffer_setting | L=64编译失败+更慢 |
| cube_l1_reuse_setting | unaligned编译失败+更慢 |

## 目录结构

```
.
├── chunked_gated_delta_rule_impl.py     # 算子实现（含 wrapper + aligned/unaligned kernels）
├── chunked_gated_delta_rule_golden.py   # Golden 参考实现（精度基准）
├── test_chunked_gated_delta_rule.py     # 自包含测试入口（精度+性能，内联 perf_utils 和 test_cases）
└── docs/
    ├── SPEC.md                          # 算子规格
    ├── DESIGN.md                        # 设计方案 + 性能优化归档
    ├── API_REPORT.md                    # API 映射报告
    └── README.md                        # 本文件
```

## 运行方式

```bash
# 设置环境
export PTO_TILE_LIB_CODE_PATH=/path/to/pto-isa
export TILE_FWK_DEVICE_ID=0

# 运行精度+性能测试
python3 tests/ops/chunked_gdr/test_chunked_gated_delta_rule.py --perf

# 运行单个用例
python3 tests/ops/chunked_gdr/test_chunked_gated_delta_rule.py aligned_gqa --perf

# 列出所有用例
python3 tests/ops/chunked_gdr/test_chunked_gated_delta_rule.py --list
```

## 精度要求

| 指标 | 值 |
|------|-----|
| rtol | 1e-3 |
| atol_abs | 0 |
| atol_rel | 1e-3 |
| tolerance 公式 | tolerance = atol_abs + atol_rel * |expected| |
| dtype | float32 (全链路) |