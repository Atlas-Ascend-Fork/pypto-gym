# Qwen3-1.7B PyPTO 融合算子

Qwen3-1.7B 模型的 PyPTO 融合 kernel 算子库。当前实际集成的算子为 Q/K channel 的 RMSNorm + RoPE 部分融合 kernel。

## 产品支持情况

- Ascend 910B2：支持（当前环境）

## 目录结构

```
qwen3_1_7b/
├── __init__.py                          # 顶层入口: USE_PTO_ROPE, qk_rope_wrapper
└── rope/
    ├── __init__.py                      # 子包入口, 重导出 qwen3_qk_rope_q / qwen3_qk_rope_k
    └── rrms_norm_rope_impl.py           # kernel 工厂与具体实现
```

## 算子列表

| 算子名称 | 融合范围 | 输入 shape | 输出 shape | 精度 | 对应 eager 代码 |
|---------|---------|-----------|-----------|------|---------------|
| `qwen3_qk_rope_q` | Q per-head RMSNorm + RoPE | `[S, 16, 128]` | `[S, 16, 128]` | BF16 | `q_norm()` + `rotate_half()` + RoPE multiply |
| `qwen3_qk_rope_k` | K per-head RMSNorm + RoPE | `[S, 8, 128]` | `[S, 8, 128]` | BF16 | 同上, N_kv=8 |

两个 kernel 均由 `_make_qk_rope_kernel(N)` 工厂函数生成。

## 输入 / 输出详解

| 参数 | shape | dtype | 含义 |
|------|-------|-------|------|
| `x` | `[S, N, 128]` | BF16 | q_proj 或 k_proj 的输出 |
| `cos` | `[S, 128]` | BF16 | RoPE cosine |
| `sin` | `[S, 128]` | BF16 | RoPE sine |
| `w_norm` | `[128]` | BF16 | q_norm/k_norm 权重 |
| `out` | `[S, N, 128]` | BF16 | RMSNorm + RoPE 结果 |

## 融合范围

```
q_proj/k_proj (PyTorch)  -->  [RMSNorm + RoPE] (PyPTO)  -->  Attention (PyTorch)
```

## 开关变量

| 变量 | 默认值 | 定义位置 | 含义 |
|------|--------|---------|------|
| `USE_PTO_ROPE` | `False` | `__init__.py` 第 11 行 | 启用 PyPTO 融合 RoPE kernel |

## 关键实现细节

- **JIT 编译**: `_make_qk_rope_kernel()` 返回 `@pypto.frontend.jit` 装饰的 kernel
- **runtime 配置**: `stitch_function_max_num=128`, `device_sched_mode=1`
- **Tile 大小**: `BS_TILE=8` (sequence 维度), `D=128` (head dim)
- **Epsilon**: `1e-6`，与 Qwen3Config `rms_norm_eps` 一致
- **中间精度**: FP32 计算，输入/输出 BF16

## 模型集成

模型代码位于 `src/pypto_gym/transformers/qwen3_1_7b/`，wrapper 位于本目录 `__init__.py`。

```bash
# 基线推理
python modeling/transformers/qwen3_1_7b/ask_Qwen3-1.7B.py --device 0 --prompt "你好"

# PTO 推理
python modeling/transformers/qwen3_1_7b/ask_Qwen3-1.7B.py --device 0 --prompt "你好" --use-pto
```

## 配置常量速查

| 常量 | 值 | 位置 |
|------|-----|------|
| D (head_dim) | 128 | `rrms_norm_rope_impl.py:35` |
| HALF_D | 64 | `rrms_norm_rope_impl.py:36` |
| EPS | 1e-6 | `rrms_norm_rope_impl.py:37` |
| BS_TILE | 8 | `rrms_norm_rope_impl.py:38` |
| N_q (Q heads) | 16 | `rrms_norm_rope_impl.py:166` |
| N_kv (KV heads) | 8 | `rrms_norm_rope_impl.py:167` |
