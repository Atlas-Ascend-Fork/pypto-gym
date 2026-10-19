# BSA (Block Sparse Attention) PyPTO 实现说明文档

## 1. 概述

BSA（Block Sparse Attention）是面向华为昇腾 NPU 的块稀疏注意力算子的 PyPTO 实现。采用 **Mask2Idx 风格紧凑张量**策略：在 Python wrapper 层预计算每个 Q 块对应的有效 KV 块集合，构建紧凑的 K/V 张量，使 kernel 内层循环仅迭代 `maxSelectNum` 而非 `numKB`。无效迭代使用 `valid_mask=0.0` 将 S 设为大负数，使 exp 下溢为零，对在线 softmax 产生中性贡献。

**已实现功能**：
- 前向传播（含在线 softmax + LSE）
- 反向传播（合并 dQ+dK/dV 单 kernel）
- 块稀疏掩码支持（dense 掩码也走 sparse 路径，简化实现）
- GQA（Grouped Query Attention）兼容
- 非对齐序列支持（per-batch actual_seq_lengths 边界掩码）
- BNSD 布局：`[batch, headNum, seqLen, headDim]`
- FP16 数据类型（Q/K/V/O/dO/dQ/dK/dV = FLOAT16, softmaxLse = FLOAT32）
- head_dim = 128（固定）
- 动态轴模式（B, Hq, Hkv, Sq, Skv 通过 hint tensor 传入，单 kernel 覆盖所有 shape）

**数据类型与精度规范**：Q/K/V/O/dO/dQ/dK/dV = FLOAT16，softmaxLse = FLOAT32，输出形状 `[B, H_q, Sq]`，GQA 约束 Hq >= Hkv 且 Hq % Hkv == 0，dV 无 scale 而 dQ/dK 乘以 scaleValue (1/sqrt(d))，前向/反向容差均为 atol=0.0001, rtol=0.0078125。

---

## 2. 文件结构与职责

```
models/experimental/attention/BSA/
├── common/                        # 共享模块
│   ├── bsa_common.py              # 共享配置（BSAConfig）、golden/impl 辅助函数、紧凑张量构建器（cached）、掩码生成
│   └── bsa_test_utils.py          # 共享测试基础设施（环境检查、性能追踪、输入生成、比较辅助）
│
├── FWD/                           # 前向算子
│   └── bsa_fwd_impl.py            # 前向 NPU kernel 实现（single-phase, auto-configured l1/sched）
│
├── BWD/                           # 反向算子
│   └── bsa_bwd_impl.py            # 反向 NPU kernel 实现（merged dQ+dK/dV 单 kernel）
│
├── TEST/                          # 测试脚本与 golden 参考基准
│   ├── bsa_fwd_golden.py          # 前向 CPU 精度参考基准（纯 PyTorch）
│   ├── bsa_bwd_golden.py          # 反向 CPU 精度参考基准（纯 PyTorch）
│   ├── test_bsa_fwd.py            # 前向测试套件（13 aligned + 4 non-aligned 用例）
│   └── test_bsa_bwd.py            # 反向测试套件（15 aligned + 4 non-aligned 用例）
│
├── docs/                          # 历史文档（设计、优化记录等）
├── BSA_README.md                  # 本文档（总体说明）
└── PROMPT.txt                     # 开发需求描述
```

> **说明**：本目录不使用 `__init__.py`。测试脚本通过 `_resolve_bsa_root()` 定位 BSA 根目录，
> 再将 `common/`、`FWD/`、`BWD/` 分别加入 `sys.path`，以直接模块名（如 `from bsa_common import ...`）导入。
> Golden 文件与测试脚本同在 `TEST/` 下，Python 自动将脚本目录加入 `sys.path[0]`，无需额外配置。

### 文件职责

| 文件 | 职责 | 关键导出 |
|------|------|---------|
| `common/bsa_common.py` | 共享配置、golden/impl 辅助函数、紧凑张量构建（cached）、掩码生成 | `BSAConfig`, `DEFAULT_CONFIG`, `BSAForwardResult`, `BSABackwardResult`, `generate_block_sparse_mask()`, `_build_sparse_kv_cached()`, `_build_sparse_q_dkdv_cached()`, `_prepare_qkv_2d()`, `_make_jit_opts()`；namedtuple 配置类型：`_SparseKVConfig`, `_QKVPrepared`, `_KVFillConfig`, `_CollectKVCfg`, `_CollectQCfg`, `_KVBoundaryCfg`, `_QBoundaryCfg`, `_QBoundaryInnerCfg` 等 |
| `common/bsa_test_utils.py` | 环境检查、性能数据采集、输入生成、比较辅助 | `_check_env()`, `gen_inputs()`, `gen_inputs_non_aligned()`, `_compare_tensors()`, `_compare_grads()`, `_compare_grads_non_aligned()`, `_find_and_parse_swimlane()`, `_save_perf_case_data()`；环境检查拆分：`_check_ascend()`, `_check_pypto_path()`, `_check_npu()`, `_check_imports()`；namedtuple 配置：`GenInputsConfig`, `GenInputsNonAlignedConfig`, `BSATestInputs` |
| `TEST/bsa_fwd_golden.py` | 前向 CPU 精度参考基准（纯 PyTorch） | `bsa_forward_golden()`, `BSAForwardInputs` |
| `FWD/bsa_fwd_impl.py` | 前向 NPU kernel 实现（single-phase, auto-configured） | `block_sparse_attention_forward()`, `BSAForwardCallInputs`；内部 namedtuple：`_FwdHintTensors`；辅助：`_auto_configure_fwd_opts()`, `_make_fwd_masks()`, `_make_fwd_hint_tensors()` |
| `TEST/bsa_bwd_golden.py` | 反向 CPU 精度参考基准（纯 PyTorch） | `bsa_backward_golden()`, `BSABackwardInputs`, `BSABackwardBlockInputs` |
| `BWD/bsa_bwd_impl.py` | 反向 NPU kernel 实现（merged dQ+dK/dV） | `block_sparse_attention_backward()`, `BSABackwardCallInputs`；内部 namedtuple：`_BwdHintTensors`, `_BwdMasks`；辅助：`_make_bwd_hint_tensors()`, `_make_bwd_masks()` |
| `TEST/test_bsa_fwd.py` | 前向自动化测试套件（precision / perf / non-aligned） | `main()` |
| `TEST/test_bsa_bwd.py` | 反向自动化测试套件（precision / perf / non-aligned） | `main()` |

### 依赖关系

```
common/bsa_common         ← (无内部依赖)
common/bsa_test_utils     ← bsa_common

FWD/bsa_fwd_impl          ← bsa_common

BWD/bsa_bwd_impl          ← bsa_common（独立于 fwd_impl，无交叉依赖）

TEST/bsa_fwd_golden       ← bsa_common
TEST/bsa_bwd_golden       ← bsa_common

TEST/test_bsa_fwd         ← bsa_test_utils, bsa_fwd_golden, bsa_fwd_impl
TEST/test_bsa_bwd         ← bsa_test_utils, bsa_fwd_golden, bsa_fwd_impl,
                             bsa_bwd_golden, bsa_bwd_impl
```

---

## 3. 架构设计概览

> 本节为高层概览。详细的设计文档请参阅 `FWD/docs/DESIGN.md` 和 `BWD/docs/DESIGN.md`。

### 3.1 整体调用链

```
用户代码
  │
  ├─ block_sparse_attention_forward(BSAForwardCallInputs)
  │    ├─ _prepare_qkv_2d(Q, K, V, block_shape, cfg) → _QKVPrepared namedtuple
  │    │    (pad + reshape 4D→2D, shared by FWD and BWD)
  │    ├─ _build_sparse_kv_cached(mask, k_2d, v_2d, _SparseKVConfig, asq, askv)
  │    │    → compact K/V + valid_mask + max_sel
  │    ├─ _make_fwd_hint_tensors(...) → _FwdHintTensors namedtuple
  │    ├─ _make_fwd_masks(valid_mask, scale, large_neg) → scaled/neg_inf FP16 masks
  │    ├─ _auto_configure_fwd_opts(Sq) → l1/sched optimization
  │    ├─ _get_fwd_kernel(cfg) → single-phase online softmax kernel
  │    └─ 裁剪填充, reshape output
  │
  └─ block_sparse_attention_backward(BSABackwardCallInputs)
       ├─ _prepare_qkv_2d(Q, K, V, block_shape, cfg) → _QKVPrepared namedtuple
       │    (same pad + reshape step as FWD)
       ├─ pad + reshape dO, O; flatten LSE to 2D
       ├─ _build_sparse_kv_cached(mask, k_2d, v_2d, _SparseKVConfig, asq, askv)
       │    → compact K/V + valid_mask + max_sel (for dQ)
       ├─ _build_sparse_q_dkdv_cached(mask, q_2d, do_2d, o_2d, lse_2d, _SparseKVConfig, asq)
       │    → SparseQDkdvResult namedtuple (for dK/dV)
       ├─ _make_bwd_hint_tensors(...) → _BwdHintTensors namedtuple
       ├─ _make_bwd_masks(...) → _BwdMasks namedtuple (scaled/neg_inf/d_row for both phases)
       ├─ _get_bwd_kernel(cfg) → merged dQ+dK/dV single kernel
       └─ 裁剪填充, cast FP32→FP16, reshape outputs
```

### 3.2 核心设计策略

| 策略 | 说明 |
|------|------|
| **Mask2Idx 紧凑张量** | wrapper 层预收集有效 KV 块索引，构建紧凑 K/V 张量，kernel 仅迭代有效块 |
| **Online Softmax** | 逐 KV 块迭代维护 m/l/o 累积器，保证分块与全量数学等价 |
| **Merged BWD Kernel** | dQ 和 dK/dV 合并为单一 kernel（Phase 1: dQ, Phase 2: dK/dV），减少 kernel dispatch |
| **Dynamic Axis (Hint Tensor)** | B, Hq, Hkv, Sq, Skv 通过 hint tensor 传入，单 kernel 覆盖所有 shape 组合 |
| **Mask Structure Cache** | `_build_sparse_kv_cached()` 和 `_build_sparse_q_dkdv_cached()` 缓存 mask 结构，避免重复遍历 |
| **Auto-Configured l1/sched** | FWD 根据 Sq 阈值自动选择最优 l1/sched 配置（Sq≥1024: l1=64/sched=1 或 sched=3，Sq<1024: l1=16/sched=3） |
| **重计算策略** | 反向不存储 S/P，仅保存 O 和 LSE，反向时重计算 P = exp(scale*S - LSE) |
| **Per-batch Boundary Mask** | 非对齐序列通过 per-batch actual_seq_lengths 计算边界掩码，零化填充行 |
| **3-Layer E-fix Defense** | BWD dK/dV padded rows: LSE=1e30 + Q/dO=0 + d_row=0，确保填充行不贡献梯度 |

### 3.3 数据布局与分块

- **布局**: BNSD `[B, H, S, 128]`，kernel 内部 reshape 为 2D `[B*H*S, 128]`
- **分块**: block_shape_x=256（Q 块），block_shape_y=512（KV 块）
- **TileShape**: vec `(128, 128)`，cube `(128, 128, 128)`

---

## 4. 代码规范说明

本代码遵循以下命名与结构规范：

- **命名规范（G.NAM.01）**：函数和变量使用 `lower_with_under`，类使用 `CapWords`，常量使用 `CAPS_WITH_UNDER`。以 `_` 开头的名称表示模块内部使用，不作为公开 API。
- **多参数封装（G.FNM.03）**：超过 5 个参数的函数使用 namedtuple（如 `_SparseKVConfig`、`GenInputsConfig`）封装参数组，避免冗长的参数列表。
- **多返回值封装（G.FNM.05）**：超过 5 个返回值的函数使用 namedtuple（如 `_FwdHintTensors`、`_BwdMasks`、`SparseQDkdvResult`）封装返回值组，使每个字段有明确语义。
- **去重提取（G.DUP.01/02）**：FWD/BWD 共享的 `_prepare_qkv_2d()` 和测试共享的 `_find_and_parse_swimlane()`/`_save_perf_case_data()` 提取到公共模块，避免重复实现。

---

## 5. PyPTO API 约束与规避方案

> 详细的 API 映射请参阅 `FWD/docs/API_REPORT.md` 和 `BWD/docs/API_REPORT.md`。

| # | 限制 | 规避方案 | 状态 |
|---|------|---------|------|
| 1 | ~~不支持 `pypto.DYNAMIC`~~ | 工厂函数 + 字典缓存，每种 shape 编译一个 kernel | **已解除**：BSA 使用 `pypto.frontend.dynamic()` 创建的 SymbolicScalar 作为 tensor 维度，等同于 `pypto.DYNAMIC`；BWD kernel 使用单一合并内核（cache key = `("bwd")`）无需每种 shape 重编译 |
| 2 | 不支持 `pypto.cond()` | 裸 `if pypto.is_loop_begin(v):` | **编译失败确认**：`pypto.cond()` 在 BSA 内核中触发 `REGISTER_COPY tile shape not set` 错误（BiSheng 编译器无法为条件分支间的数据拷贝设置 tile shape），BSA 的 `parallel=True` + SUB_SPLIT + 多 tile shape 上下文切换模式与此机制不兼容；DeepSeek V32 可用是因为其内核结构更简单（无 SUB_SPLIT、无 parallel、无显式 tile shape 切换） |
| 3 | `SymbolicScalar ** -0.5` 不支持 | kernel 外计算为 Python float，闭包捕获 | **已规避** |
| 4 | `pypto.where` 仅支持单轴广播 | 预展开 mask 到 `[BLOCK, BLOCK]` | **已规避** |
| 5 | `pypto.where(mask, tensor, scalar)` CCE 失败 | 使用 `S*mask + (1-mask)*neg` 算术替代 | **已规避** |
| 6 | `pypto.cast(BOOL, FP32)` kernel 内数据损坏 | wrapper 中预转换 `mask.float()` | **已规避** |
| 7 | `pypto.assemble` + `reshape` 冲突 | 输出 tensor 传 3D，避免 kernel 内 reshape | **已规避** |
| 8 | `pypto.sum(FP16)` 返回 FP16 | cast 到 FP32 后再 sum | **已规避** |
| 9 | `pypto.assemble` 是覆写非累加 | 拆分 kernel，局部累积后一次性写出 | **已规避** |
| 10 | `and` + `is_loop_begin` 触发 `ValueError` | 嵌套 if 代替 and | **已规避** |
| 11 | Cube tile `[256, 128]` 不支持 | 必须使用 `[128, 128]` | **已规避** |
| 12 | `pypto.view` 2D 偏移对 LSE 不正确 | 反向时 flatten LSE 到 `[B*Hq*Sq, 1]`，用 1D 偏移 | **已规避** |
| 13 | wrapper 数据与 kernel 存在竞争 | `torch.npu.synchronize()` 确保 wrapper 数据写入完成 | **已规避** |

---

## 6. 精度评测结果

### 6.1 测试环境

- **硬件**: Ascend 910B3, 22 AICore
- **CANN**: 9.0.0
- **数据类型**: FP16 (Q/K/V/O/dO/dQ/dK/dV), FP32 (softmaxLse)
- **容差**: `atol=0.0001, rtol=0.0078125`

### 6.2 前向对齐精度（13 个测试用例，全部 PASSED）

| 测试 | 配置 | 稀疏率 | O max_diff | LSE max_diff | 结果 |
|------|------|--------|------------|-------------|------|
| S256 Sparse50 | B=1, Hq=4, Hkv=2, Sq=256 | 50% | ≤0.000122 | ≤0.000024 | ✅ |
| S512 Sparse70 | B=1, Hq=4, Hkv=4, Sq=512 | 70% | ≤0.000061 | ≤0.000020 | ✅ |
| S1024 Sparse30 | B=1, Hq=8, Hkv=1, Sq=1024 | 30% | ≤0.000061 | ≤0.000021 | ✅ |
| S2048 Sparse30 | B=1, Hq=4, Hkv=2, Sq=2048 | 30% | ≤0.000061 | ≤0.000020 | ✅ |
| GQA group4 | B=1, Hq=8, Hkv=2, Sq=256 | 40% | ≤0.000061 | ≤0.000020 | ✅ |
| GQA Hq32 | B=1, Hq=32, Hkv=8, Sq=256 | 50% | ≤0.000122 | ≤0.000024 | ✅ |
| Dense 100% | B=1, Hq=4, Hkv=4, Sq=256 | 100% | ≤0.000122 | ≤0.000023 | ✅ |
| B2 MHA S256 | B=2, Hq=4, Hkv=4, Sq=256 | 50% | ≤0.000122 | ≤0.000023 | ✅ |
| B2 GQA S256 | B=2, Hq=8, Hkv=2, Sq=256 | 50% | ≤0.000061 | ≤0.000020 | ✅ |
| B2 Dense S256 | B=2, Hq=4, Hkv=4, Sq=256 | 100% | ≤0.000122 | ≤0.000023 | ✅ |
| B2 Sparse S512 | B=2, Hq=4, Hkv=4, Sq=512 | 70% | ≤0.000061 | ≤0.000020 | ✅ |
| B2 MHA S1024 | B=2, Hq=4, Hkv=4, Sq=1024 | 70% | ≤0.000061 | ≤0.000020 | ✅ |
| B4 MHA S256 | B=4, Hq=4, Hkv=4, Sq=256 | 50% | ≤0.000122 | ≤0.000023 | ✅ |

### 6.3 前向非对齐精度（4 个测试用例，全部 PASSED）

| 测试 | 配置 | asq | O max_diff | 结果 |
|------|------|-----|------------|------|
| S300 NonAligned | B=1, Hq=4, Hkv=2 | [300] | ≤0.000061 | ✅ |
| B2 VarLen S256-300 | B=2, Hq=4, Hkv=4 | [256,300] | ≤0.000122 | ✅ |
| S400 NonAligned | B=1, Hq=8, Hkv=2 | [400] | ≤0.000122 | ✅ |
| B2 MixAligned | B=2, Hq=4, Hkv=4 | [256,400] | ≤0.000122 | ✅ |

### 6.4 反向对齐精度（15 个测试用例，全部 PASSED）

| 测试 | 配置 | dQ max_diff | dK max_diff | dV max_diff | 结果 |
|------|------|------------|------------|------------|------|
| S256 Sparse50 | B=1, Hq=4, Hkv=2 | ≤0.000031 | ≤0.000061 | ≤0.000122 | ✅ |
| S512 Sparse70 | B=1, Hq=4, Hkv=4 | ≤0.000031 | ≤0.000031 | ≤0.000061 | ✅ |
| S1024 Sparse30 | B=1, Hq=8, Hkv=1 | ≤0.000031 | ≤0.000061 | ≤0.000244 | ✅ |
| MHA Dense | B=1, Hq=4, Hkv=4 | ≤0.000031 | ≤0.000031 | ≤0.000061 | ✅ |
| GQA | B=1, Hq=8, Hkv=2 | ≤0.000031 | ≤0.000031 | ≤0.000122 | ✅ |
| MHA sparse0.3 | B=1, Hq=4, Hkv=4 | ≤0.000031 | ≤0.000031 | ≤0.000061 | ✅ |
| MHA sparse0.7 | B=1, Hq=4, Hkv=4 | ≤0.000031 | ≤0.000031 | ≤0.000061 | ✅ |
| MHA S512 | B=1, Hq=4, Hkv=4 | ≤0.000031 | ≤0.000031 | ≤0.000061 | ✅ |
| MHA S1024 | B=1, Hq=4, Hkv=4 | ≤0.000031 | ≤0.000031 | ≤0.000061 | ✅ |
| B2 MHA S256 | B=2, Hq=4, Hkv=4 | ≤0.000031 | ≤0.000031 | ≤0.000122 | ✅ |
| B2 GQA S256 | B=2, Hq=8, Hkv=2 | ≤0.000031 | ≤0.000031 | ≤0.000122 | ✅ |
| B2 Dense S256 | B=2, Hq=4, Hkv=4 | ≤0.000031 | ≤0.000031 | ≤0.000122 | ✅ |
| B2 Sparse S512 | B=2, Hq=4, Hkv=4 | ≤0.000031 | ≤0.000031 | ≤0.000061 | ✅ |
| B2 MHA S1024 | B=2, Hq=4, Hkv=4 | ≤0.000031 | ≤0.000031 | ≤0.000061 | ✅ |
| B4 MHA S256 | B=4, Hq=4, Hkv=4 | ≤0.000031 | ≤0.000031 | ≤0.000122 | ✅ |

### 6.5 反向非对齐精度（4 个测试用例，全部 PASSED）

| 测试 | 配置 | asq | dQ max | dK max | dV max | 结果 |
|------|------|-----|--------|--------|--------|------|
| S300 NonAligned | B=1, Hq=4, Hkv=2 | [300] | ≤0.000031 | ≤0.000061 | ≤0.000122 | ✅ |
| B2 VarLen | B=2, Hq=4, Hkv=4 | [256,300] | ≤0.000031 | ≤0.000031 | ≤0.000122 | ✅ |
| S400 NonAligned | B=1, Hq=8, Hkv=2 | [400] | ≤0.000031 | ≤0.000061 | ≤0.000122 | ✅ |
| B2 MixAligned | B=2, Hq=4, Hkv=4 | [256,400] | ≤0.000031 | ≤0.000031 | ≤0.000122 | ✅ |

---

## 7. 运行方式

### 环境准备

```bash
export ASCEND_HOME_PATH=/home/developer/Ascend/cann-9.0.0
export TILE_FWK_DEVICE_ID=0
export PYTHONPATH=/mnt/workspace/gitCode/cann/pypto/python:$PYTHONPATH
```

### 运行测试

测试脚本内置三级路径解析（`_resolve_bsa_root()`），可从任意目录运行：

1. **`BSA_ROOT` 环境变量**（显式指定）
2. **自动检测**：从脚本位置向上搜索 `BSA_README.md` + `common/bsa_common.py`
3. **仓库相对路径**：在含 `tests/` 与 `src/` 的项目根下，按 `tests/ops/` 相对路径定位 `ops/` 下的 BSA 实现

```bash
# 前向精度测试（对齐）
python tests/ops/experimental/attention/BSA/test_bsa_fwd.py --mode precision

# 前向精度测试（非对齐/变长）
python tests/ops/experimental/attention/BSA/test_bsa_fwd.py --mode non-aligned

# 前向性能测试（泳道图）
python tests/ops/experimental/attention/BSA/test_bsa_fwd.py --mode perf

# 反向精度测试（对齐）
python tests/ops/experimental/attention/BSA/test_bsa_bwd.py --mode precision

# 反向精度测试（非对齐/变长）
python tests/ops/experimental/attention/BSA/test_bsa_bwd.py --mode non-aligned

# 反向性能测试（泳道图）
python tests/ops/experimental/attention/BSA/test_bsa_bwd.py --mode perf

# 快速测试（仅 S256+S512+B2S256）
python tests/ops/experimental/attention/BSA/test_bsa_fwd.py --cases quick
python tests/ops/experimental/attention/BSA/test_bsa_bwd.py --cases quick
```

### 注意事项

1. **确保无残留进程占用 NPU**：运行前检查 `ps aux | grep python | grep test_bsa`，如有残留进程需 `kill -9` 清理
2. **首次运行较慢**：每种 shape 组合需要 JIT 编译，后续运行命中缓存会快很多
3. **性能数据**：测试会自动采集泳道图数据到 `output/` 目录

---

## 8. 性能数据

### 8.1 测试环境

- **硬件**: Ascend 910B3, 22 AICore
- **CANN**: 9.0.0, bisheng 15.0.5
- **配置**: block_shape_x=256, block_shape_y=512, head_dim=128

### 8.2 前向性能（MHA Dense, B=1, Hq=Hkv=4）

| Shape | Task Time | AICore Time | AICore Util |
|-------|-----------|-------------|-------------|
| 256×256 | ~75 us | ~1.3 ms | ~37% |
| 512×512 | ~92 us | ~2.6 ms | ~43% |
| 1024×1024 | ~180 us | ~6.5 ms | ~60% |
| 2048×2048 | ~820 us | ~26.9 ms | ~55% |

### 8.3 反向性能（MHA Dense, B=1, Hq=Hkv=4）

| Shape | dQ Task Time | dK/dV Task Time | dQ AICore Util | dK/dV AICore Util |
|-------|-------------|-----------------|----------------|-------------------|
| 256×256 | ~79 us | ~85 us | ~38% | ~36% |
| 512×512 | ~100 us | ~112 us | ~44% | ~43% |
| 1024×1024 | ~183 us | ~200 us | ~60% | ~61% |

### 8.4 性能优化措施

| 优化项 | 说明 |
|--------|------|
| **Auto-configured l1/sched** | Sq≥1024 时自动切换 l1=64/sched=1（S1024 性能提升 ~13.5%） |
| **Mask Structure Cache** | `_build_sparse_kv_cached()` 缓存 mask 遍历结果，跨调用复用 |
| **D_row Precompute** | d_row = (dO · O).sum() 在 wrapper 中预计算，kernel 内免做 sum+mul |
| **Scaled/neg_inf Precompute** | valid_mask 预乘 softmax_scale 和 large_neg，kernel 内直接使用 FP16 mask |
| **E-fix 3-Layer Defense** | padded rows: LSE=1e30 + Q/dO=0 + d_row=0，避免数值泄漏 |
| **Merged BWD Kernel** | dQ+dK/dV 合并为单 kernel，减少 dispatch overhead |
| **parallel=True** | 外层循环使用 parallel=True 提升多核调度 |

### 8.5 已知性能瓶颈

1. **小 shape 并行度不足**: S256 dQ 仅有 8 外层任务（sub-block 后），22 核利用率 ~37-42%
2. **累积模式限制**: dK/dV 需跨 Q 块累积，无法简单增加外层并行
3. **Cube tile 限制**: `[128,128]` 是 CANN 9.0.0 支持的最大安全配置
4. **S1024+ sched=1 限制**: sched=1 对 S2048 反而有害（scheduling wait），需回退 sched=3
