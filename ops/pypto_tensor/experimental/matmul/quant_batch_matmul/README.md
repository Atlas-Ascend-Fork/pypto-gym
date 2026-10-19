# quant_batch_matmul 算子说明


## 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 算子语义

`quant_batch_matmul` 目录提供 **量化 Batch MatMul** 的 PyPTO 实现，对齐 CANN `aclnnQuantMatmulV5` 的多种量化布局。

当前已落地变体：

| 变体 | 模块 | 布局 | 量化粒度 | 输入 → 输出 |
|------|------|------|----------|-------------|
| **F-T** | `quant_batch_matmul_impl.py` | F-T | 全局标量 scale | FP8 E4M3 → INT8 |

### pertensor（F-T）数学公式

```
out[b, m, n] = quant( x1[b, m, k] @ x2[b, n, k]^T * x2Scale * x1Scale )
```

**展开形式**：

```
acc[b, m, n] = Σ(k=0..K-1) x1[b, m, k] * x2[b, n, k]
out[b, m, n] = round( clamp( acc[b, m, n] * combined_scale, -128, 127 ) )
```

其中 `combined_scale = mask_fixpipe(x1Scale * x2Scale)`；`x1Scale` 可为 `None`（仅使用 `x2Scale`）。

### 计算流程（pertensor）

1. **Host 侧 scale 合并**：`compute_combined_scale` 读取标量 `x1Scale`/`x2Scale`，应用 fixpipe 位掩码。
2. **张量展平**：`x1` `[batch, m, k]` → `[batch*m, k]`，`x2` `[batch, n, k]` → `[batch*n, k]`（inplace reshape）。
3. **Batch 循环**：`LOOP_BATCH` 按 batch 切片 `x1` 块 `[m, k]`。
4. **N 分块循环**：`LOOP_N` 按 `n_block_size` 切片 `x2` 块 `[n_block, k]`，`b_trans=True`。
5. **Cube matmul + fixpipe**：`pypto.matmul(..., DT_INT8, extend_params={"scale": combined_scale})`。
6. **结果组装**：`reshape` + `assemble` 写回 `out[b, m, n_off:n_off+valid_n]`。

---

## 输入输出规格（F-T pertensor）

### 输入张量

| 名称 | Shape | DType | 说明 |
|------|-------|-------|------|
| `x1` | `[batch, m, k]` | FP8 E4M3 | 左矩阵，不转置 |
| `x2` | `[batch, n, k]` | FP8 E4M3 | 右矩阵，kernel 内 `b_trans=True` |
| `x1_scale` | `(1,)` 或 `None` | FP32 | 可选左矩阵全局 scale |
| `x2_scale` | `(1,)` | FP32 | 右矩阵全局 scale（必填） |

### 输出张量

| 名称 | Shape | DType | 说明 |
|------|-------|-------|------|
| `out` | `[batch, m, n]` | INT8 | 量化后的 batch matmul 结果 |

### 配置参数（`QuantBatchMatmulConfig`）

| 字段 | 说明 |
|------|------|
| `ori_shape` | `[batch, m, k, n]` 逻辑形状 |
| `m_tile_shape` / `k_tile_shape` / `n_tile_shape` | Cube 分块 |
| `vector_tile_shape` | Vector/assemble 分块 |
| `n_block_size` | N 轴任务切分粒度（常用 128/256/512） |
| `combined_scale` | Host 写入的 fixpipe scale（由 wrapper 填充） |

---

## Shape 范围与约束

### 动态轴（已验证范围）

| 轴 | 典型范围 | 说明 |
|----|----------|------|
| batch | 1 ~ 8 | batch 维 |
| m | 1 ~ 32 | 小 M 场景已覆盖 |
| k | 128 ~ 7168 | 含 5120、6144 等大 K |
| n | 128 ~ 4096 | N 分块循环覆盖 |

### 约束条件

1. **布局固定为 F-T**：`x1` 为 `[batch, m, k]`，`x2` 为 `[batch, n, k]`，仅支持 `b_trans=True`。
2. **dtype 固定**：`in_dtype=DT_FP8E4M3`，`out_dtype=DT_INT8`（`__post_init__` 校验）。
3. **scale 为 pertensor 标量**：`x1_scale`/`x2_scale` shape 均为 `(1,)`。
4. **fixpipe scale 对齐**：kernel 与 golden 均对 scale 应用 `0xFFFFE000` 位掩码。
5. **输入需 contiguous**：Host wrapper 在 NPU 上调用 `.contiguous()`。

---

## 实现特点

### 性能优化

1. **F-4 展平**：batch 与 m/n 合并为一维，减少 kernel 内 3D view/reshape。
2. **N 轴分块**：`n_block_size` 控制 cube 任务粒度，大 N 用 512 等块。
3. **scale 融合进 matmul**：通过 `extend_params` 传入 fixpipe scale，避免 matmul 后再做 vector `mul`。
4. **双缓冲**：`cube_nbuffer_setting`、`vec_nbuffer_setting` 在 JIT 装饰器中配置。

### 内存访问模式

- `x1_flat` / `x2_flat`：按 batch 偏移 + N 块连续 view。
- `out`：按 `[b_idx, 0, n_off]` 分块 assemble，尾块使用 `valid_shape` 处理对齐。

---

## 精度验证

### 容差设置

- **相对容差 (RTOL)**：0.001（逐点 `|diff| <= rtol * |golden|`）
- **绝对容差 (ATOL)**：0

### 验证方法

1. **Golden**：`tests/ops/experimental/matmul/quant_batch_matmul/quant_batch_matmul_golden.py` 中 `gen_golden`。
2. **PyPTO**：`quant_batch_matmul()` 调用 JIT kernel。
3. **对比工具**：`numpy.testing.assert_allclose`。

### 运行测试

```bash
export TILE_FWK_DEVICE_ID=0
export PTO_TILE_LIB_CODE_PATH=/path/to/pto-isa
cd cry/pypto-gym
python tests/ops/experimental/matmul/quant_batch_matmul/test_quant_batch_matmul.py
python tests/ops/experimental/matmul/quant_batch_matmul/test_quant_batch_matmul.py --golden-only
```

---

## 文件清单

| 文件 | 职责 |
|------|------|
| `docs/SPEC.md` | 需求规格 |
| `docs/DESIGN.md` | 设计文档 |
| `docs/API_REPORT.md` | API 映射报告 |
| `quant_batch_matmul_impl.py` | F-T kernel 与 host wrapper |
| `tests/.../quant_batch_matmul_golden.py` | Golden 参考实现 |
| `tests/.../test_quant_batch_matmul.py` | 精度验证入口 |
