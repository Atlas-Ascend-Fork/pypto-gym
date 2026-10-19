# grouped_matmul_swiglu_quant 算子说明


## 产品支持情况

- Ascend 950PR：不支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持

## 算子语义

`grouped_matmul_swiglu_quant` 是 MoE 场景中的 grouped matmul 后处理融合算子，用于完成 MXFP8 grouped matmul、SwiGLU 激活以及 per-token INT8 动态量化。

### 数学公式

```
gmm_out_i = ScaledMatmul(a_i, b_i, scaled_a_i, scaled_b_i)
value_i, gate_i = chunk(gmm_out_i, 2, dim=-1)
swiglu_i = value_i * sigmoid(value_i) * gate_i
scale_i = max(abs(swiglu_i), dim=-1, keepdim=True) / 127
out_i = clamp(round(swiglu_i / scale_i), -127, 127).to(int8)
out_quant_i = scale_i.squeeze(-1)
```

**展开形式**：

```
gmm_out[m, n] = Σ(k=0..K-1) dequant(a[m, k]) * dequant(b[expert(m), k, n])
output[m, d] = Quant(SiLU(gmm_out[m, d]) * gmm_out[m, d + N/2])
```

其中 `dequant` 由 MXFP8 输入值和 E8M0FNU scale 共同决定。

### 计算流程

1. **Grouped Matmul 计算**：按 `group_list` 切分 token，逐 expert 调用 `pypto.scaled_mm`。
   - `a_i: [M_i, K]`
   - `b_i: [K, N]` 或 `[N, K]`
   - 输出：`gmm_out_i: [M_i, N]`

2. **SwiGLU 激活**：将 `gmm_out_i` 沿最后一维二等分。
   - `value: [M_i, N/2]`
   - `gate: [M_i, N/2]`
   - `swiglu = value * sigmoid(value) * gate`

3. **Per-token 量化**：按 token 求最大绝对值并生成 scale。
   - `scale = max(abs(swiglu), dim=-1) / 127`
   - `output = round(swiglu / scale).to(int8)`

4. **输出组装**：将每个 expert 的 INT8 输出和 FP32 scale 组装回全局输出。

---

## 输入输出规格

### 输入张量

| 名称 | Shape | DType | 说明 |
|------|-------|-------|------|
| `a` | `[M, K]` | FP8 E4M3 | 路由后 token 输入 |
| `b` | `[E, K, N]` 或 `[E, N, K]` | FP8 E4M3 | expert 权重 |
| `scaled_a` | `[M, K/64, 2]` | E8M0FNU | token scale |
| `scaled_b` | `[E, K/64, N, 2]` 或 `[E, N, K/64, 2]` | E8M0FNU | expert 权重 scale |
| `group_list` | `[E]` | int/list | 每个 expert 的 token 数 |

### 输出张量

| 名称 | Shape | DType | 说明 |
|------|-------|-------|------|
| `out` | `[M, N/2]` | int8 | SwiGLU 后 per-token 量化输出 |
| `out_quant` | `[M]` | float32 | 每个 token 对应的量化 scale |

---

## Shape 范围与约束

### 动态轴

| 轴 | 当前覆盖范围 | 说明 |
|----|--------------|------|
| M | 16 | 路由后 token 总数 |
| K | 512 | matmul K 维 |
| N | 7168 | matmul 输出列数，必须为偶数 |
| E | 2 | expert 数量 |

### 约束条件

1. **N 必须为偶数**：SwiGLU 需要将最后一维切分为 value/gate 两半。
2. **group_list 和 M 一致**：`sum(group_list) == M`。
3. **当前测试覆盖 b_trans=False**：代码保留 transpose 配置，内置 case 使用非转置权重布局。
4. **K 与 scale 布局匹配**：当前 scale 按 `K / 64` 构造。
5. **输出 scale 为 per-token 粒度**：`out_quant` shape 为 `[M]`。

---

## 实现特点

### 性能优化

1. **新前端 JIT**：kernel 使用 `@pypto.frontend.jit`。
2. **Cube + Vector 融合**：`scaled_mm` 使用 cube，SwiGLU 和量化使用 vector。
3. **按 expert 分段处理**：通过 `group_list` 保持 grouped matmul 语义。
4. **显式 tile 配置**：使用 `set_cube_tile_shapes` 和 `set_vec_tile_shapes` 控制分块。

### 内存访问模式

- `a` 和 `scaled_a` 按 expert token 范围连续切片。
- `b` 和 `scaled_b` 按 expert 维度读取。
- `out` 和 `out_quant` 使用 `pypto.assemble` 按 token 起始偏移写回。

---

## 精度验证

### 容差设置

- **INT8 输出相对容差 (RTOL)**：0.001
- **INT8 输出绝对容差 (ATOL)**：1
- **Scale 相对容差 (RTOL)**：0.0001
- **Scale 绝对容差 (ATOL)**：0.0001

### 测试用例

| 测试名称 | M | K | N | group_list | 说明 |
|---------|---|---|---|------------|------|
| testcase6 | 16 | 512 | 7168 | `[7, 9]` | 非均匀 2 expert grouped matmul + SwiGLU quant |

### 验证方法

1. **Golden 实现**：`tests/ops/experimental/matmul/grouped_matmul_swiglu_quant/gmm_swiglu_quant_golden.py` 中 `gen_golden`。
2. **PyPTO 实现**：`gmm_swiglu_quant_impl.py` 中 `gen_mxfp8` 调用 `scaled_matmul_kernel`。
3. **对比工具**：`numpy.testing.assert_allclose`。

运行单测（需在单测目录或配置好 `PYTHONPATH`）：

```bash
cd tests/ops/experimental/matmul/grouped_matmul_swiglu_quant
PYTHONPATH=/path/to/pypto-gym/src python test_gmm_swiglu_quant.py
python test_gmm_swiglu_quant.py testcase6
```
