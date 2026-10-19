# inplace_add_rms_norm

PyPTO 自定义算子：融合 elementwise add + RMSNorm，所有计算结果**原地写回**输入 buffer。


## 产品支持情况

- Ascend 950PR：不支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 算子概述

**计算公式**

```
x_add = x1 + x2
ms    = mean(x_add^2, dim=-1, keepdim=True)
rstd  = 1 / sqrt(ms + eps)
y     = x_add * rstd * gamma
```

**Inplace 写回（核心语义）**

| Buffer | 写入内容 | 备注 |
|--------|---------|------|
| `x1` | `y` (RMSNorm 最终输出) | 同 GM；`x1.data_ptr()` 调用前后不变 |
| `x2` | `x_add` (add 中间结果) | 同 GM；`x2.data_ptr()` 调用前后不变 |
| `rstd` | `1/sqrt(mean+eps)` | 新建独立 buffer (`torch.empty`) |

## 输入输出

| 名称 | shape | dtype | 角色 |
|------|-------|-------|------|
| x1 | [B, S, 7168] | bfloat16 | 输入 + inplace 输出 |
| x2 | [B, S, 7168] | bfloat16 | 输入 + inplace 输出 |
| gamma | [7168] | bfloat16 | 只读 |
| eps | scalar | float | 数值稳定项，默认 1e-6 |
| **返回** | (x1, x2, rstd) | (bf16, bf16, bf16) | x1/x2 是输入 alias；rstd 是新建 [B,S,1] |

动态轴：B ∈ [1,144], S ∈ [1,8192]；约束 `B*S ∈ [1024,8192]` 或 `(B∈[16,144] 且 S==1)`。

## 目录结构

```
custom/inplace_add_rms_norm/
├── SPEC.md                              # 算子规格
├── API_REPORT.md                        # API 探索报告
├── DESIGN.md                            # 设计方案（API/Tiling/Loop/inplace 时序）
├── inplace_add_rms_norm_golden.py       # PyTorch golden 参考实现
├── inplace_add_rms_norm_impl.py         # PyPTO kernel + wrapper
├── test_inplace_add_rms_norm.py         # 测试入口
├── test_cases.json                      # 测试用例配置
├── README.md                            # 本文件
└── .orchestrator_state.json             # orchestrator 状态文件
```

## 运行方式

### 环境准备

```bash
# 必须设置 NPU 卡 ID（Ascend910 phy-id 14 / 15）
export TILE_FWK_DEVICE_ID=0
export PTO_TILE_LIB_CODE_PATH=$(pwd)/pto-isa
```

### 运行测试

```bash
cd custom/inplace_add_rms_norm

# 列出所有测试用例
python3 test_inplace_add_rms_norm.py --list

# 跑全部用例（level0..level5）
python3 test_inplace_add_rms_norm.py

# 仅跑 first-run 烟测（level0+level1+level4）
python3 test_inplace_add_rms_norm.py --quick

# 跑单个用例
python3 test_inplace_add_rms_norm.py level1
```

### 三态结果协议

| 输出 | exit code | 含义 |
|------|-----------|------|
| `[PRECISION_PASS]` | 0 | 精度通过 |
| `[PRECISION_FAIL]` | 1 | 精度失败（数值不匹配） |
| 无标记 | 2 | 编译/运行/inplace 语义功能问题 |

## 关键实现要点

### 1. torch.library mutable schema

```python
"inplace_add_rms_norm(Tensor(a!) x1, Tensor(b!) x2, Tensor gamma, float eps)"
" -> (Tensor, Tensor, Tensor)"
```

`Tensor(a!)` / `Tensor(b!)` 标记 `x1` / `x2` 为 inplace mutable 输入，让 PyTorch 视图机制和 graph capture 正确处理别名。

### 2. Wrapper 严格透传

```python
def npu_inplace_add_rms_norm(x1, x2, gamma, eps):
    rstd = torch.empty([x1.size(0), x1.size(1), 1], dtype=torch.bfloat16, device=x1.device)
    inplace_add_rms_norm_kernel(x1, x2, gamma, x1, x2, rstd, eps)
    return x1, x2, rstd
```

- 不做 reshape / broadcast / cast / contiguous / view / unsqueeze / squeeze
- 不注册任何 PyTorch hook
- 仅做：分配 rstd buffer、调 kernel、返回

### 3. Kernel 内 inplace 写回

- 输入 `x1`、`x2`，输出 `x1_out`、`x2_out` 是不同形参（不同 SSA 变量）
- wrapper 把同一个 `torch.Tensor` 同时作为 `x1` 输入和 `x1_out` 输出传入，复用同一段 GM
- kernel 用 `pypto.assemble(...)` 写入 `x1_out` / `x2_out` / `rstd_out`
- 数据流时序：先 `view+cast` 读入 → 计算 → 最后 `assemble` 写出，保证读后写，无图回环

### 4. Tiling

- Vector 算子，`set_vec_tile_shapes(1, 7168)` 单行 fp32 = 28 KB ≤ 64 KB（满足 `pypto.sum` 约束）
- 双动态轴 (B, S) 在 kernel 入口先 `pypto.reshape` 折成 2D `[B*S, H]` 单动态轴
- 沿 `B*S` 维 `pypto.loop_unroll` + `pypto.view([1, H], [bs_idx, 0])` 切固定 int tile

## 验证要点

测试代码必须验证：

1. **inplace data_ptr**：`x1.data_ptr()` / `x2.data_ptr()` 调用前后保持不变
2. **alias**：返回值 `final_y` / `x_add` 与 `x1` / `x2` 同 `data_ptr`
3. **rstd 新建**：`rstd.data_ptr()` 不与 x1/x2 重合
4. **内容覆写**：`x1` 内容被 RmsNorm 输出覆盖；`x2` 内容被 add 中间结果覆盖
5. **shape & dtype**：x1=[B,S,H] bf16、x2=[B,S,H] bf16、rstd=[B,S,1] bf16
6. **精度**：与 golden 对比，atol/rtol 来自 test_cases.json（默认 0.01）

## 已知限制

- **dtype**：仅支持 bfloat16
- **H 固定 7168**：其它 H 需修改 `H_CONST` 并重编译
- **shape 约束**：`B*S ∈ [1024, 8192]` 或 `B∈[16,144] 且 S==1`
- **必须 NPU**：当前 wrapper 假设 NPU 设备；CPU 模式下 PyPTO kernel 不可用

## 参考资料

- 教程：`docs/tutorials/distributed/matmul_allreduce_rmsnorm.md`（add+rmsnorm 同业务）
- 示例：`examples/02_intermediate/basic_nn/layer_normalization/layer_norm.py`（rms_norm_kernel 骨架）
- 生产参考：`models/deepseek_v4/hc_pre_impl.py`（双 DYN reshape + assemble + torch.library）
