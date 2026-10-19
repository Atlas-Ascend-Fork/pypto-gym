# LLaDA2 LigerCrossEntropyLoss PyPTO 融合算子

LLaDA2 模型的 PyPTO 融合 CrossEntropyLoss 算子库。对齐 liger-kernel 的 LigerCrossEntropyLoss，将前向（online softmax + loss + gradient）与反向（逐行加权）融合为单 kernel，减少 HBM 读写。

## 产品支持情况

- Ascend 910B2：支持（当前环境）

## 目录结构

```
llada2/liger_cross_entropy/
├── __init__.py                              # 顶层入口, 重导出 LigerCrossEntropyLoss 等
├── liger_cross_entropy_loss.py              # 公共入口: autograd.Function + nn.Module + functional API
├── liger_cross_entropy_loss_fwd_impl.py     # 前向 kernel + host wrapper
└── liger_cross_entropy_loss_bwd_impl.py     # 反向 kernel + host wrapper
```

## 算子列表

| 算子名称 | 融合范围 | 输入 shape | 输出 shape | 精度 | 对应 eager 代码 |
|---------|---------|-----------|-----------|------|---------------|
| `_ce_fwd_kernel_bf16` | online softmax + loss + z_loss + argmax + gradient | `[BT, V]` bf16 | `[BT,1]` loss + `[BT,V]` saved_input | BF16 计算中间 FP32 | `F.cross_entropy` + `lse` + `softmax` |
| `_ce_bwd_kernel_bf16` | saved_input × grad_vec 逐行加权 | `[BT,V]` bf16 + `[BT,1]` grad | `[BT,V]` grad_input | BF16 | `softmax_grad × grad_output` |

## 输入 / 输出详解

### 前向

| 参数 | shape | dtype | 含义 |
|------|-------|-------|------|
| `_input` (x) | `[BT, V]` | BF16 | logits |
| `target` | `[BT]` | int64 → FP32 | 目标类别索引 |
| `weight` | `[V]` | BF16 → FP32 | 类别权重（可选） |
| `col_idx` | `[1, V]` | FP32 | 列索引（argmax 用） |
| `weight_vec` | `[1, V]` | FP32 | 权重向量（label_smoothing 用） |

| 输出 | shape | dtype | 含义 |
|------|-------|-------|------|
| `loss` | scalar / `[BT]` | BF16 | cross entropy loss |
| `z_loss` | scalar / `[BT]` | BF16 | L2 regularization on lse |
| `saved_input` | `[BT, V]` | BF16 | 梯度预计算结果（供反向用） |
| `token_accuracy` | scalar / `[BT]` | FP32 | top-1 准确率 |
| `predicted_tokens` | `[BT]` | int64 | 预测 token |

### 反向

| 参数 | shape | dtype | 含义 |
|------|-------|-------|------|
| `saved_input` | `[BT, V]` | BF16 | 前向保存的梯度预计算 |
| `grad_vec` | `[BT, 1]` | BF16 | 梯度乘子（标量广播或逐行） |

| 输出 | shape | dtype | 含义 |
|------|-------|-------|------|
| `grad_input` | `[BT, V]` | BF16 | 输入梯度 |

## 融合范围

```
                        ┌─── Pass1: online softmax (amax→sub→exp→sum→merge) ────┐
logits ──→ [Fwd Kernel] ┼─── Loss: lse - ori_x_y + z_loss + mask + reduction ────┼──→ loss, z_loss
                        ├─── Argmax: eq(r,gmax)→where→amin→minimum (可选) ──────┼──→ pred_tokens
                        └─── Pass2: gradient exp(r-lse)→mul→where→assemble ─────┘──→ saved_input

saved_input ──→ [Bwd Kernel: mul(saved × grad_vec) → assemble] ──→ grad_input
```

## 支持参数

| 参数 | 类型 | 默认值 | 含义 |
|------|------|--------|------|
| `ignore_index` | int | -100 | 忽略的目标类别 |
| `lse_square_scale` | float | 0.0 | z_loss 系数 |
| `label_smoothing` | float | 0.0 | 标签平滑系数 |
| `reduction` | str | "mean" | 归约方式: mean/sum/none |
| `softcap` | float? | None | tanh 软上限 |
| `weight` | Tensor? | None | 类别权重 |
| `return_z_loss` | bool | False | 返回 z_loss |
| `return_token_accuracy` | bool | False | 返回 token 准确率 |
| `return_predicted_tokens` | bool | False | 返回预测 token |

## 关键实现细节

### 前向 kernel

- **V=STATIC、BT=DYNAMIC**: V 用 `pypto.STATIC` 标注，编译期静态轴；BT 用 `pypto.loop` + valid_shape 尾块
- **online softmax**: 分 tile 遍历 V 轴，维护 running max/sum，避免全局 softmax 的 HBM 读写
- **sg_set_scope 合图**: Pass1 softmax 链和 Pass2 gradient 链分别用 `sg_set_scope=1/-1` 包裹，减少子图间调度开销
- **label_smoothing=0 死代码消除**: `lse_square_scale=0` 和 `eps_f=0` 时跳过冗余计算
- **host wrapper 优化**: `target * target_mask` 复用（替代 `torch.where + new_zeros`），`torch.gather` 替代高级索引

### 反向 kernel

- **统一乘子路径**: path2（reduction='none'，逐行加权）和 path3（mean/sum，标量广播）共用同一 kernel
- **path1 短路**: `torch.equal(grad_output, 1.0)` 时直接返回 saved_input，跳过 kernel
- **sg_set_scope 合图**: V 循环内 `view→mul→assemble` 合并为单子图

## 配置常量速查

### 前向

| 常量 | 值 | 位置 |
|------|-----|------|
| TILE_BT | 8 | `liger_cross_entropy_loss_fwd_impl.py` |
| TILE_V | 1024 | `liger_cross_entropy_loss_fwd_impl.py` |
| unroll_list | [4, 1] | `liger_cross_entropy_loss_fwd_impl.py` |
| combine_axis | True | `liger_cross_entropy_loss_fwd_impl.py` |
| device_sched_mode | 1 | `liger_cross_entropy_loss_fwd_impl.py` |
| stitch_function_max_num | 128 | `liger_cross_entropy_loss_fwd_impl.py` |

### 反向

| 常量 | 值 | 位置 |
|------|-----|------|
| TILE_BT | 4 | `liger_cross_entropy_loss_bwd_impl.py` |
| TILE_V | 8192 | `liger_cross_entropy_loss_bwd_impl.py` |
| unroll_list | [4, 1] | `liger_cross_entropy_loss_bwd_impl.py` |
| device_sched_mode | 1 | `liger_cross_entropy_loss_bwd_impl.py` |
| stitch_function_max_num | 128 | `liger_cross_entropy_loss_bwd_impl.py` |

## 使用方式

```python
from ops.pypto_tensor.llada2.liger_cross_entropy import LigerCrossEntropyLoss

# 与 torch.nn.CrossEntropyLoss 接口一致
criterion = LigerCrossEntropyLoss(reduction="mean", ignore_index=-100)
loss = criterion(logits, target)
loss.backward()
```

```python
# functional 形式
from ops.pypto_tensor.llada2.liger_cross_entropy import liger_cross_entropy

result = liger_cross_entropy(logits, target, return_z_loss=True, return_token_accuracy=True)
# result.loss, result.z_loss, result.token_accuracy, result.predicted_tokens
```
