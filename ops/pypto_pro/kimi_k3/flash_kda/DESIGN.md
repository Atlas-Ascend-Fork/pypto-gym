# KDA 设计

本目录包含同一套仿射摘要 CP 正反向实现。完整 shape、dtype、布局、缓存和参数定义见
[README](README.md)；本次目录整理不改变 kernel 计算、分核、流水或数值口径。

## 模块职责

| 文件 | 职责 |
|---|---|
| [flash_kda.py](op_kernel/flash_kda.py) | 正向两个 kernel 与两个 wrapper，生成反向训练缓存 |
| [flash_kda_bwd.py](op_kernel/flash_kda_bwd.py) | 反向两个 kernel 与两个 wrapper，包含激活梯度链 |
| [chunk_kda.py](op_kernel/chunk_kda.py) | PyTorch 自动求导、保存或重建缓存、head bucket 调度 |
| [kda_cp.py](op_kernel/kda_cp.py) | CP 元数据准备和一致性检查、摘要 AllGather、参数梯度 SUM 归约 |
| [op_kernel/__init__.py](op_kernel/__init__.py) | 导出自动求导、CP context 和四个 wrapper |

每个 head bucket 的常规训练路径为正向两个 kernel、反向两个 kernel。
`disable_recompute=False` 时反向额外执行两个正向 kernel 重建缓存。
通信位于 wrapper 外，由自动求导 adapter 或直调方负责。

## 正向

```text
raw q/k/v/g/beta + a_log/dt_bias + initial_state
    → Stage1：L2Norm、gate、beta sigmoid、chunk 预计算、局部 (E,T)
    → AllGather(E,T)（CP>1）
    → Stage2：prefix 合并、真实状态递推、out/final_state/train_cache
```

V-first 状态摘要表达为 `H_out = H_in @ T + E`。
Stage1 保存 `qd/kr/w/gam/mqk/upre` 等预计算结果，Stage2 复用它们。
Stage2 按前序 rank 顺序合并摘要，恢复入站状态，再计算本地输出和训练缓存。
逻辑序列起点使用本地 initial state；延续片段使用摘要恢复的状态。

packed 正向摘要只描述最后一个本地片段。跨 rank 延续边界必须对齐 chunk，
不同逻辑序列由片段边界与延续标志隔离。只有摘要参与通信，prepared 和训练缓存保留在本 rank。

## 反向

```text
train_cache + do + 本地 d_final_state
    → Stage1：反向 (E_grad,R)、参数梯度 buffer 清零
    → AllGather(E_grad,R)（CP>1）
    → Stage2：suffix 合并、状态递推、chunk 梯度、激活梯度链
    → 所有 head bucket 完成后：dA_log/ddt_bias SUM AllReduce（CP>1）
```

反向摘要表达为 `D_in = D_suffix @ R + E_grad`。
`E_grad` 来自本地输出和终态的上游梯度，`R` 表示后继状态梯度向本地起点的传递。
无后继延续依赖时 `R` 为零，相应路径可跳过无效递推。
packed 反向摘要描述第一个本地片段；多片段之间不会引入跨序列的状态依赖。

Stage2 内完成 L2Norm、gate 激活、beta sigmoid 的梯度链，返回 raw 输入梯度。
调用方不得再次应用激活梯度，也不得提前把后继贡献加入本地 `d_final_state`。
`dA_log/ddt_bias` 写入 Stage1 返回的同一参数 buffer；它在 Stage1 后还不是最终梯度。

CP=1 时直接传递原摘要对象，不需要通信。部分 shape 的 Stage1 在该对象上保存私有
`du/dH` 缓存供 Stage2 复用；复制摘要可能丢失该缓存并触发重建。这些私有属性不是接口 ABI。

## 数值与缓存

- 固定 `D=128`、chunk size `64`、V-first 状态，支持动态 B/S/N、GVA、tail 和 packed。
- 矩阵乘使用 BF16 操作数与 FP32 累加；摘要存储为 FP32，仿射表达不保证浮点逐位线性。
- `q_c/k_c/v_c/akk/h/v_new` 为 BF16，`gk_c/beta_c/gam` 为 FP32。
- `v_new` 的 chunk 内物理布局是 `[V,C]`，`h` 的状态布局是 `[V,K]`。
- 外部 initial/final state 和状态递推保留 FP32；不能用 BF16 `h` 的终项替代 FP32 final state。
- 正常路径直接复用正向 `kr/gam/qk_rstd`；兼容路径允许 `kr/gam` 同时缺失，或重建缺失的 norm 信息。

完整训练缓存版本为 `flash_kda_cp_train_v3_bf16_state`，字段表见
[训练缓存](README.md#flash_kda_stage2_wrapper-训练缓存-train_cache)。
公开接口的输出布局和 dtype 不受这次文件迁移影响。

## 调度与通信边界

核数根据设备可用 Cube 核数和任务数选择，不绑定 36 核。
head bucket 必须包含完整的 key/value-head 组，保留 GVA 对应关系；各 bucket 串行调度，
可以共享完整参数梯度 buffer，只清零和累积自身区间。

每个 bucket、每个方向各一次 Tensor 摘要 AllGather；所有 bucket 完成后默认一次参数梯度
SUM AllReduce。`param_grad_reduce='external'` 将参数归约交给外部，token 梯度不在 adapter 中归约。
准备阶段还有 `all_gather_object` 元数据校验。HCCL 路径用于 NPU 通信，Gloo CPU 中转仅用于验证。

原 `kimi_k3` 包级入口薄转发到 `flash_kda/op_kernel`，没有第二份 kernel 实现。
测试中的 CPU Golden 和精度判据分别放在 `tests/assets`，均不依赖设备 kernel。
