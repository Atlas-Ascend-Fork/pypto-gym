# Flash Attention HiFP8 Forward (PyPTO Kernel)

基于 PyPTO 框架实现的 Flash Attention 前向传播算子，输入采用 HiFP8 (HF8) 量化格式，运行于 Ascend NPU。

与 BF16 版本 (`flash_attention_mha`) 的核心区别在于：Q/K/V 以 HF8 格式输入，配合 per-token 反量化 scale (`d_scale`) 还原为 FP32 参与计算；注意力概率 P 在 P@V 之前量化为 HF8（使用 `p_scale`），最终 `p_scale` 在 `O = O / L` 中被约去。


## 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

> Kernel 内部含 `DAV_3510` 平台专属分支（`set_pass_options(sg_set_scope=...)`），在该架构下启用对应的 cube/vec 调度策略。

## 文件说明

| 文件 | 说明 |
|------|------|
| `flash_attention_fp8_impl.py` | Kernel 实现 |

> 当前目录未提供测试用例与 Golden 参考实现，精度校验方法见下文「精度校验」一节。

## 算法概述

实现多头注意力的前向传播，采用分块 (tiling) 策略降低中间注意力矩阵的内存占用，并输出 softmax 中间量 L/M。使用 **online softmax algorithm** 实现跨 KV tile 的累加。与 BF16 版本不同，本算子在 HF8 量化域中工作：输入反量化、P 再量化，以降低访存带宽与显存占用。

### 语义约定

- **Q 侧 (s1_size)**: Q/O/L/M — 序列长度为 `s1_size`
- **KV 侧 (s2_size)**: K/V — 序列长度为 `s2_size`
- **Q_TILE/K_TILE**: 分块大小，Q 和 KV 序列维度分别按此值分块迭代
- **d_scale**: per-token 反量化 scale（乘性），形状 `[total_seq, N, 1]`
- **p_scale**: P 量化 scale（标量），形状 `[1]`

### 循环结构

```
batch_loop               — 遍历 batch，偏移从 cu_seqlens_q/cu_seqlens_k 动态获取
  head_loop              — 遍历 head（步进 2，外层 h_num = num_heads // 2）
    q_tile_loop          — Q 序列按 Q_TILE 分块
      k_tile_loop        — KV 序列按 K_TILE 分块
        内层 range(2)    — 每个外层 head 迭代处理 2 个 head（h_act_idx = h_idx*2 + h_s_idx）
```

### 计算流程 (per q_tile, per k_tile)

```
1. Q_fp32 = cast(Q_hf8, FP32) * d_scale_q         [sq, D]   HF8 → FP32 (反量化)
2. K_fp32 = cast(K_hf8, FP32) * d_scale_k         [sk, D]   HF8 → FP32 (反量化)
3. S_tile = Q_fp32 @ K_fp32^T * scale             [sq, sk]  FP32
4. M_tile = max(S_tile, dim=-1)                   [sq, 1]   FP32
5. P_tile = exp(S_tile - M_tile)                  [sq, sk]  FP32
6. P_hf8  = cast(cast(P_tile * p_scale, HF8), FP32) [sq, sk] FP32 (P 再量化)
7. L_tile = sum(P_hf8, dim=-1)                    [sq, 1]   FP32 (含 p_scale)
8. V_fp32 = cast(V_hf8, FP32) * d_scale_v         [sk, D]   HF8 → FP32 (反量化)
9. O_tile = P_hf8 @ V_fp32                        [sq, D]   FP32 (经中间 BF16 cast)

Online Softmax 累加逻辑:
  - 首 KV tile:  初始化 O, L, M 累加器
  - 中间 tile:   更新累加器 (mi_new, li_new, oi_tmp)
  - 末 KV tile:  累加完成，O = oi_tmp / li_new (p_scale 在 L 中被约去)，cast BF16 写回
```

> 说明：第 9 步的 P@V matmul 在 Kernel 中将 `pij` 与 `v_fp32` 先 cast 到 BF16 再做 BF16 matmul（`out_dtype` 视首/末 tile 不同取 FP32 或 BF16），由 `pypto.cast` 显式控制。

## Kernel 签名

```python
flash_attention_fp8_varlen_forward_kernel(
    q,             # [total_seq, N, D]         HF8   — Q 输入
    k,             # [total_seq, N, D]         HF8   — K 输入
    v,             # [total_seq, N, D]         HF8   — V 输入
    d_scale_q,     # [total_seq, N, 1]         FP32  — Q per-token 反量化 scale
    d_scale_k,     # [total_seq, N, 1]         FP32  — K per-token 反量化 scale
    d_scale_v,     # [total_seq, N, 1]         FP32  — V per-token 反量化 scale
    p_scale,       # [1]                       FP32  — P 量化 scale（标量）
    output,        # [total_q, hidden_dim]     BF16  — 输出 O
    l_output,      # [total_q, N]              FP32  — softmax 分母 L
    m_output,      # [total_q, N]              FP32  — softmax 最大值 M
    cu_seqlens_q,  # [batch_size + 1]          INT32 — 累积 Q seqlen
    cu_seqlens_k,  # [batch_size + 1]          INT32 — 累积 KV seqlen
)
```

其中 `total_seq`（Q/K/V 与对应 d_scale 的首轴）为动态轴 (`pypto.DYNAMIC`)，通过 `cu_seqlens` 指定各 batch 的序列长度边界；`hidden_dim = num_heads * head_dim`，`N = num_heads` 在 Kernel 内部从 `q.shape[1]`、`q.shape[2]` 读取。

**与 BF16 版本的布局区别**：
- BF16 (`flash_attention_mha`): Q/K/V 为 2D `[total_q, hidden_dim]`
- HiFP8 (本算子): Q/K/V 为 3D `[total_seq, N, D]`，配合 3D 的 per-head 反量化 scale `[total_seq, N, 1]`

**布局约定** (与 BF16 版本一致):
- Forward 使用 `cu_seqlens` (累积序列长度): `[0, s1, s1+s2, ...]`

## Dtype 转换流程

Kernel 内部严格控制 HF8/FP32/BF16 转换以在量化精度与性能间取得平衡：

| 阶段 | 操作 | Dtype |
|------|------|-------|
| 输入 | Q/K/V | HF8 |
| 输入 | d_scale_q/k/v, p_scale | FP32 |
| 反量化 | cast(Q/K/V, FP32) * d_scale | HF8 → FP32 |
| S matmul | Q_fp32 @ K_fp32^T * scale | FP32 → FP32 (out_dtype=FP32) |
| softmax | max → exp → (P*p_scale→HF8→FP32) → sum | FP32 全程，P 中途量化为 HF8 |
| P@V matmul | cast(P, BF16) @ cast(V_fp32, BF16) | FP32 → BF16 (matmul out_dtype 视首/末 tile 取 FP32/BF16) |
| 输出 | O, L, M | O: BF16, L/M: FP32 |

## 量化说明

- **反量化 (dequant)**: Q/K/V 的 HF8 值乘以 per-token 的 `d_scale` 还原为 FP32 真值，`d_scale` 形状为 `[total_seq, N, 1]`，对每个 token、每个 head 独立。
- **P 再量化**: 注意力概率 `P = exp(S - M)` 乘以标量 `p_scale` 后量化为 HF8，再 cast 回 FP32 参与 P@V。
- **p_scale 约去**: 由于 `L = sum(P * p_scale)` 与 `O = (P * p_scale) @ V` 均含 `p_scale` 因子，末 tile 执行 `O = O / L` 时 `p_scale` 被约去，最终输出不受 `p_scale` 量级影响。

## 分块配置

**实现文件默认配置** (`flash_attention_fp8_impl.py`):
```python
Q_TILE = 320  # Kernel 内部默认 Q 序列分块大小
K_TILE = 320  # Kernel 内部默认 KV 序列分块大小
```

**内部 tile 形状**（在循环内分阶段设置）:
```python
v1_tile = [64, 512]
v2_tile = [512, 64]

# 初始
pypto.set_cube_tile_shapes([128, 128], [128, 256], [128, 128])
pypto.set_vec_tile_shapes(64, 256)
# S = Q@K^T matmul 前
pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
pypto.set_vec_tile_shapes(64, 512)   # v1_tile
# P@V matmul 前
pypto.set_cube_tile_shapes([128, 512], [256, 512], [64, 64])
```

**JIT 装饰器选项**:
```python
@pypto.frontend.jit(
    debug_options={"runtime_debug_mode": 1},
    runtime_options={"device_sched_mode": 0, "stitch_function_max_num": 1024},
    pass_options={
        "cube_l1_reuse_setting": {-1: 8},
        "vec_nbuffer_setting": {-1: 8},
        "cube_nbuffer_setting": {-1: 8},
    },
)
```

## 精度校验

当前目录未提供测试文件与 Golden 参考实现。若需自行校验，建议按下述方式构建 Golden：

1. **Golden 参考**: 将 HF8 输入按 `d_scale` 反量化为 FP32，按标准 attention 公式（`scale = 1/sqrt(head_dim)`，online softmax）计算 `O / L / M`，并严格模拟 Kernel 内部的 dtype 转换（P 经 `*p_scale → HF8 → FP32` 再量化、P@V 经 BF16 cast）以确保对比基准与硬件行为一致。
2. **校验输出**: `O (output)`、`M (m_output)`、`L (l_output)` 三个张量。
3. **容差**: 由于涉及 HF8 量化，容差应较 BF16 版本（`rtol=1/128`）适当放宽，具体取值需结合 `p_scale` 与 `d_scale` 的量级评估。

## 运行方式

```bash
# 设置设备 ID
export TILE_FWK_DEVICE_ID=0

# 需自行编写测试入口（含 Golden 参考）后运行，例如：
python test_flash_attention_fp8.py
```

## 与反向传播配合

输出的 L 和 M 用于反向传播。当前仓库 `ops_transformer/` 目录下未提供 HiFP8 的反向实现（BF16 版本反向见 `flash_attention_mha_grad/`）。

## 依赖

- Python 3.x
- PyTorch + torch_npu
- PyPTO (`pypto` 包)
- NumPy
