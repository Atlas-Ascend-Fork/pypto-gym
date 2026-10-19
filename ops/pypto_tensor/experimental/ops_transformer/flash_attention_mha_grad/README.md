# Flash Attention MHA GRAD (PyPTO Kernel)

基于 PyPTO 框架实现的 Flash Attention 反向传播算子，运行于 Ascend NPU。

## 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持

## 文件说明

| 文件 | 说明 |
|------|------|
| `flash_attention_mha_grad.py` | Kernel 实现 + 多规格测试用例 |

## 算法概述

实现标准多头注意力 (Multi-Head Attention) 的反向传播，采用分块 (tiling) 策略降低中间矩阵的内存占用。

### 语义约定

- **Q 侧 (s1_size)**: Q/O/dO/L/M/dQ — 序列长度为 `s1_size`
- **KV 侧 (s2_size)**: K/V/dK/dV — 序列长度为 `s2_size`
- **S2_TILE**: 分块大小，Q 和 KV 序列维度均按此值分块迭代

### 循环结构

```
batch_loop (parallel)       — 遍历 batch，偏移从 actual_q/actual_kv 动态获取
  head_loop                 — 遍历 head (每次处理 2 个 head)
    q_tile_loop             — Q 序列按 S2_TILE 分块
      kv_tile_loop          — KV 序列按 S2_TILE 分块
        h_s_idx (unroll 2)  — 处理 2 个 head
```

### 计算流程 (per q_tile, per kv_tile)

```
1. S_tile  = Q_tile @ K_tile^T           [sq, s2]     BF16 matmul → FP32
2. P_tile  = exp(S * scale - M) / L      [sq, s2]     FP32
3. dP_tile = dO_tile @ V_tile^T          [sq, s2]     BF16 matmul → FP32
4. D       = sum(O_tile * dO_tile, -1)   [sq, 1]      BF16 → FP32
5. dS_tile = P * (dP - D)                [sq, s2]     FP32 → cast BF16
6. dK_tile += dS^T @ Q_tile * scale      [s2, D]      BF16 matmul → FP32 → BF16
7. dV_tile += P^T @ dO_tile              [s2, D]      BF16 matmul → BF16
8. dQ_partial = dS @ K_tile * scale      [sq, D]      BF16 matmul → FP32
   dQ 跨 kv_tile 累加 (FP32)，最终 cast BF16 写回
```

## Kernel 签名

```python
flash_attention_varlen_backward_kernel(
    q,          # [bs, N, D]     BF16  — Q 输入
    k,          # [bs, N, D]     BF16  — K 输入
    v,          # [bs, N, D]     BF16  — V 输入
    o,          # [bs, N, D]     BF16  — 前向输出 O
    do,         # [bs, N, D]     BF16  — dO (输出梯度)
    l_input,    # [bs, N, 1]     FP32  — softmax 分母 L
    m_input,    # [bs, N, 1]     FP32  — softmax 最大值 M
    dq,         # [bs, N*D]      BF16  — dQ 输出
    dk,         # [bs, N*D]      BF16  — dK 输出
    dv,         # [bs, N*D]      BF16  — dV 输出
    actual_q,   # [batch_size]   INT32 — 每个 batch 的 Q seqlen
    actual_kv,  # [batch_size]   INT32 — 每个 batch 的 KV seqlen
)
```

其中 `bs` 为动态轴 (`pypto.DYNAMIC`)，`N` (num_heads) 和 `D` (head_dim) 为静态轴。

Kernel 入口处将三维输入 `[bs, N, D]` reshape 为 `[bs, N*D]`，`[bs, N, 1]` reshape 为 `[bs, N]`，以便按 head 做 view 切片。

## Dtype 转换流程

Kernel 内部严格控制 BF16/FP32 转换以平衡精度和性能：

| 阶段 | 操作 | Dtype |
|------|------|-------|
| 输入 | Q/K/V/O/dO | BF16 |
| D 计算 | cast(O, FP32) * cast(dO, FP32) → sum | BF16 → FP32 |
| S/dP matmul | Q@K^T, dO@V^T | BF16 → FP32 (out_dtype=FP32) |
| softmax | exp(S*scale - M) / L | FP32 全程 |
| 中间 cast | dS, P → BF16 | FP32 → BF16 |
| dK matmul | dS^T@Q → FP32 * scale | BF16 → FP32 → BF16 |
| dV matmul | P^T@dO | BF16 → BF16 (out_dtype=BF16) |
| dQ matmul | dS@K → FP32 * scale，累加 FP32 | BF16 → FP32 → BF16 |

## 测试用例

通过独立的 `test_XX` 函数定义，每个函数指定不同的 `batch_size`、`num_heads`、`s1_size`、`s2_size`、`dim`：

| 用例 | batch | heads | Q:s1 | KV:s2 | dim | 说明 |
|------|-------|-------|------|-------|-----|------|
| test_01 | 8 | 8 | 320 | 320 | 64 | 默认配置 |
| test_02 | 8 | 8 | 2432 | 2432 | 64 | 大序列长度 (skip) |
| test_03 | 8 | 16 | 32 | 32 | 32 | 多头小维度 |
| test_04 | 8 | 16 | 64 | 64 | 32 | 多头中等序列 |
| test_05 | 8 | 8 | 32 | 32 | 64 | 短序列 |
| test_06 | 8 | 4 | 64 | 64 | 128 | 少头大维度 |

### 精度校验

使用 `numpy.testing.assert_allclose` 进行精度验证：

```python
rtol = 0.0078125   # 1/128，等于 BF16 machine epsilon
atol = 0.0001
```

Golden reference 严格模拟 kernel 内部的 dtype 转换流程（包括中间 BF16 cast），确保对比基准与硬件行为一致。

## 运行方式

```bash
# 设置设备 ID
export TILE_FWK_DEVICE_ID=0

# 运行全部测试用例
python flash_attention_mha_grad.py
```

在 `main()` 中通过注释/取消注释 `test_funcs` 列表控制要执行的用例：

```python
test_funcs = [
    test_01,    # batch=8, heads=8, s1=320, s2=320, dim=64
    # test_02,  # batch=8, heads=8, s1=2432, s2=2432, dim=64 (跳过)
    test_03,    # batch=8, heads=16, s1=32, s2=32, dim=32
    ...
]
```

## 添加新用例

```python
def test_07(device):
    """batch=4, heads=8, s1=128, s2=256, dim=64"""
    return run_test(device, batch_size=4, num_heads=8, s1_size=128, s2_size=256, dim=64)
```

`run_test` 支持的可选参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `batch_size` | 批次数量 | 1 |
| `num_heads` | 注意力头数 | NUM_HEADS (8) |
| `s1_size` | Q 序列长度 | 320 |
| `s2_size` | KV 序列长度 | = s1_size |
| `dim` | 每个头维度 | HEAD_DIM (64) |
| `tile_config` | TileConfig 分块配置 | S2_TILE=320 |

## 依赖

- Python 3.x
- PyTorch + torch_npu
- PyPTO (`pypto` 包)
- NumPy
