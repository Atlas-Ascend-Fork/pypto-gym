# Flash Attention MHA Forward (PyPTO Kernel)

基于 PyPTO 框架实现的 Flash Attention 前向传播算子，运行于 Ascend NPU。


## 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 文件说明

| 文件 | 说明 |
|------|------|
| `flash_attention_mha_impl.py` | Kernel 实现 |
| `test_flash_attention_mha.py` | 测试用例 + Golden 参考实现 |

## 算法概述

实现标准多头注意力 (Multi-Head Attention) 的前向传播，采用分块 (tiling) 策略降低中间注意力矩阵的内存占用，并输出 softmax 中间量 L/M 用于反向传播。使用 **online softmax algorithm** 实现跨 KV tile 的累加。

### 语义约定

- **Q 侧 (s1_size)**: Q/O/L/M — 序列长度为 `s1_size`
- **KV 侧 (s2_size)**: K/V — 序列长度为 `s2_size`
- **Q_TILE/K_TILE**: 分块大小，Q 和 KV 序列维度分别按此值分块迭代

### 循环结构

```
batch_loop               — 遍历 batch，偏移从 cu_seqlens_q/cu_seqlens_k 动态获取
  head_loop              — 遍历 head
    q_tile_loop          — Q 序列按 Q_TILE 分块
      k_tile_loop        — KV 序列按 K_TILE 分块
```

### 计算流程 (per q_tile, per k_tile)

```
1. S_tile  = Q_tile @ K_tile^T * scale    [sq, sk]     BF16 matmul → FP32
2. M_tile  = max(S_tile, dim=-1)          [sq, 1]      FP32
3. P_tile  = exp(S_tile - M_tile)         [sq, sk]     FP32
4. L_tile  = sum(P_tile, dim=-1)          [sq, 1]      FP32
5. P_norm  = P_tile / L_tile              [sq, sk]     FP32 → cast BF16
6. O_tile  = P_norm @ V_tile              [sq, D]      BF16

Online Softmax 累加逻辑:
  - 首 KV tile:  初始化 O, L, M 累加器
  - 中间 tile:   更新累加器 (mi_new, li_new, oi_tmp)
  - 末 KV tile:  累加完成，normalize 并写回
```

## Kernel 签名

```python
flash_attention_varlen_forward_kernel(
    q,             # [total_q, hidden_dim]     BF16  — Q 输入
    k,             # [total_kv, hidden_dim]    BF16  — K 输入
    v,             # [total_kv, hidden_dim]    BF16  — V 输入
    output,        # [total_q, hidden_dim]     BF16  — 输出 O
    l_output,      # [total_q, n]              FP32  — softmax 分母 L
    m_output,      # [total_q, n]              FP32  — softmax 最大值 M
    cu_seqlens_q,  # [batch_size + 1]          INT32 — 累积 Q seqlen
    cu_seqlens_k,  # [batch_size + 1]          INT32 — 累积 KV seqlen
)
```

其中 `total_q` 和 `total_kv` 为动态轴 (`pypto.DYNAMIC`)，通过 `cu_seqlens` 指定各 batch 的序列长度边界。

**布局区别**:
- Forward 使用 `cu_seqlens` (累积序列长度): `[0, s1, s1+s2, ...]`
- Backward 使用 `actual_q/actual_kv` (各 batch 序列长度): `[s1, s1, ...]`

## Dtype 转换流程

Kernel 内部严格控制 BF16/FP32 转换以平衡精度和性能：

| 阶段 | 操作 | Dtype |
|------|------|-------|
| 输入 | Q/K/V | BF16 |
| S matmul | Q @ K^T * scale | BF16 → FP32 (out_dtype=FP32) |
| softmax | max → exp → sum → div | FP32 全程 |
| 中间 cast | P_norm → BF16 | FP32 → BF16 |
| O matmul | P_bf16 @ V | BF16 → BF16 (out_dtype=BF16) |
| 输出 | O, L, M | O: BF16, L/M: FP32 |

## batch一致性
该算子支持batch一致性，当batch大小变化或者样本在batch中的位置变化时，所有涉及浮点加减乘除计算类型的reduce轴的分块情况与分块间累加顺序保持不变。
### 约束
为了保证batch一致性，算子需要满足以下约束：

1、Set_tile_shape：算子需要保证涉及浮点计算的reduce轴的tile shape设置不能随batch大小而变化（比如不能根据batch大小范围进行tile shape特化调优，在batch轴变化时调用不同tile shape配置，甚至实现逻辑不同的pypto算子kernel）
2、pypto的atomic_add和index_add两个接口不保证确定性，在需要保证确定性和batch一致性的场景不应该使用此接口

## 测试用例

通过独立的 `test_XX` 函数定义，每个函数指定不同的 `batch_size`、`num_heads`、`s1_size`、`s2_size`、`dim`：

| 用例 | batch | heads | Q:s1 | KV:s2 | dim | 说明 |
|------|-------|-------|------|-------|-----|------|
| test_01 | 8 | 8 | 320 | 320 | 64 | 默认配置 |
| test_02 | 2 | 8 | 4096 | 4096 | 64 | 大序列长度 (skip) |
| test_03 | 8 | 16 | 32 | 32 | 32 | 多头小维度 (skip) |
| test_04 | 8 | 16 | 64 | 64 | 32 | 多头中等序列 (skip) |
| test_05 | 8 | 8 | 32 | 32 | 64 | 短序列 (skip) |
| test_06 | 8 | 4 | 64 | 64 | 128 | 少头大维度 (skip) |

### 精度校验

使用 `numpy.testing.assert_allclose` 进行精度验证：

```python
rtol = 0.0078125   # 1/128，等于 BF16 machine epsilon
atol = 0.0001
```

校验输出包括：
- **O (output)**: 前向输出
- **M (m_output)**: softmax 最大值（用于反向传播）
- **L (l_output)**: softmax 分母（用于反向传播）

Golden reference 严格模拟 kernel 内部的 dtype 转换流程（包括中间 BF16 cast），确保对比基准与硬件行为一致，返回三个值 `(o, m, l)`。

## 运行方式

```bash
# 设置设备 ID
export TILE_FWK_DEVICE_ID=0

# 运行全部测试用例
python test_flash_attention_mha.py
```

在 `main()` 中通过注释/取消注释 `test_funcs` 列表控制要执行的用例：

```python
test_funcs = [
    test_01,    # batch=8, heads=8, s1=320, s2=320, dim=64
    test_02,    # batch=2, heads=8, s1=4096, s2=4096, dim=64 (跳过)
    test_03,    # batch=8, heads=16, s1=32, s2=32, dim=32 (跳过)
    test_04,    # batch=8, heads=16, s1=64, s2=64, dim=32 (跳过)
    test_05,    # batch=8, heads=8, s1=32, s2=32, dim=64 (跳过)
    test_06,    # batch=8, heads=4, s1=64, s2=64, dim=128 (跳过)
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

## 分块配置

**实现文件默认配置** (`flash_attention_mha_impl.py`):
```python
Q_TILE = 320  # Kernel 内部默认 Q 序列分块大小
K_TILE = 320  # Kernel 内部默认 KV 序列分块大小
```

**说明**: 
- Kernel 函数参数 `q_tile` 和 `k_tile` 为320


## 与反向传播配合

输出的 L 和 M 用于反向传播：

```python
# 前向传播 (使用 cu_seqlens)
flash_attention_varlen_forward_kernel(q, k, v, o, l, m, cu_seqlens_q, cu_seqlens_k)

# 反向传播 (使用 actual_q/actual_kv)
flash_attention_varlen_backward_kernel(q, k, v, o, do, l, m, dq, dk, dv, actual_q, actual_kv)
```

**注意**: Forward 和 Backward 使用不同的序列长度表示方式：
- Forward: `cu_seqlens` (累积序列长度数组)
- Backward: `actual_q/actual_kv` (各 batch 序列长度)

详见 `flash_attention_mha_grad/` 目录。

## 依赖

- Python 3.x
- PyTorch + torch_npu
- PyPTO (`pypto` 包)
- NumPy
- pytest (可选，用于 skip 标记)