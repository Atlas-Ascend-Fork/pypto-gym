# aclnnSparseFlashAttention

## 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 功能说明

- 接口功能：实现稀疏Flash Attention前向计算（TND格式），用于DeepSeek V3模型的长序列推理场景。对每个query token，仅关注由topk_indices指定的稀疏KV子集，大幅降低注意力计算的复杂度。

- 核心功能：
  - 输入采用TND（Token-Head-Dim）格式，支持多batch动态序列长度。
  - Q和K分别包含nope（无旋转位置编码）和pe（含旋转位置编码）两部分，注意力分数为两者之和。
  - 支持GQA（Grouped Query Attention），每个KV head对应group个Q heads。
  - 外层循环为batch维度（B），内层循环为per-batch query token维度（s）。
  - npu_actual_q_len和npu_actual_kv_len均为前缀和格式。

## 计算公式

对于每个query token $t$和KV head $h$：

$$
\begin{aligned}
S_{nope} &= Q_{nope}[t] \times KV_{nope}[\text{topk}]^T \\
S_{rope} &= Q_{pe}[t] \times K_{pe}[\text{topk}]^T \\
S &= (S_{nope} + S_{rope}) \times \text{scale} \\
\text{softmax\_max} &= \max(S, \dim=-1) \\
P &= \frac{\exp(S - \text{softmax\_max})}{\sum \exp(S - \text{softmax\_max})} \\
O &= P \times V[\text{topk}]
\end{aligned}
$$

### 符号说明

| 符号 | 含义 |
|------|------|
| $Q_{nope}$ | Query nope部分，shape为$(T_1, N_1, D)$ |
| $Q_{pe}$ | Query pe（RoPE）部分，shape为$(T_1, N_1, D_{ROPE})$ |
| $KV_{nope}$ | Key/Value nope部分（compressed_kv_norm），shape为$(T_2, N_2, D)$ |
| $K_{pe}$ | Key pe（RoPE）部分，shape为$(T_2, N_2, D_{ROPE})$ |
| topk | topk_indices指定的稀疏KV索引 |
| $\text{scale}$ | 注意力缩放因子，通常为$1/\sqrt{D+D_{ROPE}}$ |
| $N_1$ | Query head数 |
| $N_2$ | KV head数 |
| $\text{group}$ | $N_1 / N_2$，GQA分组数 |

## 函数原型

```python
def sfa_forward_tnd(
    q_nope, compressed_kv_norm, topk_indices, 
    q_pe, k_pe, npu_actual_q_len, npu_actual_kv_len, 
    core_attn_out, softmax_max_out, softmax_sum_out, 
    nq, n_kv, scale, sparse_size, tile_config, max_total_kv=1024 * 1024
):
    """工厂函数：创建JIT编译的SFA前向TND kernel。

```

## 参数说明

### Kernel输入参数

| 参数名 | 输入/输出 | 描述 | 数据类型 | 维度(shape) |
|:--- |:--- |:--- |:--- |:--- |
| q_nope | 输入 | Query nope部分 | BFLOAT16 | (T1, N1, D)，T1为动态维度 |
| compressed_kv_norm | 输入 | 归一化后的compressed KV | BFLOAT16 | (T2, N2, D)，T2为动态维度 |
| topk_indices | 输入 | 稀疏TopK索引 | INT32 | (T1, N2, sparse_size) |
| q_pe | 输入 | Query pe（RoPE）部分 | BFLOAT16 | (T1, N1, D_ROPE) |
| k_pe | 输入 | Key pe（RoPE）部分 | BFLOAT16 | (T2, N2, D_ROPE) |
| npu_actual_q_len | 输入 | Query长度前缀和 | INT32 | (B,)，B为动态维度 |
| npu_actual_kv_len | 输入 | KV长度前缀和 | INT32 | (B,) |
| core_attn_out | 输出 | 注意力输出 | BFLOAT16 | (T1, N1, D) |
| softmax_max_out | 输出 | Softmax最大值 | FLOAT32 | (N2, T1, group) |
| softmax_sum_out | 输出 | Softmax求和值 | FLOAT32 | (N2, T1, group) |

### 分片配置（SaTileShapeConfig）

| 参数名 | 描述 | 典型值 |
|:--- |:--- |:--- |
| s_kv_tile | KV序列分片大小 | 2048 |
| c1_tile_shape | 第一次矩阵乘分片参数 | [128, 128, 128, 256, 256, 256] |
| v1_tile_shape | Softmax向量运算分片参数 | [8, 2048] |
| c2_tile_shape | 第二次矩阵乘分片参数 | [128, 128, 128, 256, 128, 128] |
| v2_tile_shape | 输出向量运算分片参数 | [64, 128] |

## 约束说明

### 确定性计算

- aclnnSparseFlashAttention默认采用确定性实现，相同输入多次调用结果一致。

### 公共约束

1. 仅支持BFLOAT16数据类型，注意力分数以FP32进行累加。
2. 输入Tensor的数据格式仅支持ND。
3. npu_actual_q_len和npu_actual_kv_len均为前缀和格式。
4. topk_indices中的索引值为T2维度上的全局索引。
5. compressed_kv_norm同时用作Key和Value（MLA架构特性）。
6. 当effective topk ≤ sparse_size时，直接view而非index_select以提升性能。

### 规格约束

| 规格项 | 规格 | 规格说明 |
|:--- |:--- |:--- |
| kv_lora_rank (D) | 512 | KV LoRA秩 |
| qk_rope_dim (D_ROPE) | 64 | QK RoPE维度 |
| sparse_size (K) | 2048 | 稀疏TopK大小 |
| N1 (nq) | 2、48、128等 | Query head数 |
| N2 (n_kv) | 1 | KV head数 |
| max_total_kv | ≥T2*N2 | T2*N2的静态上界 |

## 调用示例

```python
import torch
from sfa_sparse_attention_impl import sfa_forward_tnd, SaTileShapeConfig

# ========== 参数配置 ==========
B = 2
nq, n_kv = 48, 1
kv_lora_rank, qk_rope_dim = 512, 64
sparse_size = 2048
scale = 0.07216878364870322
group = nq // n_kv

npu_actual_q_len_list = [2, 1]
npu_actual_kv_len_list = [4096, 4096]
T1 = sum(npu_actual_q_len_list)
T2 = sum(npu_actual_kv_len_list)

# ========== 构造输入数据 ==========
q_nope = torch.randn(T1, nq, kv_lora_rank, dtype=torch.bfloat16).npu()
q_pe = torch.randn(T1, nq, qk_rope_dim, dtype=torch.bfloat16).npu()
compressed_kv_norm = torch.randn(T2, n_kv, kv_lora_rank, dtype=torch.bfloat16).npu()
k_pe = torch.randn(T2, n_kv, qk_rope_dim, dtype=torch.bfloat16).npu()
topk_indices = torch.randint(0, T2, (T1, n_kv, sparse_size), dtype=torch.int32).npu()

# 前缀和格式
npu_actual_q_len = torch.tensor([2, 3], dtype=torch.int32).npu()
npu_actual_kv_len = torch.tensor([4096, 8192], dtype=torch.int32).npu()

core_attn_out = torch.empty(T1, nq, kv_lora_rank, dtype=torch.bfloat16).npu()
softmax_max_out = torch.empty(n_kv, T1, group, dtype=torch.float32).npu()
softmax_sum_out = torch.empty(n_kv, T1, group, dtype=torch.float32).npu()

# ========== 创建并调用kernel ==========
tile_config = SaTileShapeConfig(
    s_kv_tile=2048,
    c1_tile_shape=[128, 128, 128, 256, 256, 256],
    v1_tile_shape=[8, 2048],
    c2_tile_shape=[128, 128, 128, 256, 128, 128],
    v2_tile_shape=[64, 128]
)

kernel = sfa_forward_tnd(kv_lora_rank, qk_rope_dim, nq, n_kv,
                          scale, sparse_size, tile_config)
kernel(q_nope, compressed_kv_norm, topk_indices, q_pe, k_pe,
       npu_actual_q_len, npu_actual_kv_len,
       core_attn_out, softmax_max_out, softmax_sum_out)

torch.npu.synchronize()
print(f"Attention output shape: {core_attn_out.shape}")
```
