# aclnnSparseFlashAttentionGrad

## 产品支持情况

- Ascend 950PR：不支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 功能说明

- 接口功能：实现稀疏Flash Attention的反向梯度计算（TND格式）。根据前向计算的中间结果（softmax_max、softmax_sum）和输出梯度（d_out），计算Q、K、V的梯度。

- 核心功能：
  - Q和K分别拆分为nope（无旋转位置编码）和pe（含旋转位置编码）两部分。
  - 使用前向保存的softmax_max和softmax_sum恢复softmax概率矩阵P，避免重新计算。
  - 对dK和dV使用scatter-add（index_add_）将梯度累加到正确位置。
  - 支持多batch动态序列长度，actual_seq_qlen/actual_seq_kvlen为前缀和格式。
  - 对每个(b, s)位置批量处理G个query heads。

## 计算公式

对于每个query token $t$和KV head $h$（处理G个query heads）：

$$
\begin{aligned}
\textbf{Step 1: Gather} \\
sel\_k &= \text{concat}(K_{nope}[\text{idx}], K_{pe}[\text{idx}]) & (K, D+D_R) \\
Q_{group} &= \text{concat}(Q_{nope}[t], Q_{pe}[t]) & (G, D+D_R) \\
\\
\textbf{Step 2: 恢复softmax概率} \\
S &= Q_{group} \times sel\_k^T \times \text{scale} & (G, K) \\
P &= \frac{\exp(S - m_i)}{l_i} & (G, K) \\
\\
\textbf{Step 3: 计算dP, dV} \\
dP &= dO \times V[\text{idx}]^T & (G, K) \\
dV_{local} &= P^T \times dO & (K, D) \\
\\
\textbf{Step 4: 计算dS, dQ, dK} \\
D_{val} &= \sum(dO \odot O, \dim=-1) & (G, 1) \\
dS &= P \odot (dP - D_{val}) & (G, K) \\
dQ_{local} &= dS \times sel\_k \times \text{scale} & (G, D+D_R) \\
dK_{local} &= dS^T \times Q_{group} \times \text{scale} & (K, D+D_R) \\
\\
\textbf{Step 5: Scatter-add写回} \\
dQ_{nope}[t] &\mathrel{+}= dQ_{local}[:, :D] \\
dQ_{pe}[t] &\mathrel{+}= dQ_{local}[:, D:] \\
dK_{nope}[\text{idx}] &\mathrel{+}= dK_{local}[:, :D] + dV_{local} \\
dK_{pe}[\text{idx}] &\mathrel{+}= dK_{local}[:, D:]
\end{aligned}
$$

### 符号说明

| 符号 | 含义 |
|------|------|
| $Q_{nope}$ | Query nope部分，shape为$(T_1, N_1, D)$ |
| $Q_{pe}$ | Query pe（RoPE）部分，shape为$(T_1, N_1, D_R)$ |
| $K_{nope}$ | Key nope部分，shape为$(T_2, N_2, D)$ |
| $K_{pe}$ | Key pe（RoPE）部分，shape为$(T_2, N_2, D_R)$ |
| $V$ | Value（等于K_nope），shape为$(T_2, N_2, D)$ |
| $dO$ | 输出梯度，shape为$(T_1, N_1, D)$ |
| $O$ | 前向输出，shape为$(T_1, N_1, D)$ |
| $m_i$ | 前向softmax最大值，shape为$(N_2, T_1, G)$ |
| $l_i$ | 前向softmax求和值，shape为$(N_2, T_1, G)$ |
| $\text{idx}$ | sparse_idx指定的稀疏KV索引 |
| $\text{scale}$ | 注意力缩放因子，$1/\sqrt{D+D_R}$ |
| $G$ | GQA分组数，$N_1/N_2$ |

## 函数原型

### JIT编译kernel

```python
@pypto.frontend.jit(...)
def sparse_flash_attention_grad(
    q_nope, q_pe, k_nope, k_pe, value, 
    sparse_idx, d_out, out, sm_max, sm_sum, 
    actual_seq_qlen, actual_seq_kvlen, 
    dq_nope_out, dq_pe_out, dk_nope_out, dk_pe_out, dv_out, 
    dk_nope_in, dk_pe_in, dk_out, 
    dk_2d, n_1, n_2, d, dr, k, group, scale_value
):
    """JIT编译的SFA反向kernel，TND格式，nope/rope拆分，动态shape。"""
```

### 封装接口

```python
def npu_sfa_sparse_attention_grad(
    q_nope, q_pe, k_nope, k_pe, value,
    sparse_idx, d_out, out, sm_max, sm_sum,
    actual_seq_qlen, actual_seq_kvlen, scale_value
) -> (dq_nope, dq_pe, dk_nope, dk_pe, dv)
```

## 参数说明

### Kernel输入参数

| 参数名 | 输入/输出 | 描述 | 数据类型 | 维度(shape) |
|:--- |:--- |:--- |:--- |:--- |
| q_nope | 输入 | Query nope部分 | BFLOAT16 | (T1, N1, D)，T1为动态维度 |
| q_pe | 输入 | Query pe（RoPE）部分 | BFLOAT16 | (T1, N1, DR) |
| k_nope | 输入 | Key nope部分 | BFLOAT16 | (T2, N2, D)，T2为动态维度 |
| k_pe | 输入 | Key pe（RoPE）部分 | BFLOAT16 | (T2, N2, DR) |
| value | 输入 | Value张量 | BFLOAT16 | (T2, N2, D) |
| sparse_idx | 输入 | 稀疏TopK索引 | INT32 | (T1, N2, K) |
| d_out | 输入 | 输出梯度 | BFLOAT16 | (T1, N1, D) |
| out | 输入 | 前向输出 | BFLOAT16 | (T1, N1, D) |
| sm_max | 输入 | 前向softmax最大值 | FLOAT32 | (N2, T1, G)，T1为动态维度 |
| sm_sum | 输入 | 前向softmax求和值 | FLOAT32 | (N2, T1, G) |
| actual_seq_qlen | 输入 | Query长度前缀和 | INT32 | (B,)，B为动态维度 |
| actual_seq_kvlen | 输入 | KV长度前缀和 | INT32 | (B,) |
| dq_nope_out | 输出 | Q nope梯度 | BFLOAT16 | (T1*N1, D) |
| dq_pe_out | 输出 | Q pe梯度 | BFLOAT16 | (T1*N1, DR) |
| dk_nope_out | 输出 | K nope梯度（含dV累加） | FLOAT32 | (T2*N2, D) |
| dk_pe_out | 输出 | K pe梯度 | FLOAT32 | (T2*N2, DR) |
| dv_out | 输出 | V梯度 | FLOAT32 | (T2*N2, D) |

### 标量/编译时参数

| 参数名 | 描述 | 数据类型 |
|:--- |:--- |:--- |
| N1 | Query head数 | INT |
| N2 | KV head数 | INT |
| D | nope维度 | INT |
| DR | rope维度 | INT |
| K | 稀疏TopK大小 | INT |
| G | GQA分组数（N1/N2） | INT |
| scale_value | 注意力缩放因子 | FLOAT |

## 约束说明

### 确定性计算

- aclnnSparseFlashAttentionGrad默认采用确定性实现，相同输入多次调用结果一致。

### 公共约束

1. 仅支持BFLOAT16数据类型，梯度计算中间结果以FP32进行累加。
2. 输入Tensor的数据格式仅支持ND。
3. actual_seq_qlen和actual_seq_kvlen均为前缀和格式。
4. sparse_idx中的索引值为T2维度上的全局索引。
5. dk_nope_out中已包含dV的累加（即dk_nope + dv）。
6. 输出梯度dk_nope_out和dk_pe_out为FLOAT32类型，外部需转换回BFLOAT16。
7. KV-side tensor使用MAX_TOTAL_KV（128K）作为静态上界。
8. 使用index_add_进行dK/dV的scatter-add操作。

### 输入shape约束

| 约束项 | 描述 |
|:--- |:--- |
| q_nope.dim() == 3, q_nope.size(2) == 512 | Q nope必须为3维，最后一维为512 |
| q_pe.dim() == 3, q_pe.size(2) == 64 | Q pe必须为3维，最后一维为64 |
| k_nope.dim() == 3, k_nope.size(2) == 512 | K nope必须为3维，最后一维为512 |
| k_pe.dim() == 3, k_pe.size(2) == 64 | K pe必须为3维，最后一维为64 |
| value.dim() == 3, value.size(2) == 512 | Value必须为3维，最后一维为512 |
| sm_max.dim() == 3, sm_max.size(0) == 1 | softmax_max第一维为N2（通常为1） |

### 规格约束

| 规格项 | 规格 | 规格说明 |
|:--- |:--- |:--- |
| D | 512 | nope维度（kv_lora_rank） |
| DR | 64 | rope维度（qk_rope_dim） |
| K | 1024、2048等 | 稀疏TopK大小 |
| N1 | 2、16等 | Query head数 |
| N2 | 1 | KV head数 |
| MAX_TOTAL_KV | 128*1024 | KV侧静态上界 |

## 调用示例

```python
import torch
from sparse_flash_attention_grad_impl import npu_sfa_sparse_attention_grad

# ========== 参数配置 ==========
actual_q_lens = [4]
actual_kv_lens = [32768]
N1, N2 = 2, 1
D, DR = 512, 64
K = 2048
G = N1 // N2
scale_value = 1.0 / (D + DR) ** 0.5

T1 = sum(actual_q_lens)
T2 = sum(actual_kv_lens)

# ========== 构造输入数据 ==========
q_nope = torch.randn(T1, N1, D, dtype=torch.bfloat16).npu()
q_pe = torch.randn(T1, N1, DR, dtype=torch.bfloat16).npu()
k_nope = torch.randn(T2, N2, D, dtype=torch.bfloat16).npu()
k_pe = torch.randn(T2, N2, DR, dtype=torch.bfloat16).npu()
value = k_nope.clone()
d_out = torch.randn(T1, N1, D, dtype=torch.bfloat16).npu()

# 前向输出和softmax中间结果（需从前向保存）
out = torch.randn(T1, N1, D, dtype=torch.bfloat16).npu()
sm_max = torch.randn(N2, T1, G, dtype=torch.float32).npu()
sm_sum = torch.randn(N2, T1, G, dtype=torch.float32).npu()

sparse_idx = torch.randint(0, T2, (T1, N2, K), dtype=torch.int32).npu()
actual_seq_qlen = torch.tensor(actual_q_lens, dtype=torch.int32).cumsum(0).to(torch.int32).npu()
actual_seq_kvlen = torch.tensor(actual_kv_lens, dtype=torch.int32).cumsum(0).to(torch.int32).npu()

# ========== 调用反向kernel ==========
dq_nope, dq_pe, dk_nope, dk_pe, dv = npu_sfa_sparse_attention_grad(
    q_nope, q_pe, k_nope, k_pe, value,
    sparse_idx, d_out, out, sm_max, sm_sum,
    actual_seq_qlen, actual_seq_kvlen, scale_value)

torch.npu.synchronize()
print(f"dQ_nope shape: {dq_nope.shape}")
print(f"dQ_pe shape: {dq_pe.shape}")
print(f"dK_nope shape: {dk_nope.shape}")
print(f"dK_pe shape: {dk_pe.shape}")
print(f"dV shape: {dv.shape}")
```
