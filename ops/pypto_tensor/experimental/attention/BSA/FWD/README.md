# BSA Forward 算子说明


## 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 算子语义

BSA Forward（`aclnnBlockSparseAttention`）实现块稀疏注意力的前向计算。通过块级稀疏掩码 `blockSparseMask` 标记有效块对，仅对掩码为 1 的 Q-KV 块对执行注意力计算，大幅降低长序列场景下的计算与内存开销。采用 Online Softmax 分块迭代策略，逐 KV 块维护运行最大值和指数和，保证分块计算与全量计算数学完全等价。

---

## 数学公式

### 核心计算

对每个 Q 块 $u$，沿 KV 方向逐块迭代（仅遍历掩码为 1 的 KV 块）：

$$S_{uv} = \frac{Q_u K_v^\top}{\sqrt{d}}, \quad M[u][v] = 1$$

### Online Softmax 迭代

$$m_j = \max(m_{j-1},\ \text{rowmax}(S_j / \sqrt{d}))$$

$$l_j = l_{j-1} \cdot \exp(m_{j-1} - m_j) + \text{rowsum}(\exp(S_j / \sqrt{d} - m_j))$$

$$\tilde{O}_j = \tilde{O}_{j-1} \cdot \exp(m_{j-1} - m_j) + \exp(S_j / \sqrt{d} - m_j) \cdot V_j$$

### 最终输出

$$O_u = \tilde{O}_j / l_j$$

$$\text{LSE}_u = m_j + \log(l_j)$$

其中：
- $Q_u$ / $K_v$ / $V_v$ 为按 `blockShapeX` / `blockShapeY` 划分的 Q/K/V 块
- $M \in \{0,1\}^{B \times H_q \times B_q \times B_k}$ 为块稀疏掩码
- $d = 128$（固定 head_dim）
- $\text{LSE}$ 为行级 Log-Sum-Exp 值

---

## 计算流程

### 步骤 1：输入预处理

1. 将 Q/K/V 的序列长度填充（pad）到块对齐大小
2. Reshape 4D `[B, H, S, D]` → 2D `[B*H*S_pad, D]`
3. 判断是否为 dense 掩码，选择路径

### 步骤 2：Sparse 路径 — 紧凑 KV 构建

1. 遍历每个 Q 块 `(b, h_q, u)`，收集掩码为 1 的 KV 块索引
2. 构建紧凑 K/V 张量：仅包含有效 KV 块的数据
3. 构建 valid_mask：有效位置为 1.0，填充位置为 0.0
4. 同步确保 wrapper 数据就绪

### 步骤 3：Kernel 计算（双路径）

**Sparse 路径**：
```
for outer in loop(B * Hq * numQB):        # 外层: 每个 Q 块
  初始化累积器: mi, li, oi (FP32)
  for v_idx in loop(maxSel):               # 内层: 仅有效 KV 块
    ① 加载 Q_u, K_v, V_v, valid_mask
    ② S = Q_u @ K_v^T * scale             # Cube matmul (FP32)
    ③ S_masked = S * mask + (1-mask) * neg # 算术掩码
    ④ Online softmax 更新 (m, l, o)
    ⑤ P_fp16 @ V → o_ij                   # Cube matmul (FP32)
  O = o_new / l_new, LSE = m_new + log(l_new)
  assemble → output_3d, lse_2d
```

**Dense 路径**：同上但无 mask 步骤，内层遍历 `numKB`。

### 步骤 4：输出后处理

1. 裁剪填充：`output_3d[:, :Sq, :]`
2. Reshape：`[B*Hq, Sq, D]` → `[B, Hq, Sq, D]`
3. 返回 `(attention_out, softmax_lse)`

---

## 输入输出规格

### 输入张量

| 名称 | 数据类型 | 形状 | 说明 |
|------|----------|------|------|
| query | FP16 | `[B, H_q, Sq, 128]` | Query 张量，BNSD 布局 |
| key | FP16 | `[B, H_kv, Skv, 128]` | Key 张量，BNSD 布局 |
| value | FP16 | `[B, H_kv, Skv, 128]` | Value 张量，BNSD 布局 |
| blockSparseMask | BOOL/UINT8 | `[B, H_q, ceil(Sq/X), ceil(Skv/Y)]` | 块稀疏掩码，1=有效块对 |
| actualSeqLengths | INT64 | `[B]` | 每 batch 的 Q 实际序列长度（可选） |
| actualSeqLengthsKv | INT64 | `[B]` | 每 batch 的 KV 实际序列长度（可选） |
| blockShape | INT64 | `[2]` | `[blockShapeX, blockShapeY]`（可选） |

### 输出张量

| 名称 | 数据类型 | 形状 | 说明 |
|------|----------|------|------|
| attentionOut | FP16 | `[B, H_q, Sq, 128]` | 注意力输出，与 Q 同 Shape |
| softmaxLse | FLOAT32 | `[B, H_q, Sq]` | 行级 Log-Sum-Exp 值 |

### 算子属性

| 名称 | 类型 | 约束 | 说明 |
|------|------|------|------|
| scaleValue | FLOAT | = 1/√128 ≈ 0.0884 | Softmax 缩放因子 |
| numKeyValueHeads | INT | = H_kv | KV 头数 |

---

## Shape 范围与约束

| 约束维度 | 约束值 |
|----------|--------|
| head_dim | 128（固定） |
| blockShapeX | 64 的整数倍（最小 64） |
| blockShapeY | ≥128 且为 64 的整数倍 |
| 数据类型 | Q/K/V/O = FP16，softmaxLse = FP32 |
| 数据布局 | BNSD |
| GQA 头数 | Hq ≥ Hkv，Hq % Hkv == 0 |
| 序列长度 | 支持非 blockShape 整数倍（自动填充） |
| 稀疏掩码 | `[B, H_q, ceil(Sq/X), ceil(Skv/Y)]` |

---

## 实现特点

### 性能优化

- **Dense/Sparse 双路径**：掩码全为 1 时自动切换到 dense kernel，省去 mask 开销
- **Mask2Idx 紧凑张量**：内层循环仅遍历有效 KV 块，跳过无效计算
- **L1 缓存复用**：Q 块在内层 KV 循环中复用 L1 缓存
- **Kernel 缓存**：工厂函数 + 字典缓存，避免重复编译

### 精度验证

- 前向输出 O 容差：`atol=0.0001, rtol=0.0078125`
- softmaxLse 容差：`atol=0.0001, rtol=0.0078125`
- FP32 累积路径保证数值稳定

### 测试用例

覆盖 10 个前向测试用例：不同序列长度（256/512/1024/2048）、GQA 分组（1/2/4）、稀疏率（30%/40%/50%/70%/100%）、Batch（1/2）、Dense 路径。
