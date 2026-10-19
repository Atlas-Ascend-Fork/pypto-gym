# MLA Prolog PyPTO 算子

Multi-Head Latent Attention (MLA) 前处理算子的 PyPTO 实现，支持动态序列长度。


## 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 功能说明

MLA Prolog 是 MLA (Multi-Head Latent Attention) 的前处理算子，用于推理场景。主要计算 Query 和 Key 的预处理，支持动态序列长度。

### 计算公式

#### RmsNorm 公式
```
RmsNorm(x) = gamma * (x_i / RMS(x))
RMS(x) = sqrt((1/N) * sum(x_i^2) + epsilon)
```

#### Query 计算路径
```
c^Q = RmsNorm(token_x @ W^DQ)                    # 下采样 + RmsNorm
q^C = c^Q @ W^UQ_QR[:, :N*D]                      # 上采样部分
q^R_raw = c^Q @ W^UQ_QR[:, N*D:]                  # ROPE 部分

# 对每个 head 单独上采样
q[i] = q^C[:, i*D:(i+1)*D] @ W^UK[i]             # i = 0,1,2,3
query = concat([q[0], q[1], q[2], q[3]])         # shape: [T, N, Hckv]

# 对每个 head 应用 ROPE
q^R[i] = ROPE(q^R_raw[:, i*Dr:(i+1)*Dr])        # i = 0,1,2,3
query_rope = concat([q^R[0], q^R[1], q^R[2], q^R[3]])  # shape: [T, N, Dr]
```

#### Key 计算路径
```
c^KV_kr = token_x @ W^DKV_KR                     # 下采样 + ROPE
c^KV = RmsNorm(c^KV_kr[:, :Hckv])                # KV latent
k^R_raw = c^KV_kr[:, Hckv:]                      # Key ROPE 部分
k^R = ROPE(k^R_raw)                              # shape: [T, Dr]
```

## 参数说明

### 输入张量

| 参数名 | 形状 | 数据类型 | 说明 |
|--------|------|----------|------|
| token_x | (T, He) | BF16 | 输入 token，T 为 token 数量，He 为隐藏维度 |
| weight_dq | (He, Hcq) | BF16 | Query 下采样权重 |
| weight_uq_qr | (Hcq, N*(D+Dr)) | BF16 | Query 上采样和 ROPE 权重 |
| weight_uk | (N, D, Hckv) | BF16 | Query 最终上采样权重 |
| weight_dkv_kr | (He, Hckv+Dr) | BF16 | Key 下采样和 ROPE 权重 |
| rmsnorm_gamma_cq | (Hcq,) | BF16 | c^Q 的 RmsNorm gamma 参数 |
| rmsnorm_gamma_ckv | (Hckv,) | BF16 | c^KV 的 RmsNorm gamma 参数 |
| rope_sin | (T, Dr) | BF16 | ROPE 正弦参数 |
| rope_cos | (T, Dr) | BF16 | ROPE 余弦参数 |

### 输出张量

| 参数名 | 形状 | 数据类型 | 说明 |
|--------|------|----------|------|
| query | (T, N, Hckv) | BF16 | Query 输出 (q^N) |
| query_rope | (T, N, Dr) | BF16 | Query ROPE 输出 (q^R) |
| c_kv_out | (T, Hckv) | BF16 | KV latent 输出 |
| k_r_out | (T, Dr) | BF16 | Key ROPE 输出 |

### 维度说明

- **T**: Token 数量（动态支持，支持任意正整数序列长度）
- **He**: Hidden dimension (输入隐藏层维度) = 256
- **Hcq**: Latent Query dimension = 64
- **Hckv**: Latent KV dimension = 32
- **N**: Number of heads (头数量) = 4
- **D**: Dimension per head (每个头的维度) = 16
- **Dr**: ROPE dimension per head (ROPE 维度) = 8

### Tiling 参数

- **TILE_T**: Token 维度的 tiling 大小 = 8
- 支持动态序列长度，自动处理非 TILE_T 整数倍的序列长度

## 编译运行

### 环境准备

```bash
# 设置 NPU 设备 ID
export TILE_FWK_DEVICE_ID=14
```

### 运行测试

```bash
python3 mla_prolog.py
```

或指定运行模式：

```bash
python3 mla_prolog.py --run_mode npu
```

## 测试结果

### 测试配置

```
固定维度参数: He=256, Hcq=64, Hckv=32, N=4, D=16, Dr=8
动态序列长度测试: T=[8, 16, 32]
```

### 测试输出

```
============================================================
Test: MLA Prolog (Dynamic Sequence Length)
============================================================

测试 seq_len=8 (dynamic)
✓ seq_len=8 所有精度对比通过

测试 seq_len=16 (dynamic)
✓ seq_len=16 所有精度对比通过

测试 seq_len=32 (dynamic)
✓ seq_len=32 所有精度对比通过
```

### 输出形状验证

| 序列长度 (T) | query | query_rope | c_kv_out | k_r_out |
|--------------|-------|------------|----------|---------|
| 8 | [8, 4, 32] | [8, 4, 8] | [8, 32] | [8, 8] |
| 16 | [16, 4, 32] | [16, 4, 8] | [16, 32] | [16, 8] |
| 32 | [32, 4, 32] | [32, 4, 8] | [32, 32] | [32, 8] |

### 精度对比

与 Golden 参考实现对比（BF16精度容差）：
- query: rtol=0.005, atol=0.0078125, threshold=0.005
- query_rope: rtol=0.005, atol=0.0078125, threshold=0.05
- c_kv_out: rtol=0.005, atol=0.0078125, threshold=0.005
- k_r_out: rtol=0.005, atol=0.0078125, threshold=0.05

## 实现要点

### 1. 动态序列长度支持
使用 PyPTO 的动态特性支持任意序列长度：
- 使用 `pypto.DYNAMIC` 声明动态维度
- 通过 `pypto.loop` 进行动态 tiling 循环
- 使用 `pypto.view` 的 `valid_shape` 参数处理边界情况
- 使用 `pypto.assemble` 将 tile 结果拼接成完整输出

### 2. Tiling 优化
采用 TILE_T=8 的 tiling 策略：
- 减少内存占用，提高缓存命中率
- 支持非 TILE_T 整数倍的序列长度（通过 valid_shape 处理）
- 配合 `pypto.set_codegen_options(support_dynamic_aligned=True)` 启用动态对齐支持

### 3. RmsNorm 实现
使用手动实现的 RmsNorm，避免精度损失：
- 计算 x^2 的均值
- 添加 epsilon=1e-5 避免除零
- 归一化并乘以 gamma

### 4. ROPE 实现
对输入张量的相邻维度对进行旋转位置编码：
```python
对于维度对 (2i, 2i+1):
out[2i] = in[2i] * cos[i] - in[2i+1] * sin[i]
out[2i+1] = in[2i+1] * cos[i] + in[2i] * sin[i]
```
- 使用 sin 和 cos 的前半部分（HALF_DR）进行旋转
- 对每个 head 的 ROPE 部分独立应用

### 5. 权重拆分
- `weight_uq_qr` 拆分为：
  - `mm_qc`: 前 N*D 维，用于 query 上采样
  - `mm_qr`: 后 N*Dr 维，用于 query ROPE
- `weight_dkv_kr` 拆分为：
  - `mm_ckv`: 前 Hckv 维，用于 KV latent
  - `mm_kr`: 后 Dr 维，用于 key ROPE

### 6. 多头处理
对每个 head 单独进行矩阵乘法，然后通过 concat 组合结果：
- Query: 4 个 head 独立计算 q[i] = mm_qc[:, i*D:(i+1)*D] @ weight_uk[i]
- Query ROPE: 4 个 head 独立应用 ROPE 后拼接
- 最终通过 reshape 调整为 [T, N, Hckv] 和 [T, N, Dr] 形状

## 已知限制

1. 当前版本仅支持非量化场景
2. 仅支持 ND 格式输入
3. 不包含 Cache 更新功能（需要单独实现）
4. BF16 精度损失在预期范围内（通过 compare 工具验证）
5. Tiling 大小固定为 TILE_T=8，可根据硬件特性调整

## 代码结构

```
mla_prolog/
├── mla_prolog.py              # PyPTO kernel 实现和测试
│   ├── mla_prolog_kernel()    # PyPTO 算子核心实现
│   ├── mla_prolog_golden()    # Golden 参考实现
│   ├── test_mla_prolog()      # 测试函数
│   └── main()                 # 主入口
└── README.md                  # 本文档
```

## 依赖说明

- PyPTO: 华为昇腾 AI 处理器自定义算子开发框架
- PyTorch: 用于 Golden 实现和数据生成
- torch_npu: NPU 设备支持
- utils.compare: 精度对比工具（位于 `deepseek_v32_exp/utils/compare`）
