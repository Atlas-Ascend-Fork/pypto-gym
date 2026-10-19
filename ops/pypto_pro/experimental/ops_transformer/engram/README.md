# engram

## 概述

`engram` 是一个面向 transformer 模型的高性能 **Key-Value 记忆门控算子**，基于 `pypto_pro` DSL 在 Ascend NPU 上实现。该算子从 token embedding 出发计算 key/value 投影，通过 RMS 归一化的点积打分导出内容感知的门控信号，并将门控作用于 value 投影，为下游层产出门控输出。

算子采用 **Cube-Vector (CV) 并行执行**架构 —— Cube 子核执行矩阵乘法（key/value 投影），Vector 子核并发执行 RMS 归一化、打分、门控计算和广播乘法，两个子核间通过乒乓事件同步。

## 接口签名

```python
value_out, score_back, key_back, value_back, gate_back = engram_forward_wrapper(
    hidden_states,       # [B, S, M, H]  bf16
    embeddings,          # [B, S, De]    bf16
    key_proj_weights,    # [M, De, H]    bf16
    value_proj_weights,  # [De, H]       bf16
    key_gamma,           # [M, H]        bf16
    query_gamma,         # [M, H]        bf16
    clamp_value=1e-6,    # 可选: sign-sqrt gate 的 |score| 下限
    eps=1e-6,            # 可选: RMSNorm epsilon
)
```

### 输入

| 名称                  | 形状            | 类型   | 说明                                   |
|-----------------------|-----------------|--------|----------------------------------------|
| `hidden_states`       | `[B, S, M, H]`  | BF16   | Query 激活（每个头一个）               |
| `embeddings`          | `[B, S, De]`    | BF16   | Token 嵌入，作为 key/value 投影的输入  |
| `key_proj_weights`    | `[M, De, H]`    | BF16   | 按头的 key 投影权重矩阵                |
| `value_proj_weights`  | `[De, H]`       | BF16   | 共享的 value 投影权重矩阵              |
| `key_gamma`           | `[M, H]`        | BF16   | 按头的可学习 key 缩放系数 (γ_k)        |
| `query_gamma`         | `[M, H]`        | BF16   | 按头的可学习 query 缩放系数 (γ_q)      |

**维度说明**：`B` = 批次大小，`S` = 序列长度，`M` = 头数，`De` = 嵌入维度，`H` = 隐藏维度（1280 / 2560 / 2048 / 1536，TilingKey 特化）。

### 输出

| 名称          | 形状           | 类型   | 说明                                  |
|---------------|----------------|--------|---------------------------------------|
| `value_out`   | `[B, S, M, H]` | BF16   | 门控 value 输出 (gate ⊙ value_proj)   |
| `score_back`  | `[B, S, M]`    | FP32   | 按头 score，用于反向传播              |
| `key_back`    | `[B, S, M, H]` | BF16   | Key 投影，用于反向传播                |
| `value_back`  | `[B, S, H]`    | BF16   | Value 投影，用于反向传播              |
| `gate_back`   | `[B, S, M]`    | FP32   | 门控值，用于反向传播                  |

## 算法

对每个头 `h` 和每个 token（第 `m` 行），依次执行：

1. **Key/Value 投影**（Cube 子核）：
   - `key_proj = emb · key_proj_weights[h]` — 按头 key 投影
   - `value_proj = emb · value_proj_weights` — 共享 value 投影

2. **RMS 归一化**（Vector 子核）：
   - `rms_k = 1 / sqrt(mean(key_proj²) + ε)`
   - `rms_q = 1 / sqrt(mean(query²) + ε)`

3. **Score 计算与门控**（Vector 子核）：
   - `score = sum(γ_k ⊙ rms_k ⊙ key_proj ⊙ γ_q ⊙ rms_q ⊙ query) / sqrt(H)`
   - `gate = sigmoid(sign(score) · sqrt(max(|score|, clamp_value)))`

   门控函数采用"有符号平方根 + sigmoid"的非线性形式，输出值域为 (0, 1)，可根据内容自适应地放行或抑制 value 信息。

4. **门控输出**（Vector 子核）：
   - `value_out = gate · value_proj`

## 架构设计

### CV 并行执行

每个 block 内的 2 个子核分工如下：

| 子核            | Section | 职责                                              |
|-----------------|---------|---------------------------------------------------|
| 子核 0 (AIC)    | Cube    | MatMul：emb × weight → key_back、value_back，写入 GM |
| 子核 1 (AIV)    | Vector  | 加载 key_back/value_back/hidden，四阶段 VF 流水线 → value_out |

### 跨核同步（乒乓协议）

Cube 和 Vector 之间按头同步，采用深度为 2 的乒乓机制防止死锁：

```
Cube（每个 head h）:                        Vector（每个 head h）:
  计算 key_back[h]                            wait K_READY[h%2]
  set K_READY[h%2]  ───────────────────→      加载并处理 head h
  wait ACK[h%2]     ←───────────────────      set ACK[h%2]
```

- `K_READY{0,1}`（cube→vec）：通知 key_back 已写入 GM
- `ACK{0,1}`（vec→cube）：背压信号，确保 Vector 消费完毕前 Cube 不会覆写同一槽位的事件 ID

### AIV Split

Vector section 内部将奇偶行 tile 分配到 2 个子核（`sub_idx`）进一步并行处理。

### H 维全量 tile（不分片）

每个 HMode 使用**整 H 单 tile**（`n_h_chunks = 1`）：H=1280/2560/2048/1536 分别绑定 `[rows, H]` 的编译期 tile 形状（rows = 8/4/4/8），RMS 累加和 score 点积在单 tile 内完成，无需跨 chunk 累加。UB 布局按 HMode 特化地址（HMode 0/1 字节同构共享，2/3 独立），峰值占用见下文 Tiling 常量。

### Cube 段 L1 复用

- `emb` 宽 tile `[TILE_M, 1280]` 常驻 L1，每 M-tile GM→L1 一次，后续 value + 每头 key 投影只做 `move(offset=)` 取 K 子块；
- 右矩阵（W_k/W_v）L1 三缓冲 `[256,128]`：一次 load 覆盖 2 个 K 子块（128 行），段内 `move(offset=)` 拆分，GM 读量减半并隐藏 MTE2 延迟。

## Tiling 常量

| 符号          | 值           | 说明                             |
|---------------|--------------|----------------------------------|
| `TILE_M`      | 128 / 64     | Cube M 方向 tile（CMode 选择：`B·S > 2048` → 128，否则 64） |
| `TILE_K`      | 128          | Cube K 方向 tile（内积维度）          |
| `TILE_N`      | 128          | Cube N 方向 tile（输出列数）          |
| `TILE_M_VEC`  | 8 / 4        | Vector M 方向 tile（HMode 选择：H=1280/1536 → 8，H=2560/2048 → 4） |
| `H_CHUNK`     | 1280/2560/2048/1536 | 整 H 单 tile（HMode 选择，`n_h_chunks=1`） |
| `KW`          | 256          | 右矩阵 L1 宽 tile 行数（2 个 K 子块/次 load） |
| `clamp_value` | 1e-6（默认） | 开方前 score 绝对值下限（运行时标量参数） |
| `eps`         | 1e-6（默认） | RMS 归一化的 epsilon（运行时标量参数）    |

UB 总占用：按 HMode 特化布局，高水位约 246 KB（H=1536，< 256 KB 上限）。

## 支持的配置

- 隐藏维度 `H`：1280 / 2560 / 2048 / 1536（TilingKey HMode 特化，wrapper 校验）
- 头数 `M`：任意（受显存限制）
- 嵌入维度 `De`：任意（支持至 1280，受 emb L1 宽 tile 约束）
- 多核：最多 32 核，自适应 `num_cores = min(32, ceil(B·S / TILE_M))`；TilingKey `CMode` 按 `B·S` 选 TILE_M=128/64

## 文件结构

```
engram/
├── engram_forward_impl.py    # 算子 kernel（pypto_pro DSL）+ host 封装
└── README.md    # 本文件
```

## 使用方法

```python
from ops.pypto_pro.experimental.ops_transformer.engram import engram_forward_wrapper

# 所有输入需在 NPU 设备上，除标注外均为 BF16；返回值顺序见「接口签名」
value_out, score_back, key_back, value_back, gate_back = engram_forward_wrapper(
    hidden_states, embeddings,
    key_proj_weights, value_proj_weights,
    key_gamma, query_gamma,
)
```

首次调用时通过 `@pl.jit` 触发 JIT 编译。

## 精度说明

- **内部计算**：全程 FP32（Vector 运算、累加器、MatMul 累加器）
- **I/O**：输入和 `value_out` 使用 BF16；`score_back` / `gate_back` 输出 FP32（保留梯度精度）；`key_back` / `value_back` 内部走 FP32 GM 工作区，wrapper 出口降为 BF16（与反向 BF16 入参、tensor 版 I/O 一致）
- 门控计算中的 sigmoid 和 sqrt 使用 FP32 以保证数值稳定性

---

# engram_backward

## 概述

`engram_backward` 是 `engram`（正向 Key-Value 记忆门控算子）的**反向传播算子**，基于 `pypto_pro` DSL 在 Ascend NPU 上实现。它接收上游对门控输出 `value_out` 的梯度 `grad_output`，以及正向缓存的中间量（`scores` / `gates` / `keys` / `value`），沿正向 7 步计算链**逆向求导**，输出 6 个梯度：对 `hidden_states`、`embeddings`、`key_proj_weights`、`value_proj_weights`、`key_gamma`、`query_gamma` 的梯度。

算子延续正向的 **Cube-Vector (CV) 并行执行**架构，但数据流方向相反：**Vector 段先跑**（产出 `grad_value_ws` / `grad_key_ws` 等立方操作数工作区与 `grad_hidden` / `grad_γ`），通过分波 / 分头两套乒乓事件（`GV_WAVE` / `GK`）将工作区交接给 **Cube 段**（消耗工作区，产出 `grad_emb` / `grad_W_v` / `grad_W_k`）。Vector 段利用 32 核 × 2 AIV = 64 个 worker，Cube 段利用 32 个 AIC。

## 接口签名

```python
(grad_hidden_states, grad_embeddings, grad_key_proj_weights,
 grad_value_proj_weights, grad_key_gamma, grad_query_gamma) = engram_backward_wrapper(
    grad_output,         # [B, S, M, H]  bf16   上游梯度 ∂L/∂value_out
    hidden_states,       # [B, S, M, H]  bf16   正向输入（query 激活）
    embeddings,          # [B, S, De]    bf16   正向输入
    key_proj_weights,    # [M, De, H]    bf16   正向输入（host 内部预转置）
    value_proj_weights,  # [De, H]       bf16   正向输入（host 内部预转置）
    key_gamma,           # [M, H]        bf16   正向输入
    query_gamma,         # [M, H]        bf16   正向输入
    scores,              # [B, S, M]     fp32   正向缓存 (Step5 输出)
    gates,               # [B, S, M]     fp32   正向缓存 (Step6 输出)
    keys,                # [B, S, M, H]  bf16   正向缓存 (Step2 输出)
    value,               # [B, S, H]     bf16   正向缓存 (Step1 输出)
    clamp_value=1e-6,
    eps=1e-6,
)
```

### 输入

| 名称                  | 形状            | 类型   | 说明                                       |
|-----------------------|-----------------|--------|--------------------------------------------|
| `grad_output`         | `[B, S, M, H]`  | BF16   | 上游梯度 ∂L/∂value_out                     |
| `hidden_states`       | `[B, S, M, H]`  | BF16   | 正向 Query 激活                            |
| `embeddings`          | `[B, S, De]`    | BF16   | 正向 Token 嵌入                            |
| `key_proj_weights`    | `[M, De, H]`    | BF16   | 正向按头 key 投影权重（host 预转置为 wk_t）|
| `value_proj_weights`  | `[De, H]`       | BF16   | 正向共享 value 投影权重（host 预转置为 wv_t）|
| `key_gamma`           | `[M, H]`        | BF16   | 正向 key 缩放系数 γ_k                      |
| `query_gamma`         | `[M, H]`        | BF16   | 正向 query 缩放系数 γ_q                    |
| `scores`              | `[B, S, M]`    | FP32   | 正向缓存：Step5 score（**必须 FP32**）     |
| `gates`               | `[B, S, M]`    | FP32   | 正向缓存：Step6 gate（**必须 FP32**）      |
| `keys`                | `[B, S, M, H]`  | BF16   | 正向缓存：Step2 key 投影                   |
| `value`               | `[B, S, H]`     | BF16   | 正向缓存：Step1 value 投影                 |

> Step 编号按 kernel 7-step 链（Step1 value 投影、Step2 key 投影、Step5 score、Step6 gate）。

**约束**：`scores` 和 `gates` 必须为 `torch.float32`（wrapper 会校验并报错）；其余激活/权重输入为 BF16。`H` 仅支持 1280 / 2560 / 2048 / 1536（TilingKey 特化）。**注意**：inv_rms 不再由正向传入（正向无 rms cache 输出），反向在 kernel 内从 `keys` / `hidden_states` FP32 重算 `1/rms`。

### 输出

| 名称                     | 形状            | 类型   | 说明                                       |
|--------------------------|-----------------|--------|--------------------------------------------|
| `grad_hidden_states`     | `[B, S, M, H]`  | BF16   | ∂L/∂hidden_states（Step3 逆向，对 query） |
| `grad_embeddings`        | `[B, S, De]`    | BF16   | ∂L/∂embeddings（Step6+Step1 逆向累加）    |
| `grad_key_proj_weights`  | `[M, De, H]`    | BF16   | ∂L/∂key_proj_weights（Step1 逆向）        |
| `grad_value_proj_weights`| `[De, H]`       | BF16   | ∂L/∂value_proj_weights（Step6 逆向）      |
| `grad_key_gamma`         | `[M, H]`        | BF16   | ∂L/∂key_gamma（Step2 逆向）               |
| `grad_query_gamma`       | `[M, H]`        | BF16   | ∂L/∂query_gamma（Step3 逆向）             |

**维度说明**：与正向一致 —— `B` 批次、`S` 序列、`M` 头数（动态 1–16）、`De` 嵌入维度、`H` 隐藏维度（1280 / 2560 / 2048 / 1536）。内部 `B·S` 合并为 `BS` 轴。

## 算法

沿正向 7 步链**逆向**求导（Step7 → Step1），逐头 `m ∈ [0, M)`：

1. **Step7 逆向**（Vector，门控广播乘法 `O = g · V`）：
   - `grad_gates[m] = Σ_h(grad_output · value)` → `[B,S,M,1]`
   - `grad_value    = Σ_m(grad_output · gates)` → `[B,S,H]`

2. **Step6 逆向**（Cube，`V = E @ W_v`）：
   - `grad_emb_from_v, grad_value_proj_weights = linear_bw(grad_value, E, W_v)`

3. **Step5 逆向**（Vector，signed_sqrt_gate）：
   - `grad_score = grad_gate · g(1−g) · mask / (2·√max(|s|,c) + 1e-12)`，`mask = (|s|>c)`

4. **Step4 逆向**（Vector，缩放点积）：
   - `grad_normed_key   = grad_score · (1/√H) · normed_query`
   - `grad_normed_query = grad_score · (1/√H) · normed_key`（对称）

5. **Step3 / Step2 逆向**（Vector，RMSNorm：`inv_rms` kernel 内重算 + rmean/gradx）：
   - `(grad_hidden_m, grad_γ_q) = rms_norm_bw(grad_normed_query, hidden, γ_q)`
   - `(grad_key_m,    grad_γ_k) = rms_norm_bw(grad_normed_key,   key,    γ_k)`

6. **Step1 逆向**（Cube，`K = E @ W_k[m]`）：
   - `grad_emb_from_k, grad_W_k[m] = linear_bw(grad_key_m, E, W_k[m])`

7. **跨头累加**：`grad_embeddings = grad_emb_from_v + Σ_m grad_emb_from_k`

其中 `linear_bw` 对应 `grad_x = grad_output @ W^T`、`grad_W = x^T @ grad_output`；`rms_norm_bw` 按 `rms = √(mean(x²)+ε)`、`n = x/rms` 重算后求 `grad_x = (grad_n − n·mean(grad_n·n))/rms`、`grad_γ = Σ grad_x̂·n`。

## 架构设计

### 单 kernel 双 section（Vector 先 → Cube 后）

与正向「Cube 产 key/value、Vector 消费」相反，反向是 **Vector 先产工作区、Cube 后消费**：

| Section | 子核        | 职责                                                                 |
|---------|-------------|----------------------------------------------------------------------|
| Vector  | AIV (×64 worker) | 产 `grad_value_ws` / `grad_key_ws` / `grad_hidden` / `grad_γ`        |
| Cube    | AIC (×32)   | 消耗工作区，产 `grad_emb` / `grad_W_v` / `grad_W_k`                  |

### Vector 段：Pass A（分波）+ fused Phase B

Vector 段切分为两相，各自产出 Cube 所需的工作区：

| 相            | 产出                                   | 事件         | Cube 消费者（可提前启动）|
|---------------|----------------------------------------|--------------|--------------------------|
| Pass A（分波） | `grad_value_ws`（按波逐段就绪）        | `GV_WAVE{0,1}` | Nest1-val（按波启动）+ Nest2 |
| Phase B（fused，head-major） | 逐 head 产出 `grad_key_ws[m]` + `grad_hidden[m]` + `grad_γ` | `GK{2,3}` per-head | Nest1-key[m] + Nest3[m]（逐 head 启动） |

- **Pass A 按波（wave）生产**：wave-major 行映射 —— 一个 wave = 一个 cube Nest1 完整 tile 轮次（`num_cores×128 = 4096` 行），64 个 worker 各持波内连续 64 行；`bs_start` 内层与旧版相同（`[TILE_BS_VEC_A, H_CHUNK]` 大 tile）。波内仍一次循环完成 Step7a（grad_value）+ Step7b（grad_gate）+ Step5（grad_score）——不拆 7a/7b（历史上拆分需重搬 `go`，实测变差已回退）。
- **Phase B 为 KEY+QUERY 融合单循环**：每 `(tile, m_head)` 内先重算双方 `inv_rms`（`vf_compute_inv_rms`，FP32），再做 rmean / gradx 两段 RMS 反向，KEY 与 QUERY 链共享一次 `grad_score` load 与 keys/hidden 加载（独立的 B-query pass 已删除，`GK_DONE` 在融合 pass 全部完成后才 fire）。
- **grad_score 缓存**：Pass A 算完 `gs` 后存入 `gscore_ws`，Phase B 直接 load，省掉重算 Step7b+Step5。Phase B 读其它 worker 产的 `gscore_ws` 行，由**最后一波的 AIV-only barrier** 保证（wave-major 下末波 barrier == 全部 Pass A 落盘）。

### 跨核同步（V → Cube：GV 分波乒乓 + GK 全局）

`GV` 侧为**奇偶双 event + ACK 背压的乒乓协议**（与正向 `K_READY/ACK` 同构；二值事件两次 set 之间无 wait 会合并，故槽位复用前必须先收 ACK）：

```
Vector（每波 w）:                          Cube（Nest1 每轮 t，wave(t)==t）:
  ...本波 share 行 7a+7b+5 + 存储...          wait GV_WAVE[t%2]   (MTE2)
  sync_all(AIV_ONLY)   ← 全 worker 波栅栏     set  GV_ACK[t%2]    (FIX, 立即回)
  if w>=2: wait GV_ACK[w%2]  ← 背压           ...本轮 128 行 tile 的 loads/mma...
  set  GV_WAVE[w%2]   (MTE3)
```

- **波粒度**：`wave_span = num_cores×128` —— 信号只在"够全部 32 个 AIC 各拿一个新 tile"时发出（用户要求），cube 不必为 2 个 tile 的零头提早启动。
- **wave(t) == t 恒等式**：`iters_per_core == n_waves`，Nest1 第 t 轮的 tile 行恰好属于第 t 波 → cube 无需 cursor，每轮一次 wait+ACK 即可；空尾 tile 的核也统一执行 wait/ACK（序列全核一致，且背压只需 ACK 到 `n_waves-2`）。Nest2 在 Nest1 之后跑，全部波已消费，**无需任何等待**。
- **早 ACK 安全性**：`grad_value_ws` 行是 write-once，cube 在 wait 通过后立即回 ACK（不必等本轮 matmul 做完），vector 最早可在 wave t+2 复用槽位。
- **GK per-head 乒乓（head-major Phase B）**：Phase B 外层为 `for m_head`，每个 head 完成后 `sync_all(AIV_ONLY) + set GK{m%2}`（`EVENT_GK_BASE=2`）；cube 逐 head `wait GK{m%2} → set GK_ACK{4,5}` 后才跑该 head 的 Nest1-key+Nest3。同样需要 ACK 背压——vector 每 head 的 Phase B 远快于 cube 的 per-head Nest，无背压时奇偶槽两次 set 合并、cube 末次 wait 必挂死。`grad_key_ws`/`grad_hidden` 按 head 维 write-once，早 ACK 安全。
- 旧的全局 `GV_DONE`/`GK_DONE`（各带一对 `sync_all(MIX)`）已被上述两套 AIV-only 乒乓取代，kernel 内不再有 MIX 栅栏（两侧对称删除）。

### Cube 段：3 个 Nest（K-reduction cover-write / atomic-add）

Cube 段用 BF16 `[128,128,128]` tile，L0C FP32 累加器。**L1 四缓冲 / L0A-L0B 双缓冲 / acc 四缓冲**（acc 多缓冲使跨 N-tile 的 L0C store 与下一段 matmul 重叠），让 K 链的 GM→L1 (MTE2) + L1→L0 (MTE1) 与上一片 matmul (M) 三流水重叠。三个 Nest 的数学对应：

| Nest      | 数学操作                                          | 输出                    | 写法                |
|-----------|---------------------------------------------------|-------------------------|---------------------|
| Nest1-val | `grad_value_ws @ wv_t`                            | `grad_emb_acc` (FP32)   | atomic-add（同槽）  |
| Nest2     | `emb_t @ grad_value_ws`                           | `grad_value_proj_weights`| cover-write (K=BS) |
| Nest1-key | `Σ_m grad_key_ws[m] @ wk_t[m]`                    | `grad_emb_acc` (FP32)   | atomic-add（同槽）  |
| Nest3     | `emb_t @ grad_key_ws[m]`                          | `grad_key_proj_weights` | cover-write (K=BS) |

- **Scheme A（grad_emb）**：value 贡献（Nest1-val）+ 每头 key 贡献（Nest1-key）都 atomic-add 进**同一个** `grad_emb_acc[BS, De]` FP32 槽（M-strided 划分保证每地址只被一个核写，atomic-add 退化为本地 FP32 累加，无跨核竞争）。host 端一次性 `grad_emb_acc → BF16` 得到 `grad_embeddings`。替代了旧的 `[M_H+1, BS, De]`（~71MB）cover-write workspace + Phase3 vector 求和。
- **Nest2 / Nest3 分解（无 K 轴跨核拆分）**：按输出 tile `(de, h)` 分给 32 核轮转（`flat = core_id + i·num_cores`，共 `n_de×n_h` 个单元），每个单元 **K=BS 全链在单核内完成**（逐 `[128,128]` k-tile 从 GM 加载 `emb_t` / `grad_*_ws` 进 L1 双缓冲）。每个输出 tile 仅一个写者 → cover-write 直接落 BF16，结果确定、无 atomic。
- **grad_γ RMW**：`grad_γ_q` / `grad_γ_k` 按 per-subblock FP32 GM workspace（`[num_cores·2, M, H]`）做 tile RMW，host 端 `sum(dim=0)` 归约后 cast。

### TilingKey 双字段自适应（4×4 = 16 个编译变体）

- **HMode**（2 bit）：`0/1/2/3 → H=1280/2560/2048/1536`，绑定 Vector sub-tile 行数与整 H 单 tile 的 UB 布局（HMode 0/1 字节同构 `[8,1280]≡[4,2560]` 共享 (A) 组地址；2/3 破坏常数积，独立特化）。
- **BSMode**（2 bit）：按 `BS` 阈值选 `V_TILE`（`0/1/2/3 → 128/64/32/16`，阈值 `BS ≥ 8192/4096/2048/<2048`）。大 BS 档位恰好 `ceil(BS/V_TILE) = 64` 填满全部 worker；`BS < 2048` 一律用最小 `V_TILE=16` 最大化占用（worker 数不足 64 由 `bsi < n_bs_v` 守卫跳过空槽，结果正确但 vector 利用率下降）。

`HMode`/`BSMode` 经 parser ConstInt fast path 只编译 taken 分支，使 tile 形状成为编译期常量。`[rows,64]` / `[1,H_CHUNK]` tile 的 UB 地址按 HMode 用 parser-foldable 算术选择特化（行数翻倍的 key 需 0x800 stride 避免 overlap）。

## Tiling 常量

| 符号              | 值         | 说明                                          |
|-------------------|------------|-----------------------------------------------|
| `CUBE_MN`         | 128        | Cube M/N 输出 tile（BF16 `[128,128]`=32KB）   |
| `CUBE_K`          | 128        | Cube K tile（BF16；L1 四缓冲、L0A/L0B 双缓冲、acc 四缓冲） |
| `V_TILE_BS0..3`   | 128/64/32/16| Vector worker BS-row tile（BSMode 选择）     |
| `TILE_BS_VEC`     | 4          | Vector sub-row tile（所有 HMode 统一为 4 行） |
| `TILE_BS_VEC_A`   | 8 (H=1280) / 4 (其余) | Pass A 专用大 tile              |
| `H_CHUNK`         | 1280/2560/2048/1536 | Vector 整 H 单 tile（HMode 选择）   |
| `clamp_value`/`eps` | 1e-6（默认） | 运行时标量参数，与正向同名参数一致（见正向 Tiling 常量表） |
| `GATE_EPS`        | 1e-12      | signed_sqrt_gate 分母防零（对齐 golden）       |

UB 总占用：按 HMode 特化，高水位约 243 / 252 / 203 / 154 KB（H=1280/2560/2048/1536），均 < 256 KB 上限。Cube L1 占用 256 KB（`a_l1` / `b_l1` 各 4×32KB 双缓冲组，基址 0x00000 / 0x20000，L1 总量 512 KB 留有余量）；L0C FP32 acc 四槽共 256 KB（独立 L0C 空间）。

## 支持的配置

- 隐藏维度 `H`：**仅** 1280 / 2560 / 2048 / 1536（TilingKey 特化，wrapper 校验）
- 头数 `M`：动态 1–16
- 嵌入维度 `De`：任意
- `BS = B·S`：任意；小 BS（< 2048）时 Vector 利用率下降（worker 难以填满）
- 多核：固定 32 核（32 AIC × 2 AIV = 64 vector worker），靠 `bsi < n_bs` / `flat < total_*` 边界守卫跳过空槽

## 文件结构

```
src/.../ops/pypto_pro/experimental/ops_transformer/engram/
├── engram_forward_impl.py     # 正向算子 kernel + host 封装
├── engram_backward_impl.py    # 反向算子 kernel（pypto_pro DSL）+ host 封装
└── README.md                  # 本文件

# golden 参考实现（torch，CPU FP64 真值）位于：
tests/ops/experimental/ops_transformer/engram/engram_golden.py
# PyPTO-Pro 精度测试：
tests/ops/pypto_pro/experimental/ops_transformer/engram/test_engram_{forward,backward}_pypto_pro.py
```

## 使用方法

```python
from ops.pypto_pro.experimental.ops_transformer.engram import (
    engram_forward_wrapper,
    engram_backward_wrapper,
)

# 1) 正向：除 value_out 外的 4 个返回值全部作为反向缓存（见正向「输出」表）
value_out, score_back, key_back, value_back, gate_back = engram_forward_wrapper(
    hidden_states, embeddings,
    key_proj_weights, value_proj_weights,
    key_gamma, query_gamma,
)

# 2) 反向：传入 grad_output 与正向缓存（score/gate 出来即是 FP32，直接传）
grad_hs, grad_emb, grad_kpw, grad_vpw, grad_kg, grad_qg = engram_backward_wrapper(
    grad_output,
    hidden_states, embeddings,
    key_proj_weights, value_proj_weights,
    key_gamma, query_gamma,
    score_back, gate_back,           # scores / gates（FP32）
    key_back, value_back,            # keys / value（BF16）
    clamp_value=1e-6,                # 可选: 与正向一致
    eps=1e-6,                        # 可选: 与正向一致
)
```

首次调用时通过 `@pl.jit` 触发 JIT 编译。

## 精度说明

- **内部计算精度**：Vector 段运算（inv_rms 重算 + rmean/gradx RMS 反向、grad_γ RMW、grad_score/grad_gate）全程 FP32；Cube 段为 BF16 操作数 × FP32 L0C 累加器（对齐 A5 BF16 matmul 4× 算力）。
- **关键中间量存储**：`grad_value_ws` / `grad_key_ws` 由 Vector 段以 FP32 算出，**存入 GM 时降为 BF16**（作为 Cube matmul 操作数）。这是性能取舍（避免 FP32 cube 的 4× 算力损失），但在极端抵消点会带来精度边界（见下）。
- **I/O**：输入激活/权重/缓存为 BF16（`scores`/`gates` 强制 FP32）；6 个梯度输出为 BF16（由内部 FP32 结果在写出/host 端一次性 cast 得到）。
- **精度参考**：`engram_golden.py` 为前向/反向 CPU FP64 数学真值；板上跑 `tests/ops/pypto_pro/experimental/ops_transformer/engram/test_engram_{forward,backward}_pypto_pro.py` 做 kernel / benchmark / FP64 golden 三方精度对比。
- `1/√H` 在 FP32 vector 寄存器内构造（避免 BF16 标量路径丢精度）；`H` 为精确整数，`mean = sq / H` 用整数除法避免倒数误差。
