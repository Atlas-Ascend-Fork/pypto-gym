# IndexerPrologueK 设计说明（DESIGN）

> 实现：`op_kernel/indexer_prologue_k.py`；测试/golden：`tests/`、`tests/assets/`。
> 运行平台：Ascend 950（arch 3510，soc 260）。

## 1. 概述

对 `latent[T, H]` 做投影，再依次做 RMSNorm、尾部 RoPE、block-32 MXFP4（E2M1 + UE8M0）量化，
并按 `cache_index` 原地写分页 `k_cache`：

```text
latent -- BF16 GEMM(FP32 累加) --> projected -- RMSNorm --> 尾部 RoPE
       --> block-32 MXFP4 打包 --> 按 cache_index 原地写 uint8 k_cache
```

中间张量 `projected[T, D]` 驻留 device，两阶段之间无 host 同步。

## 2. 计算阶段划分

### 阶段 1：投影（Cube）

- `projected[T, D] = BF16(GEMM_FP32_ACC(latent, wkᵀ))`，K 方向（H）分块累加，累加器为 FP32。
- **BF16 发布边界①**：L0C(FP32) → GM(projected, BF16)。

### 阶段 2：后处理（Vector / SIMT，块协作）

每命中一行 cache（`cache_index != -1`）把整行 `projected` 搬入 UB，块内协作完成：

1. **RMSNorm**：每线程一个元素，`warp_reduce_add` 累加 `x²`，每 warp 一次 `atomic_add` 到共享
   scratch，`t0` 求 `inverse_rms = rsqrt(Σx²/D + eps)` 广播。
2. **发布 RMSNorm**（BF16 边界②）：`norm[t] = bf16(fp32(proj[t]) · inverse_rms · γ[t])`。
3. **尾部 RoPE**（仅尾部 `Dr`，相邻偶奇；BF16 边界③）：
   `local = t-(D-Dr)`；偶 `r = norm[t]·cos - norm[t+1]·sin`，奇 `r = norm[t]·cos + norm[t-1]·sin`。
4. **MXFP4**（每 32 元素一组）：一个 warp 恰好覆盖一个 group；`warp_reduce_max` 求组内幅值 → UE8M0
   scale（exponent 位操作 + 倒数乘 `safe_amax·(1/6)`）；每线程按 E2M1 阈值链编码；偶 lane 用
   `warp_shfl_xor(code,1)` 与奇 lane 合成 1 字节（低 nibble 偶、高 nibble 奇）。
5. **Scatter**：按 cache ABI（见 README）直接写 `k_cache` / `k_scale_cache`。

## 3. 下发形态与流水（按 T 派发，`FUSED_T_MAX` 默认 2048）

### 3.1 两 kernel（`T > FUSED_T_MAX`）

`indexer_projection_kernel` 与 `indexer_postprocess_kernel` 顺序下发到**同一条 stream**，
`projected` 驻留 GM，两者之间无 host 同步（依赖 stream 顺序）。

### 3.2 单 kernel 融合（`T <= FUSED_T_MAX`）

`make_fused_kernel` 在同一 `@pl.jit` 内：

```text
with section_cube():    <投影：写 projected 到 GM>
pl.system.sync_all(core_type=SyncCoreType.MIX)   # AIC/AIV 全局屏障
with section_vector():  <后处理：读 projected 做 norm/rope/mxfp4/scatter>
```

- 混合核：每 block = 1 AIC + 2 AIV；`section_cube` 由 AIC 执行、`section_vector` 由 AIV 执行，
  两者在 `sync_all(MIX)` 汇合（cube 段末与 vector 段首各一次，配对）。
- 收益：省一次 kernel 下发的固定开销（小/中 T 约 1.3–2.1×）；大 T 时全局屏障无收益甚至略慢，
  故 `> FUSED_T_MAX` 回退两 kernel。

两条路径共用同一套后处理 SIMT 逻辑（`_make_postprocess_row` 生成），输出逐字节一致。

## 4. 分核策略

| 阶段 | 分工 | 核数 |
|---|---|---|
| 投影 | 把输出切成 `m_tiles=ceil(T/16)` × `n_tiles=ceil(D/base_n)` 个 tile，每核以 grid 步长处理若干 tile | `min(AIC, m_tiles·n_tiles)` |
| 后处理 | **块级 owner**：每个 vector 核按输入顺序扫描全部 token，只处理 `owner_key % num_blocks == block_idx` 的目标行 | `min(AIV, 工作项)`；mode0 工作项 = T，mode1 = `ceil(T/G)` |
| 融合 | 混合核（1 AIC + 2 AIV） | `min(AIC, AIV/2)` |

- owner 语义：每个目标行恰好一个 owner 核、核内按输入顺序处理，因此**重复 `cache_index` 天然
  「最后写入」**。
- 工作项不足满核时不空起核；owner 扫描在 kernel 内按运行期实际块数自适应。

## 5. Buffer 划分

### 5.1 投影（L1 → L0A/L0B → L0C）

| 缓冲 | 布局 | 地址 | 说明 |
|---|---|---|---|
| `a_l1`（L1 Mat） | `[BASE_M, base_k]` BF16 | `0x00000` | latent tile，双缓冲（`mutex [0,1]`）|
| `b_l1`（L1 ZN） | `[base_k, base_n]` BF16 | `0x10000` | wk 转置搬入（`order=[1,0]`，表达 `Bᵀ`），双缓冲 |
| `a_left`（L0A） | `[BASE_M, base_k]` BF16 | `0x00000` | compact=1 |
| `b_right`（L0B） | `[base_k, base_n]` BF16 | `0x00000` | compact=1 |
| `acc`（L0C） | `[BASE_M, base_n]` FP32 | `0x00000` | K 分块累加（首块 `matmul`、其余 `matmul_acc`）|

分块：`BASE_M=16`；`base_n = 128 if D%128==0 else 64 if D%64==0 else 16`；`base_k = 128 if H%128==0 else 16`。
尾块用 `pl.set_validshape` 限定有效窗口。

### 5.2 后处理（UB）

| 缓冲 | 形状 / dtype | 地址 | 说明 |
|---|---|---|---|
| `proj_g` | `[1, D]` BF16 | `0x0000` | 整行 projected |
| `norm_g` | `[1, D]` BF16 | `0x1000` | RMSNorm 发布 |
| `pub_g` | `[1, D]` BF16 | `0x2000` | RoPE 发布 / MXFP4 输入 |
| `gamma_g` | `[1, D]` FP32 | `0x3000` | RMSNorm γ（循环外载入一次）|
| `red_g` | `[8, 1]` FP32 | `0x4000` | 归约 scratch（`red[0,0]` 求和、`red[1,0]` inverse_rms）|

线程数 = `D`（每线程一个元素），一个 warp = 32 lane 恰好覆盖一个 MXFP4 group；要求 `D % 32 == 0` 且 `D <= 2048`。

数据流：`proj_ub → norm_ub → pub_ub → cache(GM)`；`gamma` 只载入一次，`rope_sin/cos` 与 `cache` 直接读/写 GM。

## 6. 接口

### 6.1 公开接口

```python
from ...indexer_prologue_k.op_kernel.indexer_prologue_k import indexer_prologue_k_wrapper

indexer_prologue_k_wrapper(
    latent, wk, norm_weight, rope_sin, rope_cos, k_cache,
    k_scale_cache=None, *, cache_index, storage_mode, norm_eps, combined_block_size=-1)
```

原地更新 `k_cache`（及可选 `k_scale_cache`）并返回同一个 `k_cache`；参数/dtype/shape 见 README「公开接口」。
`wk` 以 ND `[D, H]` 传入（转置在 GM→L1 搬入时完成），`k_cache`/`k_scale_cache` 需连续。

### 6.2 内部 kernel

- `indexer_projection_kernel(latent, wk, projected)`：`@pl.jit(tiling_key=IndexerProjectionKey)`，`T` 动态。
- `indexer_postprocess_kernel(projected, gamma, rope_sin, rope_cos, cache_index, cache, scale_cache, epsilon, row_count)`：`@pl.jit`，形状固化。
- `indexer_fused_kernel(latent, wk, gamma, rope_sin, rope_cos, cache_index, cache, scale_cache, projected, epsilon)`：`@pl.jit`，cube+vector 混合。

## 7. 编译期固化与缓存

| 路径 | key | 说明 |
|---|---|---|
| 投影 | `(NMode, KMode)` | 仅编码 L1 tile 形状类别；`(H,D)`、`T` 运行时读取，一份二进制服务该类别下任意形状 |
| 后处理 | `(T, D, Dr, storage_mode, G, write_scale, cache_blocks, cache_slots)` | SIMT 入口要求静态 shape，故按此固化 |
| 融合 | 后处理 key + `head_dim` | 固化维度与后处理一致 |

编译实例在 host 侧按 key 记忆化（`_FUSED_CACHE` / `_POSTPROCESS_CACHE`）；`projected` 由 wrapper
分配（`torch.empty`）。

## 8. 数值规则

三处 BF16 发布边界与参考实现一致（GEMM 输出、RMSNorm 输出、尾部 RoPE 输出）；MXFP4 使用参考实现的
exponent 位操作与 E2M1 阈值链。详见 README「数值规则」与 `tests/assets/indexer_prologue_k_golden.py`。
