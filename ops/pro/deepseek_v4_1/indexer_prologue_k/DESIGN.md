# IndexerPrologueK 设计说明（ACLNN 交付）

> 实现：`op_kernel/indexer_prologue_k.py`、`indexer_prologue_k_projection.py`、
> `indexer_prologue_k_postprocess.py`；JIT 参考：`indexer_prologue_k_jit.py`。
> 运行平台：Ascend 950（arch 3510）。

## 1. 概述

对 `latent[T, H]` 做投影，再依次做 RMSNorm、尾部 RoPE、block-32 MXFP4（E2M1 + UE8M0）量化，
并按 `cache_index` 原地写分页 `k_cache`：

```text
latent -- BF16 GEMM(FP32 累加) --> projected -- RMSNorm --> 尾部 RoPE
       --> block-32 MXFP4 打包 --> 按 cache_index 原地写 uint8 k_cache
```

ACLNN 交付拆成三个 OpDef，共用一套 host plan：

| OpDef | 计算域 | 用途 |
|---|---|---|
| `IndexerPrologueK` | Cube + Vector（MIX） | `T <= 2048` 融合 |
| `IndexerPrologueKProjection` | Cube | 两阶段投影 |
| `IndexerPrologueKPostprocess` | Vector / SIMT | 两阶段后处理 |

## 2. 阶段划分

### 2.1 投影（Cube）

`projected[T, D] = BF16(GEMM_FP32_ACC(latent, wkᵀ))`，K 方向（H）分块累加，累加器为 FP32。
BF16 发布边界①：L0C(FP32) → GM(projected, BF16)。

### 2.2 后处理（Vector / SIMT，块协作）

每命中一行（`cache_index != -1`）把整行搬入 UB，块内协作完成：

1. RMSNorm：每线程一个元素，`warp_reduce_add` 累加 `x²`，每 warp 一次 `atomic_add` 到
   `red`，`t0` 求 `inverse_rms = rsqrt(Σx²/D + eps)` 广播；
2. 发布 RMSNorm（BF16 边界②）；
3. 尾部 RoPE（仅尾部 `Dr`，相邻偶奇；BF16 边界③）；
4. MXFP4：每 32 元素一组，一个 warp 覆盖一个 group；`warp_reduce_max` 求组内 amax，
   生成 UE8M0 scale，按 E2M1 阈值链编码，偶 lane 用 `warp_shfl_xor(code,1)` 合成字节；
5. Scatter：按 cache ABI 直接写 `k_cache` / `k_scale_cache`。

## 3. 下发形态

### 3.1 融合（`T <= 2048`）

`IndexerPrologueK` 在一个 `@pl.jit` 内：

```text
with section_cube():    <投影：写 projected 到 ACLNN workspace>
pl.system.sync_all(core_type=SyncCoreType.MIX)
with section_vector():  <后处理：读 projected 做 norm/rope/mxfp4/scatter>
```

- 混合核：每 block = 1 AIC + 2 AIV；`blockDim = min(AIC, AIV/2)`；
- `projected` 使用 ACLNN workspace，大小 `T*D*2`；
- 省一次 kernel 下发的固定开销。

### 3.2 两阶段（`T > 2048`）

测试桥按 `T` 路由：

1. `aclnnIndexerPrologueKProjection`：`latent`、`wk` → `projected`；
2. `aclnnIndexerPrologueKPostprocess`：读 `projected` → `k_cache` / `k_scale_cache`。

`projected[T, D]` 由 bridge 在 device 上分配，两阶段顺序下发到同一条 stream，无 host 同步。

## 4. 共享 plan 与 tiling key

`op_host/indexer_prologue_k_plan.h` 统一维护：

- `BASE_M=16`、`BASE_N_WIDE=128`、`BASE_N=64`、`BASE_K=128`、`NZ_ALIGN=16`；
- `FP4_GROUP_SIZE=32`、`MAX_DIM=2048`、`T_MAX=262144`；
- `NextPow2`、`DimCapIndex`、`DynUBufSize`；
- `ProjectionAxes`、`NModeOf`、`KModeOf`。

| 入口 | tiling key 字段 |
|---|---|
| Fused | `NMode(2b) | KMode(1b) | DimCap(2b)` |
| Projection | `NMode(2b) | KMode(1b)` |
| Postprocess | `DimCap(2b)` |

`DimCap` 取值 `[512, 1024, 2048]`；host 取 `dimCap = max(512, NextPow2(D))`，按档位选择
kernel 变体和 UB 布局。

## 5. 分核策略

| 阶段 | 分工 | 核数 |
|---|---|---|
| 投影 | `m_tiles=ceil(T/16)` × `n_tiles=ceil(D/base_n)` 个 tile，每核以 grid 步长处理 | `min(AIC, m_tiles*n_tiles)` |
| 后处理 | 块级 owner：每核扫描全部 token，只处理 `owner_key % num_blocks == block_idx` | `min(AIV, 工作项)`；mode0=T，mode1=`ceil(T/G)` |
| 融合 | 混合核（1 AIC + 2 AIV） | `min(AIC, AIV/2)` |

owner 语义保证每个目标行只有一个 owner，重复 `cache_index` 天然“最后写入”。

## 6. 后处理 UB 布局（按 DimCap）

后处理使用 9 个 UB tile：

| tile | dtype | 形状 | 512 地址 | 1024 地址 | 2048 地址 |
|---|---|---|---|---|---|
| `proj_g` | BF16 | `[1, DimCap]` | `0x00000` | `0x00000` | `0x00000` |
| `norm_g` | BF16 | `[1, DimCap]` | `0x01000` | `0x00800` | `0x01000` |
| `pub_g` | BF16 | `[1, DimCap]` | `0x02000` | `0x01000` | `0x02000` |
| `gamma_g` | FP32 | `[1, DimCap]` | `0x03000` | `0x01800` | `0x03000` |
| `red_g` | FP32 | `[8, 1]` | `0x04000` | `0x02800` | `0x05000` |
| `cos_g` | FP32 | `[1, DimCap]` | `0x05000` | `0x03000` | `0x05080` |
| `sin_g` | FP32 | `[1, DimCap]` | `0x06000` | `0x04000` | `0x07080` |
| `packed_g` | UINT8 | `[1, DimCap/2]` | `0x07000` | `0x05000` | `0x09080` |
| `scale_g` | UINT8 | `[1, DimCap/32 按 32B 对齐]` | `0x07100` | `0x05200` | `0x09480` |

| DimCap | UB 高水位 | `DYN_UB_SIZE` |
|---|---|---|
| 512 | `0x7120` | 20768 |
| 1024 | `0x5220` | 12832 |
| 2048 | `0x94C0` | 29888 |

512 保持原交付布局；1024/2048 使用独立、无重叠布局。`scale_g` 按 32B 对齐，2048 时为 64B。

SIMT 入口按档位使用 `max_threads=512/1024/2048`，kernel 内按 `t < dim` 裁掉多余线程。

## 7. 投影 L1/L0 布局

| 缓冲 | 布局 | 地址 | 说明 |
|---|---|---|---|
| `a_l1` | `[BASE_M, base_k]` BF16 | `0x00000` | latent tile，双缓冲 |
| `b_l1` | `[base_k, base_n]` BF16 ZN | `0x10000` | wk 转置搬入 |
| `a_left` | `[BASE_M, base_k]` BF16 | `0x00000` | L0A，compact |
| `b_right` | `[base_k, base_n]` BF16 | `0x00000` | L0B，compact |
| `acc` | `[BASE_M, base_n]` FP32 | `0x00000` | L0C 累加 |

`base_n = 128 if D%128==0 else 64 if D%64==0 else 16`；`base_k = 128 if H%128==0 else 16`。

## 8. Workspace ABI

ACLNN 生成的 kernel 入口会把**倒数第二个参数**当作 workspace。Projection/Postprocess 的真实
输出参数不能放在这个位置，因此两个 kernel 在 `tiling` 之前显式增加 `workspace_ptr`：

```text
IndexerPrologueKProjection(..., projected, workspace_ptr, tiling)
IndexerPrologueKPostprocess(..., scale_out, workspace_ptr, tiling)
```

host tiling 对应设置 `workspace[0] = 32`。这样 `projected` / `scale_out` 不会被
`usrWorkspace` 替换。

Fused 入口的 `projected` 本身就是 workspace，大小 `T*D*2`，因此不需要额外 dummy。

## 9. 数值规则

三处 BF16 发布边界与参考实现一致（GEMM 输出、RMSNorm 输出、尾部 RoPE 输出）；MXFP4 使用
参考实现的 exponent 位操作与 E2M1 阈值链。参考实现位于
`tests/assets/indexer_prologue_k_golden.py`。
