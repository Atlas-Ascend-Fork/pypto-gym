# IndexerPrologueK（PyPTO-Pro）

Indexer Key 前处理算子：对 `latent[T, H]` 做投影，再依次做 RMSNorm、尾部 RoPE、block-32
MXFP4 量化，并按 `cache_index` 原地写入分页 `k_cache`。运行于 Ascend 950（arch 3510）。

- 实现（kernel）：`op_kernel/indexer_prologue_k.py`
- 测试与 golden：`tests/`（golden 在 `tests/assets/`）
- 设计说明：`DESIGN.md`；结果/精度/性能：`RESULT.md`

## 公开接口

```python
from ops.pypto_pro.deepseek_v4_1.indexer_prologue_k.op_kernel.indexer_prologue_k import (
    indexer_prologue_k_wrapper,
)

result = indexer_prologue_k_wrapper(
    latent, wk, norm_weight, rope_sin, rope_cos, k_cache,
    k_scale_cache=None,          # mode 0 可选
    cache_index=cache_index,     # keyword-only, INT64, -1 跳过
    storage_mode=0,              # 0: 数据/scale 分离; 1: combined
    norm_eps=1e-6,
    combined_block_size=-1,      # mode 1 为 G
)
```

原地更新 `k_cache`（及可选 `k_scale_cache`）并返回同一个 `k_cache`。

| 参数 | shape / dtype | 说明 |
|---|---|---|
| `latent` | `[T,H]` BF16 ND | 输入 hidden states |
| `wk` | `[D,H]` BF16 ND | 投影权重 |
| `norm_weight` | `[D]` FP32 | RMSNorm γ |
| `rope_sin` / `rope_cos` | `[T,Dr]` FP32 | 相邻偶奇 RoPE 系数 |
| `cache_index` | `[T]` INT64 | 目标逻辑行，`-1` 跳过 |
| `k_cache` | mode0 `[B,C,1,D/2]`；mode1 `[B,C,1,G(D/2+D/32)]` UINT8 | 原地输出 |
| `k_scale_cache` | mode0 可选 `[B,C,1,D/32]` UINT8 | 原地输出 |

## 计算流程

```text
latent -- BF16 GEMM(FP32 累加) --> projected -- RMSNorm --> 尾部 RoPE
       --> block-32 MXFP4(E2M1 + UE8M0) --> 按 cache_index 原地写 k_cache
```

中间张量 `projected[T,D]` 驻留 device，两者之间无 host 同步。

## 下发形态（按 T 派发，`FUSED_T_MAX` 默认 2048）

- `T <= FUSED_T_MAX`：**单 kernel 融合**（`make_fused_kernel`）——同一 `@pl.jit` 内 `section_cube()`
  投影 → `sync_all(MIX)` 全局屏障 → `section_vector()` 后处理；混合核每 block = 1 AIC + 2 AIV，
  block 数 = `min(AIC, AIV/2)`。省一次 kernel 下发的固定开销（小/中 T 约 1.3–2.1×）。
- `T > FUSED_T_MAX`：**两 kernel**——`indexer_projection_kernel` + `indexer_postprocess_kernel`。
  大 T 时融合的 MIX 屏障无收益甚至略慢，故保留两 kernel。

两条路径由同一套 `postprocess_row`（`_make_postprocess_row` 生成）与投影代码构成，输出逐字节一致
（`test_fused_matches_two_kernel_dispatch` 覆盖）。

## 实现结构

### Kernel 1 — 投影（Cube）

`projected[T,D] = BF16(GEMM_FP32_ACC(latent, wkᵀ))`。

- `T` 为 `pl.DYNAMIC` 轴：运行时读 `latent.shape[0]`，故同一模型形状 `(H,D)` 下**一份二进制服务任意 T**。
- 每核处理若干 `(m,n)` tile（`m_tiles=ceil(T/16)`、`n_tiles=ceil(D/base_n)`）；K 方向分块累加
  （`k_idx==0` 用 `matmul` 初始化 L0C，其余用 `matmul_acc`）。
- 权重 `wk`（ND `[D,H]`）在 GM→L1 时以 `order=[1,0]` **转置搬入** ZN tile，得到 `Bᵀ`。
- 数据通路：`GM →L1(a_l1/b_l1, 双缓冲) →L0A/L0B →L0C(FP32) →GM(projected, BF16)`。
- 分块：`base_n = 128 if D%128==0 else 64 if D%64==0 else 16`；
  `base_k = 128 if H%128==0 else 16`；`BASE_M = 16`。

### Kernel 2 — 后处理（Vector / SIMT，块协作）

**块级 owner 调度**：每个 vector 核按输入顺序扫描全部 token，只处理
`owner_key % num_blocks == block_idx` 的目标行；每个目标行恰好一个 owner 核，核内串行，
因此重复 `cache_index` 天然「最后写入」。

每命中一行：把整行 `projected` 搬入 UB，块内协作完成：

1. **RMSNorm**：每线程一个元素，`warp_reduce_add` 累加 `x²`，每 warp 一次 `atomic_add`
   到共享 scratch，`t0` 求 `inverse_rms = rsqrt(Σx²/D + eps)` 广播。
2. **发布**：`norm[t] = bf16(proj[t] · inverse_rms · γ[t])`（BF16 发布边界）。
3. **RoPE（仅尾部 Dr）**：`local = t-(D-Dr)`；偶 `r = norm[t]·cos - norm[t+1]·sin`，
   奇 `r = norm[t]·cos + norm[t-1]·sin`；前缀不变（BF16 发布边界）。
4. **MXFP4**：线程数取 `D`，**一个 warp 恰好覆盖一个 32 元素 group**；
   `warp_reduce_max` 求 group 幅值 → UE8M0 scale；每线程按阈值链编码 E2M1；
   偶 lane 用 `warp_shfl_xor(code,1)` 与奇 lane 合成 1 字节（低 nibble 偶、高 nibble 奇）。
5. **Scatter**：按 ABI 直接写 `k_cache` / `k_scale_cache` 的对应字节。

数据流：`proj_ub → norm_ub → pub_ub → cache(GM)`；`gamma` 循环外只载入 UB 一次；
`rope_sin/cos` 与 `cache` 直接读/写 GM。

## 数值规则

三处 BF16 发布边界：GEMM 输出、RMSNorm 输出、尾部 RoPE 输出。

RMSNorm：

```text
inverse_rms = rsqrt(mean(x²) + eps)
norm = bf16(fp32(x) · inverse_rms · γ)
```

RoPE（尾部 `Dr`，相邻偶奇；`local = 尾部内下标`）：

```text
out[local even] = norm[l]·cos[l] - norm[l+1]·sin[l]
out[local odd ] = norm[l]·cos[l] + norm[l-1]·sin[l]
```

MXFP4（每 32 元素 group）：

```text
amax  = max(|x|)
safe  = max(amax, 6·2^-126)
raw   = safe · (1/6)                 # 倒数乘（非除法）
bits  = uint32(raw); exp = (bits>>23)&0xFF
exp  += (mantissa(bits) != 0)        # >= raw 的下一个 2 的幂
scale = float32(exp<<23); scale_byte = uint8(exp)
n = |x| / scale; code = 7
if n<=5.0:6; <=3.5:5; <=2.5:4; <=1.75:3; <=1.25:2; <=0.75:1; <=0.25:0
if x < 0: code += 8
packed_byte = code[even] | (code[odd] << 4)
```

## Cache ABI（`V=D/2`, `S=D/32`）

```text
mode 0:  block = cr // C ; slot = cr % C
         data : k_cache[block, slot, 0, j]          (j = 0..V-1)
         scale: k_scale_cache[block, slot, 0, g]    (g = 0..S-1, 若传入)

mode 1:  raw = C*G ; block = cr // raw ; slot = cr % raw
         cg = slot // G ; sub = slot % G
         data : k_cache[block, cg, 0, sub*V + j]
         scale: k_cache[block, cg, 0, G*V + sub*S + g]
```

`cache_index[t] == -1` 跳过；重复目标以最后一个 token 为准；未触及字节保持不变。

## 约束

- `T > 0`；`H > 0` 且 `H % 16 == 0`；`D >= 32` 且 `D % 32 == 0`；`0 < Dr <= D` 且 `Dr % 2 == 0`。
- `storage_mode ∈ {0,1}`；mode 1 时 `combined_block_size = G > 0` 且 `C % G == 0`。
- 后处理按 `(T, D, Dr, storage_mode, G, write_scale, cache_blocks, cache_slots)` 编译期专门化
  （SIMT 入口要求静态 shape）；线程数 = `D`，要求 `D` 为 32 的倍数且 `D <= 2048`。
- `k_cache` / `k_scale_cache` 需连续；`latent` / `wk` 需连续。

## 性能

环境（固定）：Ascend950PR，cube/vector = 32/64；`H=7168, D=512, Dr=510, block_size=128`；
`value_bytes = D/2 = 256`，`scale_bytes = D/32 = 16`；`norm_eps=1e-6`。计时用 `torch.npu.Event`
取中位，派发阈值 `FUSED_T_MAX=2048`。逐例规格与耗时：

| case | T | storage_mode | G | `k_cache` 形状 | `k_scale_cache` | 派发 | 起核 | Pro (ms) |
|---|---|---|---|---|---|---|---|---|
| decode | 16 | 0 | -1 | `(1, 128, 1, 256)` | `(1, 128, 1, 16)` | fused | 32 混合（32 AIC + 64 AIV）| 0.781 |
| wide | 512 | 0 | -1 | `(4, 128, 1, 256)` | `(4, 128, 1, 16)` | fused | 32 混合 | 0.675 |
| prefill | 4096 | 0 | -1 | `(32, 128, 1, 256)` | `(32, 128, 1, 16)` | two | cube 32 / vec 64 | 1.637 |
| combined | 4096 | 1 | 8 | `(32, 16, 1, 2176)` | 无（scale 内嵌）| two | cube 32 / vec 64 | 1.467 |

其中 cache 第 0 维 `B = ceil(T/128)`；mode1 的 `cache_slots = 128/G`、`cache_width = G·(256+16)`。
cube 起核 `min(AIC, ceil(T/16)·ceil(D/128))`，vec 起核 `min(AIV, 工作项)`（mode0 工作项 = T，
mode1 = `ceil(T/G)`），融合路径 32 混合核。绝对值随机器负载波动。

融合相对两 kernel 的收益：**`T ≤ 2048` 约 1.3–2.1×**（小/中 T 由一次 kernel 下发的固定开销主导，
融合省掉一次）；**`T ≥ 4096` 融合的 MIX 屏障无收益甚至略慢**，故 `> FUSED_T_MAX` 走两 kernel。

## 测试

```bash
python3 -m pytest ops/pypto_pro/deepseek_v4_1/indexer_prologue_k/tests/test_indexer_prologue_k.py -q
```

无需 NPU 的 CPU 用例（golden / MXFP4 / cache ABI）始终可跑；NPU 端到端用例在无设备时跳过。
