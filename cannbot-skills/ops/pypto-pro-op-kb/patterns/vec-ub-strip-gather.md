# UB strip gather — index-driven movement on a vector-only target

**Topology:** `gather-scatter-indexing`
**Status:** conceptual addressing skeleton; the historical 12/12 runs establish
dtype compatibility only because their exact `C`/`L` geometry and runnable
artifact were not retained.
**Evidence:** [retained validation record](../examples/validation-records.md), scoped to
the recorded target and dtype-compatibility result.

## When this applies

An index tensor drives which element of a source tensor each output element
takes, along one axis, with the other axes carried through — `torch.gather`,
`GatherElements`, `take_along_dim`, and the read half of `scatter`/`index_put`.
It does **not** apply to `tf.gather`-style whole-row selection along an axis;
that is a segment copy, not this.

## The constraint that determines everything

**`vf.gather` reads UB only.** There is no GM random-access primitive, and
pypto-pro has no SIMT, so an approach that indexes GM per element does not
translate here at all. Every element a tile needs must already be in UB.

## The shape

Flatten any rank about the indexed axis `k` into three axes:

    x   -> [Px, X, W]     X = x.shape[k], W = x's inner extent
    idx -> [P,  G, V]     G = index.shape[k], V = index's inner extent
    y[p, g, v] = x[premap(p), idx[p, g, v], wmap(v)]

For one outer position and one inner column block, the elements any output can
reach are the **strip** `x[px, :, w0:w0+C]` — the entire indexed axis, `C`
columns wide. Hold it in UB as an `[Xs, C]` tile and the source for output
`(g, c)` is at UB element offset

    idx * C + c

which `vf.gather` takes directly. The following sketch covers only the hoisted
path whose premises are stated below:

```python
# HOISTING THIS BITMASK IS ONLY LEGAL WHEN C is a power of two and divides L.
lanec = vf.and_(vf.arange(0, dtype=pl.DT_INT32), cmask, mi)   # hoisted: c term
...
idx = vf.load_align(t_idx, base)
off = vf.muls(idx, C, mi)
off = vf.add(off, lanec, mi)
y   = vf.gather(t_src, off, md)
vf.store_align(t_out + base, y, md)
```

The `c` term looks free, and it is — **but only under a load-bearing premise**.
Step `r` covers `L` consecutive positions of the row-major `[GT, C]` output
tile starting at position `r*L`, so the column a lane writes is

    c = (r*L + lane) mod C

The general expression is therefore `arange(r*L, r*L+L) % C`. For a reusable
skeleton in which `r = 1` may execute, it becomes the hoistable, `r`-independent
`arange(0, L) % C` exactly when `L % C == 0`. Only when `C` is also a power of
two may that modulo be spelled with `& (C-1)`.

**This premise is load-bearing, and violating it fails silently.** At
`L = 32, C = 64`, `r*32 mod 64` alternates `0, 32, 0, 32…`, so every odd step
gathers its whole register from source columns **32 positions off**. The output
is still written to its normal contiguous
positions and remains plausibly shaped; only the selected source column is
wrong. Nothing in the DSL, the tile geometry or a capacity assertion catches
it, because the incorrect gather offsets still land inside the source tile.

**All paths require `C > 0`; before hoisting the shown bitmask, also assert
`L % C == 0 and (C & (C-1)) == 0`.** If `L % C != 0`, recompute `lanec` per
step as `(r*L + lane) % C`; otherwise `lane % C` may be hoisted. In either case,
use `& (C-1)` only for a power-of-two C. Other C values require a separately
validated modulo implementation or rejection. The second tile family below
sets `C = Irun` and follows the same rules.

## Sizing it — the measured knee

The strip load is a strided GM read of `Xs` rows, `C` elements each, at the
source's inner stride. Measured on 9579 with the moved volume held fixed so row
length is the only variable:

| row bytes | 32 | 64 | **128** | 256 | 512 | >=1024 |
|---|---|---|---|---|---|---|
| % of contiguous bandwidth | 31 | 44 | **88** | 96 | 98 | 100 |

**Target `C * esize >= 128 bytes.** Since `Xs * C * esize <= UB_budget`, that
caps the strip at about `budget / 128` rows — roughly 1024 for 128 KiB. An
indexed axis longer than that takes **passes**, not a narrower strip: below
128 B the cost roughly doubles per halving, and two passes at 128 B beat one at
32 B by a wide margin.

The kernel is DMA-bound at these sizes, not gather-bound, which is what makes
extra passes and half-empty gather lanes affordable.

## Four things that fail silently

1. **Do not use the fixed-width strip when `C > Irun`, where `Irun` is the
   flattened inner extent and `esize` is bytes per element.** A full-block count
   `ncb = floor(Irun / C)` is zero when `0 < Irun < C`; ceiling division would be
   one but would require a separately designed partial-block tail. Every 1-D
   case is `Irun = 1`. Use a second tile family for `Irun < 32/esize`: the
   `(indexed, inner)` block is contiguous when the inner extent is that small,
   so the strip becomes a flat `[1, Xs*Irun]` run — the best possible DMA — and
   the same offset expression works with `C = Irun`.
2. **The b16 index window is narrower than UB.** A b16 gather dst takes a
   **UINT16** index, and the offset must be produced by `vf.pack`, which
   truncates without saturating or erroring. `Xs * C > 65536` returns plausible
   wrong data. Make it a tiling invariant, not a runtime check.
3. **An out-of-range gather offset is undefined, not masked.** When splitting
   the indexed axis into passes, clamp `idx - pass_lo` into `[0, Xs-1]`
   *before* the gather and mask the *store*. Masking after the gather is reading
   whatever that address held.
4. **Pass loop placement decides the traffic.** Passes must sit **outside** the
   output-row-chunk loop, or the strip is re-read once per row chunk and source
   traffic multiplies. Since `pl.store` has no per-element mask, the output tile
   is read back from GM for passes after the first — `(npass-1)` extra reads and
   writes of the *output*, which is far cheaper than re-reading the *source*.

## Dtypes — every spelling is forced, none is a preference

`vf.gather` and `vf.load_align` derive a register's C++ type from the **tile's**
dtype and hand it to an intrinsic that wants another. Four spellings do not
build at all; the diagnostics and the routes around them are in
[`references/pypto-pro-framework-findings.md`](../references/pypto-pro-framework-findings.md).
The short form:

| payload | declare as | narrowing |
|---|---|---|
| fp32 | `DT_FP32` | — |
| int32 | **`DT_UINT32`** | — (the signed spelling will not build) |
| fp16 / bf16 | `DT_FP16` / `DT_BF16` | offsets in INT32, `vf.pack(dtype=DT_UINT16)` |
| int8 | **`DT_UINT8`**, widened to `DT_UINT16` in UB by `pl.cast` | `vf.pack(dtype=DT_UINT8)` — must truncate, never saturate |
| int64 | **`DT_UINT32`, inner extent doubled** | index duplicated by `vf.interleave(idx, idx)` |
| index int64 / int8 | `pl.cast` to an INT32 tile in UB | — |

Two of those are correctness, not convenience. The int8 narrowing must truncate:
the b16 gather result carries the byte zero-extended, so `-1` arrives as
`0x00FF` and a saturating cast returns `127`. And int64 needs no 64-bit gather —
viewed as UINT32 with the inner extent doubled, the source word for output word
`p` is `idx[p/2]*2C + (p mod 2C)`, the same expression with each index value
duplicated.

## Coverage the cases you can see will not exercise

The map from index coordinates to source coordinates is the identity only under
two conditions, and the draws you can inspect routinely satisfy both while the
declared range does not:

* **prefix identity** — extents agree on every axis before `k` except the
  outermost. When it fails, decode the mixed radix **on device** from a small
  block table (`px = sum_j ((o / t_inner_j) mod size_j) * x_stride_j`), once per
  outer position. A host-precomputed table is `O(P)` and `P` is unbounded.
* **inner identity** — extents agree on every axis after `k+1`. When it fails,
  the differing trailing axes fold into the outer loop and the contiguous run
  that remains becomes the inner extent. The three-axis view absorbs this with
  no new kernel code; only the host tiling changes.

## Correctness signal

**Bitwise equality, for every dtype including floats.** This topology moves
data; it computes nothing. Any nonzero error is an indexing bug, and a
tolerance-based gate cannot distinguish a correct permutation of NaNs from a
wrong one — benchmark suites for these operators reliably include an all-NaN
case and an inf case precisely because they are cheap to generate. Compare
floats through `.view()` on an integer of the same width so NaN payloads and
signed zeros are compared as bits, and keep the official gate as the formal
second verdict rather than the first.

## FP32 strip candidates / FP32 条带候选与取舍

本新增节区分上文历史vector-only实现前提与目标版本能力：它只讨论UB Tile→Reg
条带路线，不把旧实现的“无SIMT”前提扩展到所有版本。目标SDK若提供SIMT直接GM寻址，
应在相同算子合同下作为另一候选比较；转置候选见[向量约束](../constraints/vec.md)。

本节是候选设计规则，不将整张卡或任意条带几何升级为已验证骨架；
[历史验证记录](../examples/validation-records.md)仍只支持上文明确的dtype兼容范围。
128B是值得比较的行宽，不是无条件最优或所有平台的精确阈值；源轴pass数、索引和输出成本
都要计入。整行`index_select`不是本卡的GatherElements合同，不能互相替代验证。

输入`x[N,128]`、逐元素独立的`idx[64,128]`，输出`y[g,w]=x[idx[g,w],w]`。
以FP32、源条带`[chunk,C]`为例，先从目标UB预算扣除索引、输出和临时量；若为两块
`[64,C]`索引/输出及4096B临时区，则容量条件为`(chunk*C+2*64*C)*4+4096<=UB_budget`。
C=8/32/64对应32B/128B/256B行宽；加宽可能增加源轴pass，也可能在小源轴下不增加pass。
因此比较相邻候选，而不是硬编码C=32最优。该容量模型不是通用的UB大小或性能阈值。

源轴分pass时，gather前clamp局部索引，再mask更新；后续pass的输出回读写回必须计入。
完整调用还包括源读取、索引加载和输出合并，不能只套用DMA带宽表。若要给出某形状的
加速比，须另有覆盖该实现和合同的可复核测量；本节不引用未随仓库保留的测量倍数。
