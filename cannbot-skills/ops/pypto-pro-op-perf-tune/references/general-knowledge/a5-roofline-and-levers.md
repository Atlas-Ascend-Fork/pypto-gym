# A5 roofline workflow

Load this reference only after detection confirms an A5 target: raw output
`3510` (the `NpuArch` value), helper output `dav-3510`, runtime output `DAV_3510`, or build
target `dav-c310`. It must not supply constants or tuning rules to an unknown or
non-A5 target.

## Evidence gate

Before calculating a roofline:

1. Record the detected device model and architecture.
2. Record the CANN and PyPTO versions and the resolved `pypto_pro` package path.
3. Read constants from the platform files installed with that exact runtime.
4. Retain the profiler output used for every measured claim.

If any input is unavailable, leave the numerical estimate unverified and use
the profiler result as the only tuning basis.

## Cross-Skill profiling tools

For raw collection, cross-run speedup and bottleneck triage this page pairs with the
`ops-profiling` Skill ([SKILL.md](../../../ops-profiling/SKILL.md)) and its
`msopprof-visualization` sub-skill
([SKILL.md](../../../ops-profiling/msopprof-visualization/SKILL.md)) for interactive HTML reports.
Those tools supply measurements only; the derivation and evidence gate above still decide the roofline.

> Note: `ops-profiling` is not vendored in this repository; it is fetched from the
> upstream `cann/cannbot-skills` repo at install time. The sibling links above resolve
> only in the installed layout or an upstream checkout. If they dangle, locate the
> skill via `$CANNBOT_CONFIG_ROOT/skills/ops-profiling` (or `$VIS_SKILL_ROOT` for the
> `msopprof-visualization` sub-skill).

## Primary source paths

Resolve these under the installed, version-recorded PyPTO/CANN tree; do not copy
values from another checkout:

- `framework/src/platform/parser/simulation_platform/platform_config/950DT_957x.ini`
- the matching `950DT_958x.ini`, `950PR_957x.ini`, or `950PR_958x.ini` for the
  detected SKU
- `framework/src/platform/parser/platforminfo.ini`
- `python/pypto_pro/runtime/compile_config.py`
- `python/pypto_pro/runtime/platform.py`

The files are primary evidence for the matching installed version only. If a
path or key differs, search the installed source and record the replacement;
do not infer a value.

## Derive, do not transcribe

Read the cube, HBM, UB, L1 and L0 figures out of the platform ini that ships
with the runtime you are measuring, so no constant lives in this page and
detaches from the installed version:

```
<pypto root>/framework/src/platform/parser/simulation_platform/platform_config/<SKU>.ini
```

Resolve `<SKU>` from the detected device rather than assuming it, and record
which ini you read alongside the number. The derivation is deliberately stated
as arithmetic over the ini rather than delegated to a tool: it is the part that
has to stay reproducible from this repository alone.

The derivations it applies, so they can be checked rather than trusted:

```
cube FLOP/s = M*N*K (from [DtypeMKN]) * 2 * cube_core_cnt * cube_freq
HBM  B/s    = cube_core_cnt   * [AICoreMemoryRates]ddr_rate    * cube_freq
            + vector_core_cnt * [VectorCoreMemoryRates]ddr_rate * vec_freq
estimate    = max(bytes / HBM, MACs*2 / cube FLOP/s)
```

**Pick the SKU deliberately.** All four A5 inis share `NpuArch=3510`,
`cube_freq=1650`, and the same L2/UB/L1/L0 sizes, but they differ in core count
*and* by more than a factor of two in `ddr_rate`. A roofline computed against the
wrong one is wrong by that factor. `950DT` and `950PR` at the same core count are
not interchangeable, and `platform.py` reports only `DAV_3510` plus core counts,
which narrows to two SKUs rather than one. Record which ini was used.

## Calibration measurements

Measured ratios, not platform constants, recorded with provenance because the
evidence gate above requires it.

**A byte-count model is a floor, not a prediction.** It prices the traffic and
nothing else, so it will sit below any figure that also prices vector work in
the epilogue, weight streaming rather than peak bandwidth, or a cube bound that
binds before memory does. Establish *which* resource binds before optimising
against the model, and recompute per case: the same kernel changes which
resource binds at different shapes, and the spread within one operator can be
as wide as the spread between operators.

**Price causal work explicitly, or the model is wrong by up to 2x.** A model
that counts the full attention rectangle overstates a causally masked kernel.
For mask `j <= i + (S_kv - S)` the kept fraction is
`(S*(S_kv-S+1) + S*(S-1)/2) / (S*S_kv)` — about 0.5 when `S == S_kv`, about 1.0
when `S << S_kv`. So skipping masked work is required on square cases and buys
nothing on short-query ones, and a model that ignores the mask can be off by
roughly the reciprocal of that fraction.

**Achievable fraction of peak.** Measured on A5: an aligned-plane MTE2 read
reaches ~1.7 TB/s, and a tuned vector-only kernel on a memory-bound normalise +
quantise shape (8192x8192 fp16, 401 us) reaches 1.17 TB/s and is described by
its authors as sitting at the memory roof, while an untuned kernel for the same
shape manages 0.69 TB/s.

Two things to take from that, both independent of who wrote either kernel:
**the roof a real kernel reaches is well below theoretical peak**, so calibrate
against a measured ceiling rather than the peak; and **the tuned/untuned gap on
one shape is the size of the prize**, which is the number worth estimating
before committing to a tuning round.

## The vector-only ceiling

If the per-core `ddr_rate` split is physical, a kernel using only
`section_vector()` can reach at most the vector-core share — roughly half the
aggregate on `950PR`. On a memory-bound case that needs ~1.59 TB/s to hit a
given target, a ~1.48 TB/s vector-only theoretical ceiling puts the target out
of reach however clean the kernel is. Compute both numbers before committing to
a vector-only design: if the requirement exceeds the vector share, the deficit
is structural and no kernel-side work reaches it.

Treat this as a hypothesis, not a constant. The same ini block lists
`ddr_rate=31` beside `ub_to_ddr_rate=128` for AICore and `16` beside `40` for
VectorCore; the units are not self-consistent, and the reading above is simply
the one consistent with measured aggregate bandwidth.

**Settle it by measurement:** a pure-DMA copy of identical bytes, vector-only
versus cube+vector, nothing else varied, with a control variant in the same run.
Design around the answer only afterwards.

## Symbolic roofline

Use version- and SKU-specific values read from the source above:

```text
cube_time   ≈ total_MACs / detected_cube_throughput
vector_time ≈ vector_work / detected_vector_throughput
move_time   ≈ bytes_per_path / detected_path_bandwidth
estimate    ≈ max(cube_time, vector_time, move_time)
```

This ranks hypotheses; it does not establish actual latency. Reload counts,
launch overhead, dependencies, occupancy, and compiler scheduling can change
the result.

## Measurement loop

1. Freeze a passing correctness test.
2. Capture a baseline with the same input, launch geometry, warm-up, and
   profiler configuration used for later variants.
3. Read the dominant measured pipe from the retained profiler artifact —
   `aic_mac_ratio` versus `aic_mte2_ratio` on the cube side, `aiv_vec_ratio`
   versus `aiv_mte2_ratio` on the vector side.
4. Change one relevant factor: work distribution, tile shape, buffering,
   on-chip residency, accumulation structure, or documented dtype.
5. Re-run correctness and profile again.
6. Keep the change only when the measured target metric improves without
   violating the numerical contract.

Use [msprof-guide.md](../perf-analysis/msprof-guide.md) or
[msprof-op-guide.md](../perf-analysis/msprof-op-guide.md) for collection, and
[csv_fields_reference.md](../perf-analysis/csv_fields_reference.md) for the fields supported by
the current parser.

## A5 原语代价表（每 64 lane 寄存器）

这张表提供 A5 dataflow 决策所需的 target-specific 实测数量级；仅在目标门禁确认后使用。

- **target**：A5 `Ascend950PR_9579`，56 vector core
- **测得时间**：2026-07（随 A5 实测批次）
- **适用范围**：仅该 SKU。换 SKU 后必须重新测量，不能直接复用本表数值。

| 原语 | 代价 | |
|---|---|---|
| `vf.gather` | **~20 ns** | 跨 lane |
| `vf.scatter` | **~18 ns** | 跨 lane |
| UB 往返（**含必需的 `vf.mem_bar(VST_VLD)`**） | **~16 ns** | 跨 lane |
| `vf.load_align` / `vf.store_align` | **< 1 ns** | 不跨 lane |
| 算术本身 | **~0.3 ns** | 不跨 lane |

**三个跨 lane 原语彼此相差不到 25%**——ISA 不提供寄存器级 lane shift，所有 UB 中转替代品
交同样的税。

两条推论：

- **无 bank 冲突的 gather ≠ 便宜的 gather。** padding pitch 仍然必要（冲突时再差 15×），
  但消除冲突不会让它接近 `load_align`。
- **跨 lane 与不跨 lane 相差 20–35 倍。** 一个内层循环里如果有 1 次 gather + 2 次 scatter，
  即使两种写法 op 数完全相同，**UB 寻址也会占到 83%、算术只占 17%**
  （证据：一个连续扫描算子，64 元素 13 op）。

## Measured levers, ranked by what they actually returned

From two attention-family operators taken from correct-but-slow to the
bandwidth roof. Each was chosen from a per-kernel profile, never guessed, and
each figure is a device measurement on the same case before and after.

**1. Delete work the shape makes unnecessary.** Padding the token count up to a
whole M tile needs a pad kernel before and a strip kernel after — but only when
`M` is not already a multiple of `TM`. Guarding both launches on `M == Mp` and
letting the cube read and write the real tensors removed **24.5%** of the
largest case outright. This is historical diagnostic evidence; delivery must
fold equivalent tail handling into one launch or report a blocker.

**2. Size the transfer, not the tile.** A copy kernel moved 67 MB at **121 GB/s**
against a roof near 1 TB/s purely because its column tile was 512 elements — a
1 KB DMA. Widening to 4096 (8 KB) was the whole fix. Choose the column tile from
the transfer size you want.

**3. Fold the inner dimension into the task index.** The same kernel strided its
outer loop over rows with an inner loop over column tiles. At `M = 1` — the
decode shape, and a large share of any decode-heavy set — that is one row, so
one core did everything
and 31 idled. Striding over `(row, column_tile)` pairs fixes it with no change to
the body. Whenever the outer dimension can be small, fold the inner one in.

**4. One task per output row can be pure descriptor overhead.** A RoPE kernel ran
`M * N = 65536` tasks each issuing eight 128-byte DMAs: **506 us**, 33% of the
case. Batching 64 rows into one `[TRR, HALF]` tile per task — same arithmetic,
same bytes, one strided transfer instead of 64 — took it to **46 us**. The
register function only has to become a loop over registers; row boundaries do
not matter to an elementwise pass because each row's operands sit at the same
offset in every tile.

**5. Widen the cube's N tile to lengthen the DMA run.** At `TN = 64` each row of
a B tile is a 128-byte run of a 24576-wide weight, and the matmul ran at
~1 TB/s; `TN = 128` doubled the run and reached **1.72 TB/s**, at the roof.
Widths constrain which kernels can use it, and a partial N tile does **not**
fault — it silently corrupts its own share of the output — so keep the narrow
variant for any width the wide tile does not divide.

**6. Hoist the reused operand out of the inner loop — but watch parallelism.**
Flattening `(m_tile, kv_tile)` into one task index re-loads the query tile once
per KV tile: 2.1 GB of traffic where 134 MB suffices. Hoisting it out entirely
fixed that and immediately broke something else — it leaves only `n_mt` tasks,
and decode shapes have `S = 1`, so one case had **4 tasks for 32 cores** and got
*worse* (16.7 → 37.9 us). The shape that works is a task per (M tile × *group*
of KV tiles), with the host choosing `n_g = ceil(cores / n_mt)`: just enough
groups to refill the cores, so the operand is re-read `n_g` times rather than
`n_kt` times. Final: **15.7 us**.

**7. Larger M tiles cut weight re-reads, if L0A allows.** The K operand is
re-read once per M tile. Doubling `TM` halves that, but a `[64, 512]` narrow
query tile is 64 KB — the entire L0A, with no room for a second operand.
Walking the contraction dimension in blocks instead (two 256-wide blocks, both
resident in **L1** across the loop, moved L1→L0A per use) makes `TM = 64` fit and
keeps GM traffic at once per task. Worth ~10% here — smaller than the traffic
arithmetic predicted, because the re-read was already largely L2-resident, so
what improved was L2 pressure rather than DRAM traffic.

**Know when to stop.** After these, the kernels moved 604 MB in 351 us
(**1.72 TB/s**), 125 MB in 72 us (1.74 TB/s) and 234 MB in 139 us (1.68 TB/s),
against a ~1.6 TB/s roof for a weight-streaming access pattern. At that point
nothing further is
available by making a kernel faster — only by making it move fewer bytes. Say so
and stop, rather than continuing to tune.

## Scope of retained examples

The KB's
[BF16 operand-reuse implementation](../../../pypto-pro-op-kb/examples/samples/bf16_matmul_operand_reuse/bf16_matmul_operand_reuse_impl.py)
demonstrates one reuse topology and embeds a correctness test. It does not prove
that an operator is cube-bound or that the topology is faster on another shape
or target. Profile the current kernel.

## When the local levers plateau: structural levers

Every lever above preserves the algorithm — it re-times, de-duplicates, or
re-lays-out work that already exists. When those plateau against a traffic or
occupancy wall that the byte arithmetic says is still reducible, the remaining
moves change *what work and traffic exist*. Three, transferred from measured
attention-family work on the same silicon (the structure transfers; the
numbers do not):

- **Keep a loop-carried reduction on chip.** An accumulator (or running
  max / denominator) that round-trips a GM workspace every step is often the
  dominant traffic of a low-arithmetic-intensity reduction — a decode-shaped
  attention with few query rows is the canonical case. Keep it resident in UB
  (or L0C) across the loop and touch GM once, at the contract boundary.
  Precondition: it fits the on-chip budget at the target tile — check the real
  headroom, since bring-up kernels often leave UB largely free once wide
  temporaries are gone. Risk: on-chip accumulation is a reduction/cast-order
  change; validate at the new boundary, and prefer UB residency when the cube
  overlap depends on L0C slots.
- **Read each shared operand once across cores.** When every core re-reads the
  same GM region, the aggregate read is `operand_bytes × core_count`. Cheapest
  first: **measure whether L2 already absorbs it** — many cores reading
  identical addresses often hit L2 rather than HBM, and the wall may be far
  smaller than the byte count implies; then re-tile so the shared dimension is
  the inner loop; an explicit broadcast/shared-load scheme comes last, because
  it trades a read wall for a sync wall. Measured lever 6 above is this
  lever's instance, including the failure mode (hoisting fully starved the
  cores; the group form fixed it).
- **Split the reduction dimension to raise occupancy** (the flash-decoding
  shape). When the parallelised dimension is small and further splitting is
  exhausted, split the long reduction axis across cores into mergeable
  partials — for softmax: partial output, running max, running denominator —
  and combine them in a cheap final merge. Preconditions: the reduction admits
  a mergeable partial form, and the merge stays small next to the occupancy
  gained; sweep the split count, since over-splitting makes the merge
  dominate. The merge is its own cast-order change — validate the merged
  result against the reference.

These are larger edits that usually move reduction or cast order: apply one at
a time and re-verify correctness at the new boundary before measuring.

## When to stop

Stop at a wall proven with data: the moved bytes are irreducible — each read or
written once, each feeding the contract — **and** occupancy is at the device
limit for the parallelism the algorithm exposes. Record the measurement that
proves it. "`MTE2` is at 98%" alone does not; "`MTE2` is at 98% and every byte it
moves is read exactly once" does.

## Provenance of the numbers on this page

The model in "Calibration measurements" is reproducible without hardware:
recompute it from the installed platform ini. It is arithmetic over the platform
constants, not profiler output, and should be recomputed rather than quoted from
here.

The achievable-bandwidth figures are cited measurements from other A5 work, kept
because a theoretical roof alone gives no sense of what fraction is reachable. They
are scenario-specific: do not treat 1.17 TB/s as a target for a different
operator or shape. Any *new* bandwidth or duration claim added here needs a
version-tagged profiler artifact and the command that produced it.

## The per-register mask is a first-class cost, and hoisting it can be the whole win

Established with an ablation ladder -- a load/store-only floor rung, then each
compute stage added back -- with ABAB pairing inside one lock window and control
drift held under ~0.3%. Re-derive it the same way on any target before quoting
the numbers below.

**`pl.min` + `vf.update_mask` costs 10.3-13.6 ns per register group.** Against
this page's own primitive table -- arithmetic ~0.3 ns, `load_align`/`store_align`
under 1 ns -- that puts the *mask that guards the arithmetic* in `vf.gather`'s
class (~20 ns), i.e. **30-45x the work it protects**. A loop that recomputes the
predicate every register group is therefore paying for masking, not for
computing, and no amount of buffering or blocking touches that cost.

**Splitting the register loop** into a full-register path that takes the
all-lanes predicate, plus a zero-or-one-trip tail that keeps the mask, measured
on the floor rung:

| 形态 | 观察 |
|---|---|
| 大而访存受限的 shape | 提升有限（约 1.1x）：本就接近访存上限，掩码不是主要成本 |
| 中等 shape | 约 2x |
| 小而落在 L2 内的 shape | 约 3x：掩码开销占比最高，且不受访存上限压制 |

The ratio flip is the mechanism self-proof: a vector-bound rung became a
genuinely memory-bound one. The speedups differ because the 8192 case hits the
DDR wall and stops while the smaller two are L2-resident -- **a fixed
per-element saving shows up as wildly different ratios depending on which side
of the 128 MiB L2 the working set sits.**

**It is value-equivalent, so the ordinary correctness suite applies**: on a full
register `vf.update_mask(64)` *is* the all-lanes predicate and
`vf.select(v, ident, all)` is the identity, accumulation order is untouched, and
the tail keeps both. Bit-exact, not within-tolerance -- which matters when an
output is passing *at* its threshold.

**Two caveats, both measured.** This page's mask-hoisting entry elsewhere
reports only +5% and a 3-4% *loss* on 8/16-column tiles: narrow tiles amortise
the hoist over 2-4 registers and can lose. Measure the short-axis cases
separately rather than assuming, and dispatch per width if they disagree. And a
zero-length tail is a live hazard: `vf.update_mask(0)` reaching a
`vf.select`/`vf.reduce_*` inside a zero-trip tail body has produced device
fault 507035; clamp the tail extent to at least 1 (harmless, since the loop is
zero-trip exactly there).

### Two corollaries about reading a floor rung

- **A load/store-only rung is not automatically "the memory floor."** Check its
  own pipe row first: the rung above had `aiv_vec_ratio` 0.658 against
  `aiv_mte2_ratio` 0.337, so it was a *minimal-VF* floor, and every ratio taken
  against it was a ratio against vector work. Cross-DSL closure made the error
  visible: another DSL's **complete** kernel beat this do-nothing rung on the
  same shape, while the hoisted rung then beat that complete kernel.
- **Stall headroom lives in the floor rung, not the full kernel.** Measured
  bubble (`1 - aiv_vec_time/aiv_time`) was 34.2% at the floor and **0.8%** at
  the full kernel on the same shape: the bubble is absorbed as stages are added.
  "The floor has 147 us of stall, so deeper buffering can win 147 us" does not
  follow -- deeper buffering only pays where `1 - aiv_vec_ratio` is large in the
  *shipping* kernel.

## Cube cores cannot be borrowed as DMA engines on this DSL

Two independent reasons, both measured or read off the installed source, on
PyPTO-Pro 26.0:

- **L1 cannot be written back to GM.** `pypto_pro/ir/op/block_ops.py:233`
  restricts `store`/`store_tile` sources to Vec (UB) or Acc (L0C), and `:465`'s
  `move` paths (`Mat->Left, Mat->Right, Acc->Vec, Vec->Vec`) give L1 no route to
  UB either. GM->L1 loads are fine (`:668`), so the only cube->GM path is
  GM->L1->L0A/L0B->L0C->GM, i.e. through the matmul accumulator. **The doc page
  `store.md:17` lists "L1/UB Tile" as valid sources; the installed code
  disagrees, and the code is what runs.**
- **Merely declaring `pl.section_cube()` costs 1.88x on the vector path** --
  the launch drops from 56 blocks to 28 (803.5 us vs 427.2 us on the same
  vector-only work). The cube assist is a net loss before the store
  restriction even applies.

Also measured while testing this: **there is abundant spare DDR bandwidth on a
vector-bound kernel.** A read-only contention probe on the same shape took
+28.6% bytes for +7.6% time -- the extra reads were served at 26% of the
saturated rate, a marginal 2.21 TB/s. So on this operator family a
vector-issue-bound verdict cannot be re-explained as a bandwidth shortage.
