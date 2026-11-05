# 分阶段 matmul A/B 证据协议

一次分阶段 Cube matmul 的 A/B 比较要成立，必须满足下列每一项。把它当判据清单读，
不是当表格填——它约束的是"这次比较能不能作为证据"，不是记录格式。配套的精度与性能门在
[当前数值合同](../../../pypto-pro-kernel/references/api/numerics.md)、[GEMM owner](../../../pypto-pro-kernel/references/api/gemm.md) 与 [source-bound comparison](../../../pypto-pro-kernel/references/diagnostics/performance.md#test-a-mechanism-with-source-bound-evidence)。本页保留既有工程的 A/B 记录方法，不另设技术规范；具体任务的累加/阶段边界以这些 owner 和实际执行参考为准。

两条贯穿全篇的规则：

- **每次只验证一个主要假设。** 默认只改一个变量；有明确依赖依据的已登记联合候选可作为
  一个整体实验，但结果不能单独归因给其中任一分量。
- **不适用的项要写明理由，不能跳过。** 略过的项在事后审阅时等同于"已检查并通过"，
  而这正是无法复盘的开始。

下面每一项都是事后判断该轮测量是否可信所必需的；缺一项，该轮结论的强度就降一级。

## Identity

- Date
- Source SHA A / B
- Target / physical NPU
- CANN / PyPTO-Pro
- Environment setup
- Official task/checker SHA
- Lock path, device:inode, holder PID: （记录实测值）

## Frozen contract and hypothesis

- Official shapes and value domains
- Stage equation
- Input -> accumulator -> output dtype
- A reduction tree
- B reduction tree
- Only changed variable, or registered joint tuple and dependency rationale
- Expected precision effect
- Expected performance effect
- Range/shape assumptions not guaranteed by contract
## Static and generated-code gates

- [ ] Wrapper signature and output contract are unchanged.
- [ ] Wrapper performs no tensor arithmetic and preserves the allowed JIT/launch count.
- [ ] Cube LHS/RHS dtypes match; no implicit BF16 -> FP16 load or mixed matmul dtype.
- [ ] Any BF16 -> FP16 RHS is first materialized by AIV into ordinary FP16 GM.
- [ ] Generated code confirms operand dtype/layout, K extent, phases, and final cast mode.
- [ ] Every reused Acc range records exactly one handoff protocol:
  （记录实测值）.  Hardware mode pairs producer `AccPhase` with
  consumer `STPhase` and closes both sides with `Final`; software mode keeps
  producer and consumer phase-free and uses one TileGroup mutex.  The two modes
  are never mixed.
- [ ] Generated CCE proves the selected Acc protocol.  Hardware mode has a
  phase-bearing M producer and phase-bearing FIX consumer over matching blocks.
  Software mode has matched get/release around both M operations and the FIX
  drain with the same Acc mutex id.  Reject phased M + ordinary phase-less
  Acc-to-Vec `TMOV`, mismatched mutex ids, missing release, or raw `make_tile`
  without mutex ownership.
- [ ] Workspace maximum bytes and conversion GM traffic are recorded: （记录实测值）.
- [ ] Any pack/re-layout records full-shape bytes per call, writer ownership,
  barrier cost, and measured overlap; contraction FLOPs are not reported alone.
- [ ] `pl.load_tile` tile-block offsets and `pl.load` absolute element offsets
  are not mixed; non-tile-aligned GM loads have an A5 MTE/tail gate.
- [ ] Each GM/Mat/Acc region has one proved writer or a documented atomic protocol.
- [ ] A shape-specialized graph that omits a fixed partial never reads its
  `torch.empty` GM buffer; every absent term is initialized locally before use,
  with the required VF store/load barrier.
- [ ] Local-memory live intervals, mutex IDs, and capacity pass: （记录实测值）.
- [ ] Every `pl.move` has physically compatible source/destination TileTypes;
  `valid_shape` is not used to justify a physical-shape mismatch, and `offset`
  is used only to extract a destination-sized sub-rectangle from the source.
- [ ] Every TileGroup has at least as many physical slots as the maximum number
  of simultaneously live values; two `next()` values consumed together never
  come from a one-slot group.
- [ ] Reused Vec scratch in a pipelined load/cast/store loop is a TileGroup in
  the auto-mutex ownership graph, not a bare address-disjoint `make_tile`.
- [ ] Typed TileGroups that reuse one physical arena have distinct mutex IDs,
  disjoint live stages, and an unconditional hard barrier between stages.
- [ ] Tail ownership and untouched sentinel regions pass: （记录实测值）.
- [ ] For every multi-wave split, each wave's start/end is derived from the
  physical tensor equation; the wave union is disjoint and equals the total
  tile/chunk extent.  Head counts are converted through the actual per-head
  width before they are used as vector-chunk counts.
- [ ] Every launched AIC/AIV lane reaches the same unconditional synchronization points.
- [ ] A repeated local-mailbox task ends with zero outstanding READY/FREE/
  STORE-FREE credit and a `bar_all()` on both engines after their final event;
  the gate covers the real uneven per-core task depth and maximum dynamic depth.
  `bar_all()` is recorded as local `PIPE_ALL`, not MIX or event re-seeding.
- [ ] Each `pl.section_cube` / `pl.section_vector` recomputes the scalar row
  offsets, worker IDs, and worker counts it consumes; it does not capture a
  scalar first defined in another section. Target compilation has no
  undeclared SSA identifier in generated C++.
- [ ] A VF store/load dependency has the required `vf.mem_bar(VST_VLD)`.
- [ ] No runtime shape branch changes the Cube/resource topology.
- [ ] If topology varies, every `tiling_key` IR is separately inspected and contains no runtime key reference.
- [ ] The pre-lock host smoke imports every helper used by the real input
  factory and asserts required symbols without allocating an NPU Tensor; the
  first actual input-factory call remains inside the device lock.
- [ ] The staged SHA manifest includes every transitive input-factory/helper
  module needed by each isolated profiler process; a missing helper is a
  pre-launch environment failure, not a kernel or performance result.
- [ ] One outer `flock lockfile command` or re-exec owns the lock for the full
  compile/run/profile process tree.  `device:inode` and the non-`*` `lslocks`
  holder PID (not a `WRITE*` waiter) match at start/end, and no competing
  same-device process appears.  A printed
  `LOCK_ACQUIRED` line alone is not accepted; any holder/inode drift invalidates
  the complete timing run.
- [ ] The wrapper retains no Tensor-derived GM buffer, prepacked weight, or
  persistent workspace across calls; caches contain only code or
  value-independent metadata and do not key on `weakref`, `_version`, or
  `data_ptr()`.

Related templates:
[stage-task flatten](../../templates/stage-task-flatten.py.tmpl),
[exact-order shared-left output pair](../../templates/cube-shared-left-output-pair.py.tmpl), and
[tiling-key specialization](../../templates/tiling-key-resource-specialization.py.tmpl).

The historical mailbox prototype is retained only as
[negative evidence](dual-aiv-mailbox-negative-evidence.md), not as an
implementation option.

## FP16 range and special-value audit

| boundary | contract bound or observed range | smallest nonzero | NaN | +Inf/-Inf | evidence |
|---|---:|---:|---:|---:|---|
| source input/weight/gamma | （记录实测值） | （记录实测值） | （记录实测值） | （记录实测值） | （记录实测值） |
| BF16 before FP16 cast | （记录实测值） | （记录实测值） | （记录实测值） | （记录实测值） | （记录实测值） |
| FP16 workspace | 值域 | 最小非零 | NaN 计数 | Inf 计数 | （记录实测值） |
| FP32 accumulator/output | 值域 | 最小非零 | NaN 计数 | Inf 计数 | （记录实测值） |

- [ ] Every analytic bound cites a contract premise; premises true only of the
  sampled data are labelled unproven.
- [ ] FP16 overflow, underflow, and non-finite propagation are checked before promotion.
- [ ] Exact ties/adjacent values validate terminal `CAST_RINT` against Torch BF16 RNE when used.

## Precision A/B

Use identical deterministic inputs. Compare the changed intermediate FP32
stage before terminal conversion, then run the unmodified official checker.
`max_abs` is diagnostic only.

| case/domain | variant | intermediate max_abs | overall MERE / MARE | normal errors/ref | small errors/ref | cancel errors/ref | NaN positions | result |
|---|---|---:|---|---:|---:|---:|---:|---|
| 每个用例 | A | （记录实测值） | （记录实测值） | （记录实测值） | （记录实测值） | （记录实测值） | （记录实测值） | （记录实测值） |
| 每个用例 | B | （记录实测值） | （记录实测值） | （记录实测值） | （记录实测值） | （记录实测值） | （记录实测值） | （记录实测值） |

- [ ] Normal, small-value, and cancellation domains all pass.
- [ ] The comparator receives the target-dtype native reference for every
  output; if unavailable, fallback counts are labelled diagnostic-only.
- [ ] For a scheduling-only candidate, run candidate and production from
  separate processes and separate build directories on identical seeds, then
  require an exact output comparison before performance profiling.  Do not
  load same-name JIT variants into one process or reuse one build directory.
- [ ] If a synthetic absolute oracle rejects both a candidate and its known
  accepted production control with identical regional statistics, record the
  synthetic domain as an independent range-risk probe; do not attribute the
  failure to the candidate until the differential control is complete.
- [ ] Aligned/non-aligned tails and sentinels pass.
- [ ] Unsampled-shape probes preserve the actual batch/sequence pair as well as
  the flattened row count, and include reduced-K domains that change the
  number of legal partials.
- [ ] Alternating shapes/keys pass in one process without `161002` or stale state.
- [ ] A pass on the sampled shapes is not described as proof for unsampled ones.
- [ ] Bounded JIT/load/launch/synchronize liveness is reported separately from
  local-comparator precision; neither is promoted to a claim about unsampled
  shapes.
- [ ] A stalled remote stage is correlated with process/event evidence
  before it is called a kernel hang.
- [ ] The input to the changed stage was compared before changing its
  consumer; an upstream error is not attributed to the new consumer.
- [ ] If another residual term is proposed, the current remainder is nonzero
  on the failing positions and the new term changes those positions.

## Performance A/B

Keep target, physical NPU, environment, lock, inputs, warmup, and measurement
method identical. Include cast/widen/workspace traffic rather than timing only
the contraction.

- Warmup / repeats
- Timing source and synchronization
- Baseline definition
- Conversion overlap/extra traffic
- Full-weight pack/re-layout bytes per call
- Aligned tile load versus absolute-element load evidence
- Static logical bytes / task descriptors before and after
- Slowest physical-core output chain before and after
| case | A device times (us) | B device times (us) | A statistic | B statistic | B/A | notes |
|---|---|---|---:|---:|---:|---|
| 每个用例 | （记录实测值） | （记录实测值） | （记录实测值） | （记录实测值） | （记录实测值） | （记录实测值） |

- [ ] Logical L1/GM bytes, issue counts, and task descriptors are labelled as
  static upside bounds, not measured speedups.
- [ ] Replacing GM with an on-chip mailbox records physical message width,
  Acc-chain/mutex count, aligned-load and shared-left/RHS reuse deltas,
  per-core task depth, final drains and both engines' local barriers.  Reduced
  logical bytes alone cannot approve the candidate.
- [ ] A mailbox FREE/ACK is emitted immediately after the last mailbox read;
  generated code proves no later read, while residual casts/stores use disjoint
  scratch.  Early release never removes the final zero-credit drain.
- [ ] If the slowest physical-core output chain is unchanged, promotion uses a
  repeated target A5 B/A/A/B run against the frozen exact control in separate
  build directories; simulator timing or static counts cannot satisfy this gate.
- [ ] Every ABBA arm uses deterministic inputs with matching fingerprints or
  an explicitly frozen input factory and seed.  Profiler warnings and CSV
  coverage are recorded; an incomplete trace is not used as roofline evidence.
- [ ] If a shape dispatch is added, report hit-domain and fallback-domain
  ratios separately.  An aggregate result must not hide a regression in the
  newly selected kernel path.
- [ ] A local target-board ABBA is a promotion gate, not a substitute for an
  official evaluation-server terminal result when that server defines the
  aggregate.  Record the exact revision, per-case elapsed values,
  baseline and hardware floors.  If the official hit-domain result reverses
  the local sign beyond repeat drift, reject or narrow the dispatch and retain
  the frozen server-best package.
- [ ] Cross-run comparisons use the per-case elapsed values from the two named
  runs.  Do not assume that a baseline you do not control, the hardware
  floor, the task bundle or the environment is unchanged merely because the
  operator and source baseline have the same label.

## Decision

Reject B immediately for an official precision failure, NaN-position mismatch,
FP16 non-finite outside the contract, ownership/sentinel failure, resource
error, or unexplained generated-code change. Retain B only when repeated device
timing improves under the same conditions.

- Decision
- Evidence-backed reason
- Remaining unverified risk
- Logs/artifacts
- Next single-hypothesis probe
