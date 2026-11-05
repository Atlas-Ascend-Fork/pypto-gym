# Vector reduction: current owners and legacy handoff checks

This page preserves the existing Module workflow's review questions. The sole
technical authority is [pypto-pro-kernel](../../pypto-pro-kernel/SKILL.md).
Use its current owners and installed-source evidence instead of inheriting API
or performance claims from a historical reduction implementation.

## <a id="selection-rule"></a>Selection rule

Implement the task-specific dataflow and numerical boundaries in `DESIGN.md`.
The current authoring policy uses VF for Vector compute except its explicit
sorting exceptions; an old tile-operation sample supplies no extra exception.
Use [row reduction](../../pypto-pro-kernel/references/patterns/vf-row-reduce.md)
for independent rows and [outer reduction](../../pypto-pro-kernel/references/patterns/vf-outer-reduce.md)
when the reduced axis crosses rows. Confirm the input, accumulator and intermediate
dtypes against [numerics](../../pypto-pro-kernel/references/api/numerics.md).
A same-type reduction cannot recover an overflow that happened before widening.

## <a id="load-mask-placement"></a>Load, mask and output maps

Read the [VF owner](../../pypto-pro-kernel/references/api/vf.md), including the
installed signatures, physical register load extent, predicate width, narrow-source
widening and store distribution. A computation mask does not shrink a complete
register load. Derive actual input addresses, valid lanes and output writes before
adapting an example; replacing an API prefix is not a conversion.

## <a id="reduction-checklist"></a>Reduction checklist

- Record logical rows, physical pitch, all live local tiles and rotating slots.
- Initialize each row/stream state independently; keep padding out of every stage.
- Check required intermediate rounding, small magnitudes and cancellation.
- Separate transfer chunk size, register width and horizontal-reduction frequency.
- Recheck addresses, precision, tails and repeated calls after changing geometry.
- Compare the complete call using the same source, inputs, workers and metric.

## <a id="no-branch"></a>VF control flow

A branch is not universally illegal or necessarily emitted as both arms. Follow
[current VF loop-form evidence](../../pypto-pro-kernel/references/api/vf.md#keep-runtime-dispatch-out-of-vf-loops):
compare invariant dispatch outside VF with specialized bodies, then inspect the
actual lowering. Register bindings and runtime branches have their own contracts.
An empty route must genuinely issue no invalid access; a zero count or final mask
alone is not that proof. Historical instruction-cost estimates are hypotheses,
not current performance facts.

## <a id="reviewable-examples"></a>Current correctness anchors

- [Basic VF softmax](../../pypto-pro-kernel/examples/basic_api/vf/vf_softmax/).
- [LayerNorm](../../pypto-pro-kernel/examples/algorithms/layernorm/).
- [Streaming masked softmax](../../pypto-pro-kernel/examples/algorithms/streaming_masked_softmax/).
- [Example catalog](../../pypto-pro-kernel/references/tooling/examples.md).

Read each kernel/test header and rerun its relevant unchanged scope on the target
before adapting it. The old tile/VF softmax pair recorded different historical
implementations and measurements; these current anchors do not reproduce that
comparison or validate an unrun rewrite. A passing softmax does not establish that
another reduction or dtype uses the same rounding, layout or fastest path.

## <a id="scratch-barrier"></a>Scratch barriers protect dependency edges

Use [same-VF ordering](../../pypto-pro-kernel/references/api/vf.md#212-multiple-memory-accesses-to-the-same-tile-inside-vf-have-no-ordering-guarantee)
and [allocation/lifetimes](../../pypto-pro-kernel/references/api/memory-allocation.md)
for actual guarantees. For each dependency record the producer/consumer,
overlapping range or alias, RAW/WAR/WAW hazard, loop backedge and applicable barrier
or tile-group handover. Place synchronization between matched accesses and include
ring wrap and final consumption. A helper's lexical end is not a substitute for
that analysis, nor does a local VF barrier establish every cross-pipe dependency.
Inspect the installed lowering and profile the matched alternatives; do not append
a blanket barrier or claim it is free.

## <a id="primary-sources"></a>Evidence for implementation and handoff

Use target-version official docs, installed source/signatures and controlled target
probes as distinct evidence. See [evidence checks](../../pypto-pro-kernel/references/verification/evidence.md).
Preserve source/toolchain identity and exact commands/results. L1 staged evidence
covers only its implemented Modules; finalization must verify the actual final
wrapper and every applicable DESIGN invariant. Conflicting observations require a
scoped diagnosis, not a universal capability verdict.
