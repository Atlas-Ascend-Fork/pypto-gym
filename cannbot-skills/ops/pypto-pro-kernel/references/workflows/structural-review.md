# Structural review and reachable target

Use this when a kernel is correct but structure-limited: micro-tuning has stalled, or
the current structure may be the wrong one. The [coverage workflow](../workflows/optimization-coverage.md)
owns when to enter this review.

## 1. Forced review scope

Enter this review when a condition holds and the current structure looks limiting; do not
run it beside every optimization. The diagnostic-first flow in
[optimization.md](optimization.md) still owns budget and routing.

Conditions that qualify:

- the user supplied an explicit numeric performance target and the gap is significant;
- the delivered kernel is self-developed, has not been aligned with any verified reference,
  and micro-tuning has stalled;
- all pipe utilizations are low, no dominant bound exists, or active cores are far below physical cores.

Also trigger it when:

- the critical path is dominated by synchronization or dependency waits;
- the operator is built from many small contractions and fixed overhead dominates;
- after all preset sources are closed, the implementation is still an order of magnitude away from the reachable target.

## 2. Gather the best known structure

Before redesigning, establish a structure anchor:

1. search the workspace, the operator repository and its tests for the same operator or topology;
2. search installed official samples and any explicitly configured card library
   (`--cards-root`) for existing implementations;
3. record each candidate's structure: operand dtype, on-chip residency, work decomposition,
   synchronization model, graph-equivalent rewrite and stage split, with its measured performance and conditions;
4. if no reproducible measurement exists, derive a lower bound from arithmetic roofline.

Record the result in the design comment or the performance report. If a verified faster
structure exists, align with it before micro-tuning the current one.

## 3. Four structural dimensions

For each dimension, record: current value -> candidate value -> precision/compliance impact
-> expected effect. Record each in the reply or the requested audit file.

| Dimension | Decision |
| --- | --- |
| Operand numeric type | Can a Cube contraction operand move from FP32 to BF16/FP16 while keeping FP32 accumulation? |
| On-chip residency | Can loop/recurrence state and cross-iteration intermediates stay in UB or L1 instead of round-tripping through GM? |
| Work decomposition | Can independent work items be flattened as (item, head/shard) and spread over all AI cores? |
| Graph-equivalent rewrite | Can an elementwise or transcendental phase on the critical path be rewritten as a cheaper native operation without changing the math? |

The graph-equivalent rewrite is an algorithm-level dimension and usually has the largest
headroom. Check the numerical contract before adopting it.

## 4. Closure

A dimension closes only as:

- **implemented and measured**: full correctness, paired measurement, mechanism checked;
- **reproducible capability or contract block**: a minimal reproduction with source location,
  generated artifact or documentation, and the command.

"Not applicable", "expected gain is small", "profiler did not show it" and "capability gap
without a minimal reproduction" are not closure reasons. Mechanism-level negatives need
implementation variants before they close.

## 5. Reachable target

Use the best verified measured structure as `T_best`. If none exists, estimate:

```
T_lb ~= total work / (AI cores * per-core peak * target efficiency)
```

State the unit, the measured efficiency source and its scope. `T_lb` is an engineering
estimate, not a mathematical lower bound. Record the ratio of the current implementation
to `T_best`/`T_lb`.

## 6. Scope of a floor claim

A floor measured on the frozen structure bounds that structure only, not the problem.
When a forced or triggered condition holds, a frozen-structure floor is not a valid
stopping reason; complete sections 2 to 4 first.

## 7. Handoff

- Record the review as `structural` in the reply or audit file.
- Close it before source-derived candidates from a card miss.
- If a dimension changes precision, layout or the public wrapper, update the design
  comment and the acceptance matrix with it.
