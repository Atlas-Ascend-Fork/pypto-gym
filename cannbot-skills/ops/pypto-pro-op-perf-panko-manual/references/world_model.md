# From profile to structural edit

Inspect the continuation base’s warm latency, spread and launch count. For detailed counters, use the retained msprof CSV and [performance diagnosis](../../pypto-pro-kernel/references/diagnostics/performance.md).
For the selected action, cite one or two readings and the source function. Propose one edit, predict a metric change and say what would disprove it.

- **SIMD batching:** map independent chains to lanes. Preserve order and private state. Compare setup savings with packing and register costs.
- **Shared production:** reuse a result only across matching inputs, masks, types and rounding. Check lifetime, reloads and synchronization.
- **Direct layout:** have the producer emit its consumer’s layout. Check all readers, allocation feasibility and transfers saved.
- **Earlier readiness:** advance work once its inputs are ready. Check stale reads, buffer reuse and tail/skip synchronization.
- **Cost-aware placement:** compare per-core finish times. Adjust partitions, core count or slots with exact coverage and chain order. Check tail time, overhead and storage within the measured boundary.

Leave tile shapes to TPE. Compare precision, latency and counters under matched settings using the measured source hash: RECORD may already have changed the working file.
Use `evolve` to update priorities. Keep acquisition failures separate from kernel failures.
