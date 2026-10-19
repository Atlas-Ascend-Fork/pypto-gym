# PyPTO-Pro knowledge base

This directory contains reusable PyPTO-Pro design constraints, patterns, and
validated study kernels. It supplements the installed PyPTO-Pro API
documentation and official examples; it does not override them.

Start at [ROUTER.md](ROUTER.md) and open only the reference needed for the
current decision.

## Active resources

- Platform and API constraints: [constraints/README.md](constraints/README.md)
- Reusable dataflows: [patterns/pattern-index.md](patterns/pattern-index.md)
- Selector usage contract: [examples/README.md](examples/README.md)
- Validated implementation selector: [examples/kernel-index.md](examples/kernel-index.md)
- Correctness-preserving decomposition: [references/decomposition-primitives.md](references/decomposition-primitives.md)
- Measurement workflow: [`pypto-pro-op-perf-tune` evidence protocol](../pypto-pro-op-perf-tune/references/perf-analysis/evidence-protocol.md)
- Integrity checks: `check_kb_integrity.py`, with negative controls in [tests/test_check_kb_integrity.py](tests/test_check_kb_integrity.py) — every case there is a defect that once passed a full green run

## Retention gate

Keep a KB item only when all of these hold:

1. An active router or selector links to it.
2. Its purpose is reusable beyond one experiment, branch, environment, date,
   benchmark score, or session.
3. Technical claims cite an official document/source path or a retained,
   reviewable validation artifact.
4. Code is selected only when its validation state is `validated`. Failed,
   work-in-progress, and diagnostic probes do not belong in the production
   selector.

Use scenario and technique names for canonical files. Put measured values in
validation evidence, never in filenames.

## How to write a reusable rule

The retention gate above asks whether an item is reusable. This is how to *write*
one so that it is — the same principle [ROUTER.md](ROUTER.md) applies to routing:
**classify by the shape of the computation, never by the operator's name, because
an operator name tells a new operator nothing it can reuse.**

Write every rule as **trigger → rule → evidence**:

- **Trigger** — a structural property the reader can check against their *own*
  kernel with no knowledge of any other operator: tile width in lanes, pitch in
  bytes, pass count, whether the loop crosses lanes, whether a tile is aliased.
  This is what decides whether the rule applies to them.
- **Rule** — what to do, stated so it survives a change of SKU and of operator.
- **Evidence** — the measurement, with enough provenance to judge whether to
  trust it. Describe the subject **structurally** where you can ("a 4-op
  elementwise body over 128 full registers") and name the operator only as a
  source tag.

**The failure mode is making the operator the subject of the rule** — "operator X's
pitch is 80 elements" instead of "a pitch that is an even multiple of 32 B
conflicts; check it on the tile the access actually runs on." A reader then has to
reconstruct the analogy before the rule means anything, and if they cannot see it,
the rule does not transfer at all. Three kinds of operator mention are legitimate
and should not be scrubbed: **the topology's defining semantics**
(`torch.gather` on a gather-scatter page), **API names** (`vf.scatter`), and
**validation provenance** (`custom/<op>/` as the retained artifact behind a
`validated skeleton`, which the retention gate requires).

A quick self-check when writing: delete every operator name from the sentence. If
the rule still tells a reader when it applies and what to do, it was written
correctly.
