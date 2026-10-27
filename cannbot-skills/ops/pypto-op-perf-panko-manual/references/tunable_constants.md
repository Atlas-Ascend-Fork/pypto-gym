# Tunable constants — extending INIT normalisation past the tile call

Status: design agreed, not implemented.

## What exists

`bayesian_optimization.normalize` runs at INIT and already carries the whole pattern: the model
rewrites, the harness checks the rewrite is admissible, and the BO gains sites.
Its one claim is

> normalisation exposes parameters and does not change values

checked by resolving the symbolic arguments against the constant bindings and
comparing them to the literals the model wrote.

It covers the two tile calls and nothing else. Whether the deterministic core
owns a tile therefore depends on how a generated kernel happened to spell one
call — a kernel written with literals gets a BO lever, the same kernel written
with names gets none.

## The gap

The same asymmetry one level out. A named constant that drives a loop trip count
or a view extent is invisible to `bayesian_optimization.apply`, so the search cannot move it and the
model has to sweep it by hand as a structural delta — paying a full evaluation
per point for what a single BO dimension would cover.

## No source markers

An earlier draft had the kernel author annotate each tunable constant. Rejected:
PANKO must work on kernels that never opted in, including kernels written before
it existed, and the delivered kernel must not carry annotations that mean nothing
outside PANKO. The declaration belongs in `search_state.json`, where it also
survives the snapshot/restore ratchet.

## Ownership

| step | owner |
|---|---|
| enumerate integer constants and their uses | DET |
| select the admitted ones | DET, by the rule below |
| rewrite the kernel so each is a single literal site | LLM, at INIT |
| verify no value moved | DET |
| verify each incumbent value is inside its own domain | DET |
| record the declaration | DET |

The model's remaining job is the same one it already has in normalisation: decide
which uses of a name are the parameter and which are the extent, and spell the
rewrite. Selection is computable and is not asked.

## Domain

Deterministic, anchored on the incumbent — the ladder around the value the kernel
is running, the way the L1 multiplier is already anchored so a warm start is
always reachable. No author knowledge, no model guess. The domain does not have
to be correct; it has to CONTAIN the good values, and the search prunes the rest.

Generator syntax (`divisors(N)` and similar) was considered and dropped. The
binding constraint on a tile-like constant is rarely the one such a generator
expresses, so it produces illegal values while looking authoritative.

## Which constants are admitted

Two stages, and the order matters.

1. A constant read DIRECTLY inside a `@jit` type annotation or a `torch.*` call
   is part of the operator's declared interface. Excluded.
2. Of the rest, admit only those that reach a `pypto.loop` trip count, directly
   or through a derivation. Reaching a loop bound means the constant sets an
   iteration count, which is what a tiling or chunking parameter does and what a
   declared extent does not.
3. Everything else is left alone.

The rule is default-deny. Stage 2 admits a narrow, positively identified class
rather than admitting everything and gating the dangerous cases out.

Derivation matters in both directions and asymmetrically. A constant typically
reaches its loop bound one derivation removed, so stage 2 must follow the chain.
Stage 1 must NOT: a constant whose DERIVED name reaches an annotation is not
itself part of the interface, and treating it as one loses real levers.

## Legality is measured

Almost every illegal value announces itself: a compile error, a capacity refusal,
a device fault, a precision failure. Those need no declaration. They cost one
evaluation, and the existing machinery amplifies that payment — a learned ceiling
retires the whole upper half of a ladder from one paid failure, and front-end
refusals are carried across structures. This is the asymmetry the design already
adopted: a false rejection silently deletes a reachable optimisation, a false
acceptance costs one evaluation.

The dangerous case is illegality that is SILENT ON THE FIXTURE. A value that
reads outside a bound can land on memory that happens to be valid under the test
harness's own allocation pattern; the golden then passes, a paired re-measurement
agrees with itself, and a kernel that is wrong under any other allocation is
banked. Reproducibility and correctness are different properties and no amount of
re-measurement separates them.

An earlier draft proposed a static gate for it, triggered on a constant appearing
in an index expression over a parameter's extent. It is not implementable at this
level of analysis: name-based taint has no scope and no flow sensitivity, so the
closure grows until some tainted name reaches some subscript, and the flag fires
on every constant in every kernel measured. A gate that never discriminates is
worse than none, because it reads as protection.

The rule above removes the need for it. A constant admitted for setting a loop
trip count is not an index, so the silent case does not arise for the class being
tuned. This is why the gate direction was flipped rather than the gate improved.

## The known miss, and why it fails safe

Where an operator is split into modules, a tiling constant appears in the
module's own declared interface, so stage 1 excludes a lever that is real at the
operator level. Moving such a constant is a coordinated edit across the module
boundary rather than one literal, so it does not belong to a single-site BO knob
in any case.

## What this does not claim

- That the declared domain contains only legal values. It does not have to.
- That the golden catches every illegal value. It does not, which is why the
  static trigger above exists.
- That normalisation preserves semantics. Constant folding cannot know that; the
  golden answers it. Two gates, each doing the thing it can actually do.
