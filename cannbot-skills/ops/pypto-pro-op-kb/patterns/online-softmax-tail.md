# Streaming softmax across score chunks

## Status

Conceptual guidance only. The KB has no retained runnable PyPTO-Pro reference
that validates the complete streaming QK/softmax/PV pipeline. Do not copy this
page as an implementation skeleton. Confirm API signatures and synchronization
against the target SDK's official flash-attention example.

The retained [single-tile softmax sample](../examples/samples/softmax/softmax_impl.py)
validates only the case where an entire row fits one tile.

## Applies when

The reduction axis is split into score chunks and each new chunk must update a
running maximum, denominator, and unnormalized output without materializing the
whole row.

## Complete recurrence

For the old state `(m_old, l_old, o_old)` and a new score chunk `s`:

```text
m_chunk = row_max(s)
m_new   = max(m_old, m_chunk)
alpha   = exp((m_old - m_new) * scale)
p       = exp((s - m_new) * scale)
l_new   = l_old * alpha + row_sum(p)
o_new   = o_old * alpha + p @ v_chunk
```

After the final chunk:

```text
y = o_new / l_new
```

The denominator update is indivisible: multiplying `l_old` by `alpha` without
adding `row_sum(p)` is incorrect.

## Tail and precision requirements

- Exclude padding lanes from both `row_max` and `row_sum` using the runtime
  valid shape for the final chunk.
- Keep `m`, `l`, and the accumulation for `o` in FP32 unless the official
  implementation for the detected target proves another contract.
- Keep the old maximum available until `alpha` is computed.
- Treat QK/PV section handoff and cross-core synchronization as
  target-version-specific; derive it from official code, not this recurrence.

## Where the tail mask enters, and why the obvious place is too late

The requirement above — "exclude padding lanes from both `row_max` and
`row_sum`" — is correct but under-specified, and the under-specification has a
name. This section says *where* in the recurrence each mask lands.

### Padding zeros are not neutral for a maximum

When the final score chunk has `valid_n < TILE_N`, the padded columns of the
staged score tile hold **zeros**. Zero is not the identity for a maximum. Those
zeros are seen by `row_max`, so on any row whose true maximum is negative they
raise `m_chunk`, hence `m_new`, hence `alpha` — and the corruption then
propagates into `l` and into `o`. The output is wrong on rows that never touched
the padding.

**A mask applied in the `p` domain cannot repair this.** Zeroing `p`'s invalid
lanes after the exponential removes their contribution to `row_sum(p)` and
nothing else: `m_new` and `alpha` were computed upstream and are already wrong.
The rule is therefore about position, not about presence —

> the padded tail must already behave like `-inf` **before** `row_max` runs.

This is also the failure's signature: aligned shapes pass, only the last chunk is
wrong, and the corruption appears on rows that look unrelated to the tail.

### The order the two masks and the cast go in

There are two different tails and they enter at two different points. Numbering
the recurrence in [Complete recurrence](#complete-recurrence) above:

```text
1. mask the reduction-axis (score-column) tail        <-- before any max
2. m_chunk = row_max(s)
3. m_new   = max(m_old, m_chunk)
4. s - m_new                                          (shift to exponent domain)
5. mask the row-axis tail                             <-- here, not earlier
6. p = exp(...)
7. l_new = l_old * alpha + row_sum(p)                 in FP32
8. cast to the narrow output dtype                    <-- only after step 7
```

Three things this pins down that the recurrence alone does not:

- **The column-tail mask is pre-max; the row-tail mask is post-subtract.** They
  are not interchangeable, and applying both at one point loses one of them.
- **The row sum is accumulated in FP32 and the narrowing cast happens after the
  running-sum update.** Moving the cast earlier changes the numerical contract
  silently — it still runs, and it still produces plausible output. This agrees
  with [../constraints/precision.md](../constraints/precision.md), which requires
  the widened chain through the exponential for narrow dtypes.
- **Where a causal mask and a length tail coincide on the same chunk, apply the
  causal mask first and the valid-length mask second.** The second must not undo
  the first.

### Use a finite sentinel, not `-inf`

Write the invalid lanes to a large finite negative value:

```python
NEG_LARGE = -1.0e30
```

and initialise the running row-max `m` to it as well, so the first chunk's
`max(m_old, m_chunk)` is a no-op rather than a special case.

**This one has a local anchor.** In PyPTO-Pro a module-level
`float("inf")` **does not even compile**: it renders as the undeclared C++
identifier `inff` and bisheng rejects the generated `kernel.cpp`
([pypto-pro-dsl-limitations-a5.md](../references/pypto-pro-dsl-limitations-a5.md)
#19, measured on a scan kernel). So the finite sentinel is the route here regardless of
whether the representability claim transfers.

**The sentinel has one hazard, and it is the all-invalid row.** On a row where
*every* lane is masked, `m_new` is itself the sentinel. With true `-inf`,
`-inf − (-inf)` is NaN and the row fails loudly. With `NEG_LARGE`,
`exp(NEG_LARGE − NEG_LARGE) = exp(0) = 1`, so every masked lane contributes 1 to
`row_sum` and the row silently returns a uniform distribution over padding. The
finite sentinel converts a loud failure into a plausible wrong answer, so an
all-invalid row must be excluded or short-circuited explicitly rather than left
to the arithmetic. (Analysis, not a ported claim — it follows from the sentinel
substitution itself. Whether an operator's tiling can even produce a fully masked
row is shape-dependent; check before deciding the guard is unnecessary.)

## Evidence

- Current graph-frontend documentation describes the local statistics and
  state merge in
  `$PYPTO_DEVKIT_DIR/docs/zh/api/operation/pypto-experimental-online_softmax.md`
  and
  `$PYPTO_DEVKIT_DIR/docs/zh/api/operation/pypto-experimental-online_softmax_update.md`.
  Those are A5-only experimental graph APIs, not proof that the same operation
  exists on the PyPTO-Pro tile DSL surface.
- [softmax_impl.py](../examples/samples/softmax/softmax_impl.py) and its
  [golden](../examples/samples/softmax/softmax_golden.py) cover only the
  non-streaming base case.

No full streaming reference is retained, so this page must remain
`conceptual only` in [pattern-index.md](pattern-index.md).

## Independent softmax storage contract / 独立 Softmax 的存储约束

本节是独立、有限FP32输入、输出完整概率行的候选设计规则；不提升上文QK/softmax/PV
融合流水的`conceptual only`状态，也不验证其尾块、NaN或全无效行语义。
保留的[单tile Softmax样例](../examples/samples/softmax/softmax_impl.py)只支持其声明的驻留范围，
不作为以下长行/暂存路线的验证证据。

- **Resident row / 整行驻留**：输入、exp权重和临时量均可驻留UB时，输入读一次；
  先求max，再计算并保存exp权重、求sum，最后读取UB中的权重归一化。
- **Rereadable input / 输入可重读**：超UB长行可第一遍在线合并max/sum，第二遍读输入输出
  全概率行；这与逐遍分别求max、sum、输出的三遍实现是不同算法路线。
- **Output scratch / 输出暂存**：输入不可重读时，可把局部exp写入可回读的输出，并保留分块
  max/sum；最终回读输出按最终统计重缩放再写回。不能只靠常数max/sum状态直接输出整行。
  其首要价值是满足单读合同，不保证比允许重读的两遍路线更快。

### Traffic and compute / 搬运与计算分别核算

对N个FP32元素，忽略统计量等较低阶项：不缓存exp的三遍输入读+一次输出写约16N字节；
两遍读+一次写约12N；一次输入读+输出暂存的写、读、最终写约16N；整行驻留约8N。
这是逻辑外部流量估算，不是硬件计数器，也不是完整调用时延加速比的硬上限。

上述不缓存中间exp的三遍实现，在求sum和最终输出时各计算一次exp；驻留路线可保存并复用
权重，只计算一次exp。因此比较这两条完整路线时，同时改变了外部搬运、exp次数、UB访问
和循环/调度，不能把整次调用的加速全部归为存储收益，也不能仅从16N/8N推断总提速上限为2倍。
若要分别归因，需要额外控制计算/存储变量或用实际profile取证；本节不提供无对应归因实验的分项收益。

先核算UB、分块统计、输入可重读及输出暂存条件，再比较完整合法调用；少读一次输入不等于
减少总访存。具体时延和加速比须有覆盖实现与目标合同的可复核测量，本节不引述未随仓库保留的倍数。
