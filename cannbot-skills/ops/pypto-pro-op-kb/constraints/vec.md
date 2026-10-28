# Vector authoring constraints

## Applies to

PyPTO-Pro vector sections implemented with tile operations or
`@pl.vector_function` / `vf.*`.

## Conditional authoring rule

Select the level that is correct and supported by the installed API:

- Use tile operations when they express the complete dataflow and their runtime
  valid-shape semantics cover the tail.
- Use `vf.*` when explicit lane masks, register-level control, or an instruction
  composition unavailable at tile level is required.
- When both are legal, measure both on the target before choosing for
  performance.

No operator family, shape, or prior benchmark makes either level the universal
default.

## Shared correctness requirements

1. A row reduction writes a row scalar, conventionally a `[rows, 1]` result,
   which is broadcast back over the row by the matching expand operation.
2. Padding lanes must be excluded from reductions and stores.
3. Accumulation dtype follows the numerical contract; use FP32 unless a
   documented, validated alternative is allowed.
4. Input, output, reduction, and workspace tiles must fit the detected
   platform's vector-memory budget.
5. Compare/select mask dtype, layout, offset units, and supported operations are
   API-version-specific. Confirm them in the installed documentation.

## Examples

Tile-operation row softmax:

```python
pl.row_max(row_value, input_tile, workspace)
pl.row_expand_sub(output_tile, input_tile, row_value)
pl.exp(output_tile, output_tile)
pl.row_sum(row_value, output_tile, workspace)
pl.row_expand_div(output_tile, output_tile, row_value)
```

Vector-function load and mask placement:

```python
register = vf.load_align(input_tile, offset)
accumulator = vf.add(accumulator, register, predicate)
result = vf.reduce_sum(accumulator, predicate)
vf.store_align(output_tile + offset, result, predicate)
```

`vf.load_align(input_tile, offset, predicate)` is not a supported mask
placement in the retained guidance.

## Evidence

- Tile-operation implementation and golden:
  [softmax_impl.py](../examples/samples/softmax/softmax_impl.py) and
  [softmax_golden.py](../examples/samples/softmax/softmax_golden.py)
- Vector-function implementation:
  [vf_softmax_impl.py](../examples/samples/vf_vs_tileop/vf_softmax_impl.py)
- Conditional selection and source order:
  [vf-reduction-perf.md](../../pypto-pro-op-develop/references/vf-reduction-perf.md)

The installed `$PYPTO_DEVKIT_DIR/docs/pypto_pro/api/` documentation remains the
primary source for signatures and target-version behavior.

## 小输入广播

当输入为 `[M,N]` 与 `[M,1]` / `[1,N]` 时，让小输入在消费点复用，避免物化满尺寸副本。
零步长 GM 视图载入满 Tile 仍占用满 Tile 的 UB，不能据此认定减少了内存或搬运。
VF 的标量广播、加载方式和物理布局须按目标接口确认，不沿用 Tile `expand_*` 的维度约定。
基础用法不另建卡片；旧广播样例已归档，不再作为新版候选。

### 任意切分轴的广播寻址

广播的切分轴可以不在尾轴。让 lane 对应保留轴，被广播轴在循环中展开；被广播的向量在寄存器中
加载一次，跨整条切分轴复用；每一块按切分轴起点重算源/目的偏移与尾块掩码。超长尾轴须先按
物理容量分块，再逐块广播。以下是 `y = x * col`（`col` 为 `[1,N]`，按行展开）的最小形式：

```python
mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
col_reg = vf.load_align(col_tile, [0, 0])
for row in pl.range(rows):
    xv = vf.load_align(xtile, [row, 0])
    yv = vf.mul(xv, col_reg, mask)
    vf.store_align(ytile, yv, mask, [row, 0])
```

推广到非尾轴切分或多轴同时广播时，必须显式推导每块的行起点、列起点、lane 映射与尾块掩码，
不能把二维广播示例当作通用方案。该片段是寻址规则示意；本库样例未覆盖该泛化，
不据此声称任意切分轴或完整算子已经验证。

## 比较谓词与选择

VF 比较返回谓词寄存器；`vf.select(src0, src1, predicate)` 真取 `src0`，假取 `src1`。
谓词的有效位布局须匹配**消费指令读取谓词的步长/元素宽度**，不能把它理解成谓词寄存器
具有不可转换的数据类型；普通bool/UINT8数据也不能直接当作谓词寄存器。
MaskReg的固定物理位数、跨宽度位重排和证据范围见[谓词位宽转换](vec-mask-width.md)。
声明一个不同dtype的目标寄存器不等于转换已有谓词位；构造/更新掩码仍须遵循目标版本
`create_mask`/`update_mask`的dtype与pattern合同，不能由“声明是bookkeeping”推导出这些参数均可忽略。
旧Tile接口中的独立UINT8掩码Tile、FP32临时区及参数顺序，不能直接套到VF。

接口：`vf.create_mask`、`vf.update_mask`、`vf.select`、`vf.mem_bar`。

## Tiled transpose candidates / 分块转置候选

先按 tile 资源而非全张量大小判断能否驻留；非连续输入也可先连续跨度 DMA、UB 解交错，
再片上转置。按物理 stride、storage offset 与真实尾块验证地址；SIMT 与分块候选均合法时，
比较完整搬运、解交错、转置及写回成本，不能按 noncontiguous 或全张量超 UB 直接排除后者。

本节保留完整成本比较规则，不给出某形状的固定阈值或无可追溯材料的加速倍数。
SIMT是否可用由目标SDK决定；历史vector-only前提不能推广为所有版本的限制。
布局和容量约束参见[内存布局](memory-layout.md)；API以目标安装包的官方声明为准。
