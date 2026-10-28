# 算子需求规范

## 机器合同

> 下列 JSON 是公式、公开接口、验收配置的唯一机器事实源。正文只解释语义与证据，不复制字段值。

```json machine-contract
{
  "schema_version": 1,
  "op_name": "{{OP_NAME}}",
  "formula": "{{FORMULA_OR_NUMBERED_STEPS}}",
  "supported_dtypes": [],
  "inputs": [],
  "outputs": [],
  "default_params": {},
  "tolerance": {
    "atol": null,
    "rtol": null
  },
  "dynamic_axes_ranges": {},
  "shape_constraints": [],
  "p0_cases": [],
  "perf_target": null,
  "exit_criteria": null
}
```

填写约束：

- `op_name` 使用 lower_snake_case；`supported_dtypes` 按首次出现顺序列出默认输入输出及全部 P0 case 实际使用的 canonical dtype。
- `formula` 用一个 JSON 字符串定义每个公开输出；简单算子写可审查的等式，复杂递推写编号伪代码步骤（换行转义为 `\n`），且每个输出都必须是赋值或箭头目标。不要只写自然语言标签。
- `inputs` / `outputs` 按公开签名顺序填写，每项包含 `name`、`shape`、`dtype`、`value_range`，可选布尔字段 `is_list`（缺省为 `false`）；rank-0 shape 写 `[]`，有限部分的闭区间值域写 `[min, max]`。某 P0 输入还包含非有限值时，该 case 增加 `input_special_values`，按输入名列出实际出现的 `"-inf"`、`"+inf"`、`"nan"`（例如 `{"x":["-inf","+inf"]}`）；不能用有限样本替换原任务的特殊值。
- shape 维度只使用正整数，或由符号、整数、`+ - * //` 和括号构成的表达式；每个动态符号须在至少一个输入 shape 中作为独立维度出现，并在 `dynamic_axes_ranges` 中给出正整数闭区间。公开接口确实跨 rank 时可将该 tensor 的顶层 shape 写成单独的 `["..."]`，表示 rank 多态；此时 P0 case 仍须给出逐案完整具体 shape，不能由此声称覆盖未声明的 rank/shape。输出形状语义仍须由公式和独立 Golden 验证。
- `default_params` 只放公开签名中的标量默认参数，顺序与签名一致；无默认参数时写 `{}`。
- 采用 Pro `precision_compare.py` 的方案 A 时，`tolerance` 填 `{"policy":"pro_scheme_a"}`；采用已确认的逐元素容差时填写 `atol/rtol`。两种格式不可混用。
- `p0_cases` 至少一项，每项包含 `name`、`params`、`input_shapes`、`output_shapes`，可增加成对的 `input_dtypes`、`output_dtypes`，以及非有限输入的 `input_special_values`。dtype 映射必须按声明顺序覆盖全部输入/输出；缺省时继承顶层 tensor dtype。首项 `params` 等于 `default_params`，首项 dtype 等于顶层默认值；每个 shape 都是公式代入后的具体整数数组。同一公开接口的多 dtype case 放在同一 SPEC。
- TensorList 输入/输出显式声明 `is_list: true`：顶层 `shape` 是成员约束，异构 shape 可用 `["..."]`；P0 的 `input_shapes[name]` / `output_shapes[name]` 是非空成员 shape 数组，例如 `[[8,64],[16,64]]`，不能编码为 `[arity, ...]`。每个列表的成员 dtype 相同、rank 相同且至少为 1，连续且在同一设备；公共 `[min,max]` 值域逐成员生效。成员 shape 中的符号遵循已有共享符号规则；跨成员不同的维度通过 `["..."]` 和具体 P0 shapes 表达。列表特殊值通过 `input_special_values[name]` 声明，按整个列表所有成员的并集核对，有限值域仍逐成员检查。单个 TensorList 输出作为一个公开输出，不能按列表长度拆成多个输出。非有限 scalar 使用 JSON 字符串 `"inf"` / `"+inf"` / `"-inf"` / `"nan"`，作为数值参数传给 Golden 和公开 wrapper；kernel 使用显式浮点 scalar 形参，勿在 VF 内直接写裸 `inf` / `nan`。当前仍不支持空列表；未导出的长度/成员 shape 不属于交付范围。
- 不适用的可选字段使用空数组、空对象或 `null`，不要保留示例值或另建第二份机器表格。

## 语义说明

### 1. 功能与分类

{{DESCRIPTION_AND_CATEGORY}}

### 2. 公式符号与依据

{{FORMULA_NOTATION_AND_RATIONALE_WITHOUT_REPEATING_MACHINE_FIELD}}

### 3. 算法描述

{{ALGORITHM_OR_NOT_APPLICABLE}}

### 4. 数据流说明

{{DATAFLOW}}

### 5. 接口语义

{{INTERFACE_SEMANTICS_WITHOUT_REPEATING_MACHINE_FIELDS}}

### 6. 功能与可选参数依据

{{FEATURE_AND_OPTIONAL_PARAM_RATIONALE}}

### 7. 精度语义

{{PRECISION_SEMANTICS_WITHOUT_REPEATING_TOLERANCE}}

### 8. 动态 Shape 与约束依据

{{SHAPE_RATIONALE_WITHOUT_REPEATING_MACHINE_FIELDS}}

### 9. 边界条件处理

{{ZERO_EXTREME_NAN_INF_BEHAVIOR}}

### 10. P0 与性能目标依据

{{P0_AND_PERF_RATIONALE_WITHOUT_REPEATING_MACHINE_FIELDS}}

### 11. 参考与来源

{{SOURCES_AND_CONFIDENCE}}

### 12. 自动决策

{{DECISIONS_OR_NOT_APPLICABLE}}

---
*生成时间: {{TIMESTAMP}}*
*确认状态: {{CONFIRMATION_STATUS}}*
