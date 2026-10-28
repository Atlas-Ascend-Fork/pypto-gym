---
name: pypto-pro-golden-generate
description: 为 PyPTO-Pro 算子生成、规范化和验证 golden 参考实现。根据已校验 SPEC 产出 NPU torch/torch_npu `{op}_golden.py` 与 CPU `{op}_golden_cpu.py`；CPU 浮点参考使用 FP32。仅在调用方显式传入 `collect_golden_perf=true` 时采集 NPU 性能。既可由编排器调用，也可直接响应自然语言或既有参考，适用于“生成 golden / 参考实现 / 验证基准 / golden.py / normalize PyTorch 或 NumPy reference”等请求；不用于架构设计、kernel 实现或性能调优。
---

# PyPTO-Pro Golden 参考实现

负责 Stage 2 的数学参考实现，不决定 Module 划分、kernel 写法或优化策略。

## 输入、路径与交付

支持两种入口：

- 编排模式：接收已经通过 canonical validator 的 `custom/<op>/SPEC.md`，交付目录固定为 `custom/<op>/`。
- 单独使用：可接收已校验 SPEC、自然语言需求或既有 PyTorch/NumPy 参考。没有已校验 SPEC 时，先加载 `pypto-pro-intent-understand` 创建并校验 SPEC；信息不足时由该 skill 澄清或返回 `blocked`，本 skill 不猜测合同。

下文以 `<spec-path>` 表示已校验 SPEC，以 `<op-dir>` 表示交付目录。编排模式下二者分别为 `custom/<op>/SPEC.md` 和 `custom/<op>/`；单独使用时 `<op-dir>` 可由用户指定，否则取 `<spec-path>` 的父目录，不存在时先创建该目录。

SPEC 中唯一的 JSON machine-contract 是 Stage 2 的机器事实源。至少确认：

- 算子名称和数学公式；
- 全部输入、输出、参数及默认值；
- dtype、shape、边界条件和动态轴范围；
- 全部 P0 配置。

machine-contract 缺字段、类型错误或 P0 不完整时停止并退回修正 SPEC；不要从正文、动态范围或经验补造数学语义、参数、shape 或 P0。

编排模式直接复用已通过的 Stage 1 产物，不重复检索 devkit。仅当生成 Golden 所需的明确
数学事实缺失时，才按 [reference-normalization.md](references/reference-normalization.md) §1
检索仓内 PyTorch/NumPy 参考；仍不足则返回 `stage1_material_gap`，列明缺口和已核对证据。

Stage 2 必须交付并验证：

| 文件 | 职责 |
|---|---|
| `<op-dir>/<op>_golden.py` | torch + torch_npu 的 NPU 数学参考、输入工厂和自验证 |
| `<op-dir>/<op>_golden_cpu.py` | 纯 torch 的 CPU FP32 浮点参考和自验证 |
| `<op-dir>/GOLDEN_VALIDATION.json` | 与 SPEC、两份 golden SHA-256 绑定的单次自验证回执 |

`GOLDEN_PERF_REPORT.md` 仅在 `collect_golden_perf=true` 时交付。单独使用时，只有用户明确要求采集 NPU golden 性能才将该开关视为 `true`。开关缺失或为 `false` 时不得运行 profiling，也不得把报告列为门禁；SPEC 中存在性能 P0 或用户要求高性能实现本身不能开启该开关。

## 工作流

### 1. 选择生成路径

按 [references/reference-normalization.md](references/reference-normalization.md) §1 选择并规范化参考；没有可用参考时只按 SPEC 生成。

使用 skill 目录中的固定模板和脚手架，不手工重建文件骨架：

```bash
python <skill-dir>/scripts/gen_golden_scaffold.py \
  --spec <spec-path> \
  --template <skill-dir>/templates/golden-template.py.tmpl \
  --out <op-dir>/<op>_golden.py
```

脚手架通过 Intent Understand 的 `load_spec_contract()` 读取并验证该合同，将校验后的 `formula` 内容写入生成源码的 `_SPEC_FORMULA`，并按合同顺序生成全部 `p0_cases` 的签名、shape、参数和 case 名。合同无效或字段缺失时会非零退出；先修正 SPEC，再继续。

参考源与交付目标同名时，按 [reference-normalization.md](references/reference-normalization.md) 的候选流程处理。其他已存在的交付文件须先取得用户覆盖确认再写入；未确认则保留原文件并返回 `blocked`。

### 2. 实现 NPU golden

在 `<op>_golden.py` 中遵守以下合同：

- 只用 `torch` 和 `torch_npu` 表达数学计算；禁止 import `pypto` 或 `pypto_pro`。
- 函数名为 `<op>_golden`；参数顺序、可选性、默认值和返回值与 SPEC 一致，SPEC 中的每个参数都必须实现。dtype 跟随输入，不新增 dtype 参数。
- 优先采用语义等价的稳定 PyTorch API；没有等价 API 时严格按 `formula` 实现，不添加 SPEC 未定义的修正项。
- 每次 `torch.matmul` 都先把两个输入转为 FP32，例如 `torch.matmul(a.float(), b.float())`；如输出合同要求低精度，再转换输出 dtype。
- 将 SPEC 的零值、边界值、mask、索引、整除关系等语义写进数学实现或输入前置条件。

保留模板的设备逻辑：

- 通过 `TILE_FWK_DEVICE_ID` 选择卡号，未设置时使用 0；禁止硬编码 `npu:0` 或其他卡号。
- `torch_npu` 未安装时直接报错并提示安装。
- NPU 不可用或 `torch.npu.device_count() == 0` 时回退 CPU。
- 输入由 golden 函数迁移到目标 device；测试输入直接创建在同一 device 上。

### 3. 实现 `_make_inputs(device)`

每个 NPU golden 必须导出 `_make_inputs(device)`，且 `_validate()` 必须复用它。函数签名顺序决定位置参数；标量和非 tensor 参数放入 `kwargs`。

单 case 返回：

```python
return [tensor_arg_1, tensor_arg_2], {"scalar_arg": value}
```

多 case 返回：

```python
return [
    ("perf_p0_small", [tensor_arg_1, tensor_arg_2], {"scalar_arg": value}),
    ("perf_p0_large", [tensor_arg_1_b, tensor_arg_2_b], {"scalar_arg": value_b}),
]
```

构造输入时：

- 保留脚手架生成的全部合同 P0；不得删除、改名、重复或跳过 case，`_validate()` 必须拒绝缺失、额外或重名 case。单个 P0 可使用单 case 格式。
- 所有 tensor 创建时带 `device=device`，shape、dtype 和标量值来自同一 machine-contract。
- `p0_cases[].input_special_values` 列出的每一种 `-inf`、`+inf`、`nan` 都须实际出现在对应 P0 输入；顶层有限 `value_range` 只约束其余有限元素。自验证记录特殊值数量，不能用有限样本替代原始 case。
- 普通数值可用 `torch.randn`；正值、非零值、合法索引等按值域使用 `torch.rand`、变换或带有效上下界的 `torch.randint`。
- 多 tensor 依赖、状态缓存、位置编码和整除关系必须联合构造；状态 tensor 按语义初始化，不以任意随机值代替。
- 不让 `_validate()` 和 profiling 各自构造另一套 P0 输入。

### 4. 验证 NPU golden

按“全部合同 P0 → 已确认的性能/功能 P1 与可选分支 → 动态轴泛化 case”的顺序验证。在 `_validate()` 中覆盖：

1. machine-contract 中的全部 P0；
2. SPEC 已冻结的每项性能/功能 P1 和每个可选参数分支至少一个合法 case；这些是 `_validate()` 的额外检查，不得加入 `_make_inputs()` 的 exact-P0 返回列表；
3. 每个动态轴的 low/mid/high（其余轴固定为已确认的合法代表值），并覆盖 all-low、all-high 和跨轴约束所需的合法组合；约束使边界组合非法时改用对应边界上的合法组合并说明，不生成无意义的完整笛卡尔积；
4. 输出 shape、dtype/接口合同和 NaN/Inf；
5. 公式可推导的值域、边界/特殊点和数学属性；
6. 存在独立 PyTorch 等价 API 或既有参考时的数值对比。

P1 或可选分支缺少可执行的已确认值时，先退回 `pypto-pro-intent-understand` 补全 SPEC，不自行填值。容差必须与 dtype 和算子数值特性一致。验证失败时定位数学、输入或合同根因并修复；不要用放宽容差、吞掉异常或加入未定义 epsilon 掩盖失败。同一根因连续 3 次失败且没有新证据时停止重试，报告失败命令、原始错误和已尝试修复。

复杂公式没有独立 PyTorch API 或既有参考作为 oracle 时，在交付结果中标记 `semantic_review_required` 并说明缺少何种对照；该标记不替代自验证，也不使用星级置信度。

不得以 `exec(open(...).read())`、零散调用或额外 cross-check 命令替代自验证；需要的数值对比
必须写入该文件的 `_validate()`，由第 6 步统一执行一次。

### 5. 生成并验证 CPU golden

使用 [templates/golden_cpu_template.py.tmpl](templates/golden_cpu_template.py.tmpl)，
从已完成数学实现和自验证代码的 NPU golden 复制相同数学逻辑并进行以下适配：

- 函数名为 `<op>_golden_cpu`，签名、参数语义和计算公式与 NPU golden 一致。
- 只 import `torch`；移除 `torch_npu`、NPU device 选择和 `.to(npu)`。
- FP16/BF16 浮点输入提升到 FP32 计算并返回 FP32，不在末尾降回输入 dtype。
- 保留整数、布尔和索引参数的语义及 dtype。

CPU 自验证至少覆盖 FP32/低精度浮点输入可运行、输出为 FP32、shape 正确且无 NaN/Inf，
并由下一步统一执行。

### 6. 单次执行并生成回执

两份文件完成后，只通过以下固定 runner 验证：

```bash
TILE_FWK_DEVICE_ID=<id> python <skill-dir>/scripts/validate_golden_once.py --op-dir <op-dir> --spec <spec-path>
```

未指定卡号时省略环境变量；`--spec` 缺省时读取 `<op-dir>/SPEC.md`。相同 SPEC 与文件哈希下，
成功项只复用回执，不再次执行；失败项修正文件后自动重试，仅环境问题且文件未变时显式传
`--retry-failed cpu|npu|all`。两份 golden 的 exit code 均为 0 且源码在执行期间未变，
回执的 `--check` 才能通过。

### 7. 按开关处理 profiling

- `collect_golden_perf=false` 或缺失：跳过，继续完成 Stage 2。
- `collect_golden_perf=true`：先保证 NPU golden 自验证通过，再阅读并执行 [references/profiling.md](references/profiling.md)。单独使用时沿用本 skill 的显式开关定义，并把 reference 命令中的 `custom/<op>/` 换成实际 `<op-dir>/`；采集流程和判定不变。多个 P0 必须逐 case 采集。
- `profile-only=true`：仅在 `collect_golden_perf=true` 且
  `python <skill-dir>/scripts/validate_golden_once.py --op-dir <op-dir> --spec <spec-path> --check` 通过时执行
  profiling；不得重写或重新验证 golden。

Profiling 是 Stage 2 对 NPU golden 的独立可选能力；采集逻辑只存在于 `scripts/profile_golden.py`，不得写入 golden 文件，也不绑定任何下游阶段的协议。

## 完成条件

交回上层前确认：

- 两份 golden 均存在，数学公式、函数签名和全部参数一致；
- NPU golden 仅使用 torch/torch_npu，CPU golden 仅使用 torch；
- `_make_inputs()` 覆盖全部合同 P0，并满足所有输入约束；
- `_validate()` 额外覆盖已冻结的性能/功能 P1、可选参数分支和必要的动态轴组合；
- 单次验证回执与当前 SPEC、两份 golden 哈希一致，且 `--check` exit code 为 0；
- 验证覆盖 shape、finite、适用的 API 对比、边界和数学属性；
- 只有 `collect_golden_perf=true` 时才存在有效性能报告。

只报告实际产物、执行命令、exit code 和未解决的证据；不要用主观置信度替代验证结果。
