# PyPTO-Gym Tests

本目录包含 pypto-gym 的算子测试与测试公共层。Tensor 算子和部分 Pro 算子在 `tests/ops/` 下测试；带 `op_kernel/` 的 Pro 算子也会将测试放在算子自己的 `tests/` 中，需要显式指定路径。例如：

| Kernel 实现 | 对应测试 |
|---|---|
| [ops/pypto_tensor/arctic/sum_lstm.py](../ops/pypto_tensor/arctic/sum_lstm.py) | [ops/arctic/test_sum_lstm.py](ops/arctic/test_sum_lstm.py) |
| [ops/pypto_tensor/qat/qat_impl.py](../ops/pypto_tensor/qat/qat_impl.py) | [ops/qat/test_qat.py](ops/qat/test_qat.py) |
| [ops/pypto_tensor/glm_v4_5/glm_*_impl.py](../ops/pypto_tensor/glm_v4_5/) | [ops/glm_v4_5/test_glm_*.py](ops/glm_v4_5/) |
| [ops/pypto_tensor/deepseek_v32_exp/*_impl.py](../ops/pypto_tensor/deepseek_v32_exp/) | [ops/deepseek_v32_exp/test_*.py](ops/deepseek_v32_exp/) |
| [ops/pypto_tensor/qwen3_1_7b/qwen3_*.py](../ops/pypto_tensor/qwen3_1_7b/) | [ops/qwen3_1_7b/test_*.py](ops/qwen3_1_7b/) |
| [ops/pypto_tensor/qwen3_next/gated_delta_rule_impl.py](../ops/pypto_tensor/qwen3_next/gated_delta_rule_impl.py) | [ops/qwen3_next/test_gated_delta_rule_qwen3_next.py](ops/qwen3_next/test_gated_delta_rule_qwen3_next.py) |


## 运行方式

```bash
# 运行指定模型的所有算子
pytest tests/ops/arctic -v --forked

# 运行单个算子用例文件
pytest tests/ops/glm_v4_5/test_glm_gate.py -v

# 指定 NPU 设备
pytest tests/ops/arctic -v --forked --device 0
```

## 标记（markers）

- `@pytest.mark.soc("950", "910")` — 指定 SoC 版本。未标注等同于 `"910"`。
- `@pytest.mark.world_size(N)` — 指定所需 NPU 卡数，默认 1。

SoC 筛选、耗时估计重排序等调度逻辑定义在顶层的 [conftest.py](../conftest.py)。

## PyPTO-Pro 样例与仓内算子的联动

以 [pypto-pro-kernel](../cannbot-skills/ops/pypto-pro-kernel/SKILL.md) 选择完整样例，以 [性能卡片索引](../cannbot-skills/ops/pypto-pro-op-perf-tune/references/knowledge-cards/index.md) 查询机制与适用条件。先对齐两侧的公开参数、形状、dtype、默认值、布局和精度判据；共享名称不代表支持域相同。

通过 [op_caller](common/op_caller.py) 的 `get_op_caller` 获取公开入口的测试支持 `--op-entry OP=FILE.py:CALLABLE`。文件路径相对执行 pytest 时的目录解析，可重复指定不同算子；只在 JIT 模式生效。候选按独立包名加载，支持包内相对导入，不用同名 `kernel.py` 覆盖其他样例。加载失败、入口不可调用或进程内源码变化会报错，不回退到默认实现。请求的候选未被测试加载时，即使用例通过，正常执行仍以非零状态退出；`--collect-only` 只报告未加载，不要求执行。pytest 摘要显示加载的文件、公开入口与 SHA256；辅助模块也须纳入执行门禁的源码集合。

Engram Gate 保持仓内 CPU golden、输入生成与方案 A 精度标准，只替换公开调用：

```bash
# 仓内实现：五个算子用例与比较器失败路径校准
python -m pytest ops/pypto_pro/deepseek_v4_1/engram_gate/tests -q

# 相同测试与判据，调用 skill 样例的公开入口
python -m pytest ops/pypto_pro/deepseek_v4_1/engram_gate/tests -q --op-entry engram_gate=cannbot-skills/ops/pypto-pro-kernel/examples/algorithms/engram_gate/kernel.py:engram_gate_wrapper

# AttnEpilogue 的公开输出边界测试使用相同候选入口协议
python -m pytest ops/pypto_pro/deepseek_v4_1/attn_epilogue/tests -q -k test_token_boundaries --op-entry attn_epilogue=cannbot-skills/ops/pypto-pro-kernel/examples/algorithms/attn_epilogue/kernel.py:attn_epilogue
```

AttnEpilogue 的内部 workspace、tiling 和图缓存测试依赖仓内模块的私有接口，继续验证该模块；比较候选时显式选择上述公开边界测试。FlexAttention 已通过 `get_op_caller` 获取入口，也可使用相同参数；候选仍须满足它的公开接口。其它直接导入实现的测试需先接入公共调用层，参数不会全局替换 Python 导入。

板上执行这些 pytest 命令时，使用核心 skill 的 `check.py run --source ... -- <pytest command>`，声明候选、仓内实现、测试及 golden、公共调用层和相关 conftest。设备、环境与锁由执行环境配置提供。卡片查询可使用 `--cards-root cannbot-skills/ops/pypto-pro-op-perf-tune/references/knowledge-cards`；卡片与静态建议只提供有条件的实验方向，不能替代当前实现的数值验证与原生计时。

公共调用层可在无 NPU 主机上独立回归，隔离根目录的设备 fixture：

```bash
python -m pytest -c /dev/null --confcutdir=tests/common -p no:cacheprovider tests/common -q
```

## 添加新测试

参考 [README 的 "添加新算子" 小节](../README.md#添加新算子)。
