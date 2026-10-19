# PyPTO-Gym Tests

本目录是 pypto-gym 的**测试目录**，与 `ops/pypto_tensor/` 下的算子实现一一对应，例如：

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

## 添加新测试

参考 [README 的 "添加新算子" 小节](../README.md#添加新算子)。
