# Qwen3.5-9B NPU 迁移说明

| 字段 | 说明 |
|------|------|
| HuggingFace | Qwen/Qwen3.5-9B |
| 权重目录 | 由 `MODEL_PATH` 环境变量 / `--model-path` 指定（Qwen3.5-9B 权重目录） |
| 代码来源 | transformers 包 (built-in，使用 qwen3_5 架构) |
| 运行命令 | `python3 scripts/ask_Qwen3.5-9B.py --model-path <weights>` |
| transformers 版本 | 5.8.1（实测运行 / 与上方环境信息一致） |
| 代码位置 | 运行时：transformers 内置 `transformers.models.qwen3_5`；归档快照：`src/pypto_gym/transformers/qwen3_5_9b/{modeling,configuration}_qwen3_5.py` |
| 修改内容 | 运行时由 `ask_Qwen3.5-9B.py` 经 `sys.modules` 注入 + monkey-patch 接入 PyPTO（built-in `qwen3_5` 架构，`config.json` 无 auto_map）。归档快照在内置 qwen3_5 modeling 中嵌入同一 chunk 注入钩子（含 `query.device.type == "npu"` 守卫）+ 华为 NOTICE，供参考；保留上游相对导入，未改绝对导入 |

## 环境信息

| 字段 | 版本 |
|------|------|
| torch | 2.10.0 |
| torch_npu | 2.10.0 |
| torchvision | 0.25.0 |
| transformers | 5.8.1 |
| CANN | 9.0.0 |
| NPU | Ascend 910B3 |

> 下方性能对比在以上环境实测（transformers 使用内置 `qwen3_5` 架构，非 vendored 副本）。

## 已融合算子 (Fused operators)

| Operator | Toggle flag | Path |
|----------|-------------|------|
| `gated_delta_rule` | `USE_PTO_GATED_DELTA_RULE` | [`ops/pypto_tensor/qwen3_5_9b/gated_delta_rule/`](../../../ops/pypto_tensor/qwen3_5_9b/gated_delta_rule/) |

The fused operator replaces the chunk-prefill path in
`Qwen3_5GatedDeltaNet.forward` (linear-attention layers). The decode and
full-attention paths are unaffected.

`--use_pypto` 模式依赖运行时 `qwen3_5_9b_pto_kernels/` 适配层包：脚本在导入
`transformers` 之前 `sys.path.insert(0, model_path)` 并 `import
qwen3_5_9b_pto_kernels`。该适配层包预期部署在权重目录下
（`{weights_dir}/qwen3_5_9b_pto_kernels/`），由模型侧维护。

## 运行

```bash
# Baseline
python3 scripts/ask_Qwen3.5-9B.py --model-path <weights_dir>

# PyPTO fused
python3 scripts/ask_Qwen3.5-9B.py --model-path <weights_dir> --use_pypto
```

Benchmark:

```bash
# 整网 e2e GREEDY token generation（prefill→首 token 延迟 + decode tok/s + 峰值显存）
# 1 次 warmup generate + 1 次 measured generate；prompt "你好，请介绍一下自己。"，output_length=100
# 同时跑：自然 prompt（~16 tok）与合成 256-token 输入；逐 token streamer 拆分 prefill 与 decode
python3 scripts/bench_qwen3_5_9b.py --model-path <weights> [--use_pypto]
bash bench_qwen3_5_9b.sh   # eager + PyPTO 两阶段, 写 bench_baseline.json / bench_pypto.json
```

> ℹ️ `bench_qwen3_5_9b.py` 测的是**可部署**的整网 greedy 生成（真实模型、**无图捕获**），
> 反映端到端用户可感知延迟。融合 kernel 只加速 **prefill** chunk 路径——故 PyPTO 大幅降低
> **prefill / 首 token 延迟 (TTFT)**（~3.7–4.3x），而 decode（未被 PyPTO 替换、走上游 recurrent 路径）保持不变。

## 性能对比

下表为 **可部署 (deployable)** 数字：真实模型、**无图捕获**、greedy e2e 生成
（`bench_qwen3_5_9b.py`，output=100，1 次 warmup + 1 次 measured generate，单卡 910B3；实测环境见「环境信息」）。
prefill→首 token 延迟与 decode tok/s 由逐 token streamer 拆分。

### 整网 e2e（deployable，greedy 生成）
| 输入 | 模式 | prefill→首 token (ms) | decode (tok/s) | 加速 | 峰值显存 |
|------|------|----------------------|----------------|------|---------|
| 自然 prompt (~16 tok) | eager（torch chunk 回退） | 405.6 | 9.86 | 1.00x | ~18 GB |
| 自然 prompt (~16 tok) | **PyPTO**（融合 kernel） | **110.4** | 10.08 | **3.67x** | ~18 GB |
| 256-token 输入 | eager（torch chunk 回退） | 438.5 | 9.99 | 1.00x | ~18 GB |
| 256-token 输入 | **PyPTO**（融合 kernel） | **105.1** | 10.13 | **4.17x** | ~18 GB |

> **PyPTO 将 prefill / 首 token 延迟 (TTFT) 降低 ~3.7–4.3x：** eager 的 GatedDeltaRule 走纯 torch chunk
> 回退（FLA Triton 在 Ascend 不可用），即使短 prompt 的 prefill 也要数百毫秒；PyPTO 融合 kernel 替换该 prefill
> chunk 路径，故首 token 延迟大幅下降（自然 prompt 3.67x、256-token 输入 4.17x）。**decode（9B ~10 tok/s）保持不变**——
> 融合 kernel 只替换 prefill chunk 路径，decode 走未被替换的上游 recurrent 路径，两种模式相同。
> 因此 deployable 的用户可感知收益是**大幅降低 TTFT**，而稳态 decode 不受影响。

> **基线说明：** baseline 是纯 torch `chunk_gated_delta_rule` 回退（FLA 的 Triton/CUDA kernel 在 Ascend
> 不可用，是 Ascend 上的真实基线）。
>
> **Greedy 一致性说明：** eager 与 PyPTO 的 greedy generate 会产生**略有不同**的 token——融合 kernel 与
> torch chunk kernel 并非逐位一致，bf16 logit 微差会翻转 argmax 并在 100 步中累积。**计时有效**；逐 forward
> 的 logits 数值上仍非常接近。


## 归档映射

本集成在 pypto-gym 仓库中的文件布局（步骤 27 还原重建按此反向拷贝到权重目录）：

| 来源（模型部署目录 `{weights_dir}/`） | pypto-gym 归档位置 |
|---|---|
| `ask_Qwen3.5-9B.py`, `bench_qwen3_5_9b.py`, `bench_qwen3_5_9b.sh`, `README.md` | `modeling/transformers/qwen3_5_9b/` |
| `modeling_qwen3_5.py`, `configuration_qwen3_5.py`（带 PyPTO 钩子的内置 qwen3_5 归档快照）, `README.md` | `src/pypto_gym/transformers/qwen3_5_9b/` |
| `qwen3_5_9b_pto_kernels/gated_delta_rule/gated_delta_rule_impl.py`, `__init__.py`, `README.md` | `ops/pypto_tensor/qwen3_5_9b/` |
| `qwen3_5_9b_pto_kernels/gated_delta_rule/{gated_delta_rule_golden.py, test_*.py, test_cases.json}` | `tests/ops/qwen3_5_9b/` |

> 还原：按上表逆向拷贝到权重目录。模型为 built-in `qwen3_5` 架构——`config.json` 无 auto_map，
> 运行时加载 transformers **内置** qwen3_5 实现（本仓库的 `modeling_qwen3_5.py` 为带 PyPTO 钩子的
> 归档快照，默认不被 trust_remote_code 加载；如需让快照生效需在 config.json 配置 auto_map 指向它）。
> 然后 `ask_Qwen3.5-9B.py [--use_pypto]` 双模式验证。config.json 由权重目录提供，不单独归档。
