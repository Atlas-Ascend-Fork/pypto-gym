# Kimi-Linear-48B-A3B-Instruct — NPU 迁移说明

| 字段 | 说明 |
|------|------|
| HuggingFace | moonshotai/Kimi-Linear-48B-A3B-Instruct (`KimiLinearForCausalLM`) |
| 权重目录 | `/data/models/Kimi-Linear-48B-A3B-Instruct`（可通过 `--model-path` 指定） |
| 代码来源 | trust_remote_code — `modeling_kimi.py` / `configuration_kimi.py` 由 HuggingFace 仓库自带（config.json 的 auto_map） |
| 运行命令 | `python3 scripts/ask_Kimi-Linear-48B-A3B.py --num_npus 4` |
| transformers 版本 | 5.8.1（模型面向 4.57.1 — 已适配） |

## 已融合算子 (Fused operators)

| Operator | Toggle flag | Path |
|----------|-------------|------|
| `kda_fused_prefill` (prefill) | `USE_PTO_KDA` | [`src/pypto_gym/ops/pypto_tensor/kimi_linear_48b_a3b/kda/`](../../../ops/pypto_tensor/kimi_linear_48b_a3b/kda/) |
| `kda_fused_decode` (decode, T=1) | `USE_PTO_KDA` | 同上 |

The prefill fused operator replaces the entire KDA subgraph of the `mode == 'chunk'`
branch in `KimiDeltaAttention.forward` (conv1d → gate → beta → chunk core → o_norm,
via `_try_kda_fused_prefill`). Decode (T=1) takes the `kda_fused_decode` path — a
full-layer fusion including the projections (`_try_fused_decode`) — when `USE_PTO_KDA`
is enabled and the conv/recurrent states have been seeded by prefill; graph mode goes
through `StaticDecodeCache` from `kda_decode_graph.py`. For shapes outside the operator
envelope, `NotImplementedError` → fallback to the torch reference path.

`--use_pypto` 模式依赖运行时 `kimi_linear_48b_a3b_pto_kernels/` 适配层包：脚本在导入
`transformers` 之前 `sys.path.insert(0, model_path)` 并 `import
kimi_linear_48b_a3b_pto_kernels`，置 `USE_PTO_KDA = True`。该适配层包预期部署在权重目录下
（`{weights_dir}/kimi_linear_48b_a3b_pto_kernels/`），由模型侧维护。

> **缓存同步（cache-sync）N/A：** 本集成在导入 `transformers` **之前**经
> `sys.path.insert(0, model_path)` + `sys.modules` 注入适配层包，modeling/算子均从该
> 路径（部署目录或仓库）直接加载，**不经过** HuggingFace 的 `~/.cache/.../transformers_modules`
> trust_remote_code 模块缓存。因此 skill 的“缓存同步”步骤（把 pto_kernels 拷进 HF modules
> 缓存以使本地改动生效）在此**不适用**——改动适配层后直接重跑即可，无需清缓存。

## 多卡分片 (Multi-NPU sharding)

模型 ~96 GB bf16（>64 GB/卡），**必须** 跨 >=2 卡。ask 脚本基于 `--num_npus`（默认 4）
构造均衡 `device_map`：`embed_tokens` + 早期层在 npu:0，27 个 `KimiDecoderLayer` 按连续块
分布，`norm` + `lm_head` 在最后一卡。经 accelerate (device_map) 加载，4 卡峰值 HBM ~22-25 GB/卡。

## 运行

```bash
# Baseline
python3 scripts/ask_Kimi-Linear-48B-A3B.py --model-path <weights_dir> --num_npus 4

# PyPTO fused (KDA prefill + decode)
python3 scripts/ask_Kimi-Linear-48B-A3B.py --model-path <weights_dir> --num_npus 4 --use_pypto
```

Benchmark（两段式 baseline vs pypto，输出对比表格）:

```bash
bash bench_Kimi-Linear-48B-A3B.sh
```

## 环境信息

迁移与端到端验证所用环境（Ascend 910B 容器，`python3.11.15`）：

| 组件 | 版本 |
|------|------|
| torch | 2.10.0+cpu（Ascend：基础 torch 走 CPU wheel，NPU 算子由 torch_npu 提供） |
| torch_npu | 2.10.0 |
| torchvision | 0.25.0+cpu |
| transformers | 5.8.1（HF 模型面向 4.57.1 — 已在 `modeling_kimi.py` 适配，见 transformers README 适配补丁） |
| CANN | 9.0.0 |
| NPU | Ascend 910B3 |

## 性能对比

> 🟢 **可部署性能（真实路由、输出正确）= PyPTO 1.38x** —— 见下『全模型实测』表。
> ⚠️ 后面『图捕获吞吐代理』一节的**所有 graph / static-route 数字均为理想化代理**：用静态**假路由**换取可捕获性，
> **输出不正确、不是可运行模式**，仅用于量化 host 启动开销上限与「未来 PyPTO MoE kernel」可得的收益。
> **覆盖率前提：** PyPTO 的 JIT kernel 每进程绑定一张 NPU。单进程
> `ask_Kimi-Linear-48B-A3B.py`（device_map）只覆盖约 **6/20** 层（见脚本顶部注释与
> 运行时 warning）；**全量（20/20）覆盖需每卡一进程流水线** —— 仓库内
> `bench_kimi_multinpu.py`（torchrun）。

### Full-model prefill (4 NPUs / graph)

`bench_kimi_multinpu.py`, all 27 layers (20 KDA) / 4 NPUs with one process per card /
real checkpoint / bf16 / `--graph` (the static route makes the MoE segment capturable
by NPUGraph):

| Config | Old `kda_chunk` operator (removed) | ling `chunk_kda` operator | **`kda_fused_prefill` (this repo)** | vs old | vs ling |
|-------|-----------------------------------:|--------------------------:|------------------------------------:|:------:|:-------:|
| seq=300, fixed-178 | 313.9 ms | 190.7 ms | **171.4 ms** | **1.83x** | **1.11x** |
| seq=2048, uniform-256 | 1131.1 ms | 425.6 ms | **385.2 ms** | **2.94x** | **1.11x** |

> The old `kda_chunk` was the standalone PyPTO chunk operator that predated this fused
> operator (since removed); ling is the `ling_3_0_flash/chunk_kda` PyPTO operator run
> on the same bench as a reference. All three use the same static route and the same
> graph configuration, so the numbers isolate the contribution of the KDA kernel itself.

**复现：**
- seq=300: `MODEL_PATH=<weights> torchrun --nproc_per_node=4 bench_kimi_multinpu.py --mode prefill --pypto --graph --route fixed --seq 300 --report-file out.json`
- seq=2048: 同上，`--route uniform --seq 2048`
- harness smoke（随机权重，无需 checkpoint）：
  `torchrun --nproc_per_node=2 bench_kimi_multinpu.py --random-weights --layers 4 --seq 80 --iters 2 --pypto`
- 单进程快速检查（~6/20，*不会*得到上面的全覆盖数字）：
  `python3 scripts/ask_Kimi-Linear-48B-A3B.py --model-path <weights> --num_npus 4 --use_pypto`
- 算子级 msprof：`MODEL_PATH=<weights> bash prof_Kimi-Linear-48B-A3B.sh`

### Full-model decode (4 NPUs / real routing)

| Mode | Baseline (torch) | Fused (PyPTO) | gain |
|------|-----------------:|--------------:|:----:|
| eager | 88.58 ms/token | **73.30 ms/token** | **−17%** |
| graph | 29.80 ms/token | **16.38 ms/token** | **−45%** |

**Reproduce:** `MODEL_PATH=<weights> torchrun --nproc_per_node=4 bench_kimi_multinpu.py --mode decode --pypto [--graph] --report-file decode.json`

## 归档映射

本集成在 pypto-gym 仓库中的文件布局（步骤 27 还原重建按此反向拷贝到权重目录）：

| 来源（模型部署目录 `{weights_dir}/`） | pypto-gym 归档位置 |
|---|---|
| `scripts/ask_*.py`, `bench_*.sh`, `prof_*.sh`, `sample_inputs.txt`, `README.md` | `modeling/transformers/kimi_linear_48b_a3b/` |
| `config.json` | `src/pypto_gym/transformers/kimi_linear_48b_a3b/config.json` |
| `modeling_kimi.py`, `configuration_kimi.py`, `kimi_fla_compat.py`, `kda_decode_graph.py`（trust_remote_code 打补丁版） | `src/pypto_gym/transformers/kimi_linear_48b_a3b/` |
| `kimi_linear_48b_a3b_pto_kernels/kda/*_impl.py`, `_device_guard.py`, `__init__.py`, `README.md` | `ops/pypto_tensor/kimi_linear_48b_a3b/` |
| `kimi_linear_48b_a3b_pto_kernels/kda/{*_golden.py, test/*}` | `tests/ops/kimi_linear_48b_a3b/` |

> 还原：按上表逆向拷贝到权重目录，确认 `config.json` 的 `auto_map` 指向部署的
> modeling，然后 `ask_Kimi-Linear-48B-A3B.py [--use_pypto]` 双模式验证。
