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
| `kda_chunk` (prefill) | `USE_PTO_KDA` | [`ops/pypto_tensor/kimi_linear_48b_a3b/kda/`](../../../ops/pypto_tensor/kimi_linear_48b_a3b/kda/) |

融合算子替换 `KimiDeltaAttention.forward` 中 prefill 的 `chunk_kda` 调用（KDA 线性注意力
层）。Decode（T=1 recurrent，走上游 torch `fused_recurrent_kda`）与全注意力 (MLA) 路径不受影响（仅 pto 化 chunk 路径）。

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

# PyPTO fused (KDA chunk)
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
> **切勿把代理数字（如 1.27x / 939 tok/s）当作可部署性能。** 唯一可部署的加速是 PyPTO **1.38x**（真实路由）。

> **覆盖率前提：** PyPTO 的 JIT kernel 每进程绑定一张 NPU。单进程
> `ask_Kimi-Linear-48B-A3B.py`（device_map）只覆盖约 **6/20** 层（见脚本顶部注释与
> 运行时 warning）；**全量（20/20）覆盖需每卡一进程流水线** —— 仓库内
> `bench_kimi_multinpu.py`（torchrun）。

### 隔离 KDA chunk kernel（单算子实测，Eager vs PyPTO）

KDA chunk kernel 本地单算子 bench（prefill，avg-of-20，与 torch golden 对比）：

| 算子 | Eager (µs) | PyPTO (µs) | 加速比 | 精度（max\|abs diff\| vs torch golden） |
|------|-----------:|-----------:|:------:|----------------------------------------|
| `kda_chunk` (prefill, T=256) | 29,649 | **6,982** | **4.25x** | 3e-5（正确） |

> **输入形状：** q,k,v `[B,T,H,D]` bf16；g `[B,T,H,D]` fp32（逐通道门控）；beta `[B,T,H]` fp32；
> B=1, T=256, H=32, D=128；`initial_state=None`；`scale=D**-0.5`。
> **kernel 加速比随序列长度增长** —— T=1024 时为 6.18x（chunk 算子规模越大融合收益越高）。

### 全模型实测（committed bench，可由仓库脚本复现）

`bench_kimi_multinpu.py`，**全 27 层 / 4 卡每卡一进程 / seq=256 prefill / 真实 checkpoint / bf16**
（1 warmup + 1 measured，真实路由、输出正确，`/data/public_models/Kimi-Linear-48B-A3B-Instruct`）：

| 模式 | prefill wall (ms) | 吞吐 (tok/s) | KDA 覆盖 |
|------|-------------------|-------------|----------|
| baseline（torch chunk） | 1844.7 | 138.8 | 0 pypto / 0 fallback |
| **pypto** | **1335.4** | **191.7** | **20/20 层走 PyPTO，0 fallback** |
| **加速比** | **1.38x (wall)** | **1.38x** | |

> **计时口径：** wall (ms) 取自每卡一进程流水线的最慢 rank（1 warmup + 1 measured，measured iter 前
> `synchronize()+barrier()`）；吞吐 tok/s = seq(256) ÷ wall(s)。

→ 每卡一进程下 **0 fallback / 全 20/20 KDA 覆盖**（真实权重，非 smoke）。这是**唯一可部署**的加速：
真实路由、输出正确。**1.38x** 含流水线 bubble + 同步开销；隔离 KDA chunk kernel 单算子为 4.25x（见上），
端到端被 host-bound 的 MoE 逐专家分发循环摊薄。

**复现：**
- 全模型 20/20 + 上表（每卡一进程，需 checkpoint）：
  `MODEL_PATH=<weights> torchrun --nproc_per_node=4 bench_kimi_multinpu.py --pypto --seq 256 --report-file out.json`（baseline 去掉 `--pypto` 再跑一次）。
- harness smoke（随机权重，无需 checkpoint）：
  `torchrun --nproc_per_node=2 bench_kimi_multinpu.py --random-weights --layers 4 --seq 80 --iters 2 --pypto`
- 单进程快速检查（~6/20，*不会*得到上面的全覆盖数字）：
  `python3 scripts/ask_Kimi-Linear-48B-A3B.py --model-path <weights> --num_npus 4 --use_pypto`
- 算子级 msprof：`MODEL_PATH=<weights> bash prof_Kimi-Linear-48B-A3B.sh`。

---

### ⚠️ 图捕获吞吐代理（static-route graph-capture proxy — 非可部署 / NOT deployable）

> **本节以下所有数字都不是可部署性能**（静态假路由、输出不正确）。可部署数字见上方『全模型实测』(PyPTO 1.38x)。

> ⚠️ **静态路由吞吐代理：** 下表 graph 各格用**固定静态路由**（`--route fixed --active 178`）替换真实 MoE
> 路由，以消除 `moe_infer` 的 host 同步（`.argsort()`/`.bincount()`/`.cpu().numpy()`/
> 数据依赖的逐专家循环），使 per-die 计算段可被 `torch.npu.NPUGraph` 捕获。真实 router 的 AICPU
> argsort/bincount **无法**被捕获进 NPU graph，故 MoE +graph **必须**用固定假路由。**logits 与真实路由
> 不一致、输出值不正确、不可用于真实推理**（shape-faithful 但
> value-infidel 的**吞吐代理**）。真实可部署加速见上方 = **PyPTO 1.38x**（真实路由）。

4 卡 / seq=256 / 真实权重 / active=178 experts / 1 warmup + 1 measured pipeline wall（最慢 rank）：

| 模式 | wall (ms) | 吞吐 (tok/s) | 备注 |
|------|-----------|-------------|------|
| vec + graph（eager + graph，静态假路由） | 345.5 | 741 | 吞吐代理（不可部署 / 输出不正确 / 静态假路由）|
| **pypto + graph（静态假路由）** | **272.7** | **939** | 吞吐代理（不可部署 / 输出不正确 / 静态假路由）|
| **加速比（pypto+graph vs vec+graph）** | **1.27x** | **1.27x** | 仅代理，非可部署 |

> 吞吐 tok/s = seq(256) ÷ wall(s)。

**说明：** PyPTO 对 KDA 的净贡献（同静态路由、同 graph）：vec+graph 345.5ms → pypto+graph 272.7ms =
**1.27x**。**与 qwen 不同**：qwen 经 PyPTO 后整网 compute-bound、图捕获冗余；kimi 27 层流水线即使用
PyPTO-KDA 仍 **launch/dispatch-bound** —— **host-bound 的 MoE 逐专家分发循环**仍是瓶颈，图捕获将其折叠。
故图捕获在 kimi 与 PyPTO 互补，但**需静态假路由、输出不正确、不可部署**。

**复现：** `MODEL_PATH=<weights> torchrun --nproc_per_node=4 bench_kimi_multinpu.py [--pypto] --graph
--route fixed --active 178 --seq 256`（`--graph` 开图捕获；`--count-experts` 量测自然激活专家数）。

## 归档映射

本集成在 pypto-gym 仓库中的文件布局（步骤 27 还原重建按此反向拷贝到权重目录）：

| 来源（模型部署目录 `{weights_dir}/`） | pypto-gym 归档位置 |
|---|---|
| `scripts/ask_*.py`, `bench_*.sh`, `prof_*.sh`, `sample_inputs.txt`, `README.md` | `modeling/transformers/kimi_linear_48b_a3b/` |
| `config.json` | `src/pypto_gym/transformers/kimi_linear_48b_a3b/config.json` |
| `modeling_kimi.py`, `configuration_kimi.py`, `kimi_fla_compat.py`（trust_remote_code 打补丁版） | `src/pypto_gym/transformers/kimi_linear_48b_a3b/` |
| `kimi_linear_48b_a3b_pto_kernels/kda/*_impl.py`, `_device_guard.py`, `__init__.py`, `README.md` | `ops/pypto_tensor/kimi_linear_48b_a3b/` |
| `kimi_linear_48b_a3b_pto_kernels/kda/{*_golden.py, test/*}` | `tests/ops/kimi_linear_48b_a3b/` |

> 还原：按上表逆向拷贝到权重目录，确认 `config.json` 的 `auto_map` 指向部署的
> modeling，然后 `ask_Kimi-Linear-48B-A3B.py [--use_pypto]` 双模式验证。
