---
name: pypto-fused-op-integration
description: 将HuggingFace大语言模型迁移到华为昇腾NPU环境并集成PyPTO融合算子。完整工作流：NPU迁移→基线验证→算子融合→模型集成→端到端验证→性能分析→归档。触发词：NPU、昇腾、Ascend、华为AI处理器、算子融合、整网集成、replace small ops、模型算子替换（GLM、LLaMA、Qwen、MoE、Attention等）、HuggingFace模型部署、内存预估、auto_map、离线加载等。
---

## 环境约定

所有 `pip` / `npm` 命令必须使用国内源：
- **pip**：`-i https://mirrors.huaweicloud.com/repository/pypi/simple --trusted-host mirrors.huaweicloud.com`
- **npm**：`--registry=https://registry.npmmirror.com`

---

# PyPTO 融合算子整网集成 Skill

将HuggingFace大语言模型迁移到华为Ascend NPU环境，并将PyPTO融合算子替换到整网中替代原始算子实现的完整工作流程。

**强制要求：** 必须使用真实NPU硬件，最终验证推理脚本成功运行。

---

## 工作流程概览

```
阶段零：NPU 迁移与基线建立  →  模型下载 → PYPTO入网适配 → 脚本生成 → 基线验证 → Git基线 → 代码部署
阶段一：前置准备           →  需求分析 + 环境验证 ★ + 智能推荐
阶段二：理解验证           →  打点采集 ★ + Golden编写 + 场景验证 ⚠️
阶段三：经典算子开发       →  design → develop → precision-compare → perf-tune
阶段四：模型集成           →  目录结构 + 适配层 + 调用逻辑 + 缓存处理 ★
阶段五：验证与提交        →  端到端验证（必须） + 性能采集与分析 [可选] + 归档 + 提交
```

> **★ 标注：** 关键步骤，必须完成  
> **⚠️ 标注：** 需特别注意  
> **🔀 标注：** 用户选择点，必须询问  
> **[可选]：** 后续迭代进行

---

## 关联 Skill

本 Skill 仅调用当前 `pypto-gym/cannbot-skills` 子树中已存在的能力：

| Skill | 用途 |
|-------|------|
| `pypto-intent-understand` | 需求理解 |
| `pypto-api-explore` | PyPTO API 探索 |
| `pypto-golden-generate` | Golden 生成 |
| `pypto-op-design` | 算子设计 |
| `pypto-op-develop` | 算子实现 |
| `pypto-precision-compare` | 精度对比 |
| `pypto-precision-debug` | 精度调试 |
| `pypto-op-perf-tune` | 性能调优 |

## 参考文档索引

以下内容已移至 `references/` 子目录，需要时查阅：

| 文件 | 内容 | 使用场景 |
|------|------|---------|
| `references/memory_estimation.md` | NPU内存预估公式+各规模模型估算表 | 步骤1判断模型是否能在单卡运行 |
| `references/directory_structure.md` | 阶段零/四/归档目录结构 + README必须字段 + 变量定义 | 各步骤创建目录、写README时 |
| `references/troubleshooting.md` | 常见问题FAQ + 下载中断/导入修改失败恢复 | 遇到报错时查阅 |

---

## 详细步骤指南

---

### 阶段零：NPU 迁移与基线建立

---

#### 步骤 0：确认模型信息 ★

**必须询问：**
1. HuggingFace模型链接 → 提取 `repo_id`（如：Qwen/Qwen2-7B）和 `model_name`（如：Qwen2-7B）
2. 模型存放目录 `user_model_dir`（默认：/data/models）

**自动检测：**
- `pypto_repo`：当前仓库根目录（通过 pwd 或检测 .git 文件）
- 变量定义见 `references/directory_structure.md`

**检测已有模型：**
```bash
ls {model_weight_dir}/*.safetensors {model_weight_dir}/*.bin 2>/dev/null
```

- 存在权重 → 直接使用，跳过步骤 3
- 不存在 → **询问：**「无权重文件。有已下载的目录吗？（给路径则跳过下载，否则自动下载）」

**⚠️ 副本/克隆目录需验证tokenizer文件：**

模型目录权重文件存在不等于tokenizer可用。从baseline复制或symlink模型时，必须同步检查：
```bash
ls {model_weight_dir}/tokenizer.json {model_weight_dir}/vocab.json {model_weight_dir}/merges.txt 2>/dev/null
```
缺失会导致 `tokenizer(prompt)` 返回 0 tokens，后续模型因空输入崩溃。

---

#### 步骤 1：检查NPU环境与内存预估

**检查NPU状态：**
```bash
npu-smi info
```

**验证标准：** 输出显示 NPU 设备列表，且至少一张卡可用；失败时停止流程并向用户报告环境问题。

**内存预估：** 查阅 `references/memory_estimation.md` 判断模型是否能在单卡运行。

---

#### 步骤 2：安装依赖

**torch和torch-npu版本必须完全一致！**

```bash
pip install torch==2.7.1 torch-npu==2.7.1
pip install transformers accelerate sentencepiece protobuf

# 验证安装
python3 -c "import torch; import torch_npu; print(f'torch: {torch.__version__}'); print(f'NPU可用: {torch.npu.is_available()}')"
```

**⚠️ 根据模型 HF 页面确认软件版本要求：**

不同模型对 transformers 版本有具体要求。先抓取模型 README 中的版本声明：

```bash
curl -sk "https://hf-mirror.com/{repo_id}/raw/main/README.md" | grep -iE "transformers.*[0-9]+\.[0-9]+|pip install|require"
```

如 README 指定了版本（如 `transformers==4.41.2`），**必须安装对应版本**，否则可能出现 KV cache 不兼容、API 参数变更等问题。

**⚠️ 版本冲突必须先问用户：** 当新模型要求的 transformers / torch 版本与当前环境不兼容时，AI 禁止直接 `pip install`。必须展示版本差异并询问：A）当前环境升级 B）新建 conda 环境 C）放弃迁移。用户确认后才能操作。

---

#### 步骤 3：下载模型

> **前置条件：** 步骤 0 已确认 `{model_weight_dir}` 不存在权重文件。

**方法一：download_hf_model.py（推荐）**

```bash
python3 scripts/download_hf_model.py \
    --model-id {repo_id} \
    --output-dir {model_weight_dir}
```

支持 `--revision`、`--token`、`--allow-pattern`、`--ignore-pattern` 可选参数。

**方法二：snapshot_download 直接调用**

```bash
mkdir -p {model_weight_dir}
export HF_ENDPOINT=https://hf-mirror.com
nohup python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='{repo_id}', local_dir='{model_weight_dir}', max_workers=10)
" > {model_weight_dir}/download.log 2>&1 &
```

**方法三：git clone + git lfs pull（代理环境大文件更稳定）**

```bash
apt-get install -y git-lfs 2>/dev/null || yum install -y git-lfs 2>/dev/null
GIT_SSL_NO_VERIFY=1 git clone https://hf-mirror.com/{repo_id} {model_weight_dir}
GIT_SSL_NO_VERIFY=1 git -C {model_weight_dir} lfs pull
```

**检查下载进度：**
```bash
ps aux | grep "snapshot_download\|git.lfs"
du -sh {model_weight_dir}/
ls -lh {model_weight_dir}/*.safetensors {model_weight_dir}/*.bin 2>/dev/null
```

---

#### 步骤 3.5：PYPTO入网适配 ★

HF 下载的原始 `modeling_*.py` 不含 PyPTO 算子集成。检查 pypto-gym 是否已包含此模型的华为修改版：

```bash
bash scripts/restore_model_patch.sh \
    {model_weight_dir} {model_name}
```

脚本自动完成：备份 HF 原始代码 → 替换为华为修改版 → 写入 `pto_kernels/` → 确保 `auto_map` → 清除 HF 缓存。

若无对应修改版（src/pypto_gym/ 下不存在），脚本跳过 code/patch，不报错。

---

#### 步骤 4：生成脚本（ask + bench + prof）

> 合并原步骤 5/5.5/5.6，一键生成推理脚本、基准测试脚本和 msprof 采集脚本。

**生成推理脚本：**
```bash
mkdir -p {script_dir}

python3 scripts/generate_ask_script.py \
    --model-name "{model_name}" \
    --script-dir "{script_dir}" \
    --default-model-dir "{user_model_dir}"
```

生成的 `ask_{model_name}.py` 支持：`--prompt`、`--sentence_file`、`--output-length`（默认 100）、`--use-pto`、`--per-step-timing`、`--report-file`、`--profile-dir`（torch_npu.profiler trace 输出目录）

**生成基准测试脚本：**
```bash
python3 scripts/generate_bench_script.py \
    --model-name "{model_name}" --script-dir "{script_dir}" --model-dir "{model_weight_dir}"

chmod +x {script_dir}/bench_{model_name}.sh
```

**生成msprof采集脚本 [可选]：**
```bash
python3 scripts/generate_prof_script.py \
    --model-name "{model_name}" --script-dir "{script_dir}" --model-dir "{model_weight_dir}"
chmod +x {script_dir}/prof_{model_name}.sh
```

---

#### 步骤 5：验证脚本运行 ★

```bash
python3 {script_dir}/ask_{model_name}.py
```

**验证通过标准：**
- 脚本成功加载模型到 NPU
- 输出显示正在使用 NPU 设备
- 生成回复并输出，无错误退出

---

#### 步骤 6：Git基线提交 ★

> 在代码部署之前，将已验证跑通的基线代码纳入版本控制。

```bash
cd {model_weight_dir}

[ ! -d .git ] && git init

cp templates/gitignore {model_weight_dir}/.gitignore

git add scripts/ *.json *.md .gitignore
git commit -m "migrate {model_name} to NPU — baseline before pto integration"
```

> 仅提交脚本/配置/文档，排除权重/tokenizer等大文件。

---

#### 步骤 7：代码部署 ★

检测 `{model_weight_dir}/config.json` 的 auto_map 字段判断网络结构来源。

**README必须字段：模型信息 + 环境版本 + 下载方式 + PYPTO入网适配 + 性能对比。** 见 `references/directory_structure.md`
**文件版权声明：** 见 `references/directory_structure.md`

创建 README 前，先采集当前环境版本：

```bash
echo "torch:       $(python3 -c 'import torch; print(torch.__version__)')"            > /tmp/versions.txt
echo "torch_npu:   $(python3 -c 'import torch_npu; print(torch_npu.__version__)')"  >> /tmp/versions.txt
echo "torchvision: $(python3 -c 'import torchvision; print(torchvision.__version__)' 2>/dev/null || echo N/A)" >> /tmp/versions.txt
echo "transformers:$(python3 -c 'import transformers; print(transformers.__version__)')" >> /tmp/versions.txt
echo "CANN:        $(ls /usr/local/Ascend/ascend-toolkit/latest 2>/dev/null || npu-smi info 2>/dev/null | head -1)" >> /tmp/versions.txt
```

将 `/tmp/versions.txt` 内容写入 README 的环境信息部分。

> 如模型 HF README 指定了 transformers 版本（步骤 2 已检查），在 README 中标注「HF 要求: transformers==X.X.X」。

---

**情况A：auto_map 存在（trust_remote_code 模式）**

步骤 3.5 已应用融合补丁（若 pypto-gym 含修改版）。如需额外处理：
1. 确认 `config.json` 中 `auto_map` 指向的代码文件存在且正确
2. 创建 `{script_dir}/README.md`（代码来源填"HuggingFace仓库自带 + pypto-gym 补丁"）

---

**情况B：auto_map 不存在（transformers 内置模式）**

从 transformers 包复制实现文件到 `{model_weight_dir}/core/` 目录：

1. **复制代码：**
   ```bash
   mkdir -p {model_weight_dir}/core

   TRANSFORMERS_PATH=$(python3 -c "import transformers; print(transformers.__path__[0])")
   MODEL_TYPE=$(python3 -c "import json; print(json.load(open('{model_weight_dir}/config.json')).get('model_type',''))")

   cp $TRANSFORMERS_PATH/models/$MODEL_TYPE/modeling_$MODEL_TYPE.py {model_weight_dir}/core/
   cp $TRANSFORMERS_PATH/models/$MODEL_TYPE/configuration_$MODEL_TYPE.py {model_weight_dir}/core/
   ```

2. **修改导入方式：** 使用自动化脚本（推荐）或手动修改
   ```bash
   python3 scripts/fix_imports.py {model_weight_dir}/core/modeling_{model_type}.py
   python3 scripts/fix_imports.py {model_weight_dir}/core/configuration_{model_type}.py
   ```

   手动修改规则：`from ...xxx import yyy` → `from transformers.xxx import yyy`，`from .configuration_xxx` 保持不变。

3. **添加 auto_map 到 config.json：**
   ```bash
   python3 -c "
   import json
   c = json.load(open('{model_weight_dir}/config.json'))
   mt = c.get('model_type', ''); arch = c['architectures'][0]
   c['auto_map'] = {
       'AutoConfig': f'core/configuration_{mt}.{mt.capitalize()}Config',
       'AutoModelForCausalLM': f'core/modeling_{mt}.{arch}'
   }
   json.dump(c, open('{model_weight_dir}/config.json', 'w'), indent=2)
   "
   ```

4. 创建 `{script_dir}/README.md`（包含必须字段+情况B额外字段）

---

#### 步骤 8：代码部署Git提交

```bash
cd {model_weight_dir}
git add core/ config.json
git commit -m "code deployment: add core/ with fixed imports and auto_map"
```

---

### 阶段一：前置准备

---

#### 步骤 9：需求分析（智能推荐）

**目标：** 分析网络可融合部分，推荐给用户确认。

**操作：**
1. 阅读模型代码，识别算子组合模式（Attention、MoE、FFN等）
2. 搜索 `models/` 目录下的实现案例，匹配 pypto 现有算子
3. 向用户推荐候选融合点（示例格式）：
   ```
   发现可融合算子组合：
   | 位置 | 原始实现 | pypto算子 | 收益 |
   | Attention层 | Q/K/V投影+Softmax+Output | pypto.flash_attention | 减少3次matmul |
   | FFN层 | Gate+Up+SwiGLU+Down | pypto.swiglu_ffn | 减少中间存储 |
   请确认目标算子。
   ```
4. 无合适推荐 → 直接询问用户融合位置和目标

**推荐 Skill：** `pypto-api-explore`

---

#### 步骤 10：前置验证（环境+网络基线）★

**环境验证：**
- 若 NPU 环境不可用，停止流程并向用户报告
- 检查 NPU 状态：`npu-smi info`
- 确认 torch/torch-npu 版本一致

**基线确认：**
- 阶段零的步骤5已验证模型可正常运行 → 直接确认
- 否则运行 `python3 {script_dir}/ask_{model_name}.py` 验证

**验证检查点：** ✅ NPU驱动正常 ✅ 模型可正常加载和推理 ✅ 输出为自然语言（非乱码）

**CANN ops 内核包验证（⚠️ 易遗漏）：**

`npu-smi info` 只检查驱动，不检查算子库。需额外验证 ops 可用性：
```bash
python3 -c "
import torch, torch_npu; torch.npu.set_device(0)
x = torch.randn(2,2, dtype=torch.float16).to('npu:0')
y = x.to(torch.float32)           # aclnnCast 是否可用
torch.equal(x.cpu(), x.cpu())      # aclnnEqual（模型加载 tie_weights 触发）
z = x * 2.0                        # aclnnMul（基础计算）
print('CANN ops OK')
"
```
任一 op 报 `EZ9999: Op XXX does not has any binary` → 安装对应 Ascend-cann-ops-kernel 包或降级 CANN 版本。

---

### 阶段二：理解验证 ⚠️

> **为什么需要？** 推测的计算逻辑需 Golden 验证确认。

#### 步骤 11：打点采集真实 Tensor 信息 ★

**目标：** 从原始网络采集真实 shape/dtype，构造必须 pass 的测试用例。

**操作：**
1. 定位并插入 print（采集所有外部输入的 shape/dtype）
2. 运行原始网络采集
3. 采集完成后删除打印，恢复代码原状
4. 创建 test_cases.json

**注意事项：**
- ⚠️ **不要限制打印次数**：禁止使用计数器限制，否则会漏掉不同场景
- 应采集所有调用场景，覆盖不同 shape/dtype 组合

**格式参考：** `../../ops/pypto-op-verify/templates/test_cases_template.json` 和 `references/test-cases-template.md`

**输出物：** test_cases.json → 存放 `models/{model_name}/pto_kernels/xxx/test/test_cases.json`

---

#### 步骤 12：编写 Golden 脚本（场景区分）

**目标：** 编写 PyTorch 参考实现，精度对比基准。

**场景判断：** 检查**被替换逻辑**是否使用 torch_npu 融合算子（只看逻辑本身，不看文件import）。

| 场景 | 判断条件 | 策略 | 输出件 |
|------|---------|------|--------|
| **场景A** | 只使用基础算子 | 直接复制原始代码，无需理解验证 | `xxx_golden.py` |
| **场景B** | 使用融合算子 | 用 torch 重写等价实现，必须验证 | `xxx_golden.py` + `test_xxx_golden_correctness.py` |

**通用要点：** 纯 PyTorch，禁止引入 pypto/torch_npu。导出 `{op}_golden()` 函数，独立 `{op}_golden.py` 文件。

**⚠️ Golden 必须逐行匹配原始代码的操作顺序**，不仅是数学等价。例如 `gamma * x_norm.to(input_dtype)`（FP16 乘）与 `(gamma.to(FP32) * x_norm).to(input_dtype)`（FP32 乘再降精度）会差 ~1e-4，导致 `max_diff≠0` 验证失败。

**推荐 Skill：** `pypto-golden-generate`

---

#### 步骤 13：验证理解正确性（分场景验证）

**场景A：未引用 torch_npu → 无需验证**，Golden = 原始代码，直接进入步骤15。

**场景B：引用 torch_npu ★ 必须验证**

验证流程：
1. 构造测试用例（来自步骤11采集的 shape/dtype）
2. 对比 torch Golden 与 torch_npu：`assert_allclose(output_npu, output_golden, rtol=1e-3, atol=1e-3)` → `[PRECISION_PASS]`
3. 一致后替换 Golden 到整网，验证输出正确

**验证检查点：** ✅ Golden与torch_npu一致(diff < 1e-3) ✅ 整网替换后输出正常 ✅ 无NaN/Inf

---

#### 步骤 14：决策与迭代

- **通过 ✅**：场景A → 步骤14.5；场景B → torch Golden与torch_npu一致 + 整网验证通过 → 步骤14.5
- **失败 ❌**：场景B → 重新理解融合算子语义，修正 Golden

---

### 阶段三：经典算子开发

> 阶段三-B 走**逐步手动精细化控制**路线（design → develop → precision-compare → perf-tune），适合复杂/定制算子或需要深度调试的场景。

#### 步骤 15：生成设计方案 ★

**推荐 Skill：** `pypto-op-design`

**操作：** 基于阶段二采集的 test_cases.json 和 golden，编写算子设计文档，明确：
- 输入/输出 tensor 的 shape 和 dtype
- PyPTO API 选型（cast、mul、sum、add、rsqrt 等）
- 动态轴处理策略（pypto.loop / pypto.loop_unroll）
- TileShape 配置
- 精度路由（FP32 中间计算、BF16 输入输出）

**⚠️ Kernel dtype 策略（关键决策）：**

模型原生 dtype（HuggingFace 加载后的 `torch_dtype`）决定 PyPTO kernel 的 DT_* 标注。**kernel 标注 dtype 必须与模型原生 dtype 一致**，否则 wrapper 需做 expensive 的 NPU-side cast（且 CANN 环境可能不支持某些 dtype cast）：

| 模型加载 dtype | PyPTO kernel DT_标注 | 说明 |
|---|---|---|
| `float16` | `pypto.DT_FP16` | 最常见，CANN 兼容性最好 |
| `bfloat16` | `pypto.DT_BF16` | 精度更高但需 CANN bf16 ops 支持 |
| `float32` | `pypto.DT_FP32` | 仅推理用 FP32 时 |

**决策规则**：优先匹配模型 dtype，不要假设 BF16。若 CANN 不支持 BF16→FP16 cast（常见于 ops 内核包不完整的环境），kernel 直接用 FP16。

---

#### 步骤 16：实现 PyPTO Kernel ★

**推荐 Skill：** `pypto-op-develop`

**操作：**

1. 为每类融合算子独立编写 `{op}_impl.py`：
   - 使用 `@pypto.frontend.jit` 装饰 JIT 编译
   - `@allow_in_graph` 修饰 wrapper 函数
   - wrapper 内完成 torch ↔ pypto tensor 转换

2. **Golden 对比验证：** `assert_allclose(output_impl, output_golden, rtol=1e-3, atol=1e-3)` → `[PRECISION_PASS]`

**输出物：**
- `{op}_impl.py` — PyPTO kernel 实现
- `test_{op}.py` — 单算子精度测试
- 测试须覆盖 test_cases.json 的全部场景

---

#### 步骤 17：精度调试（如需要）★

**推荐 Skill：** `pypto-precision-compare`、`pypto-precision-debug`

当 golden 与 impl 的 diff 超过容差时使用。

```bash
python3 {pto_kernels_dir}/{op}/test_{op}.py
```

**差异等级：**

| 差异 | 处理 |
|------|------|
| < 1e-3 | ✅ PASS，无需调试 |
| 1e-3 ~ 1e-2 | ⚠️ 轻度不匹配，检查数值稳定性（eps、中间dtype、tile对齐） |
| > 1e-2 | ❌ 严重不匹配，使用 pypto-precision-debug 逐层定位 |

**常见问题：**
- FP32 ↔ BF16 转换丢失精度 → 中间计算全程保持 FP32
- TileShape 对齐不当 → 检查 tail axis 32B 对齐
- 数学等价但实现不同 → 对齐 golden 的计算顺序

---

#### 步骤 18：性能调优 [可选]

**推荐 Skill：** `pypto-op-perf-tune`

通过调整 Tile 配置、合图策略、向量化等手段优化性能。包括：
- TileShape 优化（block 大小调优）
- DM 搬运与计算流水线优化
- 多核并行调度

---

### 阶段四：模型集成

#### 步骤 20：创建算子库目录

创建 pto_kernels 目录结构。目录结构和命名规则见 `references/directory_structure.md`。

---

#### 步骤 21：配置适配层

**关键要点：**
- 开关设计：按算子粒度 `USE_PTO_{OP}`，便于渐进式验证
- 函数命名：与原始算子同名
- 文档注释：包含目标文件、目标类、替换代码片段
- `allow_in_graph`：适配层函数 `{op}_pto` 调用 `@allow_in_graph` 修饰的 `{op}_wrapper`，禁止越级调 JIT kernel

**参考案例：** https://gitcode.com/songle1/glm-net/blob/main/glm_pto_kernels/__init__.py

---

#### 步骤 22：修改模型调用逻辑 + sys.modules注入 ★

**推荐方案：sys.modules注入**

**原理：** Python的 `sys.modules` 是全局模块注册表。脚本预导入算子库→注入→modeling自动获取，无缓存依赖。

**实施步骤：**

**步骤A：脚本注入（在transformers导入前）**
```python
parser.add_argument("--use-pto", action="store_true", help="启用PyPTO算子")

if args.use_pto:
    sys.path.insert(0, args.model_path)
    import {model}_pto_kernels
    sys.modules["{model}_pto_kernels"] = module
    module.USE_PTO_{OP} = True

from transformers import AutoModelForCausalLM   # 之后加载模型
```

**步骤B：modeling获取**
```python
pto_kernels = sys.modules.get("{model}_pto_kernels")

def forward(self, hidden_states):
    if pto_kernels is not None and pto_kernels.USE_PTO_{OP}:
        return pto_kernels.{op}(self, hidden_states)
    # 原始torch实现（fallback）
    ...
```

**关键要点：** 注入位置在 transformers 导入前，必须保留原始 torch 实现作为 fallback。

**⚠️ PTO 注入时序规则（易错）：**

PTO 注入（`USE_PTO_XXX=True`）必须发生在模型加载到 NPU **之后**：
```
# ✅ 正确
model = AutoModelForCausalLM.from_pretrained(...)
model.to('npu')
_pto.USE_PTO_RMS_NORM = True       # ← 注入在 NPU 就绪后

# ❌ 错误（JIT kernel 收到 CPU tensor）
_pto.USE_PTO_RMS_NORM = True       # ← 太早！
model = AutoModelForCausalLM.from_pretrained(...)
```
建模内路由已加 `device.type == "npu"` 守卫，但依赖同步问题仍应按此顺序。

---

#### 步骤 23：aclgraph适配 ★

**目标：** 使用 torch.compile 将入口 model 编译成 aclgraph。

**步骤A：使能开关**
```python
parser.add_argument("--use-acl-graph", action="store_true")

if args.use_acl_graph:
    module.USE_ACL_GRAPH = True
```

**步骤B：图编译配置**
```python
if pto_kernels is not None and pto_kernels.USE_ACL_GRAPH:
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig

    compiler_config = CompilerConfig()
    compiler_config.experimental_config.frozen_parameter = True
    compiler_config.experimental_config.tiling_schedule_optimize = True
    npu_backend = tng.get_npu_backend(compiler_config=compiler_config)
    self.model = torch.compile(self.model, dynamic=True, fullgraph=True, backend=npu_backend)
```

---

### 阶段五：验证与提交

#### 步骤 24：端到端验证 ★

**目标：** 整网推理验证 + 问题排查。

- 运行整网推理，确认输出正常
- 与原始实现对比

**⚠️ 单算子替换性能预期：**

当只替换单个独立算子（如 RMSNorm）而未做前后融合时，**PTO 模式通常比基线慢 2-3x**。原因：(1) JIT kernel 首次编译；（2）kernel launch + dtype cast 开销 > 原 CANN 融合 kernel；（3）单算子无法抵消调用开销。

**这属于正常现象**，不是 bug。RMSNorm 的收益来自与相邻算子的融合（如 pre-attn RMSNorm + QKV projection），单算子替换仅验证**路由逻辑正确性和精度一致性**。真正的性能提升见后续 round：融合 pre-attn、post-attn 等复合算子。

**问题排查 Skill：** `pypto-precision-debug`

---

#### 步骤 25：性能采集与分析 [可选]

> **优先级：** 精度验证（必须） > 性能对比（可选）

**方法一：整网计时+显存对比**
```bash
bash {model_weight_dir}/scripts/bench_{model_name}.sh
```
bench脚本自动解析两份 JSON，生成对比表格（推理耗时、吞吐、峰值显存）。

**方法二：msprof kernel级采集**
```bash
bash {model_weight_dir}/scripts/prof_{model_name}.sh
```
输出：op_statistic CSV + Chrome trace + AI Core利用率，脚本自动diff Top15算子耗时。

**方法三：msprof Task Wait深度分析**
```bash
msprof --application="python3 scripts/ask_{model_name}.py --use-acl-graph" --output=PROF/
python3 scripts/analyze_msprof_taskwait.py PROF_xxx/
```

**方法四：torch_npu profiler + parse_prof 整网 PyPTO kernel 级分析**

适用于同时替换了多个算子、需要逐 kernel 统计 PYPTO 耗时占比的场景。

前置条件：推理脚本已支持 `--profile-dir` 参数（内置 `torch_npu.profiler.profile()` 包裹 `model.generate()`，设置 `analyse_flag=True`），并在指定目录下生成包含 `trace_view.json` 的分析产物。

```bash
# 采集 PyPTO 模式 trace（一次 generation，不需 warmup/iters 循环）
python3 scripts/ask_{model_name}.py --use-pto \
    --profile-dir ./prof_pypto --output-length 100 \
    --report-file ./prof_pypto/bench_pypto.json

# parse_prof 解析（过滤 tilefwk / PYPTO 命名的 kernel）
python3 -c "
import os, json
prof = {}
exec(open('/path/to/parse_prof.py').read(), {'os': os, 'json': json, '__builtins__': __builtins__})
result = prof['parse_prof']('./prof_pypto')
print(json.dumps(result, indent=2))
"
```

**输出：**
```json
{
  "prof": {
    "aicore_e2e": 884527.4,        // PYPTO kernel 总耗时 (μs)
    "aicore_e2e_jitter": 2.06,     // 抖动系数 (整网推理 token 间差异大，>1 正常)
    "aicpukernel_gap": 0.0         // AICPU 阻塞间隙
  }
}
```

> **注意：** `aicore_e2e` 是 PYPTO/tilefwk 命名 kernel 的累计耗时，非单次均值。如需单 kernel 耗时分布（avg/min/max），直接解析 `trace_view.json` 按 `name` 字段聚合。

**进一步调优 Skill：** `pypto-op-perf-tune`（Tile配置、合图策略、向量化等）

**记录结果：** 将对比表格追加到 `{model_weight_dir}/scripts/README.md` 末尾。

---

#### 步骤 26：归档到 pypto-gym 仓库 [可选]

**触发：** 询问用户是否归档。

**变量定义：** 见 `references/directory_structure.md`

**归档前检查已有文件：**

```bash
ls -d {pypto_gym_repo}/ops/pypto_tensor/*/ {pypto_gym_repo}/tests/ops/*/ 2>/dev/null
```

如已有 `{model_name}` 或近邻名称的归档，**必须先询问用户确认**，再删除。**这一步不可跳过——已有归档意味着之前做过方案，不确认就直接覆盖会丢失旧实现、引入不兼容变更。**：

```bash
rm -rf {pypto_gym_repo}/ops/pypto_tensor/{model_name} \
       {pypto_gym_repo}/src/pypto_gym/transformers/{model_name} \
       {pypto_gym_repo}/tests/ops/{model_name}
find {pypto_gym_repo}/modeling/transformers/{model_name} -mindepth 1 -delete 2>/dev/null
```

**文件映射（文件夹级别）：**

| 来源 (`{model_dir}/`) | 目标 (`{pypto_gym_repo}/`) |
|---|---|
| `scripts/` | `modeling/transformers/{model_name}/` |
| `config.json` | `src/pypto_gym/transformers/{model_name}/` |
| `core/` | `src/pypto_gym/transformers/{model_name}/` |
| `{model}_pto_kernels/` | `ops/pypto_tensor/{model_name}/` |

**归档后写入 `modeling/transformers/{model_name}/README.md`**，追加归档映射记录（内容同上表，保持文件夹粒度）和当前环境版本信息。

> ⚠️ **强制检查：** 归档完成后 grep 确认 `## 归档映射` 存在；同时检查每个算子子目录是否有 `README.md`，缺失按 `references/directory_structure.md` 模板补齐。

**test 文件 import 改造：**

移至 `tests/ops/{model_name}/` 后，impl 不在同级目录，通过 `sys.path` 跨目录引用：
```python
from pathlib import Path; import sys; import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_IMPL = Path(__file__).resolve().parents[3] / "ops/pypto_tensor/{model_name}/{op}"
sys.path.insert(0, str(_IMPL))
from {op}_impl import {op}_wrapper
from {op}_golden import {op}_golden
```

**归档验证：** `python3 tests/ops/{model_name}/test_{op}.py` → `[PRECISION_PASS]` + `All tests passed!`

**⚠️ 工号/个人路径检查（强制）：** 提交前扫描硬编码个人路径
（`/npu/xxx/`、`/home/zhangsan/` 等）：

```bash
cd {pypto_gym_repo}
grep -rPn "(/[nN][pP][uU]/|/[hH][oO][mM][eE]/)" \
    modeling/transformers/{model_name}/ \
    src/pypto_gym/transformers/{model_name}/ \
    ops/pypto_tensor/{model_name}/ \
    tests/ops/{model_name}/
```

**处理：** 硬编码路径 → 环境变量或占位符。
无法自动处理时询问用户。通过标准：输出为空。

**提交 PR [可选]：** 按目标仓库的贡献流程手工创建 PR。

---

#### 步骤 27：还原重建指南 [可选]

将 pypto-gym 归档文件 + 已下载的权重重建为可运行的 PTO 融合模型。

> ⚠️ **强制前置：先对齐 Python 环境，再拷贝文件。** 仅 `transformers` 大版本必须与归档 README 一致（大版本 API 不兼容会导致 modeling 代码报错）。`torch` / `torch_npu` / `CANN` 不强绑，但 `torch` 与 `torch_npu` minor 版本须一致。

**① 安装 Python 环境 ★（不可跳过）** — 读取 `{pypto_gym_repo}/modeling/transformers/{model_name}/README.md` 环境版本表。**先向用户展示版本对比，确认后再安装：**

```bash
# 版本冲突时新建 conda 环境代替直接 pip
pip install torch==<README torch版本> torch-npu==<README torch-npu版本> \
    transformers==<README transformers版本> accelerate sentencepiece protobuf \
    -i https://mirrors.huaweicloud.com/repository/pypi/simple --trusted-host mirrors.huaweicloud.com

python3 -c "import torch; import torch_npu; print(f'torch: {torch.__version__} NPU: {torch.npu.is_available()}')"
```

**② 反向归档映射拷贝 ★** — 按步骤 26 的表逆向拷贝：

| 来源 (`{pypto_gym_repo}/`) | 目标 (`{model_weight_dir}/`) |
|---|---|
| `modeling/transformers/{model_name}/` | `scripts/` |
| `src/pypto_gym/transformers/{model_name}/` | 按文件类型：`.py` → `core/`，`config.json` → 根目录 |
| `ops/pypto_tensor/{model_name}/` | `{model}_pto_kernels/` |

拷贝后调整测试脚本的 `sys.path` 使其引用 `pto_kernels/` 内的 impl：

```python
# 测试脚本开头改为：
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from rms_norm_golden import rms_norm_golden
from rms_norm.rms_norm_impl import rms_norm_wrapper
```

**③ 运行验证** — 确认 auto_map 指向 `core/`，然后双模式跑通：

```bash
python3 {model_weight_dir}/scripts/ask_{model_name}.py --prompt "你好" --device <NPU卡号>           # baseline
python3 {model_weight_dir}/scripts/ask_{model_name}.py --prompt "你好" --device <NPU卡号> --use-pto  # PTO
```

通过标准：两次均加载成功、输出自然语言、PTO 模式无 `pto_kernels` import 错误。

---

#### 步骤 28：Git提交 PTO融合变更

```bash
cd {model_weight_dir}
git add {model}_name_pto_kernels/ core/ scripts/ config.json
git commit -m "pto integration: add fused kernel for {model_name}"
```

---

#### 步骤 29：提交与文档

按目标仓库的贡献流程创建 Issue 和 PR。

---

**遇到问题？** 查阅 `references/troubleshooting.md`

---

**Skill 版本：** v3.14  
**最后更新：** 2026-06-21  
**维护者：** PyPTO Team  
**更新说明：** 
- v3.14: 步骤3新增 download_hf_model.py 推荐；新增步骤3.5 restore_model_patch.sh — HF 下载后自动注入 PyPTO 融合代码/算子；步骤7情况A修正 trust_remote_code 模式描述
- v3.13: 步骤26新增算子 README 检查 — 归档后检查每个算子子目录是否有 README.md，缺失按模板补齐
- v3.12: 步骤26/27归档映射精简为文件夹级别 — 去掉冗余子文件列表和"操作"列，映射每行是目录/文件名，不展开内部文件；README 映射同步简化
- v3.11: 步骤26归档映射记录改为强制项 — README 缺失 `## 归档映射` 视为遗漏，步骤27(还原重建)依赖此映射
- v3.10: 步骤26新增工号/个人路径扫描检查 — grep 工号模式后强制改为环境变量或占位符
- v3.9: 步骤27强制前置环境对齐 — 归档 README 环境版本必须优先检查并安装，否则跳过直接拷贝会导致 torch_npu/CANN 版本冲突或 transformers 大版本 API 不兼容而失败
- v3.8: 步骤27全面重写 — 增加从 README 提取环境版本→安装 Python 环境→归档反向映射文件拷贝→import 适配→验证的完整还原流程
- v3.7: 步骤3新增方法二 git clone + git lfs pull 下载方式 — 代理环境大文件截断/SSL报错的稳定替代方案，自带断点续传
- v3.6: 步骤25新增方法四 — torch_npu profiler + parse_prof 整网 PyPTO kernel 级分析；步骤4补充 --profile-dir 参数说明
- v3.5: 8路并行迁移交叉验证 — 补充 golden 操作顺序要求、FP16 JIT 首编崩溃重试、CANN 多版本 set_env.sh 选择
- v3.4: 三路并行迁移实测 — tokenizer检查、CANN多版本诊断、单op延迟预期
- v3.3: CANN ops 验证、PTO 注入时序规则、kernel dtype 策略
- v3.2: 阶段三算子开发拆分为两条路线
