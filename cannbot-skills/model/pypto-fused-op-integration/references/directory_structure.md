# 目录结构参考

## 阶段零：模型部署目录结构

所有文件集中在模型权重目录（默认 `{user_model_dir}/{model_name}`）：

```
{model_weight_dir}/                  # 默认：/data/models/{model_name}
├── config.json                      # 模型配置（auto_map指向core/下文件）
├── model.safetensors                # 模型权重
├── tokenizer.json                   # tokenizer
├── tokenizer.model                  # tokenizer模型
├── core/                            # 网络结构代码（仅transformers内置模式需要）
│   ├── modeling_xxx.py              # 导入：from transformers.xxx, from .configuration_xxx
│   └── configuration_xxx.py         # 导入：from transformers.xxx
└── scripts/                         # 脚本目录
    ├── ask_{model_name}.py          # 推理脚本（内嵌计时/显存采集）
    ├── bench_{model_name}.sh        # 基准测试脚本（双模式对比，含 JSON 汇总）
    ├── prof_{model_name}.sh         # msprof 核级采集脚本 [可选]
    ├── sample_inputs.txt            # 示例提示词文件（bench使用）
    └── README.md                    # 迁移说明（必须）
```

**trust_remote_code模式：** core/目录不存在，网络结构代码已包含在HuggingFace下载的文件中。

## 阶段四：算子库目录结构（pto_kernels）

```
pto_kernels/                        # 算子库顶层
├── __init__.py                     # USE_PTO开关 + 导入所有算子
│
├── xxx/                            # 算子目录（如 rms_norm、ffn、softmax）
│   ├── __init__.py                 # 导出 xxx_wrapper
│   ├── xxx_impl.py                 # PyPTO kernel（带前缀）
│   ├── xxx_golden.py               # Golden参考（带前缀）
│   ├── README.md                   # 算子文档（必须）
│   └── test/
│       ├── test_xxx.py             # 测试脚本（带前缀）
│       └── test_cases.json         # 测试用例
│
└── utils/                          # 通用工具（可选）
    └── DESIGN.md                   # 设计文档
```

**算子 README 模板（`{op}/README.md`）：**

```markdown
# {算子} 算子集成 ({model_name})

## 概述
{一句话说明替代了什么原始实现。D={hidden_size}，{dtype}。}

## 测试
```bash
export TILE_FWK_DEVICE_ID={device_id}
python3 test/test_{op}.py
```

## 测试用例来源
从模型打点采集的真实 shape/dtype：{列举各场景}

## 技术说明
| 项目 | 说明 |
|------|------|
| 场景 | A/B — {golden 来源说明} |
| 实现 | {PyPTO 实现要点（tiling、动态轴等）} |
| ACLGraph | {是否已注册 torch.library，支持 --use-acl-graph} |

## 状态
✅ 单算子精度 | ✅ 整网集成 | {ACLGraph 状态} | ⏳ 性能调优
```

> 必填项：概述、测试命令、场景判断、状态。其余按实际内容裁剪。

**命名规则：**
- 目录名：抽象命名（如 `rms_norm`、`ffn`）
- 文件名：带算子前缀（如 `rms_norm_impl.py`）
- 模块名：`{model}_pto_kernels`（如 `qwen3_pto_kernels`），避免通用名称

## 归档到 pypto-gym 仓库后的结构

```
ops/pypto_tensor/{model_name}/
  __init__.py              # USE_PTO 开关 + 适配层函数
  {op}/
    {op}_impl.py            # PyPTO kernel 实现
    {op}_pypto_impl.py      # ModelNew 桥接类（可选）
src/pypto_gym/transformers/{model_name}/
  config.json               # 模型配置（含 auto_map、dtype 等）
  modeling_*.py             # 融合后网络结构（trust_remote_code 模式）
  configuration_*.py        # 模型配置类（trust_remote_code 模式）
tests/ops/{model_name}/
  {op}_golden.py            # PyTorch 参考实现
  test_{op}.py              # 单算子精度测试
modeling/transformers/{model_name}/
  ask_{model_name}.py       # 推理脚本
  bench_{model_name}.sh     # 基准测试脚本
  prof_{model_name}.sh      # msprof 核级采集脚本 [可选]
  sample_inputs.txt         # 示例提示词
  README.md                 # 迁移说明 + 环境版本 + 归档映射
```

## 文件版权声明

| 文件类型 | 版权模版 | 参考 |
|---------|---------|------|
| `configuration_*.py`、`modeling_*.py`（开源库拷贝） | 保留原始 Apache 2.0，末尾追加华为修改声明 | `src/pypto_gym/transformers/qwen3_1_7b/configuration_qwen3.py` |
| `ask_*.py`、`bench_*.sh`、`prof_*.sh`、`*.md` 等自有文件 | CANN Open Software License | `modeling/transformers/qwen3_1_7b/ask_Qwen3-1.7B.py` |
| `*_impl.py`、`__init__.py`（PyPTO kernel） | CANN Open Software License | `ops/pypto_tensor/qwen3_1_7b/__init__.py` |

> 开源库拷贝的文件必须在保留原始版权行的前提下，于末尾 `# NOTICE:` 标注华为修改。

## 变量定义

| 变量 | 定义 | 说明 |
|------|------|------|
| `model_weight_dir` | `{user_model_dir}/{model_name}` | 模型权重目录（运行时文件） |
| `script_dir` | `{model_weight_dir}/scripts` | 脚本目录 |
| `core_dir` | `{model_weight_dir}/core` | 代码目录（transformers内置模式） |
| `pto_kernels_dir` | `{model_weight_dir}/{model}_pto_kernels` | 算子库目录 |

## README.md 必须包含的字段

| 字段 | 说明 |
|------|------|
| HuggingFace | 模型的 repo_id |
| 权重目录 | 模型权重存放的实际路径 |
| 代码来源 | trust_remote_code 或 transformers包 |
| 运行命令 | 执行脚本的具体命令 |

**环境信息（必须）：**

| 字段 | 说明 |
|------|------|
| torch | `python3 -c "import torch; print(torch.__version__)"` |
| torch_npu | `python3 -c "import torch_npu; print(torch_npu.__version__)"` |
| torchvision | `python3 -c "import torchvision; print(torchvision.__version__)"` |
| transformers | `python3 -c "import transformers; print(transformers.__version__)"` |
| CANN | `ls /usr/local/Ascend/ascend-toolkit/latest` 或 `npu-smi info` 显示的版本 |
| NPU | `npu-smi info` 显示的芯片型号 |

> 如模型 HF README 指定了 transformers 版本要求，应在环境信息中注明「HF 要求: transformers==X.X.X」。

**下载方式（必须）：**

README 中必须包含模型下载命令，方便其他用户从零重建：

```markdown
## 下载模型

```bash
python3 ../download_hf_model.py \
    --model-id {repo_id} \
    --output-dir {model_weight_dir}
```
```

> 使用 skill 内置脚本 `download_hf_model.py` 或直接引用 `snapshot_download` 调用。

**PyPTO 入网适配（必须）：**

README 中必须包含 PYPTO 入网适配命令，说明如何将 HF 原始模型替换为华为修改版：

```markdown
## PYPTO入网适配

```bash
bash ../scripts/restore_model_patch.sh \
    {model_weight_dir} {model_name}
```

脚本自动完成：备份 HF 原始代码 → 替换为华为修改版 → 写入 `pto_kernels/` → 确保 `auto_map` → 清除 HF 缓存。

> 若 pypto-gym 仓库尚未包含此模型的修改版代码（`src/pypto_gym/transformers/{model_name}/`），可跳过此步骤，改为手动编写 `<script_dir>/restore_pypto_patch.sh`。

**性能对比（必须）：**

README 末尾追加性能对比表格和复现命令，方便用户 run：

```markdown
## 性能对比

| 模式 | 命令 | 模型加载 | 推理耗时 | 吞吐 | 峰值显存 |
|------|------|---------|---------|------|---------|
| baseline | `python3 scripts/ask_{model_name}.py --prompt "你好" --device <id> --output-length 50` | 7.4s | 3.1s | 9.7 tok/s | 7313 MB |
| pto | `python3 scripts/ask_{model_name}.py --prompt "你好" --device <id> --output-length 50 --use-pto` | 7.0s | 10.0s | 3.0 tok/s | 7313 MB |

> 单算子替换时 PTO 比基线慢 2-3x 属正常（JIT 首编 + kernel launch 开销），收益来自多算子融合。
```

> 表格数据从 `--report-file` 输出的 JSON 中提取，命令与表格一一对应。如被替换算子名不同或使用 `--use-acl-graph`，应收录对应命令和数据行。

**情况B（transformers内置模式）额外必须包含：**

| 字段 | 说明 |
|------|------|
| transformers版本 | 复制代码时的 transformers 版本号 |
| 代码位置 | 复制后代码的实际路径 |
| 修改内容 | 导入方式修改、auto_map添加等 |