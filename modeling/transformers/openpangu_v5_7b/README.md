# openPangu-Embedded-7B NPU 迁移说明 (cann-recipes runner)

| 字段 | 说明 |
|------|------|
| HuggingFace | FreedomIntelligence/openPangu-Embedded-7B |
| 权重目录 | `{model_weight_dir}`（如 `/path/to/openPangu-Embedded-7B`） |
| 代码来源 | cann-recipes-infer `models/pangu-7b/models/modeling_openpangu_dense.py`，适配为 pypto-gym 目录结构 |
| transformers版本 | 4.55.0 |
| 代码位置 | `src/pypto_gym/transformers/openpangu_v5_7b/`（modeling + configuration + config.json） |
| 修改内容 | 解码路径整层融合：`PanguEmbeddedModel.forward` 增加 PyPTO fused-layer dispatch（`USE_PTO_FUSED_LAYER` 开关，仅 decode 生效；prefill 退原生 PyTorch 解码层） |
| 运行方式 | 经 cann-recipes `PanguEmbeddedRunner` + YAML 驱动（非 `AutoModelForCausalLM.from_pretrained`） |

## 环境信息

| 组件 | 版本 |
|------|------|
| torch | 2.6.0 |
| torch_npu | 7.2.RC1.alpha002 |
| transformers | 4.55.0 |
| CANN | 8.3.RC1.alpha002 |
| NPU | Atlas A3 / A5 系列 |
| pypto | 9.1.0 |

## 环境准备

运行前先加载环境（CANN、PyPTO pto-isa 路径、NPU 设备号等均已集中在 `env_setup.sh`）：

```bash
cd /data/h50058642/h00949854/pypto-gym/modeling/transformers/openpangu_v5_7b
source env_setup.sh
```

`env_setup.sh` 中可通过环境变量覆盖默认值（如 `CANN_HOME`、`PTO_TILE_LIB_CODE_PATH`、
`TILE_FWK_DEVICE_ID`、`CANN_RECIPES_PATH` 等），无需修改脚本。

## 下载模型

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="FreedomIntelligence/openPangu-Embedded-7B",
    local_dir="/path/to/openPangu-Embedded-7B",
    max_workers=8,
)
```

## 运行命令

`ask_openpangu_v5_7b.py` 是 cann-recipes `infer.py` 的薄封装：把
`pypto_gym.transformers.openpangu_v5_7b` 的 modeling/configuration 注入到 runner 的
`models.*` 命名空间，并用 `--use-pto` 打开融合算子开关。需要一个 `cann-recipes-infer`
检出提供 `executor`/`module`/`runner_openpangu_dense`。

```bash
# 0. 加载环境（CANN + PyPTO + NPU 设备）
source env_setup.sh

# 基线推理（原生 PyTorch 解码层）
python3 scripts/ask_openpangu_v5_7b.py \
    --recipes-path $CANN_RECIPES_PATH \
    --model-path  /path/to/openPangu-Embedded-7B \
    --prompt "你好"

# PyPTO 整层融合推理
python3 scripts/ask_openpangu_v5_7b.py \
    --recipes-path $CANN_RECIPES_PATH \
    --model-path  /path/to/openPangu-Embedded-7B \
    --prompt "你好" --use-pto
```

> 默认 YAML 为本目录 `openpangu_v5_7b.yaml`（`model_path` 占位符，会被 `--model-path` 覆盖）。
> 也可用 `--yaml` 指向 cann-recipes 的 `models/pangu-7b/config/openpangu_v5_7b.yaml`。

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model-path` | （必填） | openPangu-Embedded-7B 权重目录 |
| `--recipes-path` | `$CANN_RECIPES_PATH` | cann-recipes-infer 检出根目录 |
| `--yaml` | 本目录 `openpangu_v5_7b.yaml` | runner YAML 配置 |
| `--prompt` | attention 续写 prompt | 输入文本 |
| `--device` | `$TILE_FWK_DEVICE_ID` | NPU 卡号 |
| `--max-new-tokens` | YAML 中的值 | 覆盖生成 token 数 |
| `--use-pto` | 关闭 | 启用 PyPTO 融合 decode kernel |
| `--no-warmup` | 关闭 | 跳过 warmup 运行 |

## 目录结构

```
openPangu-Embedded-7B/
├── modeling_openpangu_dense.py          # 模型实现（含 PyPTO fused-layer dispatch）
├── configuration_openpangu_dense.py     # PanguEmbeddedConfig
├── config.json                          # auto_map（model_type=PanguEmbedded）
└── ...模型权重 / tokenizer 文件
```

## PyPTO 融合范围

| 算子 | 融合? | 说明 |
|------|:---:|------|
| 整解码层（decode） | ✅ | RMSNorm+QKV(+bias)+RoPE+KV-cache+GQA Attention+O proj(+bias)+RMSNorm+SwiGLU FFN 融合进单个 kernel |
| Prefill | ❌ | 多 token，走原生 `PanguEmbeddedDecoderLayer` |
| Embedding / LM head | ❌ | `VocabParallelEmbedding`（TP 分片） |

## 开关变量

`ops.pypto_tensor.openpangu_v5_7b.USE_PTO_FUSED_LAYER`（bool，默认 `False`）

ask 脚本在构造模型前 `--use-pto` 时置 `USE_PTO_FUSED_LAYER = True`。

