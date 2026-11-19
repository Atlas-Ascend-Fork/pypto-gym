# DeepSeek-V2-Lite-Chat MLA KV Prolog 迁移说明

## 基本信息

| 项目 | 值 |
|------|------|
| HuggingFace | [deepseek-ai/DeepSeek-V2-Lite-Chat](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat) |
| 权重目录 | /path/to/models/DeepSeek-V2-Lite-Chat |
| 代码来源 | transformers包（内置） + PyPTO自定义算子 |
| transformers版本 | 4.45.0+ |
| PyPTO路径 | /path/to/pto-isa |
| 运行命令 | `python3 scripts/ask_DeepSeek-V2-Lite-Chat.py --device 7 --use_kv_fusion --use_acl_graph` |
| 核心优化 | 动态选择策略（短序列PyPTO，长序列Baseline） |

## 目录结构

```
DeepSeek-V2-Lite-Chat/
├── config.json
├── model*.safetensors
├── model.safetensors.index.json
├── tokenizer.json / tokenizer_config.json
├── generation_config.json
├── docs/
│   ├── benchmark_bs_comparison.md
│   └── long_seq_optimization_success.md
├── pto_kernels/
│   ├── __init__.py
│   ├── rms_norm/
│   ├── rope/
│   └── mla_prolog/
│       ├── __init__.py
│       ├── mla_prolog_dynamic_selection.py ⭐ 核心优化
│       ├── mla_prolog_baseline_torch.py
│       ├── mla_prolog_pypto_hybrid_optimized.py
│       └── ... (其他实现)
├── scripts/
│   ├── ask_DeepSeek-V2-Lite-Chat.py ⭐ 主推理脚本
│   ├── benchmark_bs_comparison.py ⭐ 性能对比测试
│   ├── analyze_bs_results.py ⭐ 结果分析
│   └── ... (其他脚本)
├── results/
│   ├── benchmark_bs_results.json
│   └── ... (配置文件)
└── README.md (归档说明)
```

## 使用方法

### 基本用法（Baseline ACLGraph）

```bash
# 不使用PyPTO融合算子
python3 scripts/ask_DeepSeek-V2-Lite-Chat.py --device 7 --use_acl_graph

# 自定义prompt
python3 scripts/ask_DeepSeek-V2-Lite-Chat.py --device 7 --use_acl_graph --prompt "介绍一下Python语言"

# 指定输出长度
python3 scripts/ask_DeepSeek-V2-Lite-Chat.py --device 7 --use_acl_graph --output_length 50
```

性能对比脚本仅在推理子进程正常退出且完整指标可解析时记录成功。即使日志已经打印吞吐量，非零退出也会记录 `success: false` 和退出码，避免将失败运行用于性能对比。

在本仓库根目录运行 `python3 modeling/transformers/deepseek-v2-lite-chat/test_benchmark_child_status.py` 可验证该规则。Host 回归使用临时模型目录与真实 Python 子进程，覆盖 stdout/stderr 指标、两种融合模式、非零退出和缺失指标；不加载模型、PyPTO 或 NPU，不代表设备性能测量。

### PyPTO融合算子（动态选择策略）

```bash
# 启用PyPTO KV融合 + ACLGraph（推荐）
python3 scripts/ask_DeepSeek-V2-Lite-Chat.py --device 7 --use_kv_fusion --use_acl_graph

# 自定义测试
python3 scripts/ask_DeepSeek-V2-Lite-Chat.py --device 7 --use_kv_fusion --use_acl_graph --prompt "你好，请详细介绍一下人工智能的发展历程" --output_length 100
```
