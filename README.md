# PyPTO-Gym

## 🔥最新动态
- 2026/06：PyPTO-Gym 项目首次上线，本仓原为 [PyPTO](https://gitcode.com/cann/pypto) 仓的 `models` 目录，现已拆分为独立仓，与主仓解耦演进。

## 🚀概述

PyPTO-Gym 是基于 PyPTO 编程框架构建的样例仓库，面向华为昇腾（Ascend）AI 处理器，使用 PyPTO 的 Tile 编程模型，提供融合算子开发样例和大模型适配样例。这些样例均基于 CANNBot 完成设计、开发与调优工作。本仓作为 PyPTO 的"算子训练场"，方便开发者学习、复用与对比。

- **编程框架**：PyPTO，基于 Tile 的编程模型
- **内容**：融合算子样例 + 大模型适配样例

## 核心特性

- **大模型核心算子样例**：覆盖 DeepSeek 系列、Qwen3 系列、GLM V4.5、Gemma4-31B-it、LLaDA2-MoE 等模型的关键算子实现
- **实验性算子样例**：基于实验性目录 `experimental`，收录 Attention、Matmul、Vector 等基础算子的开发态样例
- **大模型适配样例**：提供融合算子入网适配样例，以及端到端模型推理与性能基准脚本
- **Agent 能力**：提供模型整网适配 skills，提升大模型对接易用性

## 目标用户

- **算法开发者**：主要使用Tensor层次编程，快速实现和验证算法，专注于算法逻辑
- **系统开发者**：可在Tensor和PTO虚拟指令集层次上进行三方框架对接或集成，以及工具链开发

## 环境准备

当前仓库支持的 CANN 版本如下：

| CANN 版本 | 支持情况 |
|----------|----------|
| 9.1.0 | 支持 |

pypto-gym 无需单独安装。请先参照 PyPTO 文档完成环境部署：

- [环境部署](https://gitcode.com/cann/pypto/blob/master/docs/zh/install/prepare_environment.md)：介绍项目基础环境的搭建，包括软件包和第三方依赖的获取和安装。
- [编译安装](https://gitcode.com/cann/pypto/blob/master/docs/zh/install/build_and_install.md)：环境部署后，介绍如何快速获取或编译 PyPTO 软件包并安装。

PyPTO 环境就绪后，克隆本仓并设置运行时环境变量：

```bash
# 加载 CANN 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 指定运行的 NPU 设备 ID（根据实际可用 chip 设置）
export TILE_FWK_DEVICE_ID=0
```

推荐将上述内容保存为 `env_setup.sh`，每次执行 `source env_setup.sh` 即可。

## ⚡️快速上手

### 算子

#### 1. 运行部分算子测试

```bash
# 使用 pytest 执行 GLM V4.5 Attention
pytest tests/ops/glm_v4_5/test_glm_attention.py

# 使用 pytest 执行 GLM V4.5 目录所有算子测试
pytest tests/ops/glm_v4_5 -v --forked
```

#### 2. 运行全部算子测试

```bash
pytest -v --forked
```

具体测试范围参考 [pytest.ini](./pytest.ini) 配置

#### 3. 指定 SoC

```bash
# 指定 NPU device id（覆盖 TILE_FWK_DEVICE_ID 环境变量）
pytest tests/ops/glm_v4_5 -v --forked --device 1
```

#### 4. 用例筛选说明

测试用例通过 `@pytest.mark.soc` 标注适用芯片，conftest.py 会根据当前设备的 soc_version 自动过滤不适配的用例（显示为 `SKIPPED`）。部分规模较大的用例默认已使用 `@pytest.mark.skip(reason="large test case")` 标注，需手动移除 skip 标注后运行。

### 整网

#### 1. 下载模型权重

```bash
python3 cannbot-skills/model/pypto-fused-op-integration/scripts/download_hf_model.py \
    --model-id Qwen/Qwen3-1.7B \
    --output-dir /path/to/models/Qwen3-1.7B
```

#### 2. 入网适配

```bash
bash cannbot-skills/model/pypto-fused-op-integration/scripts/restore_model_patch.sh \
    /path/to/models/Qwen3-1.7B qwen3_1_7b
```

#### 3. 验证

```bash
python3 modeling/transformers/qwen3_1_7b/ask_Qwen3-1.7B.py \
    --device 0 --prompt "你好" --use-pto \
    --model-path /path/to/models/Qwen3-1.7B
```

#### 4. 使用限制

当前 PyPTO 在整网场景下存在以下限制：

- **不支持多流并行**：PyPTO 算子无法跨流并行执行，相关任务需在单流内串行调度。
- **不支持 superkernel**：整网场景下 PyPTO 算子不支持 superkernel 融合能力。

## 常见问题排查

**Q: 报错 `NPU out of memory`**

部分算子（如 sparse_flash_attention、gated_delta_rule）的 `stitch_function_max_num` 参数会影响 workspace 大小，公式为 `workspace = totalSlot × (stitch_function_max_num + 1) × parallelism`。可在 impl 文件对应 `@pypto.frontend.jit` 的 `runtime_options` 中降低该值（如从 128 降至 1）以减少内存占用，代价是降低并行度。

**Q: 报错 `npu_format_cast ACL error 500001`**

TBE（Tensor Boost Engine）初始化失败，通常由缺少 Python 依赖导致。执行以下命令修复：

```bash
pip install scipy decorator -i https://mirrors.aliyun.com/pypi/simple/
```

**Q: 编译 PyPTO 时找不到 pto-isa 头文件**

确认 `PTO_TILE_LIB_CODE_PATH` 指向 pto-isa 仓库根目录（含 `include/` 子目录），且 pto-isa 版本与 PyPTO 兼容（建议两仓同步到最新）。

## 目录结构

```
pypto-gym/
├── modeling/                                # 模型端到端执行脚本
│   └── transformers/                        # 大模型推理示例
│       ├── infer.py                         # 通用推理入口
│       ├── download_hf_model.py             # HuggingFace 模型下载
│       ├── runtime_patch.py                 # PyPTO 运行时补丁覆盖层生成
│       ├── bench_qwen3_1_7b.sh              # Qwen3-1.7B 基准脚本
│       ├── deepseek-v2-lite-chat/           # DeepSeek V2 Lite Chat
│       ├── gemma4_31b_it/                   # Gemma4-31B-it
│       ├── gutenocr_3b/                     # GutenOCR-3B
│       ├── kimi_linear_48b_a3b/             # Kimi-Linear-48B-A3B
│       ├── llada2_moe/                      # LLaDA2-MoE
│       ├── minimax/                         # MiniMax M2.7
│       ├── openpangu_v5_7b/                 # openPangu-Embedded-7B
│       ├── phi_3_mini_4k_instruct/          # Phi-3-mini-4k-instruct
│       ├── qwen3_1_7b/                      # Qwen3-1.7B
│       ├── qwen3_5_9b/                      # Qwen3.5-9B
│       ├── qwen3_6_27b/                     # Qwen3.6-27B
│       ├── qwen3_vl_8b_instruct_unredacted_max/  # Qwen3-VL-8B
│       ├── spatial_ssrl_3b/                 # Spatial SSRL 3B
│       └── sample_inputs/                   # 公共样例输入
├── src/
│   └── pypto_gym/
│       ├── ops/                             # 算子样例根目录
│       │   └── pypto_tensor/                  # Tile 算子实现
│       │       ├── arctic/                  # Arctic LSTM Speculator
│       │       ├── common_utils/            # 公共工具（对比、日志、设备查询等）
│       │       ├── deepseek_v2_lite_chat/   # DeepSeek V2 Lite Chat MLA Prolog
│       │       ├── deepseek_v32_exp/        # DeepSeek V3.2 MLA / Sparse Attention / Lightning Indexer
│       │       ├── deepseek_v4/             # DeepSeek V4 MLA / Compressor / Window Attention
│       │       ├── gemma4_31b_it/           # Gemma4-31B-it GQA Decode Attention / Softmax
│       │       ├── glm_v4_5/                # GLM V4.5 Attention / MoE / FFN / Gate
│       │       ├── gutenocr_3b/             # GutenOCR-3B RMSNorm / SwiGLU MLP / MRoPE
│       │       ├── kimi_linear_48b_a3b/     # Kimi-Linear-48B-A3B KDA
│       │       ├── llada2_moe/              # LLaDA2-MoE Gate / Expert FFN / Grouped GEMM
│       │       ├── ling_3_0_flash/          # Ling 3.0 Flash KDA
│       │       ├── minimax/                 # MiniMax M2.7 MoE Grouped GEMM
│       │       ├── mtgr_segment_attention/  # MTGR Ragged Segment Attention
│       │       ├── openpangu_v5_7b/         # openPangu-Embedded-7B 整层融合（单层 BSH）
│       │       ├── phi_3_mini_4k_instruct/  # Phi-3-mini-4k-instruct RMSNorm
│       │       ├── qat/                     # 量化感知训练（对称/非对称，per-tensor/channel/group）
│       │       ├── qwen3_1_7b/              # Qwen3-1.7B RMSNorm + RoPE
│       │       ├── qwen3_5/                 # Qwen3.5 GDR 前向/反向
│       │       ├── qwen3_5_9b/              # Qwen3.5-9B Gated Delta Rule
│       │       ├── qwen3_6_27b/             # Qwen3.6-27B Gated Delta Rule
│       │       ├── qwen3_next/              # Qwen3-Next Gated Delta Rule
│       │       ├── qwen3_vl_8b_instruct_unredacted_max/  # Qwen3-VL-8B RMSNorm
│       │       ├── spatial_ssrl_3b/         # Spatial SSRL 3B RMSNorm + RoPE
│       │       └── experimental/            # 实验性算子
│       └── transformers/                    # 各个模型结构定义
│           ├── gemma4_31b_it/
│           ├── gutenocr_3b/
│           ├── kimi_linear_48b_a3b/
│           ├── llada2_moe/
│           ├── minimax_m27/
│           ├── minimax_m3/
│           ├── openpangu_v5_7b/
│           ├── phi_3_mini_4k_instruct/
│           ├── qwen3_1_7b/
│           ├── qwen3_5_9b/
│           ├── qwen3_6_27b/
│           ├── qwen3_vl_8b_instruct_unredacted_max/
│           └── spatial_ssrl_3b/
├── tests/                                   # 测试用例
│   ├── ops/                                 # 算子测试（与 ops/ 对应）
│   │   ├── arctic/                          # Arctic LSTM 算子测试
│   │   ├── chunked_gdr/                     # Chunk Gated Delta Rule 算子测试
│   │   ├── deepseek_v2_lite_chat/           # DeepSeek V2 Lite Chat 算子测试
│   │   ├── deepseek_v32_exp/                # DeepSeek V3.2 算子测试
│   │   ├── deepseek_v4/                     # DeepSeek V4 算子测试
│   │   ├── gemma4_31b_it/                   # Gemma4-31B-it 算子测试
│   │   ├── glm_v4_5/                        # GLM V4.5 算子测试
│   │   ├── gutenocr_3b/                     # GutenOCR-3B 算子测试
│   │   ├── kimi_linear_48b_a3b/             # Kimi-Linear-48B-A3B 算子测试
│   │   ├── llada2_moe/                      # LLaDA2-MoE 算子测试
│   │   ├── ling_3_0_flash/                  # Ling 3.0 Flash 算子测试
│   │   ├── minimax_m27/                     # MiniMax M2.7 算子测试
│   │   ├── mtgr_segment_attention/          # MTGR 分段注意力算子测试
│   │   ├── phi_3_mini_4k_instruct/          # Phi-3-mini-4k-instruct 算子测试
│   │   ├── pypto_pro/                       # PyPTO Pro 算子测试
│   │   ├── qat/                             # QAT 算子测试
│   │   ├── qwen3_1_7b/                      # Qwen3-1.7B 算子测试
│   │   ├── qwen3_5/                         # Qwen3.5 算子测试
│   │   ├── qwen3_5_9b/                      # Qwen3.5-9B 算子测试
│   │   ├── qwen3_6_27b/                     # Qwen3.6-27B 算子测试
│   │   ├── qwen3_next/                      # Qwen3-Next 算子测试
│   │   ├── qwen3_vl_8b_instruct_unredacted_max/  # Qwen3-VL-8B 算子测试
│   │   ├── spatial_ssrl_3b/                 # Spatial SSRL 3B 算子测试
│   │   ├── utils/                           # 测试工具函数
│   │   └── experimental/                    # 实验性算子测试
│   └── README.md                            # 测试目录说明
├── conftest.py                              # pytest 调度（多卡 / 多 SoC 筛选）
├── CONTRIBUTING.md                          # 贡献指南
├── pytest.ini
├── pyproject.toml
├── setup.py
├── requirements.txt
├── LICENSE
├── SECURITY.md
└── README.md
```

## 算子总览

| 所属模型 | 算子 | 说明 |
|---------|------|------|
| `arctic/` | sum_lstm | LSTM 推测器（Arctic-Inference），融合输入、RMSNorm、GELU、门控与细胞状态更新 |
| `deepseek_v2_lite_chat/` | mla_prolog | MLA Prolog 预计算 |
| `deepseek_v32_exp/` | mla_prolog_quant, lightning_indexer_prolog_quant, sparse_flash_attention_quant, sparse_attention_antiquant, mla_indexer_prolog_quant, lightning_indexer_quant | MLA Prolog、Lightning Indexer、稀疏注意力等量化/非量化算子 |
| `deepseek_v4/` | mla_prolog_v4, mla_prolog_quant_v4, lightning_indexer_prolog_quant_v4, compressor, compress_flash_attention, sparse_compress_flash_attention, win_attention, hc_pre | MLA Prolog、压缩器、压缩/稀疏 Flash Attention、窗口注意力等算子 |
| `gemma4_31b_it/` | gqa_decode_attn, attn_softmax | GQA 解码注意力（KV 头均值，带宽降低 75%）、3-pass Softmax |
| `glm_v4_5/` | attention_pre_quant, attention, attention_fusion, gate, select_experts, ffn_shared_expert_quant, moe_fusion | 注意力（含量化前处理与融合）、MoE 门控/专家选择/FFN/融合等算子 |
| `gutenocr_3b/` | swiglu_mlp, rms_norm, mrope | SwiGLU MLP、RMSNorm、多模态 RoPE |
| `kimi_linear_48b_a3b/` | kda_chunk | Kimi Delta Attention（线性注意力，prefill/chunk） |
| `llada2_moe/` | gate_select, expert_ffn, moe_grouped_gemm | MoE 门控选择、单专家 FFN、分组 GEMM（9.2x 端到端加速） |
| `ling_3_0_flash/` | chunk_kda, fused_recurrent_kda | KDA 融合算子（Chunk / Recurrent） |
| `minimax_m27/` | moe_grouped_gemm | MoE Grouped GEMM（256 专家，BF16，910B 适配，7.6x 加速） |
| `mtgr_segment_attention/` | mtgr_ragged_segment_attention | 可变长 Flash Attention 前向（4 段策略 mask 组合） |
| `openpangu_v5_7b/` | pangu_fused_layer_v2_bsh | 整解码层融合（RMSNorm+QKV(+bias)+RoPE+KV-cache+GQA Attention+O proj(+bias)+RMSNorm+SwiGLU FFN）；单层 BSH 动态 kernel，仅 decode |
| `phi_3_mini_4k_instruct/` | rms_norm | RMSNorm（D=3072），支持 ACLGraph |
| `qat/` | symmetric_per_tensor, symmetric_per_channel, asymmetric_per_group | 量化感知训练三模式（前向 + 反向） |
| `qwen3_1_7b/` | rms_norm_rope | RMSNorm + RoPE 融合 |
| `qwen3_5/` | gdr_fwd, gdr_bwd | Chunk Gated Delta Rule 前向/反向 |
| `qwen3_5_9b/` | gated_delta_rule | Chunk Gated Delta Rule 注意力（D=128, Nv=32, Nqk=16） |
| `qwen3_6_27b/` | gated_delta_rule | Chunk Gated Delta Rule 注意力（D=128, Nv=48, Nqk=16） |
| `qwen3_next/` | gated_delta_rule | Chunk Gated Delta Rule，线性复杂度 O(n)，支持 1K-1M+ 序列长度 |
| `qwen3_vl_8b_instruct_unredacted_max/` | rms_norm | RMSNorm（hidden_size=2048） |
| `spatial_ssrl_3b/` | rms_norm, rope | RMSNorm + RoPE（Vision 2D / 多模态 3D） |
| `experimental/attention/` | BSA, chunked_gdr, incre_flash_attention_gqa_antiquant, incre_flash_attention_mla, incre_flash_attention, pfa_flash_attention | BSA、Chunk GDR、增量 Flash Attention（GQA/MLA）、PFA Flash Attention 等算子 |
| `experimental/matmul/` | gmm_mxfp8, grouped_matmul_finalize_routing, grouped_matmul_swiglu_quant, quant_batch_matmul, quant_grouped_matmul_inplace_add, quant_matmul_reduce_sum, transpose_quant_batch_matmul | MXFP8 GMM、量化分组矩阵乘、量化批量矩阵乘、转置量化矩阵乘等算子 |
| `experimental/ops_transformer/` | flash_attention_mha, flash_attention_mha_grad, flash_attention_score, flash_attention_score_grad, fused_swiglu, fused_swiglu_grad, lightning_indexer, mla_prolog, mla_prolog_mxfp_quant_v3, mla_prolog_quant_hifp8_v3, page_attention_quant, sparse_attention_antiquant_fp8, sparse_attention_antiquant_kv_split, sparse_attention_grad_tnd, sparse_attention_tnd, attention_worker_combine, engram, flash_attention_fp8, kimi_attention_residuals | Flash Attention（MHA/Score/FP8，含量化/反向）、MLA Prolog（含量化）、SwiGLU、Page Attention、稀疏注意力、Engram、Kimi Attention Residuals 等算子 |
| `experimental/vector/` | ApplyAdamWV2, GatherPaKvCache, QkvRmsNormRopeCache, ApplyRMSProp, BNTrainingReduce, InplaceAddRmsNorm, InterleaveRope, moe_gating_topk 等 | AdamW / RMSProp 更新、Paged Attention KV cache gather、QKV RMSNorm RoPE cache 融合、RMSNorm、RoPE、MoE 门控等算子 |

## 添加新算子

1. 在 `ops/pypto_tensor/` 下新建子目录（若是通用算子，放入 `experimental/` 对应子类）。
2. 编写 kernel 实现文件，命名建议为 `*_impl.py`，对外暴露入口函数 / 配置类。
3. 在 `tests/ops/` 下新建同名子目录，添加 `test_*.py`，通过绝对路径引用 kernel。
4. 用 `@pytest.mark.soc("950", "910")` 标注适用 SoC，用 `@pytest.mark.world_size(N)` 标注多卡需求。
5. 编写 `README.md` 说明算子语义、shape 范围与预期性能，以及对应测试文件路径。

详细规范参见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 关联资源

- [PyPTO 主仓](https://gitcode.com/cann/pypto)
- [PyPTO 文档中心](https://pypto.gitcode.com)
- [PyPTO 贡献指南](https://gitcode.com/cann/pypto/blob/master/CONTRIBUTION.md)

## Agent 能力快速上手

PyPTO-Gym 提供 PyPTO 算子开发、PyPTO-Pro 算子开发、模型适配和算子产物校验四类 Agent 能力。完整索引与安装说明见 [cannbot-skills/README.md](cannbot-skills/README.md)，各场景的快速上手指南见：

- [PyPTO 算子开发快速上手](cannbot-skills/plugins-official/pypto-op-orchestrator/quickstart.md)
- [PyPTO-Pro 算子开发快速上手](cannbot-skills/plugins-official/pypto-pro-op-orchestrator/quickstart.md)
- [PyPTO 模型工具快速上手](cannbot-skills/plugins-official/pypto-model-tools/quickstart.md)
- [PyPTO 算子产物校验快速上手](cannbot-skills/plugins-official/pypto-kernel-validator/quickstart.md)

## 相关信息

- [许可证](LICENSE)：CANN Open Software License Agreement Version 2.0
- [安全声明](SECURITY.md)

## 联系我们

- **问题反馈**：通过 [GitCode Issues](https://gitcode.com/cann/pypto-gym/issues) 提交
- **功能建议**：通过 [GitCode 讨论区](https://gitcode.com/cann/pypto-gym/discussions) 交流
