# Kimi-Linear-48B-A3B-Instruct — NPU 迁移 + PyPTO KDA 集成

| 字段 | 值 |
|------|----|
| HuggingFace | `moonshotai/Kimi-Linear-48B-A3B-Instruct` (`KimiLinearForCausalLM`) |
| 权重目录 | `/data/models/Kimi-Linear-48B-A3B-Instruct`（可通过 `--model-path` 指定） |
| 代码来源 | 情况A — trust_remote_code：`modeling_kimi.py` / `configuration_kimi.py` 由模型仓库自带（config.json 的 auto_map），本仓库收录打补丁后的版本。 |
| transformers 版本 | 5.8.1（模型面向 4.57.1 — 已适配，见下） |
| 运行命令 | `source $ASCEND_HOME_PATH/set_env.sh && python3 modeling/transformers/kimi_linear_48b_a3b/ask_Kimi-Linear-48B-A3B.py --prompt "..." --output_length 40 --num_npus 4` |

## 架构

Kimi Linear 是混合线性注意力架构：3:1 的 **KDA**（Kimi Delta Attention，细粒度逐通道
门控的 gated delta rule）线性注意力层与全局 MLA 全注意力层交替。KDA 在 prefill 走 chunk
路径（`mode == 'chunk'`），decode 走单步 recurrent 路径（`T == 1`）。

## 集成范围 (Fused operators)

| Operation | PyPTO Integrated | Toggle flag | Fallback | Notes |
|-----------|:---:|-------------|----------|-------|
| KDA chunk (prefill) | Yes | `USE_PTO_KDA` | `chunk_kda` (torch `vec_chunk_kda`) | 融合 subchunk=16 KDA chunk kernel |
| KDA decode (T=1) | No | — | `fused_recurrent_kda` (torch `_naive_recurrent_kda`) | 仅 pto 化 chunk/prefill 路径；decode 走上游 torch recurrent |
| FusedRMSNormGated | No | — | torch `FusedRMSNormGated` | 后注意力门控归一化 |
| ShortConvolution | No | — | torch causal depthwise conv1d | KDA 前处理因果卷积 |
| 全注意力 (MLA) | No | — | eager / sdpa | 标准 HuggingFace 注意力 |

KDA chunk 注入替换 `KimiDeltaAttention.forward()` 中 `mode == 'chunk'` 分支的
`chunk_kda(...)` 调用。开关开启且形状在算子包络内（K=V=128，`use_qk_l2norm_in_kernel=True`）
时由 PyPTO kernel 处理；否则 `NotImplementedError` → 回退到 torch chunk。

## `fla` 依赖如何被替换

`modeling_kimi.py` 从 `fla` (flash-linear-attention) 导入 KDA 算子，而 `fla` 是
CUDA/Triton-only。本集成 **不使用** 假 `fla` 命名空间注入，而是在 `modeling_kimi.py` 顶部
用 guarded import 直接回退到本地纯 torch 实现：

```python
try:
    from fla.modules import FusedRMSNormGated, ShortConvolution
    from fla.ops.kda import chunk_kda, fused_recurrent_kda
    ...
except ImportError:
    from .kimi_fla_compat import (
        ShortConvolution, FusedRMSNormGated, fused_kda_gate,
        chunk_kda, fused_recurrent_kda,
        prepare_cu_seqlens_from_mask, prepare_lens_from_mask, tensor_cache,
    )
```

`kimi_fla_compat.py` 提供：

| fla symbol | compat 实现 |
|---|---|
| `fla.modules.ShortConvolution` | 因果深度可分卷积 conv1d (k=4) + 可选激活，带 cache |
| `fla.modules.FusedRMSNormGated` | RMSNorm(x) * sigmoid(gate)，最后一维归一化 |
| `fla.ops.kda.gate.fused_kda_gate` | `g = -5*sigmoid(exp(A_log)*(g_raw+dt_bias))` |
| `fla.ops.kda.chunk_kda` | `vec_chunk_kda`（torch 分块参考实现） |
| `fla.ops.kda.fused_recurrent_kda` | `_naive_recurrent_kda`（torch 朴素 recurrent） |
| `fla.ops.utils.index.*`, `fla.utils.tensor_cache` | torch helpers / passthrough |

## PyPTO sys.modules 注入开关

kernel 模块通过 `sys.modules["kimi_linear_48b_a3b_pto_kernels"]` 暴露：

| Variable | Type | Default | 说明 |
|----------|------|---------|------|
| `USE_PTO_KDA` | `bool` | `False` | 启用 PyPTO 融合 KDA chunk kernel (prefill) |
| `kda_chunk_wrapper` | callable | — | FLA-layout chunk wrapper（`@allow_in_graph`） |

ask 脚本在 `from transformers import` 之前 `sys.path.insert(0, ...)` 并
`import kimi_linear_48b_a3b_pto_kernels`，`--use_pypto` 时置 `USE_PTO_KDA = True`。

## transformers 5.8.1 适配补丁（应用于本工作副本的 modeling_kimi.py）

1. `OutputRecorder` import：tf 5.x 将其从 `transformers.utils.generic` 移到
   `transformers.modeling_utils`（try/except 回退）。
2. `KimiLinearModel.forward`：将非 `KimiDynamicCache` 的 `past_key_values`
   （tf>=5 `generate` 注入通用 `DynamicCache`）强制转换为 `KimiDynamicCache`。
3. `KimiDynamicCache.get_mask_sizes`：tf>=5 传入 `int` 的 query_length（旧版为
   `cache_position` tensor）— 两种都处理。

## 文件表

| File | Description |
|------|-------------|
| `__init__.py` | 包初始化（SPDX header stub） |
| `configuration_kimi.py` | `KimiLinearConfig`（原样拷贝，未改） |
| `kimi_fla_compat.py` | `fla` 的纯 torch 替代（非 KDA helpers + torch 基线 KDA op） |
| `modeling_kimi.py` | 模型图：`KimiDeltaAttention` 等；KDA chunk/decode 的 sys.modules 注入分发 |

## 相关目录

- **Ops**: `ops/pypto_tensor/kimi_linear_48b_a3b/kda/` — PyPTO KDA kernel
- **Scripts**: `modeling/transformers/kimi_linear_48b_a3b/` — ask / bench 脚本
- **Tests**: `tests/ops/kimi_linear_48b_a3b/` — 单算子精度测试
