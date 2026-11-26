# flash_kda_cp_preprocess（PyPTO-Pro）

## 产品支持情况

已在 Ascend 950PR（32 AIC、64 AIV）验证。依赖 `torch_npu` 和 PyPTO Pro。精度和性能规格见 [RESULTS.md](RESULTS.md)。

## 功能说明

按 64 token 分块预处理 Flash-KDA。Q/K 用 `x × rsqrt(Σx² + 1e-6)` 逐行归一化后转 BF16。门控 `lower_bound × sigmoid(exp(a_log) × (g + dt_bias))` 在 FP32 中逐块累加得到 G，beta 经 sigmoid 后用于构造三角系统。每块用 FP16 幂零倍增求逆，生成 `u` 和 `w`。摘要在 FP32 中用 `φ = diag(decay) + wᵀ @ kg`、`R ← R @ φ`、`ψ ← ψ @ φ + uᵀ @ kg` 递推。

## 函数原型

```python
flash_kda_cp_preprocess_metadata(q, v, layout_qkv, cu_seqlens=None)
flash_kda_cp_preprocess(
    q, k, v, g, beta, scale, a_log, dt_bias, lower_bound, layout_qkv, metadata,
    cu_seqlens=None,
)
```

### 参数说明

B 为序列数，Hq 为 Q/K 头数，Hv 为 V 头数，S 为稠密序列长度，T 为打包的 token 总数。所有张量在同一 NPU 上，按其声明布局连续存储。D 固定为 128。

| 参数 | dtype / 类型 | 形状与含义 |
|---|---|---|
| q, k | BF16 | BNSD 为 [B,Hq,S,D]，BSND 为 [B,S,Hq,D]，TND 为 [T,Hq,D] |
| v, g | BF16 | 与 q 布局相同，Hq 换成 Hv。g 为门控输入 |
| beta | BF16 | 与 v 布局相同，省略最后一维 |
| scale | float | qg 的缩放系数，常用 D⁻⁰·⁵ |
| a_log, dt_bias | FP32 | [Hv]、[Hv,D]，门控参数 |
| lower_bound | float | 门控下界，取值 [-5,0] |
| layout_qkv | str | BNSD、BSND 或 TND |
| metadata | INT32 | 一维调度表，由同设备上的 metadata 入口生成 |
| cu_seqlens | INT32 或 None | TND 必填 [B+1]，从 0 严格递增至 T。稠密布局传 None |

### 返回值说明

返回顺序和形状如下。稠密 token 输出统一按 BNSD 存储。TND 的 token 输出按 [T,Hv,D] 存储。C 为 ⌈S/64⌉+B−1，TND 用 T 代替 S。输出由算子分配，输入保持原样。

| 输出 | dtype | 形状及含义 |
|---|---|---|
| transition, state_ext | FP32 | [B,Hv,D,D]，对应 R 和 ψ |
| decay | FP32 | [B,Hv,C,D]，块末累积衰减 |
| qg, kg | BF16 | [B,Hv,S,D] 或 [T,Hv,D]，带门控的归一化 Q/K |
| u, w | BF16 | 同 qg，三角系统求解后的块内操作数 |
| mqk | BF16 | [B,Hv,C,64,64]，下三角块内分数 |

## 约束说明

Hv 必须是 Hq 的正整数倍，V 头 h 对应 Q/K 头 `h // (Hv // Hq)`。每条序列长度为正，允许短尾块。TND 的边界校验在 metadata 入口完成，主入口须使用同一次规划且边界未改动的 `metadata` 和 `cu_seqlens`。每条序列只有前 ⌈长度/64⌉ 个 mqk 槽有效。其余 decay 槽为零，其余 mqk 槽未定义。

## 调用示例

```python
from op_kernel.flash_kda_cp_preprocess import (
    flash_kda_cp_preprocess,
    flash_kda_cp_preprocess_metadata,
)

# q/k/v/g 为 [1, 1, 1024, 128] BF16 NPU 张量，beta 为 [1, 1, 1024]。
md = flash_kda_cp_preprocess_metadata(q, v, "BNSD")
outputs = flash_kda_cp_preprocess(
    q, k, v, g, beta, 128 ** -0.5, a_log, dt_bias, -0.1, "BNSD", md,
)
```
