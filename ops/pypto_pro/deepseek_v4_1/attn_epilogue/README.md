# AttnEpilogue（PyPTO-Pro）

注意力输出端的融合算子：inverse RoPE + 回归组 + OCP MXFP8 量化 + 两级 block-scaled GEMM。
单次调用内完成 `o[T,64,512]`（bf16）到 `out[T,5120]`（bf16）的全部计算。

## 产品支持情况

| 产品 | 是否支持 |
|---|---|
| Ascend 950PR（`Ascend950PR_9589`，32 Cube / 64 Vector，1650 MHz） | **支持**（本机实测：精度 45/45，性能见 `RESULTS.md`） |
| Ascend 950PR（28 Cube 布局） | 支持（`tiling_key_for` 按 `cube_core_num` 选择布局；未在本机实测） |
| 其它产品 | 不支持（kernel 以 `arch="a5"` 编译，且分档依赖 A5 的 L0/L1/UB 容量） |

## 功能说明

记 `o` 为 attention 输出，按 64 个 head、每 head 512 维排列。算子执行：

1. **inverse RoPE**：对每 head 的 512 维向量按缓存的 `rope_sin`/`rope_cos` 做逆旋转，并回归组为
   8 个 group（每 group 1024 列，对应 `woa` 的分组维度）；
2. **MXFP8 量化**：按 32 个元素一块计算 E8M0 块缩放，得到 FP8E4M3 载荷与 scale（OCP MX 格式）；
3. **GEMM1**：分组矩阵乘 `y[t, g*1024+n] = Σ_k aq[t, g, k] · woa[g, n, k]`（K = 4096，FP8 × FP8 → FP32 累加）；
4. **y 量化**：对 `y` 再做一次 MXFP8 量化，得到 `yq`/`ys`；
5. **GEMM2**：`out[t, m] = Σ_k yq[t, k] · wob[m, k]`（K = 8192），结果以 bf16 写回。

两级 GEMM 均为 block-scaled（MX）矩阵乘，权重 `woa`/`wob` 由调用方在初始化阶段量化好（ND 布局 + E8M0 scale）。

## 函数原型

```python
attn_epilogue(o, woa, wob, descale_woa, descale_wob, rope_sin, rope_cos) -> torch.Tensor
```

## 参数说明

| 参数 | 类型/布局 | Shape | 说明 |
|---|---|---|---|
| `o` | bf16，ND，连续 | `[T, 64, 512]` | attention 输出，64 head × 512 维 |
| `woa` | FP8E4M3，ND | `[8, 1024, 4096]` | GEMM1 权重（8 个 group），由调用方预先量化 |
| `wob` | FP8E4M3，ND | `[5120, 8192]` | GEMM2 权重，由调用方预先量化 |
| `descale_woa` | FP8E8M0，ND | `[8, 1024, 64, 2]` | `woa` 的块缩放（每 32 个 K 元素一个 E8M0 scale） |
| `descale_wob` | FP8E8M0，ND | `[5120, 128, 2]` | `wob` 的块缩放 |
| `rope_sin` | fp32，ND | `[T, 64]` | inverse RoPE 的 sin 表 |
| `rope_cos` | fp32，ND | `[T, 64]` | inverse RoPE 的 cos 表 |

`T` 为运行期 token 数；除 `T` 外的形状均为固定维度。

## 返回值说明

- 返回 `torch.Tensor`：bf16、连续、shape `[T, 5120]`。
- 内部把 `T` 补齐到 16 的倍数（并行分档需要），**返回值只包含有效行**，不暴露补齐行。

## 约束说明

- `T >= 1` 的任意正整数；`o` 必须为 3 维且后两维恰为 `(64, 512)`；所有输入须为**同一 NPU 上的连续 ND 张量**。
- 权重与 scale **不接受 FRACTAL_NZ**（按 ND 行连续搬入 L1，片上再做转置视图）；量化由调用方完成，算子不做权重量化。
- 片上工作区：4 个 GM 缓冲区合计 `42240 * ceil(T/16)*16` 字节，由 wrapper 缓存复用（按 device/stream/padded 键）。
- 支持的分档由 tiling key schema 约束：两种硬件布局共 6 个合法 key，32 核布局下公开调度只产生 3 个编译版本，签名不含 `T` 的静态取值。
- 精度：与 torch_npu 原生参考**逐位一致**（`rtol=0, atol=0`），包含中间量（激活载荷/scale、`Y` 载荷/scale），见 `RESULTS.md`。

## 调用示例

```python
import torch
import torch_npu
from attn_epilogue import attn_epilogue

T = 256
o = torch.randn(T, 64, 512, dtype=torch.bfloat16, device="npu")
woa = torch.randint(-64, 64, (8, 1024, 4096), dtype=torch.int8, device="npu").view(torch.float8_e4m3fn)
wob = torch.randint(-64, 64, (5120, 8192), dtype=torch.int8, device="npu").view(torch.float8_e4m3fn)
descale_woa = torch.randint(120, 128, (8, 1024, 64, 2), dtype=torch.uint8, device="npu").view(torch.float8_e8m0fnu)
descale_wob = torch.randint(120, 128, (5120, 128, 2), dtype=torch.uint8, device="npu").view(torch.float8_e8m0fnu)
rope_sin = torch.randn(T, 64, dtype=torch.float32, device="npu")
rope_cos = torch.randn(T, 64, dtype=torch.float32, device="npu")

out = attn_epilogue(o, woa, wob, descale_woa, descale_wob, rope_sin, rope_cos)  # [T, 5120] bf16
```

运行测试与基准：

```bash
python -m pytest ops/pypto_pro/deepseek_v4_1/attn_epilogue/tests/test_attn_epilogue.py -q
python ops/pypto_pro/deepseek_v4_1/attn_epilogue/tests/test_attn_epilogue.py --benchmark --tokens 256
```

- 详细设计（阶段划分、切分策略、Buffer 与流水）见 `DESIGN.md`；
- 精度标准、绝对性能与理论评估（MFU / Bound / 利用率）见 `RESULTS.md`。
