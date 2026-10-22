# indexer_prologue_qw

DeepSeek V4.1 indexer 前置 Q/W 投影融合算子（PyPTO-Pro，`@pl.jit`）。一个 MIX kernel 完成两条
GEMM 路径与一个向量收尾：Q 路径 `qr @ wqb^T`（MXFP8）→ 末尾 `Dr=64` 列的 rotate-half RoPE →
MXFP4 量化打包；W 路径 `softmax_scale * (x @ ww^T)`（BF16）。

## 产品支持情况

| 产品 | 是否支持 |
|---|---|
| Ascend 950PR（`Ascend950PR_9589`，32 Cube / 64 Vector） | **支持**（21 个 P0 `T` 精度全通过，4 个目标 case 实测性能） |
| 其它产品 | 不支持（kernel 按 A5 的 L0/L1/UB 容量与 VF 指令集实现） |

## 功能说明

固定几何：`dim=5120`、`q_lora=1280`、`N=32`（head 数）、`D=128`（head dim）、`Dr=64`（RoPE 长度），
唯一动态轴 `T ∈ [1, 262144]`。

- **Q 路径**：`qr @ wqb^T` 的 MXFP8 GEMM——E4M3 字节、每 32 个 K 一个 E8M0 块 scale、FP32 累加；
  随后只对末尾 `Dr=64` 列做 rotate-half RoPE（两半各 32 维）；最后按每 32 列一组做 MXFP4 量化
  （E2M1 最近邻、`.5` 处远离零；E8M0 对 `amax/6` 向上取整到 2 的幂），每字节打包两个 nibble
  （偶列在低 4 bit）。
- **W 路径**：`softmax_scale * (x @ ww^T)`，K=`dim=5120`，FP32 累加后统一乘入缩放。

模板按 `n_tiles = ceil(T/BASE_M)`（`BASE_M=128`）选择：`n_tiles ≥ 32` 走 split-T（一个核独占一个
`T` tile 的 32 个 head，W 的 K 归约交错进 head 循环）；否则走 split-K（head 轴切 16 个 2-head block，
`n_tiles ≤ 12` 时 W 的 K 轴一并切分并走 workspace 确定性归约）。尾块（`T` 非 128 整数倍）全部在
kernel 内处理，调用者无需补齐。

## 函数原型

```python
from indexer_prologue_qw import indexer_prologue_qw, SOFTMAX_SCALE

q, descale_q, w = indexer_prologue_qw(
    x, qr, wqb, ww, descale_qr, descale_wqb, rope_sin, rope_cos,
    softmax_scale=SOFTMAX_SCALE,   # 默认 0.08838834764831845 (= 128**-0.5)
)
```

## 参数说明

| 参数 | 类型/布局 | Shape | 说明 |
|---|---|---|---|
| `x` | BF16，ND | `[T, 5120]`（行数可大于 `T`） | W 路径输入；多出的行按死区忽略 |
| `qr` | FP8E4M3FN 或 uint8，ND | `[T, 1280]` | Q 路径输入（MXFP8 载荷） |
| `wqb` | FP8E4M3FN 或 uint8，**FRACTAL_NZ** | `[4096, 1280]` | Q 路径权重（`4096 = 32 head × 128`） |
| `ww` | BF16，**FRACTAL_NZ** | `[32, 5120]` | W 路径权重 |
| `descale_qr` | uint8（E8M0），ND | `[T, 20, 2]` | `qr` 的成对块 scale（40 组 = `1280/32`） |
| `descale_wqb` | uint8（E8M0），ND | `[4096, 20, 2]` | `wqb` 的成对块 scale |
| `rope_sin` | FP32，ND | `[T, 64]` | rotate-half RoPE 的 sin 表 |
| `rope_cos` | FP32，ND | `[T, 64]` | rotate-half RoPE 的 cos 表 |
| `softmax_scale` | float（可选） | 标量 | W 路径缩放，默认 `128**-0.5` |

## 返回值说明

三个输出均在 `qr.device` 上：

| 名称 | Shape | 类型 | 说明 |
|---|---|---|---|
| `q` | `[T, 32, 64]` | uint8 | 打包的 E2M1 nibble（偶列低 4 bit、奇列高 4 bit） |
| `descale_q` | `[T, 32, 2, 2]` | uint8 | E8M0 指数；不足一组时用 `127`（`E8M0_BIAS`）填充 |
| `w` | `[T, 32]` | float32 | `softmax_scale * (x @ ww^T)` |

`T` 取自 `qr.shape[0]`；`x` 的多余行被忽略。

## 约束说明

- `wqb`、`ww` 必须按 **FRACTAL_NZ** 提供；其余输入为 ND。
- `T` 为唯一动态轴，范围 `[1, 262144]`；`x` 的行数可大于 `T`（多余行为死区）。
- **尾块由 kernel 内处理**：Q 累加器保持 `[BASE_M, D]` 物理窗口，L1 载入按 `set_validshape` 夹到
  有效行，向量收尾按 AIV 半边夹紧 store；W 累加器按运行期有效 `M` 裁剪 store。调用者不需要补齐。
- **单 launch**：公开 wrapper 只做 shape/dtype 校验、`torch.empty` 分配输出与 workspace、启动一次
  kernel；不含任何 host 侧计算或 reshape，host 调用链没有循环。
- `E2M1` 量化表在**片内**生成（`LUT_LEN=1088` 的中点桶表），不需要 host 传入任何私有输入。
- 精度要求见 `RESULTS.md`（`pro_scheme_a`：`w` 用 `rtol=atol=2e-2`；`q`/`descale_q` 为整型按位
  比较，并豁免三类量化边界差异）。

## 调用示例

```python
import torch
import torch_npu
from indexer_prologue_qw import indexer_prologue_qw

T = 2048
x = torch.randn(T, 5120, dtype=torch.bfloat16, device="npu")
qr = torch.randint(-64, 64, (T, 1280), dtype=torch.int8, device="npu").view(torch.float8_e4m3fn)
wqb = torch.randint(-64, 64, (4096, 1280), dtype=torch.int8, device="npu").view(torch.float8_e4m3fn)
ww = torch.randn(32, 5120, dtype=torch.bfloat16, device="npu")
descale_qr = torch.randint(120, 128, (T, 20, 2), dtype=torch.uint8, device="npu")
descale_wqb = torch.randint(120, 128, (4096, 20, 2), dtype=torch.uint8, device="npu")
rope_sin = torch.randn(T, 64, dtype=torch.float32, device="npu")
rope_cos = torch.randn(T, 64, dtype=torch.float32, device="npu")

# wqb / ww 需先转为 FRACTAL_NZ（torch_npu.npu_format_cast(ww, 29)）
q, descale_q, w = indexer_prologue_qw(x, qr, wqb, ww, descale_qr, descale_wqb, rope_sin, rope_cos)
```

运行测试与基准：

```bash
D=ops/pypto_pro/deepseek_v4_1/indexer_prologue_qw
python $D/tests/assets/indexer_prologue_qw_golden_cpu.py   # CPU golden 自检
python $D/tests/test_indexer_prologue_qw.py                # 全量精度矩阵 + 4 个目标 case 性能
python $D/tests/test_indexer_prologue_qw.py --t 1,72,2048 --skip-perf   # 精度子集
python $D/tests/test_indexer_prologue_qw.py --perf all     # 全部 21 个 P0 的时延
pytest $D/tests -q
```

- 设计（阶段划分、核间/核内切分、Buffer、流水）见 `DESIGN.md`；
- 用例、精度标准与性能（含 MFU/Bound/利用率）见 `RESULTS.md`。
