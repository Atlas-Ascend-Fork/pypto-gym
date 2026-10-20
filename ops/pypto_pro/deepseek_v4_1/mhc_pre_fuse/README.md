# mhc_pre_fuse

## 产品支持情况

| 项目 | 说明 |
| :--- | :--- |
| 硬件平台 | Ascend 950PR（`Ascend950PR_9589`，`NpuArch=3510`）；cube 32 核 / vector 64 核 |
| 使用的计算单元 | AI Vector（AIV）核，不使用 Cube |
| 软件环境 | CANN 9.3.0、PyPTO-Pro DSL（pypto 0.2.1）、torch 2.8.0、torch_npu 2.8.0.post4 |
| 设备侧 kernel 名 | `PYPTO_mhc_pre_fuse_kernel` |
| 输入数据类型 | `residual` 为 BF16；`gemm_out_mul`、`gemm_out_sqrsum`、`mhc_scale`、`mhc_base` 为 FP32 |
| 输出数据类型 | `post_mix`、`comb_mix` 为 FP32；`layer_input` 为 BF16 |
| 支持的形状 | 词元数 `T` 为任意正整数；隐藏维度 `H` 为 64 的倍数且 `H / 64 >= 5`；混合路数 `m` 固定为 4 |

不接受的输入由 wrapper 直接拒绝并给出错误信息，不做静默降级：`H` 不是 64 的倍数或
`H / 64 < 5`；`m != 4`；`gemm_out_mul`、`gemm_out_sqrsum`、`mhc_scale`、`mhc_base`
的形状与 `residual` 不匹配；`sinkhorn_repeat < 1`。

## 功能说明

`mhc_pre_fuse` 是 mHC（m = 4 路残差混合）的前处理融合算子。它把下面四个步骤合并成
一个 NPU 向量核，一次调用同时产出三个结果：

```text
① RMSNorm 缩放        mixes = gemm_out_mul × rms          （24 路）
② 切分 mixes          pre = mixes[0:4]  / post = mixes[4:8] / comb = mixes[8:24]
③ Sinkhorn            comb 的 4×4 矩阵做行列交替归一化 → comb_mix
④ 残差预混合          layer_input = Σ_m residual[m] × pre[m]
```

`rms` 由上游给出的平方和 `gemm_out_sqrsum` 算出，算子本身不做平方和归约。

三个输出分别来自：`post_mix` ← ②，`comb_mix` ← ③，`layer_input` ← ④。

- `layer_input`：把 4 路残差按 pre 系数加权求和，得到本层的输入，是后续计算的起点；
- `post_mix`：后混合系数，每个词元 4 个；
- `comb_mix`：4×4 的组合矩阵，经 Sinkhorn 归一化后每列、每行的和都接近 1。最后一轮是
  列归一化，所以列和与 1 的差只有一个稳定项那么小；行和没有收尾的那一步，偏差来自前几
  轮累积的稳定项，更大一些。`tests/assets/mhc_pre_fuse_golden_cpu.py` 的自检给出这两个
  偏差分别为 1.07e-06 和 3.11e-03。

三个输出各自的完整计算公式（含稳定项、Sinkhorn 的迭代细节）见 `DESIGN.md` 的
「计算阶段划分」一节。算子与 `tests/assets/mhc_pre_fuse_golden_cpu.py`（本目录自带的
独立 CPU 实现）在数学上等价，但**不是逐位（bit 级）相同**：归约顺序和分块方式不同，
浮点舍入的路径也就不同。一致性按 `RESULTS.md` 给出的精度标准与容差判定。

## 函数原型

```python
# 把 op_kernel 目录加入 sys.path 后：
from mhc_pre_fuse import Knobs, mhc_pre_fuse_wrapper

post_mix, comb_mix, layer_input = mhc_pre_fuse_wrapper(
    gemm_out_mul,       # FP32  [T, 24]
    gemm_out_sqrsum,    # FP32  [T]
    mhc_scale,          # FP32  [3]
    mhc_base,           # FP32  [24]
    residual,           # BF16  [T, 4, H]
    knobs=Knobs(),      # 五个标量开关，默认值见下表
)
```

### 参数说明

| 参数 | 形状 | 类型 | 说明 |
| :--- | :--- | :--- | :--- |
| `gemm_out_mul` | `[T, 24]` | FP32 | 上游 GEMM 算出的 24 路结果。若带前导 split 轴（把 K 维拆成几段并行算的那一维），只取第 0 份：上游 GEMM 已经把各段加完了，这里只会收到一份 |
| `gemm_out_sqrsum` | `[T]` | FP32 | `residual` 的平方和，按 `4 × H` 个元素归约 |
| `mhc_scale` | `[3]` | FP32 | 三组缩放系数，依次作用于 pre、post、comb |
| `mhc_base` | `[24]` | FP32 | 三组偏置，布局与 24 路 mixes 对齐 |
| `residual` | `[T, 4, H]` | BF16 | 4 路残差 |
| `knobs` | — | `Knobs` | 五个标量开关，字段见下表；不传则用默认值 |

`knobs` 的字段（都是标量，默认值即算子契约的取值）：

| 字段 | 类型 | 说明 | 默认值 |
| :--- | :--- | :--- | ---: |
| `rms_eps` | float | RMSNorm 的稳定项 | `1e-6` |
| `mhc_pre_eps` | float | 加到 pre 系数上的常数 | `1e-6` |
| `mhc_sinkhorn_eps` | float | Sinkhorn 归一化的稳定项 | `1e-6` |
| `mhc_post_mult_value` | float | post 系数的乘数 | `1.0` |
| `sinkhorn_repeat` | int | Sinkhorn 迭代轮数，需 ≥ 1 | `10` |

### 返回值说明

三个张量都在 `residual` 所在的设备上，由本函数内部分配后返回。

| 返回值 | 形状 | 类型 | 说明 |
| :--- | :--- | :--- | :--- |
| `post_mix` | `[T, 4]` | FP32 | 后混合系数 |
| `comb_mix` | `[T, 16]` | FP32 | 组合矩阵，4×4 按行展开成一维 |
| `layer_input` | `[T, H]` | BF16 | 本层输入，即 4 路残差按 pre 系数加权求和的结果 |

## 约束说明

- `m` 固定为 4：`residual` 的第 1 维必须等于 4。
- `H` 需为 64 的倍数，且 `H / 64 >= 5`。这个下限保证每个词元至少拆出 5 个隐藏维度分块，
  撑满算子内部的 5 级流水；不满足时 wrapper 直接报错而不做降级。
- `T` 为任意正整数。当 `T` 不是 8 的倍数时，词元分块按 4、2、1 依次退化，任何 `T` 都不
  丢词元。块越小，一条搬运指令覆盖的词元越少，同样的残差流量就要拆成更多条指令，效率
  相应下降。
- `sinkhorn_repeat` 必须 ≥ 1。
- `residual` 需连续存放；`gemm_out_mul` 的前导 split 维（若存在）长度需 ≥ 1。
- `T` 与 `H` 参与编译期专门化：同一进程内首次遇到某个 `(T, H)` 组合会编译一次并缓存，
  后续调用直接复用。形状种类很多时会有一次性编译开销。
- 启动核数取自 PyPTO-Pro 平台信息中的向量核数量。
- 单次调用下发一个 kernel。wrapper 只做形状校验与输出分配，不做数值计算。

## 调用示例

```python
import torch
import torch_npu
import sys

sys.path.insert(0, "ops/pypto_pro/deepseek_v4_1/mhc_pre_fuse/op_kernel")
from mhc_pre_fuse import mhc_pre_fuse_wrapper

torch.npu.set_device(0)
T, H = 8192, 5120

gemm_out_mul = torch.randn(T, 24, dtype=torch.float32).npu()
mhc_scale = torch.randn(3, dtype=torch.float32).npu()
mhc_base = torch.randn(24, dtype=torch.float32).npu()
residual = torch.randn(T, 4, H, dtype=torch.bfloat16).npu()
# sqrsum 是 residual 的平方和，要与 residual 对得上（否则 rms 不再是 1 附近）
gemm_out_sqrsum = residual.float().square().sum(dim=(1, 2)).npu()

post_mix, comb_mix, layer_input = mhc_pre_fuse_wrapper(
    gemm_out_mul, gemm_out_sqrsum, mhc_scale, mhc_base, residual,
)
torch.npu.synchronize()
print(post_mix.shape, comb_mix.shape, layer_input.shape)
# torch.Size([8192, 4]) torch.Size([8192, 16]) torch.Size([8192, 5120])
```

随本算子一起交付的文件：

```
mhc_pre_fuse/
├── README.md                       # 本文件：接口与用法
├── DESIGN.md                       # 设计说明：阶段划分、切分与流水
├── RESULTS.md                      # 结果说明：用例、精度与性能
├── op_kernel/
│   └── mhc_pre_fuse.py             # 算子本体（kernel + 公开 wrapper）
└── tests/
    ├── test_mhc_pre_fuse.py        # 正确性用例，每个用例一个 test 函数
    └── assets/
        ├── mhc_pre_fuse_golden_cpu.py  # 独立 CPU 参考实现，兼作自检入口
        └── mhc_pre_fuse_precision.py   # 精度比较器
```

正确性用例由 `tests/test_mhc_pre_fuse.py` 提供，按 pytest 方式收集或直接运行该文件均可；
它调用 `tests/assets/` 下的参考实现与比较器，不依赖其它测试资产。用例清单与精度判据见
`RESULTS.md`。
