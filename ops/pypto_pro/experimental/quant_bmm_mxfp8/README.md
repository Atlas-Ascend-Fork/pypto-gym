# QuantBatchMatmul MXFP8（PyPTO-Pro）

`quant_batch_matmul_mxfp8` 使用原生 PyPTO-Pro 实现 MXFP8 批量矩阵乘：读取已量化的 FP8 激活与权重及其 E8M0 分组比例，在 Cube 上完成分组反量化矩阵乘，可选地融合 FP32 bias，输出 BF16。算子消费的是**已经量化好的**字节，不做量化，也不调用任何 Torch 矩阵乘代算。基本块几何与片上资源地址是编译期常量；分核数由 host 按输出块数在运行期确定（上限 32）。接口、数值语义与支持范围以本目录的 kernel 和测试为依据。

## 产品支持情况

| 项 | 取值 |
| :--- | :--- |
| 芯片 | Ascend 950PR / 950DT（`NpuArch.DAV_3510`，kernel 装饰器 `arch="a5"`） |
| 能力依赖 | `pl.matmul_mx` / `pl.matmul_mx_acc`（MXFP8 + E8M0 分组量化，含可选 FP32 BiasTable 操作数）、`MemorySpace.ScaleLeft/ScaleRight/Bias`、`AccPhase.Partial/Final` 与 `STPhase.Final`、`pl.system.set_mm_layout_transform`（L0C 列优先排出）、`pl.load` / `pl.store` 的 `l2_cache_mode` 参数与 `pl.TLoadL2Hint`、`pl.set_validshape`、`pl.min`、`pl.get_block_num`、原生 `tiling_key`。这些能力在目标工具链中仅按 950 系列登记支持 |
| 分核数 | 上限 32（本机 32 Cube）。host 按输出块数在运行期确定，取 `min(总块数, 32)`；不运行期查询 `cube_core_num`、不读 SDK ini、不执行 host tiling、不构造资源计划 |
| 实测环境 | Ascend950PR_9589，32 Cube / 64 AIV，L2 = 128 MiB；CANN 9.3.0、pypto 0.3.0、pypto_pro（随 CANN 9.3.0 安装，无独立版本号）、PyTorch 2.12.0+cpu、torch_npu 2.12.0.post2、NumPy 2.5.1、Python 3.12.13 |
| 测试标注 | 独立 argparse 脚本，不经 pytest 收集，无 `soc` 标记；`--self-test` 为纯 CPU 路径，不导入 torch / torch_npu |
| 依赖 | pypto / pypto_pro、torch、torch_npu、NumPy |

## 功能说明

本文的 x1 是激活（矩阵乘左操作数），x2 是权重（右操作数），二者按 NN 逻辑存储：x1 为 `[B,M,K]`，x2 为 `[B,K,N]`。host 指 CPU 侧调用代码。

**支持域**：`BATCH`、`M`、`N` 为任意正整数（含 `M % 96 != 0`、`N % 256 != 0` 的尾块），`K` 为 `128` 的倍数；WeightNZ 另需 `N` 为 32 的倍数。逐条规则与报错见「约束说明」。

**数值契约**：输出与独立的 CPU FP64 golden 在 `rtol=0.008 / atol=0.001` 门限内**全元素通过**，这是本算子对支持域内**所有** shape 的保证。golden 在 CPU 上独立解码实际送入的 FP8/E8M0 字节、以 FP64 反量化并做矩阵乘，最后做一次 FP32→BF16 的 round-to-nearest-even，不读取设备输出。

tiling 数据是编译期常量——`BM/BN/BK = 96/256/128`、`KL1 = 256`、`SCALE_KL1 = 4096`，以及整张 L1/L0 地址偏移表。支持任意 shape 靠的**不是**按 shape 重新推导 tiling，而是把这套固定 tile 铺到 shape 上：M/N 尾块用 `pl.set_validshape` 收窄、K 尾窗用 `pl.min(窗口, 剩余)` 收缩、输出块数超过核数时用 grid-stride 多轮遍历。因此 `BATCH`、`M`、`N`、`K` 都是**运行时值而非编译期常量**，同一 tiling key **只编译一次**即可服务所有 shape。

一个公开的 host 函数服务全部 shape；每次公开调用在设备侧**只发射一个 kernel**，带 bias 也不例外——bias 由首条 MX MMAD 直接消费，host 不分配任何内部缓冲（无附带的清零 task），输出张量是唯一的设备分配。host 只校验属性、推导 shape 与启动块数，设备侧由**单个 kernel** 服务整个支持域：tile 几何与 L1/L0 地址表是编译期常量，shape 以运行时值传入而不进 tiling key，故同一 key 只编译一次。属实现细节，见 [DESIGN.md](DESIGN.md)。

MXFP8 的每个元素占 1 字节，为 E5M2 或 E4M3FN，x1 与 x2 必须同类型。E8M0 scale 每 32 个连续 K 元素一组，字节值 `u` 表示比例 `2**(u-127)`；`u=255` 不是合法编码。打包后的 scale 张量最后一维长度为 2，依次对应一个 64 元素窗口内的前、后两个 32 元素分组，**不是转置维**。

对于一个 batch 内的每个输出元素，计算：

```text
dequant(x) = FP8(byte) * 2^(E8M0_byte - 127)          # 每 32 个连续 K 元素共享一个 E8M0
out[b,m,n] = BF16( sum_k dequant(x1[b,m,k]) * dequant(x2[b,k,n]) + bias )
```

累加顺序是数值契约的一部分：**用 bias 初始化 FP32 累加器**（无 bias 时由首条 MX MMAD 直接初始化），再沿 K 升序做 `K/BK` 次 MX 乘加，最后一次累加使用 `AccPhase.Final`，并以 `STPhase.Final` 直接写出 BF16（RNE）。全程只做这一次输出舍入，中间累加始终保持 FP32。

`vector` bias 形状 `[N]`，在 B、M 两维广播；`batch` bias 形状 `[B,1,N]`，每个 batch 独立、在 M 维广播。带 bias 与不带 bias 两条路径**只差首条 MX MMAD 的一个操作数**：带 bias 时该指令消费 FP32 BiasTable（`init_with_zero=false`），不带 bias 时同一条指令直接初始化累加器；其余各次累加、相位与写出完全相同。

WeightNZ 使用原生 `pl.Tensor[..., pl.DT_FP8E4M3FN, pl.NZ]` 和 `pl.load`，以逻辑 K/N 偏移直接完成 GM NZ→L1 NZ 搬运；两份 scale 保持 ND。不做数值转换、不分配中间权重。ScaleNZ 不支持。

已验证范围是测试生成的有限 FP8 数据、E8M0 字节编码 124..128、FP32 bias 取值 `[-2,2]`。**不宣称**覆盖 NaN/Inf、极端 E8M0 指数或溢出行为。

## 函数原型

```python
import torch

def quant_batch_matmul_mxfp8(
    x1, x2, scale, *, offset=None, pertoken_scale=None, bias=None,
    output_dtype=None, x1_dtype=None, x2_dtype=None,
    pertoken_scale_dtype=None, scale_dtype=None, group_sizes=None, y_scale=None,
) -> torch.Tensor: ...
```

全部张量须位于同一 NPU。`scale` 是位置参数（x2 的 scale），`pertoken_scale` 是关键字参数（x1 的 scale）。

### 参数说明

| 参数 | dtype | shape / 类型 | 说明 |
| :--- | :--- | :--- | :--- |
| `x1` | float8_e5m2 / float8_e4m3fn | `[B,M,K]`（示例 `[8,96,4096]`） | 激活，ND 连续；必须与 `x2` 同 dtype |
| `x2` | 同 `x1` | `[B,K,N]`（示例 `[8,4096,1024]`） | 权重，ND 连续；WeightNZ 用例下为 format 29，逻辑 shape 不变且 `storage_offset()` 必须为 0 |
| `pertoken_scale` | float8_e8m0fnu / uint8 | `[B,M,K/64,2]`（示例 `[8,96,64,2]`） | x1 的 E8M0 scale，必填；末轴为一个 64 元素窗口内的两个 32 元素分组 |
| `scale` | float8_e8m0fnu / uint8 | `[B,K/64,N,2]`（示例 `[8,64,1024,2]`） | x2 的 E8M0 scale，位置参数；必须为 ND |
| `bias` | FP32 | `[N]` / `[B,1,N]` / `None`（示例 `[1024]` / `[8,1,1024]`） | 依次为 vector、batch、无 bias；作为 FP32 累加器的初值 |
| `output_dtype` | — | 必须显式为 `torch.bfloat16` | 本实现只输出 BF16，不支持 INT8 输出，故必须显式指定 |
| `group_sizes` | — | `[1,1,32]` 或 `None` | `None` 时推断为固定的 MX 分组 `[1,1,32]`；其他取值报错 |
| `pertoken_scale_dtype` / `scale_dtype` | — | `None` 或 E8M0 标识 | `None` 时对应 scale 必须是原生 `torch.float8_e8m0fnu`；显式指定 E8M0 时允许用 uint8 承载原始编码 |
| `offset` / `x1_dtype` / `x2_dtype` / `y_scale` | — | 仅接受 `None` | 传入其他值抛 `NotImplementedError`，不静默忽略 |

`B`、`M`、`N`、`K` 由 `x1`/`x2` 的实际 shape 推导，其余张量必须与之一致，否则报错。取值范围见「约束说明」。

scale 的两种载体（原生 E8M0 与 uint8 + dtype override）在 kernel 内都按 E8M0 位串解释，**不对指数编码做数值类型转换**。uint8 载体是为兼容不提供 `float8_e8m0fnu` 的 Torch 版本。

### 返回值说明

返回单个张量。

| 输出 | dtype | shape | 说明 |
| :--- | :--- | :--- | :--- |
| `out` | BF16 | `[B,M,N]`（示例 `[8,96,1024]`） | ND 连续；由 host 用 `torch.empty` 分配，不做设备侧预置 |

输出是这个算子**唯一**的设备分配：bias 由首条 MX MMAD 直接消费，host 不分配任何内部缓冲，因此返回张量上不挂内部缓冲引用，一次公开调用对应一个设备任务。

## 约束说明

**表 1 · shape 与维度（支持域）**

| 维度 | 要求 | 不满足时的报错 |
| :--- | :--- | :--- |
| `BATCH` | 任意正整数 | `BATCH/M/N/K must all be >= 1; got …` |
| `M` | 任意正整数，含 `M % 96 != 0` 的尾块 | 同上 |
| `N` | 任意正整数，含 `N % 256 != 0` 的尾块 | 同上 |
| `N`（仅 WeightNZ） | 另需 `N % 32 == 0`（NZ 按 32 列 fractal 打包）；ND 无此限制 | `WeightNZ packs x2 in 32-column fractals, so N must be a multiple of 32; got N=…` |
| `K` | `K % 128 == 0`（`BK = 128` 的倍数） | `K must be a multiple of BK=128; got K=…` |
| rank | `x1`/`x2` 必须 rank-3 | `x1 must be rank-3 [BATCH,M,K]; got rank …` |
| 各张量 shape 互洽 | 由 `x1`/`x2` 推导后逐一校验 | `x2.shape[:2] must equal (BATCH,K)=…` / `<name>.shape must be …, got …` |

| `K` 的隐含性质 | 说明 |
| :--- | :--- |
| `K % 128 == 0` ⟹ `K % 64 == 0` | 打包 scale 的 64 元素窗口整除，故 kernel **不需要** K 尾补零 |
| `K % 128 == 0` ⟹ `K % 32 == 0` | 满足 MX 组宽 32 |
| `K` **不必**是 `KL1=256` 或 `SCALE_KL1=4096` 的倍数 | 末窗按 `pl.min(窗口, 剩余)` 收缩，并用 `pl.set_validshape` 收窄对应 tile |
| `K == 128` 是合法下界 | 此时只有一个累加步，「首步」即「末步」，该步直接携带 `AccPhase.Final`（详见 [DESIGN.md](DESIGN.md)） |

**表 2 · 张量属性**

| 张量 | dtype | shape | format | 必填 | 违反时的报错 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `x1` | E5M2 或 E4M3FN，须与 `x2` 相同 | `[B,M,K]` | 仅 0（ND）或 2 | 是 | `x1/x2 must have the same E5M2 or E4M3FN dtype` |
| `x2` | 同 `x1` | `[B,K,N]` | ND（0/2）或 WeightNZ（29） | 是 | `x2 must be ND or FRACTAL_NZ (29)` |
| `pertoken_scale` | 原生 E8M0，或 uint8 + dtype override | `[B,M,K/64,2]` | 仅 0 或 2 | **是** | `pertoken_scale is required for these MXFP8 cases` |
| `scale` | 同上 | `[B,K/64,N,2]` | 仅 0 或 2 | 是（位置参数） | `x1, pertoken_scale, scale and bias must be ND; ScaleNZ is not supported` |
| `bias` | FP32 | `[N]`（vector）/ `[B,1,N]`（batch）/ `None` | 仅 0 或 2 | 否 | `bias must be None, FP32 [N] or FP32 [B,1,N]` |
| 输出 | BF16 | `[B,M,N]` | ND | — | 由 host `torch.empty` 分配 |

| 通用要求 | 违反时的报错 |
| :--- | :--- |
| 全部张量在同一 NPU 上 | `all inputs must be on the same NPU` |
| 全部张量连续、只支持 NN 逻辑 stride（不接受转置视图或 axis-0 padding） | `only NN logical strides are supported; no transposed views` |
| WeightNZ 只作用于 `x2` 且只在 E4M3FN 下支持；此时 `x1`、两份 scale、`bias` 与输出仍须为 ND | `WeightNZ supports only equal FP8_E4M3FN x1/x2` |
| WeightNZ 的 `x2` 必须 `storage_offset() == 0` | `WeightNZ x2 must use unshifted physical storage` |
| ScaleNZ 不支持 | 同 format 报错 |

**表 3 · 接口参数**

| 参数 | 接受值 | 其他值的行为 |
| :--- | :--- | :--- |
| `output_dtype` | 必须**显式**为 `torch.bfloat16` | `ValueError`（本实现只输出 BF16） |
| `group_sizes` | `[1,1,32]` 或 `None`（推断为固定 MX 分组） | `ValueError: group_sizes must be [1,1,32] or None` |
| `pertoken_scale_dtype` / `scale_dtype` | `None`（则对应 scale 必须是原生 `torch.float8_e8m0fnu`），或显式 E8M0 标识（则允许 uint8 承载原始编码） | `TypeError: …_dtype must specify FLOAT8_E8M0` / `… must carry E8M0 bits as native E8M0 or uint8` |
| `offset` / `x1_dtype` / `x2_dtype` / `y_scale` | 仅 `None` | `NotImplementedError: <name> is not supported for these MXFP8 cases; pass None`（不静默忽略） |

分核与调度、tiling key、数值契约（累加链与相位）、L2 cache hint 与核内同步等**实现细节不属于调用方约束**，见 [DESIGN.md](DESIGN.md)；本算子的实测结果（耗时、有效吞吐与带宽、pipe 占比、启动几何）见 [RESULTS.md](RESULTS.md)。

## 调用示例

若已按参数表准备好输入，可直接调用公开函数：

```python
import sys

sys.path.insert(0, "ops/pypto_pro/experimental/quant_bmm_mxfp8/op_kernel")
from quant_batch_matmul_mxfp8 import quant_batch_matmul_mxfp8

out = quant_batch_matmul_mxfp8(
    a, b, b_scale,
    pertoken_scale=a_scale,
    bias=bias_npu,
    output_dtype=torch.bfloat16,
    group_sizes=[1, 1, 32],
)
```

这里 `a/b/a_scale/b_scale/bias_npu` 是前述 shape 与 dtype 的 NPU 张量。原生 E8M0 scale 不需要 dtype override；用 uint8 承载原始编码时须同时指定 `pertoken_scale_dtype` 与 `scale_dtype` 为 E8M0 标识。

两套测试脚本都以独立 CPU FP64 golden 为判据，可直接用来构造完整调用：

```bash
# 性能实测所用的九个 shape（ND 六例 + E4M3FN WeightNZ 三例）
python ops/pypto_pro/experimental/quant_bmm_mxfp8/tests/test_quant_batch_matmul_mxfp8.py

# 只跑 ND 六例 / 只跑 WeightNZ 三例
python ops/pypto_pro/experimental/quant_bmm_mxfp8/tests/test_quant_batch_matmul_mxfp8.py --formats nd
python ops/pypto_pro/experimental/quant_bmm_mxfp8/tests/test_quant_batch_matmul_mxfp8.py --dtype e4m3fn --formats nz

# 只跑不带 bias 的两例
python ops/pypto_pro/experimental/quant_bmm_mxfp8/tests/test_quant_batch_matmul_mxfp8.py --bias-layouts none

# 覆盖整个支持域：M/N 尾块、K 单步与跨 scale 窗口、多轮 grid-stride、
# 两种 dtype、三种 bias 布局、ND 与 WeightNZ，共 19 个设备用例 + 5 条 host 拒绝规则
python ops/pypto_pro/experimental/quant_bmm_mxfp8/tests/test_quant_batch_matmul_mxfp8_general.py

# 只跑指定索引的用例（例如 3 与 12：K==BK 单步、64 块 2 轮）
python ops/pypto_pro/experimental/quant_bmm_mxfp8/tests/test_quant_batch_matmul_mxfp8_general.py --only 3 12

# 纯 CPU 自检：不导入 torch / torch_npu，不需要 NPU
python ops/pypto_pro/experimental/quant_bmm_mxfp8/tests/test_quant_batch_matmul_mxfp8.py --self-test
python ops/pypto_pro/experimental/quant_bmm_mxfp8/tests/test_quant_batch_matmul_mxfp8_general.py --self-test
```

均在仓库根目录执行，需先加载 CANN 与 PyPTO 环境。**目标 NPU 由环境变量 `TILE_FWK_DEVICE_ID` 选择，未设置时使用逻辑设备 0**（与参照算子 `quant_sparse_lightning_indexer` 的约定一致，测试脚本不提供 `--device` 参数）；例如可先执行 `export TILE_FWK_DEVICE_ID=0`，或在单条命令前加 `TILE_FWK_DEVICE_ID=1` 前缀。`--rtol` / `--atol` 覆盖判定门限（默认 `8e-3` / `1e-3`，即 [RESULTS.md](RESULTS.md) 「精度标准」给出的值），两套脚本都支持；`--seed` 固定输入随机数种子以复现同一批 FP8/E8M0 字节，`--dump-dir` 保存失败用例的完整输入位串、bias、golden 与设备输出，这两个只在性能实测所用的那套脚本上提供。退出码：0 全部通过，1 精度失败，2 环境、编译或执行异常。执行异常时立即结束，不继续后续用例、不调用 `empty_cache`，以免设备故障后的连带异常掩盖首个异常。

`test_..._general.py` 的 `--self-test` 是纯 CPU 的用例表自检：它校验用例表是否**真的**覆盖了它声称的各类边界（存在 `K == BK` 的单步用例、存在 `K > SCALE_KL1` 的跨窗口用例、存在 M 尾块与 N 尾块用例、存在多轮 grid-stride 用例、存在 WeightNZ 的 N 尾块用例、两种 dtype 与三种 bias 布局都出现），以及每条拒绝规则对应的 shape 确实违反了对应约束。

精度判据、用例清单与实测结果见同级目录的 [RESULTS.md](RESULTS.md)，实现结构与流水设计见 [DESIGN.md](DESIGN.md)。

内部 JIT 增加 `x2_nz` Tensor 参数：WeightNZ 场景传原始 E4M3FN NZ tensor，ND 场景传 `None`，对应分支由 `WeightNz` 编译期常量删除。公开接口、测试调用方式与 tiling key 不变。Pro 的 nullable Tensor 参数绑定会为 `None` 传空指针，动态维度槽补零，不读取该参数。

仍保留的边界：核内同步交给 `auto_mutex=True`，由框架按 TileGroup 的 mutex 所有权推导依赖，不使用手工事件编号。tiling 常量保持编译期固定，支持任意 shape 靠的是把这些固定 tile 铺到 shape 上，而不是按 shape 重新推导 tiling。本算子**只有 Cube 段**——没有 vector epilogue、没有 split-K、没有跨核部分和 workspace，因此也不含 ICache 预取；输出由 `pl.store(..., phase=STPhase.Final)` 从 L0C 直接写出 BF16。

**支持域尚未覆盖**：`K % 128 != 0`（需 K 方向尾块掩码或补零）、ScaleNZ、`x1` 的 NZ、以及非 NN 逻辑 stride。**性能实测只覆盖九个 shape**（`B/M/N/K = 8/96/1024/4096`，是支持域的子集）；支持域内其余 shape 的性能**尚未按测量协议采集**，因此本文不对其耗时做任何声明，[RESULTS.md](RESULTS.md) 中的性能数据也只对应这九个 shape。此外尾块工作量小于满块、且输出块数不是核数整数倍时末轮有 worker 空转，故负载均衡不由构造保证——这属为支持任意 shape 接受的代价，其性能影响未量化。
