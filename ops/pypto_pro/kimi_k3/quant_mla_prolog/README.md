# quant_mla_prolog

DeepSeek-V3 类 MLA 的 prolog 段（PyPTO-Pro，`@pl.jit`）。从量化后的 token 激活
出发，一次调用算出每个 token 的 96 个 head 的 Q，并把该 token 的 KV latent 原位
写进 Paged Attention 的 KV cache。

## 产品支持情况

| 产品 | 是否支持 |
|---|---|
| Ascend 950PR | **支持**（28 Cube / 56 Vector 与 32 Cube / 64 Vector 两种核数配置均通过全量用例；性能见 `RESULTS.md`） |
| 其它产品 | 不支持（kernel 按 A5 的 L0/L1/UB 容量与 VF 指令集实现） |

## 功能说明

固定几何：`DIM=7168`、`Q_LORA=1536`、`KV_LORA=512`、`ROPE=64`、`FEATURES=576`、
`N_HEADS=96`、`HEAD_DIM=192`、`PAGE=128`，自由轴只有 token 数 `T` 与 cache 页数
`blocks`。七个计算阶段（括号内是执行单元）：

| 阶段 | 计算 | 单元 |
| --- | --- | --- |
| M1 | `x ⊗ descale_x` @ `(wqa ⊗ descale_wqa)ᵀ`，FP32 累加 | Cube |
| M2 | `x ⊗ descale_x` @ `(wkva ⊗ descale_wkva)ᵀ`，复用同一份 x 的 L1 驻留 | Cube |
| M3 | `qa_raw` 过 RMSNorm(γ=`norm_weight_qa`, `norm_eps`) → BF16 → 逐 32 元素 MX 量化 | Vector |
| M4 | `qa_mx ⊗ qa_scale` @ `(wqb ⊗ descale_wqb)ᵀ` → BF16 | Cube |
| M5 | `kva_raw` 前 512 列过 RMSNorm(γ=`norm_weight_kva`)、后 64 列 RoPE 段绕一圈 BF16，拼成 576 → 量化 → 按 `cache_index` 散写 `kv_cache` | Vector |
| M6 | 逐 head：`q_latent[h] = q_head[:, h, 0:128] @ wkb[h]` → BF16 | Cube |
| M7 | `concat(q_latent[h], q_head[:, h, 128:192])` 逐 (token, head) 取 absmax → 量化出 `q` 与 `descale_q` | Vector |

整条链是一个 `@pl.jit`、一次 MixKernel 发射，内含 8 个 section 与 3 道全核 MIX
屏障。`T > 128` 按 `capacity = ceil(T/128) * 128` 在 kernel 内分块，尾块同样在
kernel 内补齐，host 侧没有 token 循环。

front 段的 `x` / `descale_x` 装载用动态 `valid_shape` 直接读 `[T, 7168]` 的有效
行，没有 pad 阶段。窗口外的 L1 行保留旧字节：FP8 与 E8M0 都没有 NaN 位型，而
RMSNorm 与 absmax 都是行内归约、matmul 行间不串，所以垃圾只落 pad 行；输出按
`[:T]` 切、`kv_cache` 只写 `t < T`，pad 行不外泄。

数值契约里几处不能单独改的细节（CPU 真值 `tests/assets/quant_mla_prolog_golden_cpu.py`
与此逐条对应）：

- **RMSNorm 在 FP32 做**：源序平方和、`sqrt` 之后再除，不换成乘倒数。
- **三处刻意的 BF16 绕行**：M3 的归一化结果进 MX 量化前转一次 BF16；M5 的 RoPE
  64 列与两个 γ 各绕一次 BF16；M4 的 Cube 输出按 BF16 落 GM。这几步会丢位，但
  算子契约如此，去掉会与真值分叉。γ 的绕行在 kernel 的 VF 里片上完成
  （`astype(BF16, CAST_RINT)` 与真值的 `.to(BF16)` 逐 bit 相同），host 不参与。
- **三处除法的下界**：RMSNorm 的分母吃 `norm_eps`；KV 的 per-tensor 量化用调用方
  给的 `qscale_kv`；`q` 的 per-(token, head) 量化取 `max(absmax, 1e-8)`。

## 函数原型

```python
# op_kernel 加入 sys.path 后：
from quant_mla_prolog import quant_mla_prolog

outputs = quant_mla_prolog(
    x, wqa, wqb, wkva, wkb,
    descale_x, descale_wqa, descale_wqb, descale_wkva,
    norm_weight_qa, norm_weight_kva,
    kv_cache, cache_index, qscale_kv,
    *, norm_eps, quant_mode_aw=1, quant_mode_c=0,
)
```

前 14 个参数按位置传，`norm_eps` 起的 3 个是 keyword-only。`qscale_kv` 默认 `None`，
`quant_mode_c=0` 的自然调用形式是整个不传它。`norm_eps` **无默认值**，漏传直接
`TypeError`。

### 参数说明

所有输入必须是 NPU 张量、与 `x` 同设备、且在声明排布下连续。

| 参数 | 形状 | dtype | 排布 |
| --- | --- | --- | --- |
| `x` | `[T, 7168]` | `float8_e4m3fn` | ND，行连续 |
| `wqa` | `[48, 7168, 32]` | `float8_e4m3fn` | NZ（`[O/32, K, 32]`，O=1536） |
| `wqb` | `[576, 1536, 32]` | `float8_e4m3fn` | NZ（O=18432） |
| `wkva` | `[18, 7168, 32]` | `float8_e4m3fn` | NZ（O=576） |
| `wkb` | `[96, 128, 512]` | `bfloat16` | ND，逐 head 的 `[QK_NOPE, KV_LORA]` |
| `descale_x` | `[T, 112, 2]` | `float8_e8m0fnu` | rank-3 |
| `descale_wqa` | `[1536, 112, 2]` | `float8_e8m0fnu` | rank-3 |
| `descale_wqb` | `[18432, 24, 2]` | `float8_e8m0fnu` | rank-3 |
| `descale_wkva` | `[576, 112, 2]` | `float8_e8m0fnu` | rank-3 |
| `norm_weight_qa` | `[1536]` | `float32` | γ |
| `norm_weight_kva` | `[512]` | `float32` | γ，只覆盖 NoPE 的 512 段 |
| `kv_cache` | `[blocks, 128, 1, 576]` | 见下表 | 逻辑形状，**原位输出** |
| `cache_index` | `[T]` | `int64` | 每个 token 的槽位；`< 0` 表示跳过 |
| `qscale_kv` | `[1]` | `float32` | KV 的 per-tensor 量化系数；`quant_mode_c=0` 时必须为 `None` |
| `norm_eps` | 标量 | float | RMSNorm 的 eps，必须为正且有限 |
| `quant_mode_aw` | 标量 | int | 只支持 `1`（权重与激活都走 MXFP8） |
| `quant_mode_c` | 标量 | int | KV cache 与 `q` 的量化模式，见下表 |

MX scale 的"群"宽度是 64 个元素一对（一对 = 两个 E8M0 字节，各覆盖 K 向连续 32
个元素），群数 = `ceil(K / 64)`。四路 scale 按 `[outer, groups, 2]` 逐维取等号校验。

`quant_mode_c` 决定 KV cache 的元素类型与 `q` 的终值形态；`kv_cache` 的逻辑形状
两种模式相同，物理 PA_NZ 分块数不同：

| `quant_mode_c` | `kv_cache` dtype | PA_NZ 物理分块 | `q` dtype | `descale_q` |
| --- | --- | --- | --- | --- |
| `0`（默认） | `bfloat16` | 36 块 × 16 列 | `bfloat16` | 不返回 |
| `1` | `float8_e4m3fn` | 18 块 × 32 列 | `float8_e4m3fn` | `[T, 96]` FP32 |

`blocks` 自由，但必须满足 `blocks * 128 ≥ T`。

### 返回值说明

返回 `dict`，键序固定，输出都在 `x.device` 上：

| 键 | 形状 | dtype | 说明 |
| --- | --- | --- | --- |
| `q` | `[T, 96, 576]` | 随 `quant_mode_c` | 连续；行数是实际 `T`，不是内部 capacity |
| `kv_cache_out` | 与入参同形 | 与入参同 dtype | 原位改写后的同一个张量（`data_ptr` 相同） |
| `descale_q` | `[T, 96]` | `float32` | 仅 `quant_mode_c=1` 返回 |

`quant_mode_c=0` 时 `descale_q` 这个**键不存在**，不是取值为 `None`。
`kv_cache_out` 是入参 `kv_cache` 本身，要保留原值的调用方自己先 `clone()`。

## 约束说明

- **`kv_cache` 只收逻辑 `[blocks, 128, 1, 576]`**，物理 PA_NZ 分块形状
  （`[blocks, 36, 128, 16]` / `[blocks, 18, 128, 32]`）一律 `ValueError`。两者
  字节序相同，调用方持有物理张量时按 `.view(blocks, 128, 1, 576)` 传入即可，
  零拷贝。kernel 侧以裸指针接收、按物理 PA_NZ 寻址，wrapper 不做任何 view。
- **四路 MX scale 只收逻辑 rank-3 且逐维取等号**，物理 NZ rank-4 拒收。rank-4 的
  通用还原要在 wrapper 里做 permute 搬运，与下面的零数据操作边界冲突。
- `quant_mode_aw` 只支持 `1`，传别的值 `ValueError`。
- `norm_eps` 是**运行期标量**，不是编译期特化轴，换值不会多编一份产物。
- `T ≥ 1` 任意整数，上界受 cache 容量限制（`blocks * 128 ≥ T`）。尾块由 kernel
  内处理，调用者不需要补齐。
- `cache_index` 为负的 token **整行跳过**，`kv_cache` 中未命中的槽位一个字节都
  不改。取值上界**不校验**：读回索引判上界要一次 D2H 同步，会把每次调用都卡在
  设备上，越界索引由调用方负责。同一槽位写入两次不在契约内 —— 设备与 CPU 真值
  的写入次序不可比。
- **单 launch、wrapper 零数据操作**：host 只做参数校验、读 `shape`/`dtype`/
  `device` 元数据、`torch.empty` 分配输出与 workspace、发射一次 kernel。没有
  view / cast / permute / contiguous：权重按物理 rank-3 直传并逐 fractal 装进
  L1，γ 的 BF16 绕行与 E8M0 窄化全部在 kernel 内。
- 输入校验不是形式要求：权重按物理 NZ 三维形状逐 fractal 搬运，排布传错时元素数
  仍然对得上、校验拦不住，结果是静默错值；非连续的 `x` 会按行距读到错的字节。
- **图模式注册不在本次交付范围**，只提供 eager 入口。
- `float8_e8m0fnu` 按 torch 2.10 的拼写使用，更老的 torch 未验证。
- 精度判据见 `RESULTS.md`。

## 调用示例

```python
import sys
sys.path.insert(0, "ops/pypto_pro/kimi_k3/quant_mla_prolog/op_kernel")

import torch
import torch_npu  # noqa: F401  注册 torch.npu 后端

from quant_mla_prolog import quant_mla_prolog

dev = "npu:0"
T, BLOCKS = 8, 1

def nz_weight(o, k):                      # [O/32, K, 32] E4M3
    w = (torch.randn(o, k) * 0.5).to(torch.float8_e4m3fn)
    return w.reshape(o // 32, 32, k).permute(0, 2, 1).contiguous().to(dev)

def e8m0(outer, k):                       # [outer, ceil(k/64), 2]，127 = ×1.0
    groups = (k + 63) // 64
    raw = torch.full((outer, groups, 2), 127, dtype=torch.uint8)
    return raw.to(dev).view(torch.float8_e8m0fnu)

outputs = quant_mla_prolog(
    (torch.randn(T, 7168) * 0.5).to(torch.float8_e4m3fn).to(dev),
    nz_weight(1536, 7168),                        # wqa
    nz_weight(18432, 1536),                       # wqb
    nz_weight(576, 7168),                         # wkva
    torch.randn(96, 128, 512).to(torch.bfloat16).to(dev),    # wkb
    e8m0(T, 7168),                                # descale_x
    e8m0(1536, 7168),                             # descale_wqa
    e8m0(18432, 1536),                            # descale_wqb
    e8m0(576, 7168),                              # descale_wkva
    torch.randn(1536, device=dev),                # norm_weight_qa
    torch.randn(512, device=dev),                 # norm_weight_kva
    torch.zeros(BLOCKS, 128, 1, 576,              # kv_cache（逻辑形状、原位输出）
                dtype=torch.float8_e4m3fn, device=dev),
    torch.arange(T, device=dev),                  # cache_index
    torch.tensor([0.5], device=dev),              # qscale_kv
    norm_eps=1e-06,
    quant_mode_c=1,
)
print(list(outputs), outputs["q"].shape, outputs["descale_q"].shape)
# ['q', 'kv_cache_out', 'descale_q'] torch.Size([8, 96, 576]) torch.Size([8, 96])
```

`quant_mode_c=0` 时 `kv_cache` 换成 `bfloat16`（逻辑形状不变）、`qscale_kv` 传
`None`，返回的 dict 只有 `q` 与 `kv_cache_out` 两个键。

运行测试：

```bash
D=ops/pypto_pro/kimi_k3/quant_mla_prolog
python3 $D/tests/assets/quant_mla_prolog_golden_cpu.py        # CPU 真值自检，不需要 NPU
TILE_FWK_DEVICE_ID=0 pytest $D/tests -q                       # 全量用例
TILE_FWK_DEVICE_ID=0 pytest $D/tests -q -k t8                 # 单个用例
```

`pytest.ini` 的 `testpaths` 是 `tests/ops`，裸跑 `pytest` 收不到本目录，必须显式
给路径。用例模块标了 `pytest.mark.soc("950")`；不标的话 `conftest.py` 会按 `910`
过滤，整个模块一条都不收还不报错。

- 设计（阶段划分、核间/核内切分、Buffer 划分、接口选择、流水）见 `DESIGN.md`；
- 用例、精度标准与性能结果见 `RESULTS.md`。
