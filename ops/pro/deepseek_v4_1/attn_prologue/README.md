# AttnPrologue（PyPTO Pro，KV compress 融合）

本实现在 A2 KV 支路内直接完成 `kv_compress_epilog_v2` 的 FP8 E4M3 group-32
量化，写出 544B cache 记录，不再需要单独的 kcev2 kernel，也不再经过
E4M3(scale=1) → BF16 的二次量化。

- 目标芯片：Ascend 950（`ascend950`）
- 算子入口：`AttnPrologue`（ACLNN：`aclnnAttnPrologue` /
  `aclnnAttnPrologueGetWorkspaceSize`）
- Torch 接口：`torch.ops.pypto_gym.attn_prologue`（前端
  `pypto_gym.attn_prologue`）

本目录沿用压缩 KV 契约，每条记录包含 512 字节 FP8 数据和 32 字节分组 scale。调用原 `ops/pypto_pro` 版本且按 512 字节分配 cache 的用户，切换到本接口时需要调整 cache 的容量和读取方式。

## 功能说明

本实现完成 QA/KV 投影、QR RMSNorm+动态量化、QB 投影和尾部 RoPE；KV 支路
在 kernel 内直接执行 `kv_compress_epilog_v2` 的 FP8 E4M3 group-32 量化，
写出 544B cache 记录。
这消除了“E4M3(scale=1) → BF16 → E4M3(group scale)”的双重量化。

记 `Deq` 为按 E8M0 scale 的逻辑反量化，`Quant32` 为每 32 元素一组的动态量化：

```text
X = Deq(x, descale_x)
qa = X @ Deq(wqa, descale_wqa)^T          # [T,R]
kv = X @ Deq(wkv, descale_wkv)^T          # [T,D]

qr, descale_qr = Quant32(RMSNorm(qa, norm_weight_qr))
q_proj = Deq(qr, descale_qr) @ Deq(wqb, descale_wqb)^T
q = BF16(TailRoPE(reshape(q_proj, [T,N,D])))

kv_bf16 = BF16(TailRoPE(RMSNorm(kv, norm_weight_kv)))   # [T,D]
```

KV compress（每 32 个元素一组）：

```text
amax = max(abs(kv_bf16[group]))
s0   = max(amax, 1e-4) / 448
s    = 2^ceil(log2(s0))                  # s0 恰为 2 的幂时保持不变
payload = E4M3FN(clamp(kv_bf16 / s, -448, 448))
scale   = BF16(s)

record = payload[512B] || scale[16 × BF16 = 32B]   # 544B
kv_cache[cache_index[t]] = record
```

## 函数原型

```python
q, qr, descale_qr = attn_prologue(
    x,                              # E4M3 [T,5120]
    wqa, wqb, wkv,                  # E4M3 FRACTAL_NZ
    descale_x, descale_wqa, descale_wqb, descale_wkv,
    norm_weight_qr, norm_weight_kv,
    rope_sin, rope_cos,
    cache_index,                    # INT64 [T]
    kv_cache,                       # UINT8 [P,BS,1,C>=544]，原地更新
    qr_norm=None,                   # 可选：BF16 [T,1280] 归一化 QR 输出 buffer
    norm_eps=1e-6,
)
```

`qr_norm` 是可选输出：不传时 host tiling 选择 `QrNorm=0` 变体，kernel 在编译期
去掉对应分支和 GM 写回；传入时选择 `QrNorm=1` 变体。`rope_sin` / `rope_cos`
为 `[T,64]` FP32 逐元素表，GPTJ_INTERLEAVED 布局（与 V4.1 网络及 attn_epilogue
的 inverse RoPE 一致）；网络标准表每对频率重复两次（`cos[2i]=cos[2i+1]`，sin
同理），同一 token 的表在全部 Q head 与 KV 上共享。

## 输入输出契约

`T` 为 token 数，范围 `[1, 262144]`。所有张量必须在同一个 NPU 设备上。

| 名称 | dtype | shape | 格式 | 说明 |
|---|---|---|---|---|
| `x` | E4M3FN | `[T, 5120]` | ND | 已由调用方完成 MXFP8 量化的激活 |
| `wqa` | E4M3FN | `[1280, 5120]` | FRACTAL_NZ | QA 权重 |
| `wqb` | E4M3FN | `[32768, 1280]` | FRACTAL_NZ | QB 权重 |
| `wkv` | E4M3FN | `[512, 5120]` | FRACTAL_NZ | KV 权重 |
| `descale_x` | E8M0 | `[T, 80, 2]` | ND | 激活的 group-32 scale |
| `descale_wqa` | E8M0 | `[1280, 80, 2]` | ND | QA 权重 scale |
| `descale_wqb` | E8M0 | `[32768, 20, 2]` | ND | QB 权重 scale |
| `descale_wkv` | E8M0 | `[512, 80, 2]` | ND | KV 权重 scale |
| `norm_weight_qr` | FP32 | `[1280]` | ND | QR RMSNorm gamma |
| `norm_weight_kv` | FP32 | `[512]` | ND | KV RMSNorm gamma |
| `rope_sin` | FP32 | `[T, 64]` | ND | GPTJ_INTERLEAVED sin 表 |
| `rope_cos` | FP32 | `[T, 64]` | ND | GPTJ_INTERLEAVED cos 表 |
| `cache_index` | INT64 | `[T]` | ND | 每个 token 的 cache slot；`< 0` 跳过写 |
| `kv_cache` | UINT8 | `[B, S, 1, C>=544]` | ND | 原地更新；自动读取 view stride，支持页/行 padding |
| `norm_eps` | float | 标量 | — | RMSNorm epsilon，默认 `1e-6` |
| `qr_norm`（可选） | BF16 | `[T,1280]` | ND | 传入时额外写归一化 QR 给 Indexer；不传则跳过 |
| `block_stride`（内部） | int | 标量 | — | ACLNN 内部 attr；Torch binding 自动取 `kv_cache.stride(0)` |
| `row_stride`（内部） | int | 标量 | — | ACLNN 内部 attr；Torch binding 自动取 `kv_cache.stride(1)` |

输出：

| 名称 | dtype | shape | 说明 |
|---|---|---|---|
| `q` | BF16 | `[T, 64, 512]` | 经 TailRoPE 的 Q |
| `qr` | E4M3FN | `[T, 1280]` | 量化后的低秩 Q |
| `descale_qr` | E8M0 | `[T, 20, 2]` | QR 的 group-32 scale |
| `qr_norm`（可选） | BF16 | `[T, 1280]` | 传入时写出的归一化 QR |

KV cache 单条记录 544 字节：`[0:512]` 为 E4M3 payload，`[512:544]` 为
16 个 group-32 BF16 scale。四维 cache `[B,S,1,C]` 的物理记录地址为
`(slot//S)*page_stride + (slot%S)*row_stride`；binding 自动取
`page_stride=kv_cache.stride(0)`、`row_stride=kv_cache.stride(1)`。两个 stride
必须 32 字节对齐，`row_stride>=544`、`page_stride>=S*row_stride`，底层 storage
必须覆盖 `(B-1)*page_stride + (S-1)*row_stride + 544`。`slot < 0` 跳过；非连续
view 可直接传入，不能用 `.contiguous()` 副本替代。`kv_compress_epilog_v2`
使用同一套分页协议。

约束：

- 权重必须在加载期用一次 `torch_npu.npu_format_cast(w.contiguous(), 29)` 转为
  FRACTAL_NZ；算子不会自动转换；
- 除三个权重外，其余输入/输出必须是 ND；`kv_cache` 可以是带页/行空洞的非连续
  view（自动读取 stride），其它张量需 contiguous；
- 仅提供前向计算；推理时使用 `torch.no_grad()`；
- `rope_sin/rope_cos` 为 GPTJ_INTERLEAVED 布局：相邻输入对 `(z[2i], z[2i+1])`
  旋转后写回相邻位置；网络标准表每对频率重复两次
  （`cos[2i]=cos[2i+1]`，sin 同理）。

## 调用方式

### 直接 `torch.ops`（推荐）

```python
import torch
import torch_npu
import pypto_gym

pypto_gym.load()   # 加载 libpypto_gym_attn_prologue.so，注册 torch.ops.pypto_gym

q, qr, descale_qr, kv_cache = torch.ops.pypto_gym.attn_prologue(
    x, wqa, wqb, wkv,
    descale_x, descale_wqa, descale_wqb, descale_wkv,
    norm_weight_qr, norm_weight_kv,
    rope_sin, rope_cos, cache_index, kv_cache,
    norm_eps=1e-6,
)
```

直接 `torch.ops` 返回 4 个张量：`q`、`qr`、`descale_qr`、以及被原地更新的
`kv_cache`；`descale_qr` 的 dtype 为 E8M0。调用是异步的：在当前 NPU stream
入队，读取结果或计时前请 `torch.npu.synchronize()`。直接用 `torch.ops` 前必须
先 `pypto_gym.load()`（或访问一次 `pypto_gym.attn_prologue`）。

### 前端便捷接口

```python
q, qr, descale_qr = pypto_gym.attn_prologue(
    x, wqa, wqb, wkv,
    descale_x, descale_wqa, descale_wqb, descale_wkv,
    norm_weight_qr, norm_weight_kv,
    rope_sin, rope_cos, cache_index, kv_cache,
    norm_eps=1e-6,
)
```

前端返回 3 个张量，`descale_qr` 为 int8 字节视图；`kv_cache` 仍原地更新。

### JIT 方式（直接执行 `op_kernel`）

```python
import importlib.util
import sys
from pathlib import Path

import torch
import torch_npu

kernel_path = Path("ops/pro/deepseek_v4_1/attn_prologue/op_kernel/attn_prologue.py").resolve()
spec = importlib.util.spec_from_file_location("attn_prologue_kernel", kernel_path)
mod = importlib.util.module_from_spec(spec)
sys.modules["attn_prologue_kernel"] = mod
spec.loader.exec_module(mod)

torch.npu.set_device("npu:0")
q, qr, descale_qr = mod.attn_prologue(
    x, wqa, wqb, wkv,
    descale_x, descale_wqa, descale_wqb, descale_wkv,
    norm_weight_qr, norm_weight_kv,
    rope_sin, rope_cos, cache_index, kv_cache,
    qr_norm=None,
    norm_eps=1e-6,
)
torch.npu.synchronize()
```

JIT wrapper 与 `torch.ops`/ACLNN 最终执行同一份设备计算代码，返回 3 个张量
（`q`/`qr`/`descale_qr`），`kv_cache` 原地更新；`qr_norm` 可选。它直接使用
`op_kernel/attn_prologue.py`，需要 PyPTO/PyPTO-Pro 运行时和 kernel 源码；
首次调用会编译（计入 host 时间，不是设备 `Task Duration`），计时前应先完成
编译并预热。

### ACLNN 接口

ACLNN 接口通过 `op_api/include/aclnnop/aclnn_attn_prologue.h` 提供：
`aclnnAttnPrologueGetWorkspaceSize` 后调用 `aclnnAttnPrologue`。参考
`tests/aclnn_bridge/aclnn_bridge_attn_prologue.cpp` 的 C++ 调用方式。

## 完整示例

```python
import torch
import torch_npu
import pypto_gym

dev = "npu:0"
torch.npu.set_device(dev)
pypto_gym.load()

T, H, R, D, N, DR = 96, 5120, 1280, 512, 64, 64
P, BS = 1, 128

def to_nz(w):
    return torch_npu.npu_format_cast(w.contiguous(), 29)

x = torch.randn(T, H, device=dev).to(torch.float8_e4m3fn)
wqa = to_nz(torch.randn(R, H, device=dev).to(torch.float8_e4m3fn))
wqb = to_nz(torch.randn(N * D, R, device=dev).to(torch.float8_e4m3fn))
wkv = to_nz(torch.randn(D, H, device=dev).to(torch.float8_e4m3fn))

unit_scale = torch.full((), 127, dtype=torch.uint8, device=dev).view(torch.float8_e8m0fnu)
descale_x = unit_scale.expand(T, H // 64, 2).contiguous()
descale_wqa = unit_scale.expand(R, H // 64, 2).contiguous()
descale_wqb = unit_scale.expand(N * D, R // 64, 2).contiguous()
descale_wkv = unit_scale.expand(D, H // 64, 2).contiguous()

norm_weight_qr = torch.ones(R, device=dev)
norm_weight_kv = torch.ones(D, device=dev)

# 真实部署应使用网络侧按 token 绝对位置生成的 GPTJ_INTERLEAVED 表
rope_sin = torch.randn(T, DR, device=dev)
rope_cos = torch.randn(T, DR, device=dev)

cache_index = torch.arange(T, dtype=torch.int64, device=dev)
kv_cache = torch.zeros(P, BS, 1, 544, dtype=torch.uint8, device=dev)

with torch.no_grad():
    q, qr, descale_qr, kv_cache = torch.ops.pypto_gym.attn_prologue(
        x, wqa, wqb, wkv,
        descale_x, descale_wqa, descale_wqb, descale_wkv,
        norm_weight_qr, norm_weight_kv,
        rope_sin, rope_cos, cache_index, kv_cache,
        norm_eps=1e-6,
    )
torch.npu.synchronize()
```

> 示例中的 `x/descale_x` 只是占位。生产环境必须传入网络 MXFP8 量化器产出的
> E4M3 激活和 E8M0 scale；KV cache 的 `cache_index` 应使用 paged KV cache 的
> 真实 slot id。

## 结构与 Host 接入

- 在 V1 基础上新增 `kv_compress_row_vf`，`kv_row_vf` 输出 BF16。
- A1/A2/B/B-epi 的数值路径与原实现一致；B 段在 `n_tiles==2` 增加了 pair-major/交错 K 的权重复用路径，见下。
- `op_host` 完整（def / infershape / tiling / CMake），`kv_cache` 输入 / 原地写。
- tiling 字段：`t / cache_rows / block_size / block_stride / row_stride /
  ws_rows / norm_eps / emit_qr_norm`。
- tiling key 两个 1-bit 字段：
  - `Profile`（bit0）：`T<=128` 用 `NotAllocKeep`，`T>128` 用默认 L2 策略。
  - `QrNorm`（bit1）：`0` 不请求 `qr_norm`，`1` 请求；host 设置
    `key = Profile | (QrNorm << 1)`，包内变体为 `[0, 1, 2, 3]`。
- B 段工作分配：
  - `n_tiles == 2`（T=129–256）：pair-major QR hoist + 交错 K。两个 M tile
    共用同一份 Wqb 权重块，每个 K 只做一次 L0B 权重搬运，连续做两个 tile 的
    MMA；QR 按 pair 提升，避免每个 unit 重读。
  - 三个及以上 M tile 默认按 `(tile, head, block)` 分配。达到 32 个 tile 且 tile 数是核数的倍数，或达到 32 个 tile 且每个核至少分到 4 个 tile 时，启用 split-T，由同一个核处理完整 tile。

## 测试

```bash
D=ops/pro/deepseek_v4_1/attn_prologue
python $D/tests/test_attn_prologue_routes.py --t 1 --t 72 --t 192 --t 1024
pytest $D/tests -q
```

当前 T=1/72/192/1024 与 binary/ACLNN、qr_norm 可选输出用例均通过。
KV 记录与独立 `kv_compress_epilog_v2` 的逐字节对比：T=1/72 完全一致；
T=1024 仅有 2 个 FP32 累加导致的 tie-break payload 字节（`kv_hard=0`）。

## 参考实现的历史性能（msprof，冷 L2，64M flush，5+20 中位数）

采集口径：每次预热/正式调用前用 64M BF16 元素（128 MiB/缓冲）的
`torch.neg(src, out=dst)` 读写冲刷 L2（256 MiB/次），5 次预热 + 20 次正式调用取
中位数；指标为 `Task Duration(us)`，不含 host workspace、分配和 H2D/D2H。
设备为 Ascend950PR（28 Cube / 56 Vector），包为 hybrid + `QrNorm` tiling key 版本。

| T | 中位 µs | MFU | 模型 MBU（1.6 TB/s） |
|---:|---:|---:|---:|
| 5 | 43.43 | 1.56% | 81.68% |
| 72 | 49.54 | 19.63% | 77.91% |
| 96 | 51.78 | 25.04% | 76.70% |
| 144 | 84.86 | 22.92% | 48.25% |
| 192 | 87.86 | 29.52% | 49.64% |
| 240 | 92.95 | 34.87% | 49.79% |
| 576 | 152.18 | 51.12% | 42.67% |
| 1152 | 250.23 | 62.18% | 38.73% |

> T=128→129 有 host tiling 台阶（约 57.01→83.72 µs）；这是多 tile 路径的
> 结构性开销，不是 `QrNorm`/`Profile` key 选择导致的。完整 24 例数据见交付
> 测试报告；warm-L2 数字不作为本文口径。
