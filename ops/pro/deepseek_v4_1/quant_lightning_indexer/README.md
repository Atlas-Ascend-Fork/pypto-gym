# GymQuantLightningIndexer (quant_lightning_indexer)

本算子提供 PyPTO-Pro JIT 和预编译 Torch/ACLNN 两个入口。两者共用 AICPU 任务表生成逻辑和 AI Core 计算。

## 产品支持情况

| 项目 | 已验证范围 |
|---|---|
| SoC | Ascend 950，arch 3510 |
| CANN | 9.3.0 |
| 构建 | PyPTO 5adda25c80f1269cf183a7a88fc3db08d70b741c，arch 3510 |
| 运行 | 安装后的 AI Core 二进制、native AICPU 库和 Torch 绑定 |

## 功能说明

对打包 MXFP4 Q/K 计算索引分数，逐 head 作 BF16 归约并应用 mask，返回 token TopK，以及按需返回的分数、候选块和长度。同分 TopK 的合法集合与数值规则沿用原 JIT 契约。

| 工作 | 执行位置 |
|---|---|
| 参数/shape/stride 校验，静态容量与 launch grid，分配内存，提交两次调用 | host |
| 读取设备序列、实际分核、query/key 任务边界、空输入标志 | AICPU |
| QK、逐 head BF16 归约、mask、TopK、分片归并、offset、候选归约/选择/长度、空行填充 | 同一次融合 AI Core kernel |

## 函数原型

公开 Python 入口为 `pypto_gym.quant_lightning_indexer`：

```python
def quant_lightning_indexer(
    q, k, w, descale_q, descale_k,
    cu_seqlens_q=None, cu_seqlens_k=None, seqused_q=None, seqused_k=None,
    cmp_residual_k=None, block_table=None, output_idx_offset=None, metadata=None,
    *, topk, quant_mode, max_seqlen_q=-1, mask_mode=0, cmp_ratio=1,
    layout_q="TND", layout_k="TND", return_value=False,
    candidate_topk_blocks=-1, candidate_block_size=-1,
):
    ...
```

## 参数说明

| 参数 | 类型与约束 |
|---|---|
| q / descale_q | 连续 UINT8 `[T,32,64]` / `[T,32,2,2]`；前者每字节打包两个 FP4 E2M1，后者保存 E8M0 原始字节 |
| w | 连续 FP32 `[T,32]` |
| k / descale_k，TND | UINT8 `[Tk,1,64]` / `[Tk,1,2,2]` |
| k / descale_k，PA_BBND | UINT8 `[P,S,1,64]` / `[P,S,1,2,2]`；S 为 16 的倍数，16≤S≤1024 |
| K 存储 | 仅第一轴允许 padding，完整内层 stride，行不重叠；page stride 为 64/4 字节的倍数 |
| 可选序列数组 | 同设备连续 INT32：cuQ/cuK `[B+1]`，usedQ/usedK/residual `[B]`；B 由 cuQ 的 shape 得到，缺省为 1 |
| block_table / output_idx_offset | 同设备连续 INT32 `[B,C]` / `[T,1]`；多批 PA 要求 table，TND 不接收 table，多批 TND 要求 cuK |
| metadata | 兼容原入口的可选连续 INT32 `[1024]`，校验后不读取其内容；设备任务表由 AICPU 重新生成 |
| 属性 | topk=1…8192；quant_mode=1；mask_mode=0/3；cmp_ratio=1…128；Q=TND；K=TND/PA_BBND |
| residual | 当且仅当 mask_mode=3 且 cmp_ratio≠1 时提供 |
| 候选 | `candidate_topk_blocks=-1, candidate_block_size=-1` 关闭；开启时容量为正且 block_size=8 |

### 返回值说明

| 输出 | 类型与形状 |
|---|---|
| indices | INT32 `[T,1,topk]` |
| values | 启用时 BF16 `[T,1,topk]`，关闭时为 `[0]` |
| candidates | 开启候选时 INT32 `[T,1,C]`，关闭时为 `[0]` |
| lengths | 开启候选时 INT32 `[T,1]`，关闭时为 `[0]` |

## 约束说明

T、Tk/P、B 必须为正，所有输入在同一 NPU 上。序列内容的前提与原 JIT 相同。
候选块号不叠加 output_idx_offset；未使用位置填 -1，分数 padding 为 -inf。

缺省序列参数以 dummy 指针和整数标志传入 AICPU；frontend 不用 cumsum、差分、zeros 或设备数据回读构造序列。
非连续 K 的规范化由 ACLNN 的设备拷贝完成，tiling 对规范化后的紧凑存储填 stride。
Torch 入口接受 ND、NCHW 和 NCDHW 三种平直存储格式，覆盖五维 K scale 的默认 NCDHW 存储。

## 调用说明

构建环境需要同时支持 PyPTO-Pro 离线 codegen、AICPU codegen 与 CANN ARM 交叉编译。
本轮使用 CANN 9.3.0、PyPTO `5adda25c80f1269cf183a7a88fc3db08d70b741c`，目标 ascend950 / arch 3510。
在仓库根目录执行：

```bash
python build_ci.py --pkg-type run --ops quant_lightning_indexer \
  --soc ascend950 --build_type Release -j8
mkdir -p /path/to/isolated-opp
bash build_out/run/ascend950/*.run --quiet --install \
  --install-path=/path/to/isolated-opp
source /path/to/isolated-opp/vendors/pypto_gym/bin/set_env.bash
python -m pip install /path/to/isolated-opp/vendors/pypto_gym/python/pypto_gym-*.whl
export PYTHONDONTWRITEBYTECODE=1
export TILE_FWK_DEVICE_ID=0
```

AI Core 二进制和 native AICPU `.so` 均随算子包预编译交付。首次加载时，Python 针对当前 Torch 环境编译主机侧 C++ 绑定。部署时可调用 `pypto_gym.prewarm(["quant_lightning_indexer"])` 提前完成编译。推理过程无需导入 PyPTO；wheel 的安装依赖另见仓库根目录的 `pyproject.toml`。

输入张量满足上述契约后，通过正式入口调用：

```python
import pypto_gym

indices, values, candidates, lengths = pypto_gym.quant_lightning_indexer(
    q, k, w, descale_q, descale_k,
    topk=512, quant_mode=1, layout_q="TND", layout_k="TND",
)
```

## 目录结构

```text
quant_lightning_indexer/
├── CMakeLists.txt
├── README.md
├── op_host/           # OpDef、infer、tiling 与共享静态契约
├── op_kernel/         # 唯一 AI Core @pl.jit 入口
├── op_kernel_aicpu/   # AICPU planner 与构建期 native 库生成
├── torch_extension/  # 正式 Python / torch.ops 接线
└── tests/            # 同组 golden、正式入口回归与 JIT 适配
```

`op_kernel/quant_lightning_indexer.py` 仅有一个 `@pl.jit` 入口。
参数固定为 13 个输入指针、4 个输出指针、workspace、192 字节 `QliTiling`。
运行时 shape/stride、可选项标志和 workspace offset 放在 TilingData 中，kernel 用 `pl.make_tensor` 重建逻辑 FP4/E8M0 视图。
独立 AICPU 入口使用 `pl.Tensor` 和整数属性，分核与任务表计算保持原实现。
其 native fast-launch 源码由 PyPTO 的 `generate_aicpu_sources` 生成，在构建期通过同版本 CANN 编译器产出包含 `.aicpu_binary` 的 host `.so`。生成的 `*_metadata_native.h` 是其真实 C ABI，供 Torch 绑定在构建时核对。

完整调用在同一 stream 上先执行预编译 AICPU planner，再通过主计算的 ACLNN 两段式接口执行 AI Core。此处采用 PyPTO JIT 同款 native fast-launch，以避免通用 `CpuKernelContext` 适配的固定开销；不使用虚构 executor 或在线编译回退。

TilingKey 有 432 个合法分支。短行选择容量按 256/512/1024/1536/2048/4096/8192/16384 分桶，长行使用原流式选择。
快速路径的 UB shard 容量按 4096/8192/16384 分桶。28 Cube 的 128K 快速路径在 14 分片预计将最长核的 tile 数减少至少 20% 时，使用专用的 9472 容量分支。实际 shard 起点、长度和每核任务范围均由 AICPU 写入任务表。
短行 topk=512 且候选容量不超过 512 时复用已有 FixedTopK 分支，将输出和空行填充 tile 容量收紧为 512；其余通用分支保留 8192 容量。实际循环、validshape 和 store 均按请求的 topk/候选容量执行，不新增 TilingKey 分支。
数值原语、数据布局和同步操作沿用原 kernel。

host 通过本轮生成的 `GET_TPL_TILING_KEY` 选 key，通过生成的 TilingData 类型写参数；注册布局另作字节级核对。
框架内部 compute ACLNN 的 `metadata` 参数是变长任务表，与 Torch 兼容参数 `metadata[1024]` 区分使用。

## 验证

JIT 和 ACLNN 测试共用本算子 `tests/assets/` 下的 CPU golden 和精度判据。索引检查合法同分集合，分数使用 Scheme A，候选长度与 padding 要求精确匹配。

```bash
python -m pytest -q --mode=torch \
  ops/pro/deepseek_v4_1/quant_lightning_indexer/tests/test_quant_lightning_indexer_routes.py
# 在 PyPTO 环境对改造入口运行相同用例
python -m pytest -q --mode=jit \
  ops/pro/deepseek_v4_1/quant_lightning_indexer/tests/test_quant_lightning_indexer_routes.py
```

覆盖 12 个默认 case × 两种返回模式，以及 page16/32/80、TopK=1/129/8192、大候选、98,304 长度快速分片、128K 不等长批次、K padding、缺省序列、压缩比 1/3/128 和非法属性/类型。

`--mode=torch` 交替调用 `pypto_gym.quant_lightning_indexer` 和 `torch.ops.pypto_gym.quant_lightning_indexer`。JIT 模式保留改造入口与原 golden 的回归。


### JIT 直接调用

在算子目录运行，参数与安装包前端一致。两个入口都先在 AICPU 生成任务表，再在 AI Core 完成索引计算。

```python
from op_kernel.quant_lightning_indexer import quant_lightning_indexer

indices, values, candidates, lengths = quant_lightning_indexer(
    q, k, w, descale_q, descale_k, topk=512, quant_mode=1,
)
```
