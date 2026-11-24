# mixed_quant_sparse_flash_mla（MQSMLA，aclnn / built-in 交付形态）

`mixed_quant_sparse_flash_mla`（混合量化稀疏 Flash MLA）的 **delivery-form** 交付：同一套
PyPTO-Pro 稀疏 Flash MLA 计算，kernel 以 raw Ptr + OpTiling 形态编写，既支持 Python 宿主
直调（jit）、经 CANN 框架的 aclnn 接口调用（run 包 built-in），以及 whl 安装态的
`torch.ops.pypto_gym.mixed_quant_sparse_flash_mla` Torch 绑定。

算法设计、流水结构与性能/优化记录见原 jit 形态算子目录
`ops/pypto_pro/deepseek_v4_1/mixed_quant_sparse_flash_mla/`（`DESIGN.md` / `RESULTS.md`）；
本目录只承载交付形态所需的 kernel 签名改造、host 侧（def / infershape / tiling）与测试桥。

## 产品支持情况

| 项目 | 说明 |
| :--- | :--- |
| 硬件平台 | Ascend 950PR（`NpuArch=3510`）；Cube / Vector 核数运行时取自设备属性，上限 `AIC_CORE_MAX_NUM=36` |
| 计算单元 | 单个 MIX kernel 内 Cube（QK/PV）与 AI Vector（反量化、在线 softmax、累加、写回）融合 |
| 软件环境 | CANN 9.2.0、PyPTO-Pro（`pypto_pro`）、torch / torch_npu、Python 3.12 |
| 设备侧 kernel 名 | `mixed_quant_sparse_flash_mla_kernel` |
| 输入数据类型 | `q` BF16；`ori_kv` / `cmp_kv` uint8（FP8 / FP4 编码字节）；索引、页表、长度 int32；`sinks` FP32 |
| 输出数据类型 | `out` BF16；`lse` FP32 |
| 测试标注 | `pytestmark = pytest.mark.soc("950")` |

## 功能说明

对每个 query token `t` 与 head `h`，按稀疏索引经 block table 收集两侧（原始 FP8 / 压缩 FP4）
有效列的反量化 KV 合并为 `X`，同一 `X` 既作 Key 也作 Value：

```text
z_j = softmax_scale * q[t,h,:] · X_j
Z   = exp(sinks_h) + Σ_j exp(z_j)          # sink 只进分母，不贡献 Value
O   = (Σ_j exp(z_j) · X_j) / Z              # 分子分母均 FP32 累加
LSE = log Z
```

GM 物理行号表由 attention kernel 的 vector prologue 在每个 block 内融合构建：本 block
只建自己后续消费的行，写入 FD partials 之后的 GM workspace；建表与 cube 段并发，之后
`bar_all` 再进入主循环。tilingkey 承载 5 个 1-bit 条件
（`HAS_ORI_LEN` / `HAS_CMP_LEN` / `HAS_CMP` / `FD_ENABLE` / `VDEP64`），其余数值参数
（`softmax_scale`、strides、`b_num`、`k`、`bs`、`bt`、`ori/cmp_row_tab_stride`）与全部 shape
走 OpTiling 运行时字段。`VDEP64` 选择 ValueDepend 输入的读取 dtype（int32/int64），避免每次
调用转换。

GM 行号表按

```text
row_tab[t, j] = block_table[b, idx[t, j] // bs] * bs + idx[t, j] % bs
```

解析每一列；列布局为 `[ori | cmp]`，每侧按 128 token tile 补齐，padding 列重复最后一个
有效列。attention kernel 的 prologue 只消费原始 `sparse_indices` / `block_table` / 长度，在本核
内解析物理行并写入 GM workspace；主循环每个 tile 把本 AIV 的 64 个物理行号用 MTE2
搬进 UB，VF 用 `vf.mull`/`vf.addc` 把 int32 page/off 转成 **int64 地址**（lo/hi 两个
32-bit 寄存器，`INTLV_B32` 写入 int64 pair 表），主循环标量取 `(min, gap)` 后用 int64
stride 一次搬两行（VF 写、标量读，避免 MTE2→S 依赖）。主循环不再读 block table，也不再
保留旧 1024 项 UB 块表窗口；JIT / ACLNN / Torch 三条路径共用同一份融合 kernel，调用方
直接传原始输入即可。

**分核规划（metadata）**：`int32[1024]` 的 per-core 工作描述（uniform 整行切分，或满足增益
门限时的 flat `(row, tile)` FlashDecode 切分 + 2-way partial 归约）。metadata **作为 kernel
输入张量传递，不放进 tiling 结构**：放进 tiling 会把参数区撑到 8.3 KB，实测 kernel 慢
10%–66%；tiling 保留 17 个标量字段（136 B，含 raw/padded K 与表 stride）。jit 与 aclnn 的 host tiling 仍用同一套规划器
计算 `FD_ENABLE` / `blockDim`。

## 参数说明

`T1` 为展平后的 query token 数，`B` 为 batch 数，`K1` / `K2` 为两侧 TopK，`bs` 为每页 slot
数，`bt` 为块表宽度。下表均为逻辑形状：

| 参数 | dtype | shape | 说明 |
| :--- | :--- | :--- | :--- |
| `q` | BF16 | `[T1, 64, 512]` | Query（TND） |
| `ori_kv` | uint8 | `[pages, bs, 1, 544]` | FP8 E4M3FN group32 原始 KV 编码字节 |
| `ori_sparse_indices` | int32 | `[T1, 1, K1]` | 原始稀疏索引（prologue 输入）；合法前缀之外的列被忽略，可为 -1 |
| `ori_block_table` | int32 | `[B, bt]` | 逻辑页 → 物理页；`bt ≤ 8192` |
| `ori_topk_length` | int32 / None | `[T1, 1]` | 每行有效 TopK；省略 = 全 K 有效 |
| `sinks` | FP32 | `[64]` | 每 head 的 sink |
| `cmp_kv` | uint8 / None | `[pages, bs, 1, 320]` | FP4 E2M1 group16 压缩 KV；省略 = 关闭压缩侧 |
| `cmp_sparse_indices` | int32 / None | `[T1, 1, K2]` | 压缩侧原始稀疏索引（prologue 输入） |
| `cmp_block_table` | int32 / None | `[B, bt]` | 压缩侧逻辑页 → 物理页；`bt ≤ 8192` |
| `cmp_topk_length` | int32 / None | `[T1, 1]` | 压缩侧每行有效 TopK |
| `cu_seqlens_q` | int32 | `[B+1]` | Query 前缀和，`cu[0]=0`、末值 `T1` |
| `softmax_scale` | float | 标量 | 缺省 `1/sqrt(512)`；aclnn 侧为 float 属性 |
| `out` | BF16 | `[T1, 64, 512]` | 稀疏注意力输出 |
| `lse` | FP32 | `[1, T1, 64]` | 每 token 每 head 的 `logsumexp` |

## 调用方式

| 调用方式 | 说明 |
| -------- | ---- |
| Kernel 直调（PyPTO，jit） | host 入口 `mixed_quant_sparse_flash_mla(q, *, ori_kv, ..., metadata, ...)`：调用方先用 `mixed_quant_sparse_flash_mla_metadata(...)` 在 host 侧规划出 `int32[1024]` metadata；wrapper 预打包 17 字段 `MqsmlaTiling`（device uint8，`lru_cache`），并把 metadata 作为普通输入张量传给 `mixed_quant_sparse_flash_mla_kernel[None, block_dim, key](...)`。 |
| aclnn API（run 包 built-in） | `aclnnMixedQuantSparseFlashMlaGetWorkspaceSize` / `aclnnMixedQuantSparseFlashMlaTensorGetWorkspaceSize` + `aclnnMixedQuantSparseFlashMla`。**metadata 是输入张量**（由调用方/测试桥提供）；host tiling（`op_host/mixed_quant_sparse_flash_mla_tiling.cpp`）读取 ValueDepend 输入的 Host 数据，用同一套规划器计算 `FD_ENABLE` tilingkey 位、blockDim 与 workspace 大小，但不再把 metadata 数组写入 tiling。 |

aclnn 侧要点：

- `cu_seqlens_q` / `ori_topk_length` / `cmp_topk_length` 注册为 `ValueDepend(OPTIONAL)`，
  dtype 同时声明 INT32 与 INT64——op_build 对 ValueDepend 输入生成 `aclIntArray`（int64 host
  值）形参。kernel 按 `VDEP64` tilingkey 选择 int32/int64 读取；jit wrapper 不再做
  INT32→INT64 转换，aclnn host tiling 按输入实际 dtype 设置该 key 位。
- `cu_seqlens_q` 的 Host 数据缺失时：单 batch（`B=1`）按 `[0, T1]` 合成；多 batch 直接报错
  （batch 边界无法由 shape 推导）。两个 topk 长度的 Host 数据缺失时退化为 uniform 规划
  （不选 FD），结果仍正确——kernel 照常读设备侧张量。
- workspace：FD 规划生效时为 `(72*64*512 + 2*72*64)*4` 字节，否则保留 1024 字节下限
  （PyPTO `SetSysWorkspaceForce` 路由要求非零）。
- 测试桥 `tests/aclnn_bridge/aclnn_bridge_mixed_quant_sparse_flash_mla.cpp` 与 jit 入口同签名；
  `metadata` 现在作为输入张量传给 aclnn 接口，`block_dim` / `fd_enable` 仍被接受但忽略
  （由框架 tiling 决定）。

## 目录结构

```
mixed_quant_sparse_flash_mla/
├── CMakeLists.txt                     # 算子级 dispatcher
├── op_kernel/
│   └── mixed_quant_sparse_flash_mla.py    # delivery-form kernel（raw Ptr + OpTiling）+ fused row-table prologue + host 规划器
├── op_host/
│   ├── mixed_quant_sparse_flash_mla_def.cpp        # 算子注册（含 ValueDepend）
│   ├── mixed_quant_sparse_flash_mla_infershape.cpp
│   ├── mixed_quant_sparse_flash_mla_tiling.h       # tiling 结构（与 kernel dataclass 逐字节对齐）
│   ├── mixed_quant_sparse_flash_mla_tiling.cpp     # host tiling：校验 + 规划器 C++ 移植
│   └── CMakeLists.txt
└── tests/
    ├── CMakeLists.txt
    ├── aclnn_bridge/
    │   ├── aclnn_bridge_mixed_quant_sparse_flash_mla.cpp
    │   └── CMakeLists.txt
    ├── assets/                                     # golden / 精度比较器
    └── test_mixed_quant_sparse_flash_mla.py        # 17 条 P0，jit / aclnn / torch 三模
```

## 构建与测试

```bash
# jit（whl）：构建 + 板上系统测试
python3 build_ci.py --pkg-type whl --ops mixed_quant_sparse_flash_mla -s

# aclnn（run 包）：构建 + 安装 + 板上系统测试
python3 build_ci.py --pkg-type run --ops mixed_quant_sparse_flash_mla --soc ascend950 -s

# 仅跑用例（jit，无需构建）
pytest ops/pro/deepseek_v4_1/mixed_quant_sparse_flash_mla/tests -q --mode jit

# 仅跑用例（aclnn，需已构建测试桥并安装 run 包）
pytest ops/pro/deepseek_v4_1/mixed_quant_sparse_flash_mla/tests -q --mode aclnn
```

## 约束说明

1. `layout_q` 固定 `"TND"`、`layout_kv` 固定 `"PA_BBND"`、`quant_mode` 固定 `1`；
   `num_heads_q=64`、`num_heads_kv=1`、`head_dim=512`。
2. `ori_kv` / `cmp_kv` 以 uint8 承载 FP8 / FP4 编码字节，**支持页间 padding（axis-0 非连续）**：
   页内维度必须连续，页跨度取输入真实 `stride(0)`（host tiling 校验页内连续性，以及整池跨度不溢出 int64）。
   attention 的 per-tile pair 表存 int64 地址（VF `mull/addc`），KV 池字节跨度和 pair gap
   不再受 int32 限制；2 GiB gap 用例已验证。block table 宽度不再有 1024 限制：fused
   prologue 生成完整 `[T1, k1_tab + k2_tab]` 表，有效页可分散在整张 8192 列表上。
   block table 宽度上限 8192（fused prologue 的 UB 块表 tile）；行号表 workspace 追加在
   FD partials 之后，FD 关闭时直接使用 workspace 起始地址。
   算子注册为 `FORMAT_ND` + `IgnoreContiguous`，aclnn 路径原样传递视图，不做压实拷贝。
3. sink 只进分母；两侧有效长度均为 0 的行输出严格为 0、`lse` 严格等于该 head 的 `sinks`。
4. 反量化值与概率 P 均舍入到 BF16，分子分母 FP32 累加——舍入位置是数值契约的一部分。
5. 精度在 `tests/` 的 17 条 P0 shape 上验证，包含真实 `spread_8192`（8192 个有效页、索引跨全窗口）。
