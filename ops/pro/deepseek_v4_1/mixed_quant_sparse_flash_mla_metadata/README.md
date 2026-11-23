# MixedQuantSparseFlashMlaMetadataPtoBnn (mixed_quant_sparse_flash_mla_metadata_pto_bnn)

MQSMLA 的分核计划算子，以 **AICPU built-in 形态**交付：设备侧读长度张量、直接写出主注意力算子消费的
`int32[1024]` metadata，全程不做 host 回读。计划语义与主算子 `mixed_quant_sparse_flash_mla_pto_bnn`
host tiling 内的规划器逐字一致（uniform 整行切分，或满足增益门限时的 flat `(row, tile)` FlashDecode 切分）。

## 产品支持情况

| 产品                                   | 是否支持 |
| -------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term> |    √     |
| Atlas A3 / A2 系列                     |    ×     |

## 功能说明

输入 `cu_seqlens_q`（query 前缀和）、`ori_topk_length` / `cmp_topk_length`（两侧每行有效 TopK 长度），
按设备 Cube 核数 `blocks` 规划每个核的行/tile 区间，输出 `metadata`：

- 每行 tile 数 `ceil(clamp(len) / 128)`，原始侧与压缩侧相加；
- uniform：整行按核均分（`s2_start = s2_end = 0`）；
- FlashDecode：`saved_tiles >= 4`（`rows >= 96` 时放宽到 2）且 split 合法时，按 `(row, tile)` 均匀切分，
  `metadata[900]` 写 split 数并填 72 个半片槽位；
- `k_ori` / `k_cmp` 非 0 时先对长度做 `[0, k]` clamp，0 表示不截断。

字表（`metadata[1024]`）：`0..323` 每核 9 字 FA 描述（36 槽），`324..899` 每 split 8 字 FD 半片描述（72 槽），
`900` 为 split 数，其余恒 0。主算子把这份张量当作普通输入消费。

## 函数原型

```python
import torch

def mixed_quant_sparse_flash_mla_metadata_pto_bnn(
    cu_seqlens_q, ori_topk_length, cmp_topk_length=None, *,
    blocks=None, has_cmp=1, k_ori=0, k_cmp=0,
    bs_ori=0, bs_cmp=0, bt_ori=0, bt_cmp=0,
    ori_pages=0, cmp_pages=0, ori_kv_stride0=0, cmp_kv_stride0=0,
) -> tuple[torch.Tensor, torch.Tensor]: ...
# -> (metadata: int32[1024], tiling_device: int32[16])
```

对应 C 接口（`op_api/include/aclnnop/aclnn_mixed_quant_sparse_flash_mla_metadata_pto_bnn.h`）：

```c
aclnnStatus aclnnMixedQuantSparseFlashMlaMetadataPtoBnnGetWorkspaceSize(
    const aclTensor *cuSeqlensQ, const aclTensor *oriTopkLength,
    const aclTensor *cmpTopkLengthOptional, int64_t blocks, int64_t hasCmp,
    int64_t kOri, int64_t kCmp, int64_t bsOri, int64_t bsCmp, int64_t btOri,
    int64_t btCmp, int64_t oriPages, int64_t cmpPages, int64_t oriKvStride0,
    int64_t cmpKvStride0, const aclTensor *out, const aclTensor *tilingDevice,
    uint64_t *workspaceSize, aclOpExecutor **executor);
aclnnStatus aclnnMixedQuantSparseFlashMlaMetadataPtoBnn(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
```

## 参数说明

| 参数 | dtype | shape | 说明 |
| :--- | :--- | :--- | :--- |
| `cu_seqlens_q` | int32 | `[B+1]` | 从 0 开始、非递减、末值 = `T1` |
| `ori_topk_length` | int32 | `[T1, 1]` | 每行有效 TopK；超出 `k_ori` 的部分被忽略 |
| `cmp_topk_length` | int32 / None | `[T1, 1]` | 压缩侧每行 TopK；`None` 时须 `has_cmp=0` |
| `blocks` | int | 标量 | 规划核数，`1..36`；缺省取设备 `cube_core_num` |
| `has_cmp` | int | 标量 | 1 = 启用压缩侧，0 = 仅原始侧 |
| `k_ori` / `k_cmp` | int | 标量 | 长度 clamp 上限，0 = 不 clamp |
| `bs_ori`/`bs_cmp`/`bt_ori`/`bt_cmp` | int | 标量 | 调用方 KV 页/页表形状（host 已知，透传） |
| `ori_pages`/`cmp_pages`/`*_stride0` | int | 标量 | 调用方 KV 页数与页跨度（host 已知，透传） |
| `metadata` | int32 | `[1024]` | 输出：per-core / per-split 调度表 |
| `tiling_device` | int32 | `[16]` | 输出：设备端 tiling 参数（形状标量 + planned_block_dim/fd_enable），直喂主算子 |

## 约束说明

1. `cu_seqlens_q` 必须从 0 开始且末值等于 `T1`，否则行号映射越界；
2. `has_cmp=0` 时 `cmp_topk_length` 传 `None`（桥内部以 1 元素占位避免空指针）；
3. `blocks` 不得超过设备核数上限 36，也不得小于 1；
4. 单次调用只发射 1 个 AICPU task；不产生 kernel workspace 负载。

## 调用说明

```python
import torch, torch_npu
import pypto_gym  # 安装 run 包后加载 torch.ops
md = pypto_gym.mixed_quant_sparse_flash_mla_metadata_pto_bnn(
    cu_q, ori_len, cmp_len, blocks=28, has_cmp=1, k_ori=128, k_cmp=2048)
out = pypto_gym.mixed_quant_sparse_flash_mla_pto_bnn(q, ..., metadata=md)
```

aicnn 直调走 `aclnnMixedQuantSparseFlashMlaMetadataPtoBnnGetWorkspaceSize` + `aclnnMixedQuantSparseFlashMlaMetadataPtoBnn`；
jit 直调用 `op_kernel_aicpu` 里的 `mixed_quant_sparse_flash_mla_metadata_pto_bnn_aicpu(...)`。

## 目录结构

```
mixed_quant_sparse_flash_mla_metadata_pto_bnn/
├── CMakeLists.txt                 # 算子级 dispatcher
├── __init__.py                    # 导出 AICPU kernel 与 wrapper
├── op_host/
│   ├── CMakeLists.txt             # 纯 AICPU: 无 def/proto/tiling, 只手写 aclnn 通道
│   └── op_api/aclnn_*.{h,cpp}     # l0op + aclnn 入口
├── op_kernel_aicpu/
│   ├── CMakeLists.txt             # enable_pypto_aicpu_kernel
│   └── mixed_quant_sparse_flash_mla_metadata_pto_bnn.py
├── tests/
│   ├── aclnn_bridge/              # pybind11 测试桥
│   └── test_*.py                  # 与主算子 host planner 逐字对拍
└── torch_extension/               # torch.ops 绑定（whl / 安装态）
```

## 验证

```bash
# 构建（含测试桥）
python build_ci.py --pkg-type run --ops mixed_quant_sparse_flash_mla_metadata_pto_bnn --soc ascend950 -s

# 手动跑对拍
pytest ops/pro/deepseek_v4_1/mixed_quant_sparse_flash_mla_metadata_pto_bnn/tests -q --mode aclnn
```

对拍基准是主算子 `tests/assets/mixed_quant_sparse_flash_mla_pto_bnn_golden_cpu.py` 的 host planner；
`uniform` 与 `FlashDecode` 两条路径都要覆盖。
