# CompressorV2（PyPTO-Pro）

## 产品支持情况

本算子的编译和注册目标为 Ascend 950（A5，`ascend950`）。Python JIT 调用依赖匹配的 CANN、PyPTO-Pro、PyTorch 和 torch_npu 环境；ACLNN 调用使用按该目标构建并安装的算子包。构建需要 CMake 3.19 或更高版本。

## 功能说明

CompressorV2 将一组 token 的投影结果压缩为一个向量，并原位更新历史状态缓存。记输入特征宽度为 `H`、输出特征宽度为 `D`、压缩率为 `r = cmp_ratio`。每个 token 先计算两路投影：

```text
K = X @ wkv.T
G = X @ wgate.T
```

两路权重均按 `[D, H]` 存储，矩阵乘以 FP32 累加。随后按绝对位置划分长度为 `r` 的组，在每个通道内沿 token 轴进行 softmax 加权：

```text
m[d] = max_i G[i, d]
a[i, d] = exp(G[i, d] - m[d])
Y[d] = sum_i(a[i, d] * K[i, d]) / sum_i a[i, d]
```

`i` 遍历一个完整组内的 `r` 个 token。向量计算使用 FP32；实现将 KV 先乘以 `1 / r`，加权归一化后再乘回 `r`，最终转换为与 `x` 相同的数据类型。

`start_pos[b]` 是第 `b` 条序列本次输入的绝对起点。起点未对齐组边界时，首组先从调用前缓存读取 `start_pos[b] % r` 个历史 token，再拼接本次输入。仅完整组产生有效输出，末尾不足一组的投影仍参与缓存更新。

缓存采用环形寻址。第 `b` 条序列使用物理块 `state_block_table[b]`，绝对位置 `p` 对应槽位 `p % block_size`。前 `D` 个通道保存 `K`，后 `D` 个通道保存 `G`。每条序列仅写入本次最后 `min(L_b, block_size)` 个 token 的投影，`L_b` 为实际使用的新输入长度；其余槽位、块间间隔和未使用的物理块保持原值。

## 函数原型

Python 公开入口 `compressor` 由算子包导出，实现在 `op_kernel/compressor_v2.py`，调用参数保持与 JIT 版本一致：

```python
def compressor(
    x, wkv, wgate, state_cache, cmp_ratio=4, *,
    state_block_table=None, cu_seqlens=None,
    seqused=None, start_pos=None,
):
    ...
```

ACLNN 接口由构建系统根据 OpDef 自动生成。混合核入口的头文件为 `aclnn_pypto_compressor_v2_stage.h`，纯 Vector 入口为 `aclnn_pypto_compressor_v2_vector.h`。每组接口均采用“查询工作区、执行”两段式调用：

```cpp
aclnnStatus aclnnPyptoCompressorV2StageGetWorkspaceSize(
    const aclTensor *x, const aclTensor *wkv, const aclTensor *wgate,
    const aclTensor *cacheIn, const aclTensor *table, const aclTensor *cu,
    const aclTensor *used, const aclTensor *start,
    const aclTensor *projectedIn, const aclTensor *historyIn,
    int64_t stage, int64_t cmpRatio, bool hasUsed, bool hasStart,
    const aclTensor *outOut, const aclTensor *cacheOutOut,
    const aclTensor *projectedOutOut, const aclTensor *historyOutOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnPyptoCompressorV2VectorGetWorkspaceSize(
    const aclTensor *x, const aclTensor *wkv, const aclTensor *wgate,
    const aclTensor *cacheIn, const aclTensor *table, const aclTensor *cu,
    const aclTensor *used, const aclTensor *start,
    const aclTensor *projectedIn, const aclTensor *historyIn,
    int64_t stage, int64_t cmpRatio, bool hasUsed, bool hasStart,
    const aclTensor *outOut, const aclTensor *cacheOutOut,
    const aclTensor *projectedOutOut, const aclTensor *historyOutOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnPyptoCompressorV2Stage(
    void *workspace, uint64_t workspaceSize,
    aclOpExecutor *executor, aclrtStream stream);

aclnnStatus aclnnPyptoCompressorV2Vector(
    void *workspace, uint64_t workspaceSize,
    aclOpExecutor *executor, aclrtStream stream);
```

通用路径依次执行混合入口的 `stage=0` 和 Vector 入口的 `stage=1`；片上路径只执行混合入口的 `stage=2`。调用侧负责分配中间张量、选择路径，并在同一 stream 上按序提交各阶段。Python 测试调用桥提供了完整适配，源码位于 `tests/aclnn_bridge/aclnn_bridge_compressor_v2.cpp`。

### 参数说明

| Python / ACLNN 参数 | 类型与形状 | 说明 |
| :--- | :--- | :--- |
| `x` | FP16 或 BF16，`[B,S,H]` 或 `[T,H]` | 输入激活。`B` 为序列数，`S` 为 BSH 布局每条序列本次输入长度，`T` 为所有序列本次输入长度之和。 |
| `wkv`、`wgate` | 与 `x` 同类型，`[D,H]` | KV 投影和分数投影权重，`D` 为投影及输出宽度。 |
| `state_cache` / `cacheIn`、`cacheOutOut` | FP32，`[block_count,block_size,2D]` | 原位更新的环形缓存。ACLNN 的输入和输出位置使用同一个 Tensor 描述。 |
| `cmp_ratio` / `cmpRatio` | 整数 | 完整压缩组的 token 数；Python 默认值为 `4`，ACLNN 显式传入。 |
| `state_block_table` / `table` | INT32，`[B]` | 每条序列对应的缓存物理块号，必须提供。 |
| `cu_seqlens` / `cu` | INT32，`[B+1]` | TH 布局的序列长度前缀和。Python 的 BSH 调用省略它；ACLNN 的 BSH 调用在该位置传入 `table` 作为占位张量。 |
| `seqused` / `used` | INT32，`[B]` | 实际使用的新 token 数。Python 省略时使用完整长度；ACLNN 同时设置 `hasUsed=false`，并在 `used` 位置传入 `table` 占位。 |
| `start_pos` / `start` | INT32，`[B]` | 各序列非负的绝对起点。Python 省略时按 `0` 计算；ACLNN 设置 `hasStart=false`，并在 `start` 位置传入 `table` 占位。 |
| `stage` | INT64 | 混合入口接受 `0` 或 `2`，Vector 入口接受 `1`；取值须与本次输入选中的路径一致。 |
| `hasUsed`、`hasStart` | BOOL | 表示逻辑输入 `seqused`、`start_pos` 是否提供，须与实际调用方式一致。 |
| `projectedIn`、`projectedOutOut` | 连续 FP32 | 通用路径的投影部分和，两处使用同一个 Tensor 描述；容量至少为 `2×P×T×D` 个元素，`P` 是 host 计划选中的 H 分区数。按 `P=4` 预留可覆盖全部合法分区。 |
| `historyIn`、`historyOutOut` | 连续 FP32 | 调用前历史快照，两处使用同一个 Tensor 描述；通用路径容量至少为 `2×B×cmpRatio×D` 个元素。 |
| `outOut` | 与 `x` 同类型，形状见下表 | 调用方预先分配的连续输出。两个阶段传入同一个 Tensor 描述。 |
| `workspaceSize`、`executor` | 查询接口的输出参数 | 本阶段所需的工作区字节数和执行器，供对应执行接口使用。 |
| `workspace`、`stream` | 设备内存地址、ACL stream | 容量至少为返回的 `workspaceSize`；大小为 `0` 时可传空指针。各阶段使用同一 stream。 |

Python 的可选张量用 `None` 表示省略。自动生成的 ACLNN 接口中，上表的十个 Tensor 输入位置均须传入非空描述；占位张量只用于补齐未参与计算的元数据位置。

片上路径不使用投影和历史中间结果；两个中间张量各预留至少一个 FP32 元素，并分别复用其输入、输出描述。通用路径在两个阶段之间保持中间张量及缓存描述一致。

路径选择与容量计算可复用 `op_host/compressor_v2_plan.h` 中的 `BuildPlan`。调用侧应先按实际 Tensor 的类型、形状和元素步长构造输入描述，通过 `aclrtGetDeviceInfo` 查询当前设备的 Cube、Vector 核数，再读取计划中的 `keys[i].Stage`、`projected_elements` 和 `history_elements`；host tiling 使用同一套规则校验各阶段。

### 返回值说明

Python 返回与 `x` 同设备、同类型的连续输出 `out`；ACLNN 将最终结果写入预分配的 `outOut`：

| 输入布局 | 输出分配形状 |
| :--- | :--- |
| BSH：`[B,S,H]` | `[B,ceil(S/r),D]` |
| TH：`[T,H]` | `[min(T,T//r+B),D]` |

其中 `r=cmp_ratio`。对第 `b` 条序列，提供 `seqused` 时取 `L_b=seqused[b]`，否则取该序列完整长度。当 `L_b>0` 时，有效组数为 `(start_pos[b]%r+L_b)//r`；当 `L_b=0` 时为 `0`。省略起点时按 `0` 计算。

BSH 的有效结果从各序列输出区域首行开始；TH 的有效结果按序列顺序紧密排列。未使用输出行的值不作规定，调用方按有效组数读取结果。通用路径的最终输出和缓存更新在 `stage=1` 完成后可用，片上路径在 `stage=2` 完成后可用。

ACLNN 接口返回状态码，`ACLNN_SUCCESS` 表示该次接口调用成功。执行接口向 stream 提交任务；读取结果或释放张量、中间缓冲和工作区前，应等待对应任务完成。

## 约束说明

- `H` 在 `[1024,10240]` 内且为 `512` 的整数倍；`D` 为 `128` 或 `512`。
- `r` 为 `2、4、8、16、32、64、128` 之一。
- `block_count>0`，`1<=block_size<=1024`。缓存最后两轴的元素步长为 `[2D,1]`，块间步长不小于 `block_size×2D`，允许块间间隔及非零存储偏移。ACLNN 描述须保留实际步长、偏移和底层存储容量。
- 所有张量位于同一当前 NPU，使用 ND 格式；除缓存块间步长外，各输入、输出及中间张量连续。
- ACLNN 的缓存、投影和历史张量须分别复用输入、输出描述；这些可写存储区与最终输出不得相互重叠，也不得与其他输入存储重叠。只读输入之间允许共享存储。
- `state_block_table` 的块号互不相同，并满足 `0<=block_id<block_count`，块号 `0` 有效。
- `cu_seqlens` 单调不减，首元素为 `0`、末元素为 `T`；`seqused[b]` 在 `0` 到该序列完整长度之间。元数据张量内的数值约束由调用方保证。
- `start_pos` 非负。调用方负责准备首组所需历史；环形容量不足以保留全部历史时，首组读取对应槽位当前保存的值。
- 省略起点与显式全零起点、以及省略实际长度与显式传入完整长度，分别具有相同计算语义，但可能选择不同执行路径；ACLNN 的存在性标志应反映实际提供方式。
- `B`、`S` 或 `T` 可为 `0`；总 token 数为 `0` 时，调用侧返回相应空输出，不提交计算阶段。各 Tensor 元素数、视图范围和工作区字节数须可由接口整数类型表示。

## 调用示例

在仓库模块可导入且 NPU 环境已初始化的条件下，可使用 Python JIT 入口：

```python
import torch
import torch_npu  # noqa: F401

from ops.pro.deepseek_v4_1.compressor_v2 import compressor

batch, sequence, hidden, dim = 1, 128, 1024, 128
device = "npu"
dtype = torch.bfloat16
x = torch.randn(batch, sequence, hidden, device=device, dtype=dtype)
wkv = torch.randn(dim, hidden, device=device, dtype=dtype)
wgate = torch.randn(dim, hidden, device=device, dtype=dtype)
state_cache = torch.zeros(2, 128, 2 * dim, device=device, dtype=torch.float32)
state_block_table = torch.tensor([1], device=device, dtype=torch.int32)

out = compressor(
    x, wkv, wgate, state_cache,
    cmp_ratio=4,
    state_block_table=state_block_table,
)
torch.npu.synchronize()
assert out.shape == (1, 32, 128)
```

ACLNN 调用按以下顺序组织：

1. 准备输入、原位缓存及连续输出，按 host 计划分配投影、历史中间张量；补齐未提供元数据的占位描述。
2. 通用路径先以 `stage=0` 调用 `aclnnPyptoCompressorV2StageGetWorkspaceSize`，分配本阶段工作区，再调用 `aclnnPyptoCompressorV2Stage`。
3. 以相同的缓存、投影、历史及输出描述，使用 `stage=1` 查询并调用 `aclnnPyptoCompressorV2Vector`，提交到同一 stream。片上路径以一次 `stage=2` 的混合入口调用替代上述两个阶段。
4. 检查各接口状态，等待 stream 完成，再读取有效输出和更新后的缓存并释放资源。

每个阶段均使用本次查询返回的执行器及工作区大小。两个执行阶段之间无需把数据取回 CPU，也无需额外同步；同一 stream 的提交顺序保证先完成历史读取和投影，再进行压缩及缓存更新。
