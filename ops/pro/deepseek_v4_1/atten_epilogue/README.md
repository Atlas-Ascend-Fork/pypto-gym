# AttenEpilogue（PyPTO-Pro）

## 产品支持情况

| 产品 | 验证情况 |
| --- | --- |
| Ascend 950PR，28 Cube / 56 Vector 核 | 已验证 JIT 与 ACLNN 调用 |
| Ascend 950PR，32 Cube / 64 Vector 核 | 已验证 JIT 与 ACLNN 调用 |
| Ascend 950DT | 本轮未实测 |
| Atlas A2、A3 系列 | 不支持本实现使用的 A5 MXFP8 指令 |

同一份 kernel 源码根据设备实际 Cube 核数选择分块。28 核环境使用 CANN 9.2.0；32 核环境的 Toolkit 安装信息为 9.3.0，compiler 与 OPP 组件元数据为 9.2.0。两者均使用 PyPTO 0.3.0；二进制包应在对应环境工具链下构建。

## 功能说明

在一次设备任务内完成逆 RoPE、激活 OCP MXFP8 量化、两级带块缩放的矩阵乘，生成 BF16 注意力输出投影。

输入 `o` 包含 64 个 head，每个 head 为 512 维。前 448 维保持不变；对后 64 维，令 `r=0..63`，执行

```text
rotated[t,h,448+r] = BF16(
    FP32(o[t,h,448+r]) * rope_cos[t,r]
    + (-1 if r is odd else 1)
      * FP32(o[t,h,448+(r xor 1)]) * rope_sin[t,r])
```

每 8 个 head 合成一组 4096 维输入，共 8 组。激活量化以连续 32 个值为一组，产生 FP8 E4M3FN 数据与 E8M0 缩放字节。量化器以 BF16 指数的组内最大值生成缩放，执行 BF16 归一化，再以有限饱和、最近舍入转换为 FP8；极小指数分组的归一化倒数为零。权重及其缩放由调用方预先提供。

第一层每组输出 1024 维：`Y[t,g,n] = MX(A[t,g,:], woa[g,n,:])`，K=4096。FP32 累加结果先转换为 BF16，再按相同规则量化。将 8 组输出依次拼接为 8192 维，第二层执行 `out[t,n] = MX(YQ[t,:], wob[n,:])`，K=8192，最终写回 BF16。`MX` 同时应用两个操作数的每 32 个 K 元素对应的块缩放；两层均保持每个输出元素的 K 方向累加顺序。

## 函数原型

```python
def atten_epilogue(
    o, woa, wob, descale_woa, descale_wob, rope_sin, rope_cos, out
):
    ...
```

Python 入口位于 `op_kernel/atten_epilogue.py`。CANN 交付保留 `AttenEpilogue` 注册名称及 `aclnnAttenEpilogueGetWorkspaceSize`、`aclnnAttenEpilogue` 两阶段接口，参数顺序同下表。ACLNN 的工作区大小以第一阶段实际返回值为准。

### 参数说明

所有张量均使用连续 ND 布局，最后一维 stride 为 1，其余 stride 为后续维度长度的乘积；输入与输出必须位于同一 NPU。

| 参数 | 方向 | 形状与含义 | dtype |
| --- | --- | --- | --- |
| `o` | 输入 | `[T,64,512]`，token、head、head 内特征 | BF16 |
| `woa` | 输入 | `[8,1024,4096]`，组、第一层输出、K | FP8 E4M3FN |
| `wob` | 输入 | `[5120,8192]`，第二层输出、K | FP8 E4M3FN |
| `descale_woa` | 输入 | `[8,1024,64,2]`，第一层权重的每 32 元素缩放，末两维为打包的 E8M0 字节 | E8M0 |
| `descale_wob` | 输入 | `[5120,128,2]`，第二层权重的每 32 元素缩放 | E8M0 |
| `rope_sin` | 输入 | `[T,64]`，各 token 的逆旋转 sine 系数 | FP32 |
| `rope_cos` | 输入 | `[T,64]`，各 token 的逆旋转 cosine 系数 | FP32 |
| `out` | 输出 | `[T,5120]`，调用方预分配、由算子写入 | BF16 |

没有可选参数、属性或默认值。`T` 为 token 个数，是唯一可变的形状维度。

### 返回值说明

Python 接口返回传入的 `out` 对象，结果写入该对象已有存储。接口不返回中间量；输入与输出存储重叠不属于已验收的使用范围。

## 约束说明

- `T >= 1`，其余维度、dtype 和 stride 必须符合参数表；权重不接受 NZ 存储。
- 已验证 `T=1..256` 的全部整数及补齐、分档和大尺寸边界，边界覆盖至 16384。具体测试范围与精度规则见 `RESULTS.md`。
- 输入使用有限数值；权重缩放应与其 OCP MXFP8 数据匹配。算子不对权重重新量化。
- 内部令 `M=ceil(T/16)*16`，用 `42240*M` 字节的 kernel 用户工作区保存四个中间载荷。补齐行不属于公开输出，不写入 `out` 之后的存储。
- JIT Python 入口检查公开张量契约。ACLNN 保持空输出也进入 host tiling 的配置，使 T=0 按同一非空契约被拒绝；非法形状不能依靠框架的空输出短路静默通过。

## 调用示例

在本算子目录下运行。示例使用原生 MX 量化准备调用方权重；该准备过程不属于融合 kernel。

```python
import os

import torch
import torch_npu
from op_kernel.atten_epilogue import atten_epilogue

torch.set_num_threads(4)
torch.npu.set_device(int(os.environ.get("TILE_FWK_DEVICE_ID", "0")))
t = 72
gen = torch.Generator().manual_seed(2026)
o = (torch.randn(t, 64, 512, generator=gen) * 0.3).to(torch.bfloat16).npu()
wa = (torch.randn(8, 1024, 4096, generator=gen) * 0.3).to(torch.bfloat16)
wb = (torch.randn(5120, 8192, generator=gen) * 0.2).to(torch.bfloat16)
woa, sa = torch_npu.npu_dynamic_mx_quant(wa.reshape(8192, 4096).npu(), dst_type=292)
wob, sb = torch_npu.npu_dynamic_mx_quant(wb.npu(), dst_type=292)
woa = woa.reshape(8, 1024, 4096)
sa = sa.view(torch.float8_e8m0fnu).reshape(8, 1024, 64, 2)
sb = sb.view(torch.float8_e8m0fnu)
rope_sin = torch.zeros(t, 64, dtype=torch.float32).npu()
rope_cos = torch.ones(t, 64, dtype=torch.float32).npu()
out = torch.empty(t, 5120, dtype=torch.bfloat16, device=o.device)
result = atten_epilogue(o, woa, wob, sa, sb, rope_sin, rope_cos, out)
torch.npu.synchronize()
assert result is out
print(out.shape)
```
