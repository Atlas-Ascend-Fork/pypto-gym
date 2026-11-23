# kda_cp_main

## 产品支持情况

- Ascend 950PR：支持（本机 `get_platform_info()` = `PlatformInfo(soc_version='DAV_3510', vector_core_num=64)`，32 Cube / 64 Vector）。
- Cube + Vector 混合 kernel（32 个混合核，每核 1 AIC + 2 AIV）；无 GM workspace，无跨核数据交接。
- 软件环境（本机验收环境）：CANN 9.3.0、pypto 0.3.0（源码版）、Python 3.14.6、torch 2.13.0+cpu、torch_npu 2.13.0rc1。

## 功能说明

Kimi Delta Attention CP prefill 后半段主算子：读取 CP preprocess 产生的块内张量
（`qg/kg/u/w/mqk/decay`）与外部 CP prefix scan 合成的本 rank 初始状态，完成
64-token 分块的状态递推，输出本 rank 序列结果与最终状态。

### 计算过程

序列按 `CHUNK = 64` 分块，头维度 `D = 128`。对第 c 块，令
`S_{c-1} ∈ R^{dv×dk}`（dv = dk = 128，按 [dv, dk] 存储）为进入该块的状态：

$$
\Delta_c = u_c + w_c \cdot S_{c-1}^T
$$

$$
O_c = qg_c \cdot S_{c-1}^T + M_{qk,c} \cdot \Delta_c
$$

$$
S_c = S_{c-1} \odot decay_c \;(\text{dk 列广播}) + \Delta_c^T \cdot kg_c
$$

`qg` 已含 QK scale，`w` 已含负号，`mqk` 已含对角下三角 mask（本算子不再生成 mask）。

### 实现结构

- dv 维二分：任务 = (batch, head, dv_idx)，共 `2·B·Nv` 个 item 跨步映射到核上，
  每个 item 持有独立的 `S[dv 半区]` 切片，递推环在 item 内闭合。
- **2 跳串行环**：Δ 链整体在 cube 侧完成（`w@S^T` 与恒等矩阵乘 `I@u` 替代向量加法，
  Δ 经 FIX 通道 bf16 在途截断落 L1，ZN 孪生读出），`T = Δ^T @ kg` 直达向量侧；
  向量侧每块只剩 `S = S·decay + T` 单遍融合更新与 S 发布。每块 2 次跨核事件跳变。
- 两个向量融合（V-pass）：状态更新（含 bf16 快照同遍发射，乘加经 `vmadd` 融合）与
  O 排水（`O` 经 FIX 落 UB 后由向量侧异步写回 GM，与下一块的 cube 计算重叠）。
- 恒等矩阵 tile 每核装载一次（跨 item 复用）；尾块显式清零（GM 裁剪加载不补零）；
  L0 槽位全独立、零别名。

## 函数原型

```python
# 把 op_kernel 目录加入 sys.path 后：
from kda_cp_main import flash_kda_cp_main

out, final_state = flash_kda_cp_main(
    transition, state_ext, decay, qg, kg, u, w, mqk,
    local_initial_state, layout_qkv, metadata=None, cu_seqlens=None,
)
```

### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|:---|:---|:---|:---|:---|:---|:---|
| `transition` | 输入 | Tensor | 必选 | 兼容参数：仅参与校验，不消费 | — | 任意（对齐参考接口） |
| `state_ext` | 输入 | Tensor | 必选 | 兼容参数：仅参与校验，不消费 | — | 任意（对齐参考接口） |
| `decay` | 输入 | Tensor | 必选 | 每块 dk 维衰减系数，只读 | float32 | `[B, Nv, cap, 128]` |
| `qg` | 输入 | Tensor | 必选 | 块内查询侧矩阵（已含 QK scale），只读 | bfloat16 | `[B, Nv, S, 128]` 或 `[S, Nv, 128]` |
| `kg` | 输入 | Tensor | 必选 | 块内键侧矩阵，只读 | bfloat16 | 同 `qg` |
| `u` | 输入 | Tensor | 必选 | 块内残差增量，只读 | bfloat16 | 同 `qg` |
| `w` | 输入 | Tensor | 必选 | 块内状态转移矩阵（已含负号），只读 | bfloat16 | 同 `qg` |
| `mqk` | 输入 | Tensor | 必选 | 块内下三角 mask 矩阵（已含 mask），只读 | bfloat16 | `[B, Nv, cap, 64, 64]` |
| `local_initial_state` | 输入 | Tensor | 必选 | 初始状态 `S_0`，只读 | float32 | `[B, Nv, 128, 128]` |
| `layout_qkv` | 输入 | str | 必选 | 布局：`"BNSD"` / `"BSND"` / `"TND"` | — | — |
| `metadata` | 输入 | Any | 可选 | 兼容参数：不消费 | — | — |
| `cu_seqlens` | 输入 | Tensor | 可选 | TND 必选：序列前缀和 | int32 | `[B+1]` |

其中 `cap = ceil(S / 64) + B - 1`。BNSD 与 BSND 请求的 token 输入统一按 BNSD
存储解释；BSND 仅输出布局不同（参考契约："BSND changes only the output layout"）。

### 返回值说明

返回 `(out, final_state)`：

- `out`：`[B, Nv, S, 128]`（BNSD）/ `[B, S, Nv, 128]`（BSND）/ `[S, Nv, 128]`（TND）
  bfloat16 连续 Tensor；
- `final_state`：`[B, Nv, 128, 128]` float32 连续 Tensor（fp32 递推累加的终值）。

## 约束说明

- 所有输入张量必须连续、位于同一 NPU，并满足参数表中的 dtype 与 shape；
  非法输入一律 `ValueError`，不静默给出错误结果。
- `D = 128`、`CHUNK = 64` 固定；`S ≥ 1`，尾块（`S % 64 != 0`）显式清零处理。
- 起核数 `min(32, 2·B·Nv)`；任务跨步分配，item 内逐块串行递推。
- 数值契约：状态 S 在 UB 中 fp32 逐块累加，每块起点经 bf16 截断后作 matmul
  操作数；`Δ = u + w·S^T` 以 fp32 求和后截断一次为 bf16；`O` 与最终状态按
  fp32 累加结果输出（O 转 bf16 存储）。三处截断位置与参考数值契约一致。
- 容差：`atol=3e-3, rtol=3e-2`。
- 输入含 NaN / Inf 时输出未定义（不在支持范围内）。

## 调用示例

```python
import torch
import torch_npu

from kda_cp_main import flash_kda_cp_main

B, Nv, S, D = 1, 32, 8192, 128
dev = "npu:0"
cap = (S + 63) // 64 + B - 1
f32 = lambda *s: torch.randn(*s, dtype=torch.float32, device=dev)
bf16 = lambda *s: torch.randn(*s, dtype=torch.bfloat16, device=dev)

transition = state_ext = None
decay = torch.rand(B, Nv, cap, D, dtype=torch.float32, device=dev) * 0.1 + 0.9
qg, kg, u, w = (bf16(B, Nv, S, D) for _ in range(4))
mqk = torch.tril(torch.randn(B, Nv, cap, 64, 64, dtype=torch.float32, device=dev)).bfloat16()
local_initial_state = f32(B, Nv, D, D)

out, final_state = flash_kda_cp_main(
    transition, state_ext, decay, qg, kg, u, w, mqk,
    local_initial_state, "BNSD")
torch.npu.synchronize()
# torch.Size([1, 32, 8192, 128]) torch.bfloat16
print(out.shape, out.dtype)
```

## 精度测试

在仓库根目录执行：

```bash
TILE_FWK_DEVICE_ID=<device-id> pytest ops/pro/kimi_k3/kda_cp_main/tests \
    --mode jit -q
# 结果：11 passed（本机 32 混合核环境，多轮复验通过）
```

11 个用例覆盖：三种布局（BNSD/BSND/TND）、单尾块与多 batch、奇数 S 尾块、
双任务核（Nv ≥ 30）、多序列变长 TND、重复稳定性。判定细节见 [RESULTS.md](RESULTS.md)。

## 性能

> 下表为**参考数值**，实际耗时每次运行可能有所波动。

口径：内核 `Task Duration`（设备端内核时间，不含 host 派发），torch_npu 进程内
profiler（ACL task-based）采集 `op_summary`。冷口径在每次受测 launch 前 1 GiB
驱逐 L2（4 轮 × 8 采样取中位数，保守口径）；warm 为 20 预热 + 100 采样 p50。
本机 Ascend 950PR_9589（32 混合核）、D = 128、固定种子。

| 用例 | 形状 | item 数 | 冷 p50（µs） | warm p50（µs） |
| :--- | :--- | ---: | ---: | ---: |
| BSND 1×6×16384 | 12 item / 12 核 | 12 | 284.8 | 286.1 |
| BSND 1×12×16384 | 24 item / 24 核 | 24 | 340.5 | 285.6 |
| BSND 1×96×1024 | 192 item / 32 核 | 192 | 173.3 | 116.1 |
| BNSD 1×5×577 | 10 item / 10 核 | 10 | 14.4 | 12.8 |

设计与取舍见 [DESIGN.md](DESIGN.md)、用例与判定细节见 [RESULTS.md](RESULTS.md)。

## 文件结构

```
ops/pro/kimi_k3/kda_cp_main/
├── DESIGN.md                    # 设计说明
├── README.md
├── RESULTS.md                   # 用例 / 精度 / 性能结果
├── __init__.py
├── op_kernel/
│   ├── __init__.py
│   └── kda_cp_main.py           # kernel + host wrapper
└── tests/
    ├── test_kda_cp_main.py      # 逐 case 测试
    └── assets/
        ├── kda_cp_main_golden_cpu.py    # 独立 CPU golden / 输入工厂 / F64 预言机
        └── kda_cp_main_precision.py     # 精度判据
```

> 当前交付为 **JIT 形态**（impl + tests）；`op_host/` 与 run 包暂不提供。
