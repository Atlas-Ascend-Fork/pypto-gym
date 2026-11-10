# engram_gate_bwd

## 产品支持情况

| 项目 | 说明 |
| :--- | :--- |
| 硬件平台 | Ascend 950PR |
| 使用的计算单元 | AI Vector（AIV）核，不使用 Cube |
| 输入数据类型 | `grad_out`、`x`、`key`、`value` 为 BF16；`weight` 为 FP32；`rstd_x`/`rstd_k`/`raw_dot`/`gate` 为 FP32；`image_mask` 为 BOOL |
| 输出数据类型 | `grad_x`、`grad_key`、`grad_value` 为 BF16；`grad_weight` 为 FP32 |
| 支持的形状 | `grad_out`/`x`/`key` 为 `[t, 4, dim]`，`value` 为 `[t, dim]`，`weight` 为 `[4, dim]`，统计量为 `[t, 4]`，`image_mask` 为 `[t]`；`t ≥ 1`，`dim ≤ 7168` 且为 `4·128=512` 的倍数 |

## 功能说明

DeepSeek V4.1 Engram 残差门控（`engram_gate`）的解析反向。它接收正向保存的四个逐
`(token, head)` 统计量，一次 kernel 算出四份梯度（所有归约在 FP32 中完成）：

```text
dgate = Σ_d grad_out · value
beta  = dgate · gate · (1-gate) · 0.5 · sqrt(rstd_x·rstd_k·dim^-0.5 / |raw_dot|)
c     = -beta · raw_dot / dim
grad_x     = grad_out + beta · weight · key + c · rstd_x² · x
grad_key   = beta · weight · x + c · rstd_k² · key
grad_value = Σ_hc gate · grad_out
grad_weight = Σ_t beta · x · key
```

`image_mask == True` 的 token 在正向是恒等映射，因此 `grad_x = grad_out`、
`grad_key = grad_value = 0`，且不贡献 `grad_weight`。`|dot| < clamp_value` 的 token 的
`beta` 为 0（与正向 `copysign` 分支一致）。

## 函数原型

```python
grad_x, grad_key, grad_value, grad_weight = engram_gate_bwd_wrapper(
    grad_out, x, key, value, weight,
    rstd_x, rstd_k, raw_dot, gate, image_mask,
    clamp_value=1e-6)
```

| 参数 | 形状 | 类型 | 说明 |
| :--- | :--- | :--- | :--- |
| `grad_out` | `[t, 4, dim]` | BF16 | 输出侧梯度 |
| `x` | `[t, 4, dim]` | BF16 | 正向输入 |
| `key` | `[t, 4, dim]` | BF16 | 正向输入 |
| `value` | `[t, dim]` | BF16 | 正向输入（head 维广播） |
| `weight` | `[4, dim]` | FP32 | 正向逐 `(head, dim)` 加权系数 |
| `rstd_x` | `[t, 4]` | FP32 | 正向保存的 `rsqrt(mean(x²)+eps)` |
| `rstd_k` | `[t, 4]` | FP32 | 正向保存的 `rsqrt(mean(key²)+eps)` |
| `raw_dot` | `[t, 4]` | FP32 | 正向保存的 `Σ_d x·weight·key` |
| `gate` | `[t, 4]` | FP32 | 正向保存的门控值 |
| `image_mask` | `[t]` | BOOL | 为 True 的 token 是恒等分支 |
| `clamp_value` | 标量 | float | 只接受默认值 `1e-6` |

返回 `(grad_x, grad_key, grad_value, grad_weight)`：前三个与 `x`/`key`/`value` 同形，
`grad_weight` 为 `[4, dim]` FP32。

## 约束说明

- `hc` 固定为 4；`dim ≤ 7168`（`MAXC`）且必须是 `hc·STEP = 512` 的倍数。
- `clamp_value` 是 kernel 内立即数，只接受 `1e-6`。
- 输入形状/ dtype 不匹配时 wrapper 报 `ValueError`；非连续输入在进入 kernel 前压实。
- 正向四个统计量必须来自同一 `(x, key, weight)`，本算子不再重算它们。

## aclnn 交付形态

| 调用方式 | 说明 |
| --- | --- |
| jit 直调 | 经 `op_kernel/engram_gate_bwd.py` 的 `engram_gate_bwd_wrapper` 构造 tiling 并启动 kernel。 |
| aclnn API | `aclnnEngramGateBwdGetWorkspaceSize` + `aclnnEngramGateBwd`。测试桥见 `tests/aclnn_bridge/`，`pytest --mode aclnn`。 |

实现说明：

- `t`/`dim` 是运行时 tiling 字段；tiling key 只有单值 `Schedule=0`（binary delivery 要求
  schema）。tiling 结构为 `EngramGateBwdTiling(t, dim)`。
- 四份 FP32 统计量在桥上打包成 kernel 的 `saved: [t, 16]` 输入，布局为
  `[raw_dot | gate | rstd_x | rstd_k]`；这是纯布局拼接，不做算术。
- `grad_weight` 需要跨 token 归约：kernel 写出 `grad_w_partial: [16, 4, dim]`（op 的第 4 个
  输出），桥在同一个 stream 上 `sum(0)` 得到公开的 `[4, dim]`，与 jit wrapper 完全一致。
  op 层因此只有 4 个输出，避免出现 kernel 不写的“空输出”。
- 纯 vector kernel 在当前 pypto delivery 中编成 AIV-only binary，kernel 显式声明
  `workspace` 参数；host tiling 申报 1024 B 非零 workspace。
- AIV-only 下 `SetBlockDim` 就是 AIV 块数。host 选 `nc = min(AIV, 64)` 并向下取整到 4 的
  倍数，`blockDim = nc`；kernel 内用 `get_block_num() * get_subblock_num()` 还原 AIV
  工作单元数（AIV-only 下 `get_subblock_num()` 为 1）。旧 pypto 的 MIX binary 才用
  `blockDim = nc/2`、`get_block_num()` 为 AIC 数。
- jit 热路径把 tiling 预打包成 device uint8 tensor（`lru_cache`）并缓存 workspace 占位，
  避免每次启动 2–3 ms 的 tiling H2D。
- `clamp_value` 在 aclnn 桥内校验为默认值，非默认直接报 `ValueError`。

## 目录结构

```
engram_gate_bwd/
├── op_kernel/
│   └── engram_gate_bwd.py          # delivery-form kernel + wrapper
├── op_host/
│   ├── engram_gate_bwd_def.cpp
│   ├── engram_gate_bwd_infershape.cpp
│   ├── engram_gate_bwd_tiling.cpp/.h
│   └── CMakeLists.txt
└── tests/
    ├── aclnn_bridge/
    ├── assets/
    ├── test_engram_gate_bwd.py
    └── CMakeLists.txt
```

验证：

```bash
pytest ops/pro/deepseek_v4_1/engram_gate_bwd/tests -q --mode jit
pytest ops/pro/deepseek_v4_1/engram_gate_bwd/tests -q --mode aclnn
```

测试覆盖 SPEC 的 12 个 `(t, dim)` 形状，另加 `t < ng`、`nq/nr` 非整除、partial/all-true
mask 与最小 `dim=512` 边界。

## 迁移对比（CANN 9.3.0）

环境：Ascend 950PR，AIC=28 / AIV=56，CANN 9.3.0 + 9.3.0 pypto，torch_npu
2.7.1.post11。当前 pypto 把本算子编成 AIV-only binary（`Task Type=AI_VECTOR_CORE`，
`coreType=VectorCore`）；host 的 `SetBlockDim` 是 AIV 数 `nc`。

### 接口

- 公开签名与旧 jit 一致：10 个张量参数 + `clamp_value=1e-06`，返回
  `(grad_x, grad_key, grad_value, grad_weight)`。
- 合法用例共 15 个（12 个 P0 + 3 个边界用例）：新 jit 与新 aclnn 的 4 个输出
  **逐字节完全相同**，shape/dtype/顺序全部一致；新 jit/aclnn 均 14/15 通过 golden，
  仅 `t8001_h2048` 失败（见下）。
- 旧 jit 的 `grad_x/grad_key/grad_value` 与新实现逐字节一致；`grad_weight` 在旧实现
  未清零 `grad_w_partial` 的用例上会随分配器残留变化，不能作为稳定基线。
- 非连续统计量（`rstd_x/rstd_k/raw_dot/gate`）可接受；数据张量在 jit 入口保持原有的
  连续性校验。

### 性能

口径：3 次独立执行 × 12 个 P0；device 为 profiler kernel Duration，每次测量前刷
256 MiB L2，host 为队列排空后的单次 wrapper 调用（3 次 × 30 样本）。下表为 median。

| 用例 | 旧 jit kernel | 新 jit kernel | 新 aclnn kernel | 旧 jit host | 新 jit host | 新 aclnn host |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `t512_h2048` | 57.80 | 56.72 | 57.57 | 232.67 | 220.58 | 45.16 |
| `t512_h2560` | 71.10 | 70.62 | 70.97 | 232.91 | 220.93 | 47.75 |
| `t512_h3072` | 83.43 | 83.56 | 84.02 | 229.18 | 220.76 | 44.07 |
| `t512_h4096` | 106.16 | 105.09 | 108.74 | 228.43 | 224.20 | 44.67 |
| `t512_h6144` | 155.10 | 154.24 | 153.47 | 227.75 | 221.32 | 43.45 |
| `t512_h7168` | 179.21 | 178.33 | 178.65 | 226.71 | 220.03 | 43.53 |
| `t8001_h2048` | 638.24 | 637.11 | 637.66 | 234.11 | 221.75 | 41.36 |
| `t8001_h2560` | 750.69 | 750.04 | 750.72 | 233.31 | 224.90 | 42.54 |
| `t8001_h3072` | 849.03 | 848.29 | 849.28 | 231.59 | 225.22 | 40.75 |
| `t8001_h4096` | 1061.25 | 1061.41 | 1061.82 | 235.76 | 227.60 | 47.66 |
| `t8001_h6144` | 1552.01 | 1552.19 | 1553.02 | 234.11 | 225.29 | 46.83 |
| `t8001_h7168` | 1802.01 | 1800.06 | 1801.03 | 236.48 | 226.14 | 47.22 |

- 目标 kernel：新 jit 最差 1.002×、新 aclnn 最差 1.024×，均在 5% 内；
- host：新 jit 0.947–0.981×，新 aclnn 0.176–0.205×（桥内 `at::cat`/`sum(0)` 异步下发，
  host 口径下空档更小；不代表 kernel 变快）；
- 包装层 device 总时长（含 saved 打包、`grad_w_partial` 清零、`sum(0)`）新 jit 最差
  1.049×、新 aclnn 最差 1.059×；超出部分来自为修正未写 partial 行而增加的清零 kernel，
  主要在小 shape 上明显。

## 已知问题（与原 jit 同源）

`t8001_h2048` 的 `grad_weight` 在 FP32 归约上与 golden 的 mean-abs 阈值相差约 0.8%
（1.538e-05 vs 1.526e-05）；原 `ops/pypto_pro/deepseek_v4_1/engram_gate_bwd` 的 jit 用例
同值同失败。根因是分核顺序累加的舍入顺序：

- 按 kernel 的划分（14 个 group × 组内 571 个 token 顺序累加，再求 14 个 partial 的和）
  在 CPU 上复现 1.500e-05，`max_abs` 4.272e-04 与实际完全一致；
- 组内改用 pairwise 求和降到 5.009e-06；2/4 个交错累加器分别降到 1.119e-05 / 8.558e-06；
- 剩余约 0.04e-05 来自 beta 的运算顺序与 dgate 的 FMA/reduce 顺序；
- FP64 自差分 oracle 与 FP32 golden 本身相差 7.739e-02，不能作为收紧依据。

这不是迁移引入的回退，其余 14 条用例全部通过。若要把该用例纳入门禁，可在 kernel 内用
2–4 个交错累加器，或放宽 FP32 归约输出的 `atol`。
