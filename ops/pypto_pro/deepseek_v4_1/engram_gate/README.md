# engram_gate

## 产品支持情况

| 项目 | 说明 |
| :--- | :--- |
| 硬件平台 | Ascend 950PR（`Ascend950PR_9589`，平台信息 `soc_version = 'DAV_3510'`）；cube（矩阵计算单元）32 核 / vector（向量计算单元）64 核 |
| 使用的计算单元 | AI Vector（AIV）核，不使用 Cube |
| 软件环境 | CANN 9.3.0、PyPTO-Pro DSL |
| 输入数据类型 | `x`、`key`、`value` 为 BF16；`weight` 为 FP32；`image_mask` 为 BOOL |
| 输出数据类型 | `y` 为 BF16 |
| 支持的形状 | `x` / `key` / `y` 为 `[t, hc, dim]`，`value` 为 `[t, dim]`，`weight` 为 `[hc, dim]`，`image_mask` 为 `[t]`；`t ≥ 1`，`hc` 为 `1 ~ 向量核数`（本平台向量核数为 64），`dim` 最大 8512 |
| 参考实现 | 功能说明里那条公式的独立 CPU 参考实现（FP32 计算、BF16 输出），见 `tests/assets/engram_gate_golden_cpu.py` |

未覆盖的情况（当前不支持）：`dim > 8512`、`hc` 大于向量核数、`eps` / `clamp_value` 取默认值
`1e-6` 以外的值、输入含有 NaN / Inf。

前三种调用都会在 host 侧报错，不会静默给出错误结果：`dim` 与 `eps` / `clamp_value` 由
wrapper 直接拦下；`hc` 越界时，`hc = 0` 在计算起核数时除零，`hc` 大于向量核数则在起核时
报 `InvalidType`。含 NaN / Inf 的输入不在支持范围内，输出未定义。

## 功能说明

`engram_gate` 是 DeepSeek V4.1 Engram 的残差门控（residual gate）算子，使用 PyPTO-Pro
在昇腾 NPU 的向量核上实现。计算沿最后一维 `dim`、逐 `(token, head)` 行进行：

```text
rstd = rsqrt(mean(x²) + eps) · rsqrt(mean(key²) + eps)    # x、key 两路均方根（RMS）的倒数之积
dot  = Σ(x · weight · key) · rstd / √dim                  # 加权点积，按 dim 归约
gate = sigmoid(copysign(√max(|dot|, clamp_value), dot))   # |dot| 开方后贴回 dot 的符号，再过 sigmoid（1/(1+e^(-x))）
y    = bf16(x + gate · value)                             # 残差更新
```

其中 `value` 按 token 共享、不随 head 变化，所以在 head 维上广播。

如果把这四步各写成一个 kernel，就要来回读写全局内存四趟。本算子把它们融合成**一个 kernel
的一趟扫描**：每个 `(token, head)` 行的 `x`、`key`、`value` 各只搬进片上一次，`y` 只写出
一次，两趟计算复用同一块片上数据，中间的统计量不再往返全局内存。`value` 行由同一 token 的
`hc` 个 head 各搬一次、`weight` 由每个核各搬自己那一行，这两处重复量在 `RESULTS.md` 的流量
表里单列。

`image_mask` 为 `True` 的 token 不做上述门控计算，直接原样返回 `x`（逐位相等，不重新取整）。
这类 token 在 kernel 里走一条只搬不算的分支。

## 函数原型

```python
# 把 op_kernel 目录加入 sys.path 后：
from engram_gate import engram_gate_wrapper

y = engram_gate_wrapper(x, key, value, weight, image_mask, eps=1e-6, clamp_value=1e-6)
```

### 参数说明

| 参数 | 形状 | 类型 | 说明 |
| :--- | :--- | :--- | :--- |
| `x` | `[t, hc, dim]` | BF16 | 主激活值，也是残差更新的基准项 |
| `key` | `[t, hc, dim]` | BF16 | key 激活值，与 `x` 同形状 |
| `value` | `[t, dim]` | BF16 | value 激活值，按 token 共享（在 head 维广播） |
| `weight` | `[hc, dim]` | FP32 | 逐 `(head, dim)` 的加权系数 |
| `image_mask` | `[t]` | BOOL | 为 `True` 的 token 原样返回 `x` |
| `eps` | 标量 | float | RMS 的稳定项，只接受默认值 `1e-6` |
| `clamp_value` | 标量 | float | `\|dot\|` 的软阈值，只接受默认值 `1e-6` |

### 返回值说明

| 返回值 | 形状 | 类型 | 说明 |
| :--- | :--- | :--- | :--- |
| `y` | `[t, hc, dim]` | BF16 | 门控后的残差更新结果，与 `x` 同形状；由本函数内部分配并返回 |

## 约束说明

- `dim` 最大 8512。这个上限来自 kernel 给片上数据行定的宽度 `CHUNK = 8512`：整行
  `(token, head)` 必须一次驻留在 UB（片上统一缓冲，kernel 直接寻址的片上内存）里，才能让
  两趟计算复用同一块数据。超过时 wrapper 直接报错，不做静默降级。
- `eps` 与 `clamp_value` 被编成 kernel 内的立即数，`1e-6` 以外的取值 wrapper 会报错。
- `t`、`hc`、`dim` 都是运行时动态轴，同一个 kernel 服务全部形状，不需要按形状重新编译。
- 启动核数取 `min(向量核数, t·hc)`，再向下取整到 `hc` 的倍数——因为 token 维按
  `hc` 分组轮转，核数必须是 `hc` 的整数倍才不会漏 token。
- 因此 `hc` 不能大于向量核数（本平台 64）：否则取整后起核数为 0。`hc ≤ 64` 时起核数总是
  `hc` 的整数倍且不小于 `hc`，每个核至少拿到一行。
- 输入张量的内存布局不做要求：连续张量与带步长的非连续视图（`token` 维、`dim` 维分别构成
  非连续）都得到一致的结果；`RESULTS.md` 的精度与性能数据在连续张量上采集。
- `x`、`key`、`value`、`weight`、`image_mask` 的形状必须互相匹配，否则 wrapper 报错；
  dtype 必须与上表一致，不一致时运行时报 `InvalidType`，并指出是哪个入参、期望与实际分别是
  什么 dtype。
- 单次调用只下发一个 kernel，host 侧不做任何数值计算，也不做额外的同步或后处理。
- 输出 `y` 是新分配的张量，不原地修改 `x`。

## 调用示例

```python
import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, "ops/pypto_pro/deepseek_v4_1/engram_gate/op_kernel")
from engram_gate import engram_gate_wrapper

torch.npu.set_device(0)
t, hc, dim = 5, 4, 5120

x = torch.randn(t, hc, dim, dtype=torch.bfloat16).npu()
key = torch.randn(t, hc, dim, dtype=torch.bfloat16).npu()
value = torch.randn(t, dim, dtype=torch.bfloat16).npu()
weight = torch.randn(hc, dim, dtype=torch.float32).npu()
image_mask = torch.arange(t) % 2 == 1
image_mask = image_mask.npu()

y = engram_gate_wrapper(x, key, value, weight, image_mask)
torch.npu.synchronize()
print(y.shape, y.dtype)
# torch.Size([5, 4, 5120]) torch.bfloat16

# image_mask == True 的行与 x 逐位相等
print(bool(torch.equal(y[image_mask], x[image_mask])))
# True
```

示例依赖的文件分布在算子目录下（`op_kernel/` 放 kernel 与公开 wrapper，`tests/assets/` 放
golden 与精度比较器，`tests/` 直接放测试入口）：

```text
engram_gate/
├── README.md                           # 本文件：接口与用法
├── DESIGN.md                           # 设计说明：阶段划分、切分与流水
├── RESULTS.md                          # 结果说明：用例、精度与性能
├── op_kernel/
│   └── engram_gate.py                  # 算子本体（kernel + 公开 wrapper）
└── tests/
    ├── test_engram_gate.py             # 测试入口（pytest 与命令行都能跑）
    └── assets/
        ├── engram_gate_golden_cpu.py   # 独立 CPU 参考实现
        └── engram_gate_precision.py    # 精度比较器（方案 A）
```

测试与参考实现自检都可以在仓库根目录直接运行，不需要额外的驱动脚本：

```bash
# 全部 5 个用例
pytest ops/pypto_pro/deepseek_v4_1/engram_gate/tests -q

# 只跑其中一个用例
pytest ops/pypto_pro/deepseek_v4_1/engram_gate/tests -q -k partial_image_mask

# CPU 参考实现自检
python ops/pypto_pro/deepseek_v4_1/engram_gate/tests/assets/engram_gate_golden_cpu.py
```
