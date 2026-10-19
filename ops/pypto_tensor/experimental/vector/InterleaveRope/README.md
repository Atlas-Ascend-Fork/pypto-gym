# interleave_rope

PyPTO 自定义算子：对 4D 特征张量沿最后一维 D（D=64）按 **interleave 模式** 应用 RoPE 旋转位置编码。


## 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 算子概述

中间数学（interleave 模式 RoPE，对每对相邻元素 `(x[2k], x[2k+1])`）：

```
y_origin[..., 2k]   = x[..., 2k] * cos[..., 2k]   - x[..., 2k+1] * sin[..., 2k]
y_origin[..., 2k+1] = x[..., 2k] * sin[..., 2k+1] + x[..., 2k+1] * cos[..., 2k+1]
```

### ⚠️ 输出 layout：**split-half**（不是 interleave）

为避开 PyPTO 5D 重组的 op 限制，本算子约定输出采用 split-half 排布：

```
out[..., 0:32 ] = [y_origin[0], y_origin[2], y_origin[4], ..., y_origin[62]]   # 取所有偶位
out[..., 32:64] = [y_origin[1], y_origin[3], y_origin[5], ..., y_origin[63]]   # 取所有奇位
```

**调用方契约**：下游 attention 必须在 Q 和 K 都按 split-half 排布时使用本算子。
`QK^T = Σ_d Q[d]·K[d]` 在 Q/K 同 layout 下数值与原 interleave 排布等价（点积对 D 维顺序不敏感）。
若混用一个 split-half 张量与一个 interleave 张量做点积，结果错误。

实现路径（kernel 内）：
1. `pypto.gathermask(x, PM=1/2)` 拆 x 奇偶位
2. `pypto.gathermask(cos|sin, PM=1/2)` 拆 cos/sin 奇偶位（按位置取值，不假设 cos[2k]==cos[2k+1]）
3. cast 到 fp32 → mul/sub/add 累积 → cast 回原 dtype
4. 两次 `pypto.assemble`：y_even 写到 out 左半（偏移 0），y_odd 写到 out 右半（偏移 32），全程 4D，无 5D 重组

## 输入输出

| 名称 | shape | dtype | 约束 |
|------|-------|-------|------|
| x   | `[B, N, S, 64]`     | fp16 / bf16 | 连续 ND；N ∈ {1, 128} |
| cos | `[B, 1, S\|1, 64]` | 与 x 同     | 连续 ND；N=1；S_cs ∈ {1, S} |
| sin | `[B, 1, S\|1, 64]` | 与 x 同     | 连续 ND；N=1；S_cs ∈ {1, S} |
| y   | `[B, N, S, 64]`     | 与 x 同     | 输出，连续 ND，**split-half layout**（详见上文）|

动态轴：B ∈ [1, 4]，S ∈ [1, 8192]，N ∈ {1, 128}；D 常量 64。

精度：atol=1e-4, rtol=7.8125e-3。

## 目录结构

```
custom/interleave_rope/
├── SPEC.md
├── API_REPORT.md
├── DESIGN.md
├── interleave_rope_golden.py   # pure-torch 参考
├── interleave_rope_impl.py     # PyPTO kernel + wrapper
├── test_interleave_rope.py     # 精度对比测试
├── test_cases.json             # P0 用例
└── README.md
```

## 用法

```python
import torch
import torch_npu  # noqa
from interleave_rope_impl import interleave_rope_wrapper

x   = torch.randn(1, 128, 2048, 64, dtype=torch.bfloat16, device="npu:0")
cos = torch.randn(1,   1, 2048, 64, dtype=torch.bfloat16, device="npu:0").clamp_(-1, 1)
sin = torch.randn(1,   1, 2048, 64, dtype=torch.bfloat16, device="npu:0").clamp_(-1, 1)

y = interleave_rope_wrapper(x, cos, sin)
```

## 测试

```bash
export TILE_FWK_DEVICE_ID=$(bash cannbot-skills/ops/pypto-op-develop/scripts/list_idle_chip_ids.sh | awk '{print $1}')

# 运行全部用例
python3 custom/interleave_rope/test_interleave_rope.py

# 单个用例
python3 custom/interleave_rope/test_interleave_rope.py level0
python3 custom/interleave_rope/test_interleave_rope.py --list
```

末尾 stdout 会输出 `[PRECISION_PASS]` 或 `[PRECISION_FAIL]`，并在异常时返回非零退出码。

## 已知限制

- D 必须为 64；N 仅支持 {1, 128}（与 SPEC 保持一致）。
- S 任意值（含 S<S_TILE 与不被 S_TILE 整除）均支持，通过 ceil-div + `valid_shape` 处理尾块。
- `S_cs=1` 路径在 wrapper 中显式 `expand(...).contiguous()`，对 S=1 单帧场景有少量额外内存。
- dtype 仅支持 fp16 / bf16；fp32 不支持。
- 4 套 kernel（`{N=1, N=128} × {bf16, fp16}`），首跑会触发各 kernel 编译。
