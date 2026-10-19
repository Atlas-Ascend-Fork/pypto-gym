# AttentionWorkerCombine PyPTO 算子


## 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 算子概述

将多个计算单元处理的注意力 token 数据进行融合，结合专家权重对结果进行加权，输出最终的注意力融合结果。

## 数学公式

$$
y = \sum_{i=0}^{K-1} (token\_data[:, i, :] \times expert\_scales[:, i]) + token\_data[:, K, :]
$$

其中：
- $K$ 为选中的路由专家数量（TopK）
- $token\_data[:, i, :]$ 为第 $i$ 个专家的输出 token
- $expert\_scales[:, i]$ 为第 $i$ 个专家的权重
- $token\_data[:, K, :]$ 为共享专家的输出 token（无权重）

## 规格

| 类型 | shape | dtype |
|------|-------|-------|
| 输入 token_data | [BS, K+1, H] | bfloat16 |
| 输入 expert_scales | [BS, K] | float32 |
| 输出 y | [BS, H] | bfloat16 |

## 切分策略

### 1. SplitBS（默认）
- **适用场景**：BS 较大，H 和 K 较小
- **实现方式**：向量化实现，按 batch 维度切分
- **特点**：最高效，推荐优先使用
- **动态支持**：支持动态 batch 维度，不同 batch size 无需重编译

### 2. SplitH
- **适用场景**：H 较大
- **实现方式**：按 hidden 维度切分，每次处理 H_tile 个 hidden 元素
- **特点**：适合大 hidden 维度场景
- **动态支持**：支持动态 batch 维度

## 动态轴支持

- **Batch 维度**：使用 `pypto.DYNAMIC` 标记，支持运行时变化
- **Hidden 维度**：通过参数 `h` 传入，编译时确定

## 使用方法

### 基本用法

```python
import torch
from attention_worker_combine import (
    attention_worker_combine_splitbs_kernel,
    attention_worker_combine_splith_kernel
)

# 准备数据
BS, K, H = 8, 2, 32
token_data = torch.randn(BS, K + 1, H, dtype=torch.bfloat16, device='npu:14')
expert_scales = torch.rand(BS, K, dtype=torch.float32, device='npu:14')
y = torch.zeros(BS, H, dtype=torch.bfloat16, device='npu:14')

# 执行计算 - SplitBS (推荐)
attention_worker_combine_splitbs_kernel(token_data, expert_scales, y, h=H, tile_bs=8)

# 或使用 SplitH
# attention_worker_combine_splith_kernel(token_data, expert_scales, y, h=H, tile_bs=8)
```

### 环境要求

```bash
# 设置设备 ID
export TILE_FWK_DEVICE_ID=14

# 设置库路径
export PTO_TILE_LIB_CODE_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann}/aarch64-linux
```

## 测试结果

```
Strategy     BS     K      H      Result     Max Diff
----------------------------------------------------------------------
SplitBS      8      2      32     PASS       0.00000000
SplitH       8      2      32     PASS       0.00000000
SplitBS      16     2      32     PASS       0.00000000
SplitH       16     2      32     PASS       0.00000000
----------------------------------------------------------------------
Total: 4/4 passed (动态 batch 无需重编译)
```

## 文件结构

```
attention_worker_combine/
├── attention_worker_combine.py     # PyPTO kernel 实现 (动态 batch)
└── README.md                       # 本文档
```

## 注意事项

1. **H 必须是 16 的倍数**：BF16 类型需要 32 bytes 对齐，即 16 元素
2. **只支持 BF16**：当前实现仅支持 bfloat16 数据类型
3. **向量化实现**：所有策略均使用向量化实现
4. **动态 Batch**：支持 batch 维度动态变化，无需重编译

## 开发历程

1. **Golden 开发**：完成 PyTorch 参考实现并验证通过
2. **SplitBS/SplitH 开发**：向量化实现，一次性通过
3. **动态轴支持**：添加 batch 维度动态支持，不同 batch size 无需重编译

## 相关文档

- [PyPTO API 文档](../../../docs/api/)
- [PyPTO 动态轴文档](https://gitcode.com/cann/pypto/blob/master/docs/zh/api/tensor_api/others/pypto-DYNAMIC.md)
