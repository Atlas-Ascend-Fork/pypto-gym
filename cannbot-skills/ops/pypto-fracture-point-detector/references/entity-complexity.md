# 实体复杂度定义

实体复杂度用于调整检测阈值，避免对复杂实体产生误报。

## 分类标准

按 golden 接口的 torch 主操作数个数区分：

| 复杂度 | torch 主操作数 | 阈值倍数 |
|--------|---------------|----------|
| **简单** | ≤8 个 | x1.0 |
| **中等** | 9-20 个 | x1.5 |
| **复杂** | 21+ 个 | x2.0 |

## 判定方式

1. 如果 session 中涉及的实体是一个 pypto API/算子，查看其 golden 接口中使用了多少个 torch 主操作来实现等价功能
2. 如果无法确定（如概念实体、文件实体），默认按**简单**处理
3. 如果 session 上下文中有明确的 golden 实现代码，直接计数 torch 操作数

## 什么是"torch 主操作"

指 golden 函数体中直接调用的 `torch.*` 函数，不计算以下内容：
- `torch.tensor()` 等纯构造函数
- `.shape`、`.dtype` 等属性访问
- `.to()`、`.contiguous()` 等类型/内存转换
- Python 控制流语句

## 示例

### 简单实体（≤8 个 torch 操作）

```python
# pypto.add 的 golden — 1 个 torch 操作
def golden_add(x, y):
    return torch.add(x, y)

# pypto.mul 的 golden — 1 个 torch 操作
def golden_mul(x, y):
    return torch.mul(x, y)
```

### 中等实体（9-20 个 torch 操作）

```python
# pypto.matmul 的 golden — 包含转置、乘法、累加等多步操作
# 通常涉及 10+ 个 torch 调用
```

### 复杂实体（21+ 个 torch 操作）

```python
# 模型级算子如 attention、layer_norm 组合
# 或量化相关的复杂实现
# 通常涉及 20+ 个 torch 调用
```

## 阈值对照

| 参数 | 简单 (x1.0) | 中等 (x1.5) | 复杂 (x2.0) |
|------|-------------|-------------|-------------|
| 重复操作 | ≥3 次 | ≥5 次 | ≥6 次 |
| 过度探索-搜索 | >10 次 | >15 次 | >20 次 |
| 过度探索-读取 | >15 次 | >22 次 | >30 次 |
