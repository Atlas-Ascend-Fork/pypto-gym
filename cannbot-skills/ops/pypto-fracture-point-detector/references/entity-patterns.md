# 实体识别模式

定义从 session 上下文中识别 pypto 实体的模式。

## 实体类型

### 1. API 实体

匹配 pypto 模块的 API 调用。

**模式**：`pypto\.[a-zA-Z_][a-zA-Z0-9_]*`

**示例**：
- `pypto.add` — 算子 API
- `pypto.matmul` — 算子 API
- `pypto.Tile` — Tile 类
- `pypto.Tensor` — Tensor 类
- `pypto.compile` — 编译 API

**注意**：同一 API 的不同调用形式视为同一实体（如 `pypto.add(a, b)` 和 `pypto.add(x, y)` 属于同一实体 `pypto.add`）。

### 2. 文件实体

匹配涉及的文件路径。

**模式**：`[\w./-]+\.(py|md|json|yaml|yml|rst|txt|cfg|toml)`

**注意**：
- 只关注与 pypto 相关的文件路径（包含 `pypto`、`operator`、`golden`、`docs` 等关键词的路径）
- 排除通用工具文件和配置文件

**示例**：
- `python/pypto/operation.py`
- `docs/api/add.md`
- `examples/operator/demo.py`
- `tests/test_add.py`

### 3. 概念实体

匹配 pypto 框架的核心概念。

**预定义关键词列表**：

| 概念 | 说明 |
|------|------|
| tile | Tile 编程模型 |
| tensor | Tensor 数据结构 |
| pass | 编译 pass |
| codegen | 代码生成 |
| ub | Unified Buffer |
| gm | Global Memory |
| l1 | L1 缓存 |
| tiling | 切分策略 |
| schedule | 调度 |
| operator | 算子 |
| golden | 标杆实现 |
| ascend | 昇腾硬件 |
| cann | CANN 框架 |
| npu | NPU 设备 |
| aiv / aic | AI Vector/Core |
| dtype | 数据类型 |
| shape | 张量形状 |
| broadcast | 广播机制 |
| reduce | 归约操作 |
| workspace | 工作空间 |
| kernel | 内核函数 |

**注意**：概念实体只有在 session 中围绕该概念出现问题时才算作实体。单纯提及不算。
