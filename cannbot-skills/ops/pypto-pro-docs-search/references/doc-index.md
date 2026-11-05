# PyPTO Pro 指南索引

按主题读取 `$PYPTO_DEVKIT_DIR/docs/<路径>`。下表使用上游当前 `guide` 路径；各分区的全部条目以其 `index.md` 为准。

| 主题 | 路径 |
|---|---|
| PyPTO 共享简介 | `guide/introduction.md` |
| 编程指南总览 | `guide/programming_guide/pro/index.md` |
| 编程范式与抽象硬件架构 | `guide/programming_guide/pro/programming_paradigm/index.md` |
| Kernel 函数 | `guide/programming_guide/pro/development/kernel_function.md` |
| Tile / Reg / Cube 编程、Tiling 与尾块处理 | `guide/programming_guide/pro/development/tile_based_python_programming/index.md` |
| JIT 与离线编译 | `guide/programming_guide/pro/development/compilation_and_execution/index.md` |
| 高级编程与自动流水并行 | `guide/programming_guide/pro/advanced_programming/index.md` |
| 功能调试与性能调优 | `guide/programming_guide/pro/debug/index.md` |
| 快速入门（SIMD / SIMT 样例） | `guide/quick_start/pro/index.md` |

从快速入门总索引读取当前版本的样例页面路径；不同版本可能采用子目录或同级文件布局。

按当前问题补充发现时，限定检索 `guide/programming_guide/pro/`、`guide/quick_start/pro/` 和 `guide/introduction.md`。需要目录导航时读取对应 `index.md`，不要求先生成或遍历全量资料索引。

缓存保留配套图片，未缓存的安装与通用配置页面由指南中的远端链接指向。当前缓存不含独立的错误码排障资料；报错时从上述调试指南与实际日志定位。API 参数与约束见 [`api-index.md`](api-index.md)，官方样例见 [`sample-index.md`](sample-index.md)。
