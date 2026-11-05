# PyPTO Pro 官方样例索引

唯一的文件清单是 [`official_samples.md`](official_samples.md)。按其“算子名称 / 类型 / 描述”定位资料，再读取 `$PYPTO_DEVKIT_DIR/<缓存相对路径>`；路径已包含 `pro_ops/` 前缀。

需要按 API 调用或关键词补充发现时，在 `$PYPTO_DEVKIT_DIR/pro_ops/` 使用 `Grep` / `Glob`，并对照统一清单确认命中文件。缓存只装配清单内样例，不能依赖上游 frontend 中的其他文件。

本索引只提供检索方法。增删或重命名条目时维护 `official_samples.md`，不生成项目内的第二份列表。

样例中的 Kernel、调用方式与验证逻辑以文件原文为准；Pro 缓存不另设普通 PyPTO 的 `ops/`、`tests/` 目录。确认接口语义时继续读取 [`api-index.md`](api-index.md) 指向的 API 文档，样例缺失则按 [缓存未就绪时](../SKILL.md#缓存未就绪时) 交还 orchestrator。
