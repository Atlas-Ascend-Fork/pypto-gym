---
name: pypto-pro-docs-search
description: 为 PyPTO-Pro 任务准备、校验和检索指定版本的本地文档缓存。用于离线资料准备、缓存完整性检查与资源定位；API 和能力查证衔接 pypto-pro-kernel 的核心查询。
---

# pypto-pro-docs-search

围绕当前问题定位 Pro 原始资料。已有准确路径时直接读取；核心 skill 与匹配源码可用时，API、指南和能力查证使用[核心查询](../pypto-pro-kernel/references/tooling/checks.md#official-documents-guides-and-docstrings)。需要本地文档缓存时，按下方说明准备并限定路径检索；独立装配缓存只需要本 skill 的同步脚本和资源清单。

## 资源缓存

`$PYPTO_DEVKIT_DIR` 默认是当前项目的 `.devkit/`。资料缓存按任务需要准备，使用任务明确给出的来源和路径；协作执行时传递同一缓存的实际路径。

| 子目录 | 内容 |
|---|---|
| `docs/` | Pro API、编程指南、快速入门、简介及配套图片 |
| `pro_ops/` | 官方指定算子样例，仅装配统一样例清单内文件，入口见 [`references/sample-index.md`](references/sample-index.md) |

指南保留上游 `guide` 目录结构，简介使用共享的 `introduction.md`。缓存不包含 Tensor API/专属指南目录、`docs/install/` 或普通 PyPTO 算子与测试。

## 准备缓存

需要准备或更新缓存时运行本技能的独立脚本。将 `<skill-dir>` 替换为本 skill 的实际绝对路径，沿用任务指定的源码与缓存位置：

```bash
PYPTO_SRC="/path/to/matching/pypto" \
PYPTO_DEVKIT_DIR="/path/to/.devkit" \
python "<skill-dir>/scripts/sync_devkit.py" \
    --samples "<skill-dir>/references/official_samples.md"
```

在同一调用末尾增加 `--check` 只校验已有缓存：`READY` 且退出码为 0 表示就绪；`NEED_PROVISION` 且退出码为 4 表示需要装配。正常装配按清单复制样例并校验结果，无需再清理 `pro_ops/`。

脚本默认从 `https://gitcode.com/cann/pypto.git` 获取 Pro 资料。`PYPTO_SRC` 可指定已有 PyPTO 工作树，`PYPTO_SRC_URL` 可覆盖远端源，`--pin <git-ref>` 从远端获取指定版本。成功后写入 `MANIFEST.json` 和就绪标记，记录缓存来源与版本，以目标运行环境对应的资料为准。

文档缓存与完整源码根是不同资源。核心查询的 `PYPTO_SRC` / `--pypto-root` 指向匹配版本的源码 checkout；本缓存只保留资料与清单文件，不能作为完整源码或已安装运行时的替代。

**协作路径传递**：将当前任务选定的绝对路径和版本交给执行者，命令中的路径加引号。缺少路径时核对实际配置，明确缺项；已有有效来源时直接复用。

## 检索

使用原生 `Glob` / `Grep`，按资料类型限定路径：

- **API 名称与签名**：在 `$PYPTO_DEVKIT_DIR/docs/pypto_pro/api/` 递归查找 `*<API名>*.md`，再读取参数、类型与约束全文。
- **编程与用法**：分别检索 `$PYPTO_DEVKIT_DIR/docs/guide/programming_guide/pro/`、`$PYPTO_DEVKIT_DIR/docs/guide/quick_start/pro/` 与 `$PYPTO_DEVKIT_DIR/docs/guide/introduction.md`，不扩大到整个 `docs/guide/`。
- **参考样例**：依据 [`references/sample-index.md`](references/sample-index.md) 中的统一清单，在 `$PYPTO_DEVKIT_DIR/pro_ops/` 查找关键词或 API 调用，只读取清单内样例。

命中后直接 `Read` 全文，返回实际命中的缓存相对路径和相关事实。

## 边界

已知确切路径时直接读取；本技能用于按关键词发现、跨文件定位。资料未说明的接口能力不能凭名称或普通 PyPTO 的同名 API 推断。

## 缓存未就绪时

`--check` 返回 `NEED_PROVISION`，或发现缓存资料缺失时，报告具体缺项。任务确需缓存时按“准备缓存”处理，并遵循当前协作方式。已有匹配源码根时，核心查询可以直接核对原资料。离线且无可用来源时报告缺口，不编造路径、版本或能力结论。

## 详细索引（按入口键）

- 知道 **API 名 / 功能类别** → [`references/api-index.md`](references/api-index.md)
- 查 **编程指南 / 快速入门 / 调试调优** → [`references/doc-index.md`](references/doc-index.md)
- 找 **官方指定样例** → [`references/sample-index.md`](references/sample-index.md)
