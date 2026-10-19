# Contributing to PyPTO-Gym

欢迎为 PyPTO-Gym 贡献代码、文档和算子样例！在提交之前，请阅读以下指引。

Welcome to PyPTO-Gym! We appreciate your contributions of code, documentation, and operator examples.

## 贡献者许可协议 / Contributor License Agreement (CLA)

向本仓库提交 Pull Request 即表示您确认：

- 您拥有所提交代码的完整知识产权，或已获得权利人的明确授权
- 您同意将提交内容以本仓库的许可证（CANN Open Software License Agreement Version 2.0）授权

By submitting a Pull Request, you confirm that you have the right to contribute the submitted code and agree to license it under the repository's license (CANN Open Software License Agreement Version 2.0).

## 如何贡献 / How to Contribute

### 1. 提交新算子 / Adding a New Operator

1. 在 `ops/pypto_tensor/` 下新建子目录（通用算子放入 `experimental/` 对应子类）
2. 编写 kernel 实现文件，命名建议为 `*_impl.py`，使用 `@pypto.frontend.jit` 装饰器
3. 在 `tests/ops/` 下新建同名子目录，添加 `test_*.py` 测试文件
4. 测试用例需包含：
   - 正确性验证（与 torch golden 实现对比，精度容差参考 benchmark 配置）
   - SoC 兼容性标注：`@pytest.mark.soc("950")`（910B/910C）或 `@pytest.mark.soc("910")`（910A）
   - 多卡需求标注（如需）：`@pytest.mark.world_size(N)`
5. 编写 `README.md`，包含：
   - 算子功能描述和数学公式
   - 输入/输出参数说明与 shape 范围
   - 对应测试文件路径
   - 已知限制和注意事项

### 2. 代码风格 / Code Style

- 遵循 PEP 8 规范
- 使用 4 空格缩进（非 Tab）
- 函数名、变量名使用 snake_case，类名使用 PascalCase
- 关键接口添加 docstring
- 运行 `bash format.sh` 进行格式化（若已配置 ruff）

### 3. 提交规范 / Commit Guidelines

- 使用清晰的中文或英文 commit message
- 格式建议：`<类型>: <简短描述>`
- 类型示例：`feat`（新算子）、`fix`（修复）、`docs`（文档）、`refactor`（重构）、`test`（测试）、`bench`（性能优化）

### 4. Pull Request 流程

1. Fork 本仓库并创建特性分支
2. 确保目标算子通过本地所有测试：`pytest tests/ops/<your_op> -v`
3. 提交 PR 时描述变更内容和测试结果
4. 等待 Review，根据反馈修改
5. Review 通过后由维护者合入

### 5. 运行测试 / Running Tests

```bash
# 单算子测试
pytest tests/ops/<model_op> -v

# 全量测试（排除 experimental）
pytest -v

# 实验性算子测试
pytest tests/ops/experimental/<category>/<op_name> -v
```

## 联系方式 / Contact

- **问题反馈**：通过 GitCode Issues 提交
- **功能建议**：通过 GitCode 讨论区交流
