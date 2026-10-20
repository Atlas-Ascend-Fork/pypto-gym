# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""桥接文件豁免判定（AST 版）单元测试。

判定逻辑已从子串 ``"import pypto" not in source`` 改为 AST 解析：
``import pypto`` / ``import pypto.xxx`` / ``from pypto import ...`` 均算
含 pypto；``import pypto_utils`` 等同名前缀独立包不算；语法无法解析时
保守不豁免。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pypto_op_lint.infer import (  # noqa: E402
    _imports_top_level_pypto,
    _is_bridge_adapter_file,
)

BRIDGE = "demo_pypto_impl.py"


def test_import_pypto_detected():
    assert _imports_top_level_pypto("import pypto\n")


def test_import_pypto_submodule_detected():
    assert _imports_top_level_pypto("import pypto.frontend\n")


def test_from_pypto_import_detected():
    """子串判定会漏掉 ``from pypto import ...``（不含 "import pypto" 子串）。"""
    assert _imports_top_level_pypto("from pypto import frontend\n")


def test_from_pypto_submodule_import_detected():
    assert _imports_top_level_pypto("from pypto.frontend import jit\n")


def test_import_pypto_utils_not_detected():
    """``import pypto_utils`` 是同名前缀独立包，不应算含 pypto。"""
    assert not _imports_top_level_pypto("import pypto_utils\n")


def test_pure_torch_not_detected():
    assert not _imports_top_level_pypto("import torch\nimport torch.nn as nn\n")


def test_pypto_in_comment_or_string_not_detected():
    """注释/字符串里出现 import pypto 字样不应误判（AST 只看真实 import）。"""
    source = '# import pypto\nx = "import pypto"\n'
    assert not _imports_top_level_pypto(source)


def test_syntax_error_conservatively_detected():
    """ast.parse 失败 → 保守按含 pypto 处理（不豁免）。"""
    assert _imports_top_level_pypto("def broken(:\n")


def test_bridge_file_with_pypto_utils_exempt():
    source = "import torch\nimport pypto_utils\n\nclass ModelNew(torch.nn.Module):\n    pass\n"
    assert _is_bridge_adapter_file(BRIDGE, source)


def test_bridge_file_with_from_pypto_import_not_exempt():
    """文件名匹配但含 ``from pypto import frontend`` → 不豁免。"""
    source = "from pypto import frontend\n"
    assert not _is_bridge_adapter_file(BRIDGE, source)


def test_non_bridge_filename_never_exempt():
    assert not _is_bridge_adapter_file("demo_impl.py", "import torch\n")


def test_none_source_never_exempt():
    assert not _is_bridge_adapter_file(BRIDGE, None)
