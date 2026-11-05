# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Exercise repository entry-selection hooks with CPU-only subprocess tests."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.fixture
def cli(tmp_path):
    pytest.importorskip("torch")
    root = Path(__file__).resolve().parents[2]
    hooks = (
        "pytest_addoption",
        "pytest_configure",
        "pytest_sessionfinish",
        "pytest_terminal_summary",
        "pytest_runtest_makereport",
        "pytest_runtest_logreport",
    )
    (tmp_path / "entry_hooks.py").write_text(
        "import importlib.util\n"
        f"spec = importlib.util.spec_from_file_location('repository_hooks', {str(root / 'conftest.py')!r})\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        f"for name in {hooks!r}:\n    globals()[name] = getattr(module, name)\n"
    )
    (tmp_path / "kernel.py").write_text("def entry():\n    return 23\n")
    env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(tmp_path), str(root / "tests/common"), env.get("PYTHONPATH", "")]
    )

    def run(body, extra=()):
        (tmp_path / "test_candidate.py").write_text(body)
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-c",
                "/dev/null",
                "--noconftest",
                "-p",
                "entry_hooks",
                "-q",
                "--op-entry",
                "sample=kernel.py:entry",
                *extra,
                "test_candidate.py",
            ],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=45,
        )

    return run


@pytest.mark.parametrize("mode", ["serial", "forked", "xdist"])
def test_candidate_identity_reaches_the_controller(cli, mode):
    extra = ()
    if mode == "forked":
        pytest.importorskip("pytest_forked")
        extra = ("-p", "pytest_forked", "--forked")
    elif mode == "xdist":
        pytest.importorskip("xdist")
        extra = ("-p", "xdist.plugin", "-n", "2")
    result = cli(
        "from op_caller import get_op_caller\n"
        "def test_candidate():\n    assert get_op_caller('sample')() == 23\n",
        extra,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "JIT entry sample:" in result.stdout


def test_unused_candidate_fails_execution_but_allows_collection(cli):
    body = "def test_reference_only():\n    assert 2 + 2 == 4\n"
    result = cli(body)
    assert result.returncode == 4, result.stdout + result.stderr
    assert "Requested JIT entries were not loaded: sample" in result.stdout
    assert cli(body, ("--collect-only",)).returncode == 0
