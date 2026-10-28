#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Run with: python -m unittest discover -s tests -p 'test_*.py'."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "kernel_installer", PLUGIN_ROOT / "scripts/install_opencode.py"
)
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def make_junction(path, target):
    command = shutil.which("cmd.exe")
    if command is None:
        raise unittest.SkipTest("Windows command interpreter unavailable")
    return subprocess.run(
        [command, "/d", "/c", "mklink", "/J", str(path), str(target)],
        capture_output=True,
        text=True,
        shell=False,
    )


class ActualResourceTest(unittest.TestCase):
    def test_junction_command_uses_resolved_executable_and_no_implicit_shell(self):
        command = "C:/Windows/System32/cmd.exe"
        with mock.patch.object(shutil, "which", return_value=command):
            with mock.patch.object(subprocess, "run") as run:
                make_junction(Path("link with spaces"), Path("target with spaces"))
        run.assert_called_once_with(
            [
                command,
                "/d",
                "/c",
                "mklink",
                "/J",
                "link with spaces",
                "target with spaces",
            ],
            capture_output=True,
            text=True,
            shell=False,
        )

    def test_subagent_sources_permissions_and_metadata_match_distribution(self):
        manifest = installer.read_json(PLUGIN_ROOT / ".claude-plugin/plugin.json")
        declared = manifest["agents"]
        self.assertEqual(len(declared), len(installer.AGENTS))
        self.assertEqual({Path(name).stem for name in declared}, set(installer.AGENTS))
        self.assertEqual(
            {path.stem for path in (PLUGIN_ROOT / "agents").glob("*.md")},
            set(installer.AGENTS),
        )
        for relative in [*declared, "AGENTS.md"]:
            path = PLUGIN_ROOT / relative
            self.assertTrue(path.is_file(), relative)
            frontmatter = path.read_text(encoding="utf-8").split("---", 2)[1]
            primary = relative == "AGENTS.md"
            if not primary:
                self.assertRegex(frontmatter, rf"(?m)^name: {path.stem}$")
            self.assertRegex(
                frontmatter, rf"(?m)^mode: {'primary' if primary else 'subagent'}$"
            )
            self.assertRegex(frontmatter, r"(?m)^permission:$")
            for tool in ("read", "edit", "bash", "task"):
                denied = (tool == "task" and not primary) or (
                    path.stem in {"pypto-pro-op-planner", "pypto-pro-op-verifier"}
                    and tool == "edit"
                )
                self.assertRegex(
                    frontmatter, rf"(?m)^  {tool}: {'deny' if denied else 'allow'}$"
                )
            self.assertNotRegex(frontmatter, r"(?m)^tools:")

    def test_deep_workspace_paths_do_not_leak_windows_extended_prefix_into_prompts(
        self,
    ):
        with tempfile.TemporaryDirectory(
            dir=installer.filesystem_path(tempfile.gettempdir())
        ) as temporary:
            project = (
                Path(temporary) / ("workspace-" + "x" * 90) / ("nested-" + "y" * 90)
            )
            result = installer.install(project)
            config = installer.filesystem_path(project) / ".opencode"
            self.assertTrue(
                (
                    config
                    / "skills/pypto-pro-kernel/examples/algorithms/bilinear_sampling_backward"
                ).is_dir()
            )
            primary = (config / "agents" / f"{installer.PLUGIN}.md").read_text(
                encoding="utf-8"
            )
            self.assertNotIn("//?/", primary)
            self.assertNotIn("//?/", result["config_root"])

    def test_complete_skill_primary_discovery_and_idempotence(self):
        with tempfile.TemporaryDirectory(
            dir=installer.filesystem_path(tempfile.gettempdir())
        ) as temporary:
            project = Path(temporary).resolve() / "workspace with spaces"
            result = installer.install(project)
            config = project / ".opencode"
            source = PLUGIN_ROOT.parents[1] / "ops/pypto-pro-kernel"
            ignored = {"__pycache__", ".pytest_cache", "node_modules", ".DS_Store"}
            files = {
                "skills/pypto-pro-kernel/"
                + path.relative_to(source).as_posix(): path.read_bytes()
                for path in source.rglob("*")
                if path.is_file()
                and not ignored.intersection(path.relative_to(source).parts)
            }
            files["skills/pypto-pro-kernel/LICENSE"] = (
                PLUGIN_ROOT.parents[2] / "LICENSE"
            ).read_bytes()
            installed = {
                "skills/pypto-pro-kernel/"
                + path.relative_to(config / "skills/pypto-pro-kernel").as_posix()
                for path in (config / "skills/pypto-pro-kernel").rglob("*")
                if path.is_file()
            }
            self.assertEqual(installed, set(files))
            for relative, expected in files.items():
                self.assertEqual((config / relative).read_bytes(), expected, relative)
            self.assertEqual(
                {path.name for path in (config / "skills").iterdir()}, {installer.SKILL}
            )
            self.assertEqual(
                {path.stem for path in (config / "agents").iterdir()},
                {installer.PLUGIN, *installer.AGENTS},
            )
            self.assertEqual(
                installer.read_json(config / "opencode.json")["default_agent"],
                installer.PLUGIN,
            )
            for agent in (config / "agents").glob("*.md"):
                self.assertNotIn(
                    "$CANNBOT_CONFIG_ROOT", agent.read_text(encoding="utf-8")
                )
            for obsolete in (
                "hooks",
                "plugins",
                "scriptor",
                "pypto-pro-op-kb",
                "scriptor-install.json",
            ):
                self.assertFalse((config / obsolete).exists(), obsolete)
            before = (project / "AGENTS.md").read_bytes()
            self.assertIsNone(installer.install(project)["backup"])
            self.assertEqual((project / "AGENTS.md").read_bytes(), before)
            self.assertEqual(result["skill"], installer.SKILL)
            receipt = installer.read_json(config / installer.RECEIPT)
            self.assertEqual(receipt["installed_skills"], [installer.SKILL])
            self.assertEqual(
                receipt["installed_agents"], [installer.PLUGIN, *installer.AGENTS]
            )
            self.assertEqual(result["subagents"], list(installer.AGENTS))
            for name in installer.AGENTS:
                expected = installer.render(
                    (PLUGIN_ROOT / "agents" / f"{name}.md").read_text(encoding="utf-8"),
                    config,
                )
                self.assertEqual(
                    (config / "agents" / f"{name}.md").read_bytes(),
                    expected.encode("utf-8"),
                )


class InstallationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.plugin = (
            self.root / "repo/cannbot-skills/plugins-official" / installer.PLUGIN
        )
        self.project = self.root / "workspace"
        self.config = self.project / ".opencode"
        self.source = self.plugin.parents[1] / "ops" / installer.SKILL
        write(
            self.plugin / "AGENTS.md",
            "---\nname: pypto-pro-op-orchestrator\nmode: primary\n---\n"
            "Read $CANNBOT_CONFIG_ROOT/skills/pypto-pro-kernel/SKILL.md\n",
        )
        for name in installer.AGENTS:
            write(
                self.plugin / "agents" / f"{name}.md",
                "---\nmode: subagent\npermission:\n  task: deny\n---\n"
                "Read $CANNBOT_CONFIG_ROOT/skills/pypto-pro-kernel/SKILL.md\n",
            )
        write(self.plugin.parents[2] / "LICENSE", "Repository CANN license\n")
        for relative in (
            "SKILL.md",
            "examples/add.py",
            "references/index.md",
            "scripts/check.py",
        ):
            write(self.source / relative, f"Bundled resource: {relative}\n")
        write(
            self.plugin / "references/multi-agent.md", "Explicit multi-agent opt-in.\n"
        )
        write(
            self.plugin / "references/performance-constraints.md",
            "Current shared performance reference\n",
        )
        self.independent_content = "Independent resource\n"
        self.patch = mock.patch.object(installer, "ORCHESTRATOR", self.plugin)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def install(self, **kwargs):
        return installer.install(self.project, **kwargs)

    def seed_obsolete_resource(self):
        self.install()
        relative = "references/obsolete.md"
        content = b"Previously managed reference\n"
        (self.config / relative).write_bytes(content)
        previous = installer.read_json(self.config / installer.RECEIPT)
        previous["owned_files"][relative] = installer.digest(content)
        (self.config / installer.RECEIPT).write_bytes(installer.json_bytes(previous))
        return relative, content

    def test_obsolete_owned_resource_is_backed_up_and_user_neighbors_survive(self):
        relative, content = self.seed_obsolete_resource()
        user_file = self.config / "references/user.md"
        write(user_file, "User reference\n")
        result = self.install()
        self.assertFalse((self.config / relative).exists())
        self.assertEqual((Path(result["backup"]) / relative).read_bytes(), content)
        self.assertEqual(user_file.read_text(), "User reference\n")
        current = installer.read_json(self.config / installer.RECEIPT)
        self.assertNotIn(relative, current["owned_files"])
        self.assertIsNone(self.install()["backup"])

    def test_modified_obsolete_owned_resource_blocks_before_changes(self):
        relative, _ = self.seed_obsolete_resource()
        write(self.config / relative, "User changes to a managed reference\n")
        before = self.tree_snapshot()
        with self.assertRaisesRegex(
            installer.InstallError, "modified retired installation resource"
        ):
            self.install()
        self.assertEqual(self.tree_snapshot(), before)

    def test_failed_resource_retirement_restores_receipt_and_manifest(self):
        self.seed_obsolete_resource()
        before = self.tree_snapshot()
        atomic_write = installer._atomic_write
        failed = False

        def fail_receipt_once(path, content):
            nonlocal failed
            if path.name == installer.RECEIPT and not failed:
                failed = True
                raise OSError("simulated receipt write failure")
            return atomic_write(path, content)

        with mock.patch.object(
            installer, "_atomic_write", side_effect=fail_receipt_once
        ):
            with self.assertRaisesRegex(OSError, "simulated receipt"):
                self.install()
        self.assertEqual(self.tree_snapshot(), before)

    def test_owned_named_subagents_refresh_and_protect_later_user_edits(self):
        self.install()
        before_agents = {}
        for name in installer.AGENTS:
            relative = f"agents/{name}.md"
            before_agents[relative] = (self.config / relative).read_bytes()
            source = self.plugin / relative
            source.write_bytes(source.read_bytes() + b"Updated role instructions\n")
        result = self.install()
        receipt = installer.read_json(self.config / installer.RECEIPT)
        self.assertEqual(
            receipt["installed_agents"], [installer.PLUGIN, *installer.AGENTS]
        )
        for relative, original in before_agents.items():
            self.assertEqual((Path(result["backup"]) / relative).read_bytes(), original)
            actual = (self.config / relative).read_bytes()
            self.assertIn(b"Updated role instructions", actual)
            self.assertEqual(receipt["owned_files"][relative], installer.digest(actual))
        self.assertIsNone(self.install()["backup"])
        write(self.config / "agents" / f"{installer.AGENTS[0]}.md", "User changes\n")
        before = self.tree_snapshot()
        with self.assertRaisesRegex(installer.InstallError, "unmanaged or modified"):
            self.install()
        self.assertEqual(self.tree_snapshot(), before)

    def test_unowned_named_agents_are_not_overwritten(self):
        for name in installer.AGENTS:
            with self.subTest(agent=name):
                self.project = self.root / name
                self.config = self.project / ".opencode"
                write(self.config / "agents" / f"{name}.md", "User-owned agent\n")
                before = self.tree_snapshot()
                with self.assertRaisesRegex(
                    installer.InstallError, "unmanaged or modified"
                ):
                    self.install()
                self.assertEqual(self.tree_snapshot(), before)

    def test_old_receipts_and_autoload_entries_require_new_workspace_before_writes(
        self,
    ):
        cases = (
            (
                "scriptor-install.json",
                json.dumps(
                    {
                        "schema": "cannbot.scriptor-install/1",
                        "workflows": [installer.PLUGIN],
                    }
                ),
            ),
            (
                "cannbot-manifest.json",
                json.dumps({"team": installer.PLUGIN, "version": "2.0.0"}),
            ),
            ("plugins/pypto-pro-op-lint.ts", "Old automatic lint hook\n"),
            ("plugins/pypto-pro-state-transition.ts", "Old state transition hook\n"),
            ("plugins/pypto-pro-scriptor.ts", "Old Scriptor hook\n"),
        )
        for current_receipt in (False, True):
            for index, (relative, content) in enumerate(cases):
                with self.subTest(current_receipt=current_receipt, entry=relative):
                    self.project = self.root / f"old-config-{current_receipt}-{index}"
                    self.config = self.project / ".opencode"
                    if current_receipt:
                        self.install()
                    write(self.config / relative, content)
                    before = self.tree_snapshot()
                    with self.assertRaisesRegex(
                        installer.InstallError, "new working directory"
                    ):
                        self.install()
                    self.assertEqual(self.tree_snapshot(), before)

    def test_old_primary_and_router_instructions_require_new_workspace(self):
        prompt = (
            "---\nname: pypto-pro-op-orchestrator\nmode: primary\n---\n"
            "Load pypto-pro-docs-search and pypto-pro-scriptor\n"
        )
        for index, (tool, relative, content) in enumerate(
            (
                ("opencode", "AGENTS.md", installer.OLD_MARKER + "\n" + prompt),
                (
                    "opencode",
                    "AGENTS.md",
                    (installer.OLD_MARKER + "\n" + prompt).replace("\n", "\r\n"),
                ),
                ("claude", "CLAUDE.md", prompt),
                ("opencode", ".opencode/agents/pypto-pro-op-orchestrator.md", prompt),
            )
        ):
            with self.subTest(tool=tool, entry=relative, case=index):
                self.project = self.root / f"old-prompt-{index}"
                self.config = self.project / ".opencode"
                target = self.project / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content.encode("utf-8"))
                before = self.tree_snapshot()
                with self.assertRaisesRegex(
                    installer.InstallError, "new working directory"
                ):
                    installer.install(self.project, tool=tool)
                self.assertEqual(self.tree_snapshot(), before)

    def test_independent_generic_resources_are_not_claimed(self):
        retained = (
            "hooks/pypto-pro-op-lint/tool.py",
            "user-tools/helper.py",
            "skills/pypto-pro-op-develop/SKILL.md",
            "references/user.md",
            "plugins/pypto-state-transition.ts",
            "plugins/lib/state-transition-core.ts",
            "agents/user.md",
        )
        for relative in retained:
            write(self.config / relative, self.independent_content)
        write(self.config / "cannbot-manifest.json", '{"team":"another-plugin"}\n')
        write(
            self.config / "scriptor-install.json",
            json.dumps(
                {
                    "schema": "cannbot.scriptor-install/1",
                    "workflows": ["another-plugin"],
                }
            ),
        )
        flag = self.project / ".pypto-pro-op-lint-enabled"
        flag.write_bytes(b"true\r\n")
        preserved = {
            relative: (self.config / relative).read_bytes()
            for relative in (
                *retained,
                "cannbot-manifest.json",
                "scriptor-install.json",
            )
        }
        self.install()
        self.assertIsNone(self.install()["backup"])
        owned = installer.read_json(self.config / installer.RECEIPT)["owned_files"]
        for relative, original in preserved.items():
            self.assertEqual((self.config / relative).read_bytes(), original)
            self.assertNotIn(relative, owned)
        self.assertEqual(flag.read_bytes(), b"true\r\n")

    def test_receipt_owned_wrapper_is_retired_only_when_unchanged(self):
        for modified in (False, True):
            with self.subTest(modified=modified):
                self.project = self.root / f"owned-wrapper-{modified}"
                self.config = self.project / ".opencode"
                self.install()
                relative = "plugins/pypto-pro-op-lint.ts"
                target = self.config / relative
                original = b"Previously managed wrapper\n"
                target.parent.mkdir(parents=True)
                target.write_bytes(original)
                receipt = installer.read_json(self.config / installer.RECEIPT)
                receipt["owned_files"][relative] = installer.digest(original)
                write(self.config / installer.RECEIPT, json.dumps(receipt))
                if modified:
                    target.write_bytes(b"User modifications\n")
                    before = self.tree_snapshot()
                    with self.assertRaisesRegex(
                        installer.InstallError, "modified retired installation resource"
                    ):
                        self.install()
                    self.assertEqual(self.tree_snapshot(), before)
                else:
                    result = self.install()
                    self.assertFalse(target.exists())
                    self.assertEqual(
                        (Path(result["backup"]) / relative).read_bytes(), original
                    )
                    self.assertIsNone(self.install()["backup"])

    def test_current_receipt_does_not_claim_independent_shared_skills(self):
        target = self.config / "skills/pypto-pro-op-develop/SKILL.md"
        write(target, self.independent_content)
        self.install()
        self.install()
        self.assertEqual(target.read_text(), self.independent_content)

    def test_modified_shared_reference_is_protected(self):
        write(self.config / "references/performance-constraints.md", "User notes\n")
        before = self.tree_snapshot()
        with self.assertRaisesRegex(installer.InstallError, "unmanaged or modified"):
            self.install()
        self.assertEqual(self.tree_snapshot(), before)

    def test_unowned_reference_link_is_rejected_without_changing_source(self):
        source = self.plugin / "references/performance-constraints.md"
        target = self.config / "references/performance-constraints.md"
        target.parent.mkdir(parents=True)
        try:
            target.symlink_to(source)
        except OSError as exc:
            self.skipTest(f"file symlinks unavailable: {exc}")
        original = source.read_bytes()
        with self.assertRaisesRegex(installer.InstallError, "non-regular file"):
            self.install()
        self.assertTrue(installer.is_link(target))
        self.assertEqual(target.read_bytes(), original)
        self.assertEqual(source.read_bytes(), original)

    def test_independent_directory_link_is_preserved(self):
        outside = self.root / "outside"
        write(outside / "helper.py", self.independent_content)
        write(outside / "SKILL.md", self.independent_content)
        target = self.config / "user-tools"
        self.make_symlink(target, outside)
        self.install()
        self.assertTrue(installer.is_link(target))
        self.assertEqual((target / "helper.py").read_text(), self.independent_content)

    def test_unowned_pro_autoload_wrapper_requires_new_workspace_before_writes(self):
        target = self.config / "plugins/pypto-pro-op-lint.ts"
        write(target, "Unknown owner\n")
        with self.assertRaisesRegex(installer.InstallError, "new working directory"):
            self.install()
        self.assertEqual(target.read_text(), "Unknown owner\n")
        self.assertFalse((self.config / "skills").exists())
        self.assertFalse((self.project / "AGENTS.md").exists())

    def test_custom_root_instructions_survive_refresh(self):
        write(self.project / "AGENTS.md", "# Project policy\nKeep my rules.\n")
        self.install()
        with (self.project / "AGENTS.md").open("a", encoding="utf-8") as stream:
            stream.write("\nMore project policy.\n")
        self.install()
        text = (self.project / "AGENTS.md").read_text()
        self.assertTrue(text.startswith("# Project policy\nKeep my rules."))
        self.assertIn("More project policy.", text)
        self.assertEqual(text.count(installer.BEGIN), 1)

    def test_modified_managed_block_is_rejected_without_changing_user_policy(self):
        self.install()
        root = self.project / "AGENTS.md"
        text = root.read_text(encoding="utf-8").replace(
            "Read ", "User edit inside managed block.\nRead "
        )
        text = (
            "# User policy\r\nKeep exact bytes.\r\n\r\n" + text + "\r\nMore policy.\r\n"
        )
        root.write_bytes(text.encode("utf-8"))
        receipt_before = (self.config / installer.RECEIPT).read_bytes()
        with self.assertRaisesRegex(
            installer.InstallError, "managed root instructions were modified"
        ):
            self.install()
        self.assertEqual(root.read_bytes(), text.encode("utf-8"))
        self.assertEqual((self.config / installer.RECEIPT).read_bytes(), receipt_before)

    def test_modified_current_resources_block_before_changes(self):
        self.install()
        target = self.config / "agents" / f"{installer.PLUGIN}.md"
        write(target, "User custom agent\n")
        root_before = (self.project / "AGENTS.md").read_bytes()
        receipt_before = (self.config / installer.RECEIPT).read_bytes()
        with self.assertRaisesRegex(installer.InstallError, "unmanaged or modified"):
            self.install()
        self.assertEqual(target.read_text(), "User custom agent\n")
        self.assertEqual((self.project / "AGENTS.md").read_bytes(), root_before)
        self.assertEqual((self.config / installer.RECEIPT).read_bytes(), receipt_before)

    def test_source_directory_initialization_keeps_template_unmodified(self):
        before = (self.plugin / "AGENTS.md").read_bytes()
        installer.install(self.plugin)
        self.assertEqual((self.plugin / "AGENTS.md").read_bytes(), before)
        primary = (
            self.plugin / ".opencode/agents" / f"{installer.PLUGIN}.md"
        ).read_text()
        self.assertNotIn("$CANNBOT_CONFIG_ROOT", primary)
        self.assertIn((self.plugin / ".opencode").as_posix(), primary)

    def test_all_project_and_explicit_global_tool_adapters(self):
        expected = {
            "opencode": ".opencode",
            "claude": ".claude",
            "cursor": ".cursor",
            "trae": ".trae",
            "copilot": ".github",
            "codearts": ".codeartsdoer",
        }
        with mock.patch.object(Path, "home", return_value=self.root / "home"):
            for tool, directory in expected.items():
                with self.subTest(tool=tool):
                    project = self.root / f"project-{tool}"
                    result = installer.install(project, tool=tool)
                    self.assertTrue(
                        (
                            project / directory / "skills/pypto-pro-kernel/SKILL.md"
                        ).is_file()
                    )
                    self.assertEqual(result["subagents"], list(installer.AGENTS))
                    prompt = "CLAUDE.md" if tool == "claude" else "AGENTS.md"
                    self.assertTrue((project / prompt).is_file())
                    global_config = self.root / f"global-{tool}"
                    result = installer.install(
                        global_config, tool=tool, global_level=True
                    )
                    self.assertTrue(
                        (global_config / "skills/pypto-pro-kernel/SKILL.md").is_file()
                    )
                    self.assertEqual(result["subagents"], list(installer.AGENTS))
                    for config in (project / directory, global_config):
                        receipt = installer.read_json(config / installer.RECEIPT)
                        self.assertEqual(
                            receipt["installed_agents"],
                            [installer.PLUGIN, *installer.AGENTS],
                        )
                        for name in installer.AGENTS:
                            relative = f"agents/{name}.md"
                            expected_agent = installer.render(
                                (self.plugin / relative).read_text(), config
                            ).encode("utf-8")
                            self.assertEqual(
                                (config / relative).read_bytes(), expected_agent
                            )
                            self.assertEqual(
                                receipt["owned_files"][relative],
                                installer.digest(expected_agent),
                            )
                    self.assertTrue((global_config / prompt).is_file())

    def test_existing_opencode_options_and_explicit_primary_are_preserved(self):
        write(
            self.config / "opencode.json",
            '{"model":"opencode/example","permission":{"bash":"ask"}}',
        )
        self.install()
        value = installer.read_json(self.config / "opencode.json")
        self.assertEqual(value["model"], "opencode/example")
        self.assertEqual(value["permission"], {"bash": "ask"})
        self.assertEqual(value["default_agent"], installer.PLUGIN)
        value["default_agent"] = "another-primary"
        original = json.dumps(value)
        path = self.config / "opencode.json"
        path.write_bytes(original.encode("utf-8"))
        result = self.install()
        self.assertEqual(path.read_bytes(), original.encode("utf-8"))
        self.assertEqual(result["next"], f"opencode --agent {installer.PLUGIN}")

    def test_invalid_primary_type_fails_before_installation(self):
        for invalid in (False, 1, [], {}):
            with self.subTest(default_agent=invalid):
                write(
                    self.config / "opencode.json",
                    json.dumps({"default_agent": invalid}),
                )
                before = self.tree_snapshot()
                with self.assertRaisesRegex(installer.InstallError, "string or null"):
                    self.install()
                self.assertEqual(self.tree_snapshot(), before)

    def test_jsonc_options_comments_and_formatting_survive_in_active_config(self):
        original = (
            '{\n// User comment\n"$schema":"https://opencode.ai/config.json",\n'
            '"permission":{"bash":"ask",}, /* User note */\n}\n'
        )
        path = self.config / "opencode.jsonc"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(original.encode("utf-8"))
        result = self.install()
        value = installer.read_jsonc(path)
        self.assertEqual(value["$schema"], "https://opencode.ai/config.json")
        self.assertEqual(value["permission"], {"bash": "ask"})
        self.assertEqual(value["default_agent"], installer.PLUGIN)
        self.assertEqual(
            (Path(result["backup"]) / "opencode.jsonc").read_text(), original
        )
        installed = path.read_bytes()
        expected = original.replace(
            "\n}\n", f'\n"default_agent": "{installer.PLUGIN}",\n}}\n'
        )
        self.assertEqual(installed, expected.encode("utf-8"))
        atomic_write = installer._atomic_write
        with mock.patch.object(
            installer, "_atomic_write", wraps=atomic_write
        ) as writer:
            self.assertIsNone(self.install()["backup"])
        self.assertFalse(any(call.args[0] == path for call in writer.call_args_list))
        self.assertEqual(path.read_bytes(), installed)

    def test_jsonc_top_level_edit_handles_nested_values_comments_and_line_endings(self):
        cases = (
            "{}",
            "{  } // end of file",
            "{\n// Only a comment\n}\n",
            '{"url":"https://host/a//b","items":[1,{"default_agent":"nested"},],}',
            '{\r\n\t"model": "example", // chosen model\r\n}\r\n',
            '{\n  "nested": {"default_agent":"nested"},\n  "default_agent" /* key */ : /* value */ null // keep\n}\n',
            '\ufeff{\r\n\t"items": ["escaped \\" quote", {"a": [1,2,],},],\r\n}\r\n',
        )
        path = self.config / "opencode.jsonc"
        for original in cases:
            with self.subTest(original=original):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(original.encode("utf-8"))
                before = installer.read_jsonc(path)
                name, content = installer._opencode_configuration(self.config)
                self.assertEqual(name, path.name)
                path.write_bytes(content)
                after = installer.read_jsonc(path)
                self.assertEqual(after, {**before, "default_agent": installer.PLUGIN})
                if "null" in original:
                    self.assertEqual(
                        content.decode("utf-8"),
                        original.replace("null", json.dumps(installer.PLUGIN)),
                    )
                for note in (
                    "// end of file",
                    "// Only a comment",
                    "// chosen model",
                    "/* key */",
                    "/* value */",
                    "// keep",
                ):
                    if note in original:
                        self.assertIn(note, content.decode("utf-8"))
                if "\r\n" in original:
                    self.assertNotIn(b"\n", content.replace(b"\r\n", b""))
                self.assertEqual(
                    installer._opencode_configuration(self.config)[1], content
                )

    def test_foreign_javascript_plugin_keeps_shared_dependencies(self):
        dependency = self.config / "plugins/lib/state-transition-core.ts"
        write(dependency, self.independent_content)
        foreign = self.config / "plugins/foreign.js"
        write(
            foreign,
            "import './lib/state-transition-core.ts';\nexport const Foreign = async () => ({});\n",
        )
        self.install()
        self.assertEqual(dependency.read_text(), self.independent_content)
        self.assertTrue(foreign.is_file())

    def tree_snapshot(self):
        if not self.project.exists():
            return {}
        return {
            path.relative_to(self.project).as_posix(): path.read_bytes()
            if path.is_file()
            else None
            for path in self.project.rglob("*")
        }

    def test_metadata_directories_and_backup_parent_file_fail_before_writes(self):
        for name in (
            installer.RECEIPT,
            "cannbot-manifest.json",
            "scriptor-install.json",
            f".{installer.PLUGIN}-backups",
        ):
            with self.subTest(name=name):
                path = self.config / name
                if name.endswith("-backups"):
                    write(path, "Not a directory\n")
                else:
                    path.mkdir(parents=True)
                before = self.tree_snapshot()
                with self.assertRaises(installer.InstallError):
                    self.install()
                self.assertEqual(self.tree_snapshot(), before)
                if path.is_dir():
                    path.rmdir()
                else:
                    path.unlink()

    def test_write_failure_restores_user_policy_and_preserves_independent_resources(
        self,
    ):
        write(
            self.config / "hooks/pypto-pro-op-lint/independent.py",
            self.independent_content,
        )
        write(self.project / "AGENTS.md", "# User policy\n")
        before = self.tree_snapshot()
        atomic_write = installer._atomic_write

        def fail_receipt(path, content):
            if path.name == installer.RECEIPT:
                raise OSError("simulated receipt write failure")
            return atomic_write(path, content)

        with mock.patch.object(installer, "_atomic_write", side_effect=fail_receipt):
            with self.assertRaisesRegex(OSError, "simulated receipt"):
                self.install()
        self.assertEqual(self.tree_snapshot(), before)

    def test_fresh_write_failure_removes_created_workspace_directories(self):
        self.assertFalse(self.project.exists())
        atomic_write = installer._atomic_write

        def fail_receipt(path, content):
            if path.name == installer.RECEIPT:
                raise OSError("simulated receipt write failure")
            return atomic_write(path, content)

        with mock.patch.object(installer, "_atomic_write", side_effect=fail_receipt):
            with self.assertRaisesRegex(OSError, "simulated receipt"):
                self.install()
        self.assertFalse(self.project.exists())

    def test_refresh_failure_restores_previous_receipt_manifest_and_skill(self):
        self.install()
        before = self.tree_snapshot()
        write(self.source / "examples/add.py", "Updated source resource\n")
        atomic_write = installer._atomic_write
        failed = False

        def fail_manifest_once(path, content):
            nonlocal failed
            if path.name == "cannbot-manifest.json" and not failed:
                failed = True
                raise OSError("simulated manifest write failure")
            return atomic_write(path, content)

        with mock.patch.object(
            installer, "_atomic_write", side_effect=fail_manifest_once
        ):
            with self.assertRaisesRegex(OSError, "simulated manifest"):
                self.install()
        self.assertEqual(self.tree_snapshot(), before)

    def make_symlink(self, path, target):
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.symlink_to(target, target_is_directory=True)
        except OSError as exc:
            if os.name == "nt" and hasattr(path, "is_junction"):
                # Junctions need no symlink privilege and exercise the same
                # external-directory protection on ordinary Windows accounts.
                result = make_junction(path, target)
                if result.returncode == 0:
                    return
            self.skipTest(f"directory links unavailable: {exc}")

    def test_backup_symlink_cannot_redirect_writes(self):
        outside = self.root / "outside"
        outside.mkdir()
        self.make_symlink(self.config / f".{installer.PLUGIN}-backups", outside)
        write(self.project / "AGENTS.md", "User policy\n")
        with self.assertRaisesRegex(installer.InstallError, "symlink"):
            self.install()
        self.assertEqual(list(outside.iterdir()), [])
        self.assertEqual((self.project / "AGENTS.md").read_text(), "User policy\n")
        self.assertFalse((self.config / "skills").exists())

    @unittest.skipUnless(os.name == "nt", "Windows junction fallback")
    def test_junction_is_rejected_without_pathlib_junction_detection(self):
        outside = self.root / "outside"
        outside.mkdir()
        link = self.config / "skills"
        link.parent.mkdir(parents=True)
        process = make_junction(link, outside)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.addCleanup(link.rmdir)
        with mock.patch.object(Path, "is_junction", return_value=False, create=True):
            with self.assertRaisesRegex(installer.InstallError, "symlink"):
                self.install()
        self.assertEqual(list(outside.iterdir()), [])

    def test_independent_directory_link_survives_failed_installation(self):
        outside = self.root / "outside"
        write(outside / "SKILL.md", self.independent_content)
        link = self.config / "user-tools"
        self.make_symlink(link, outside)
        atomic_write = installer._atomic_write

        def fail_receipt(path, content):
            if path.name == installer.RECEIPT:
                raise OSError("simulated receipt write failure")
            return atomic_write(path, content)

        with mock.patch.object(installer, "_atomic_write", side_effect=fail_receipt):
            with self.assertRaisesRegex(OSError, "simulated receipt"):
                self.install()
        self.assertTrue(installer.is_link(link))
        self.assertEqual(link.resolve(), outside.resolve())
        self.assertEqual((outside / "SKILL.md").read_text(), self.independent_content)
        self.assertFalse((self.config / installer.RECEIPT).exists())

    def test_invalid_receipt_paths_are_rejected(self):
        for name in (
            "",
            ".",
            "./",
            "./.",
            "../elsewhere",
            "/absolute",
            "C:/absolute",
            "a\\b",
            None,
        ):
            with self.subTest(name=name), self.assertRaises(installer.InstallError):
                installer._safe_relative(name)


if __name__ == "__main__":
    unittest.main()
