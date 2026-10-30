#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Install the PyPTO-Pro skills and their entry point for the selected tool.

Uses the existing Python environment; does not install PyPTO-Pro/CANN or run
network or device commands.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
import uuid


def filesystem_path(path):
    """Use extended Windows paths for deep bundled skill resources."""
    path = Path(path).absolute()
    value = str(path)
    if os.name == "nt" and not value.startswith("\\\\?\\"):
        value = (
            "\\\\?\\UNC\\" + value[2:]
            if value.startswith("\\\\")
            else "\\\\?\\" + value
        )
        return Path(value)
    return path


def display_path(path):
    value = str(path)
    if value.startswith("\\\\?\\UNC\\"):
        value = "\\\\" + value[8:]
    elif value.startswith("\\\\?\\"):
        value = value[4:]
    return value.replace("\\", "/")


def is_link(path):
    # Windows directory junctions are reparse points too, and can redirect writes
    # even when the account cannot create ordinary symbolic links.
    if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
        return True
    # Path.is_junction was added in Python 3.12; init.sh also supports 3.10/3.11.
    if os.name == "nt":
        try:
            return path.lstat().st_reparse_tag == stat.IO_REPARSE_TAG_MOUNT_POINT
        except (FileNotFoundError, NotADirectoryError):
            pass
    return False


ORCHESTRATOR = filesystem_path(Path(__file__).resolve().parents[1])
PLUGIN = "pypto-pro-op-orchestrator"
SKILLS = (
    "pypto-pro-kernel",
    "pypto-benchmark-fracture-aggregator",
    "pypto-fracture-point-detector",
    "pypto-session-dfx-report",
)
AGENTS = tuple(
    f"pypto-pro-op-{role}"
    for role in ("planner", "coder", "debugger", "verifier", "optimizer", "tool")
)
RECEIPT = f"{PLUGIN}-install.json"
BEGIN = f"<!-- cannbot:{PLUGIN}:begin -->"
END = f"<!-- cannbot:{PLUGIN}:end -->"
OLD_MARKER = "<!-- cannbot:pypto-pro-router:1 -->"
IGNORED = {"__pycache__", ".pytest_cache", "node_modules", ".DS_Store"}
TOOLS = ("opencode", "claude", "trae", "cursor", "copilot", "codearts")


class InstallError(Exception):
    """An installation conflict or validation failure."""


def digest(content):
    return hashlib.sha256(content).hexdigest()


def read_json(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise InstallError(f"expected a JSON object: {path}")
    return value


def _regular_file_or_missing(path, description):
    if is_link(path) or (path.exists() and not path.is_file()):
        raise InstallError(f"{description} must be a regular file: {path}")


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def read_jsonc(path):
    """Read OpenCode's JSONC without treating URL/comment text in strings as syntax."""
    source = path.read_bytes().decode("utf-8-sig")
    output = []
    index = 0
    quoted = False
    escaped = False
    while index < len(source):
        char = source[index]
        if quoted:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
            output.append(char)
        elif source.startswith("//", index):
            end = source.find("\n", index)
            index = len(source) if end < 0 else end
            output.append("\n")
            continue
        elif source.startswith("/*", index):
            end = source.find("*/", index + 2)
            if end < 0:
                raise InstallError(f"unterminated JSONC comment: {path}")
            index = end + 2
            output.append(" ")
            continue
        else:
            output.append(char)
        index += 1
    text = "".join(output)
    output = []
    quoted = escaped = False
    for index, char in enumerate(text):
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char == "," and text[index + 1 :].lstrip().startswith(("}", "]")):
            continue
        output.append(char)
    value = json.loads("".join(output))
    if not isinstance(value, dict):
        raise InstallError(f"expected a JSONC object: {path}")
    return value


def render(content, config):
    return content.replace("$CANNBOT_CONFIG_ROOT", display_path(config))


def config_directory(project, tool, global_level=False):
    home = Path.home()
    if tool == "trae":
        variant = next(
            (
                name
                for name in (".trae-cn", ".marscode", ".traecli")
                if (home / name).is_dir()
            ),
            ".trae-cn",
        )
        name = ".trae" if not global_level and variant == ".trae-cn" else variant
    else:
        name = {
            "opencode": ".opencode",
            "claude": ".claude",
            "cursor": ".cursor",
            "copilot": ".github",
            "codearts": ".codeartsdoer",
        }[tool]
    if global_level:
        if tool == "opencode":
            return (
                Path(os.environ.get("XDG_CONFIG_HOME", str(home / ".config")))
                / "opencode"
            )
        if tool == "copilot":
            return home / ".copilot"
        return home / name
    return project / name


def _safe_relative(name):
    if not isinstance(name, str) or not name or name == ".":
        raise InstallError(f"unsafe installation receipt path: {name!r}")
    path = Path(name)
    if (
        not path.parts
        or path.anchor
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in name
        or ":" in name
    ):
        raise InstallError(f"unsafe installation receipt path: {name}")
    return path


def _no_link_parents(path, root):
    """Never write or retire a resource through an arbitrary directory symlink."""
    for parent in (path, *path.parents):
        if parent == root:
            return
        if is_link(parent):
            raise InstallError(f"installation path is a symlink: {parent}")
        if parent.exists() and not parent.is_dir():
            raise InstallError(f"installation parent must be a directory: {parent}")
    raise InstallError(f"installation path is outside config root: {path}")


def _resources(source, prefix, config=None):
    if not source.is_dir():
        raise InstallError(f"required installation resource is missing: {source}")
    result = {}
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if any(part in IGNORED for part in relative.parts) or not path.is_file():
            continue
        content = path.read_bytes()
        if config is not None and path.suffix == ".md":
            content = render(content.decode("utf-8"), config).encode("utf-8")
        result[(Path(prefix) / relative).as_posix()] = content
    return result


def _load_previous(config):
    path = config / RECEIPT
    _regular_file_or_missing(path, "installation receipt")
    if not path.is_file():
        return {}
    previous = read_json(path)
    if (
        previous.get("plugin") != PLUGIN
        or previous.get("schema") != "cannbot.kernel-install/1"
    ):
        raise InstallError(f"unrecognized installation receipt: {path}")
    owned = previous.get("owned_files", {})
    if not isinstance(owned, dict) or any(
        not isinstance(value, str) or len(value) != 64 for value in owned.values()
    ):
        raise InstallError(f"invalid owned-file fingerprints: {path}")
    for name in owned:
        _safe_relative(name)
    return previous


def _reject_legacy_installation(config, root_target, previous, tool):
    """Refuse old active entry points; do not migrate or claim shared resources."""

    def reject(path):
        raise InstallError(
            f"legacy PyPTO-Pro installation detected: {path}; "
            "initialize in a new working directory (or a separate global config directory)"
        )

    old_receipt = config / "scriptor-install.json"
    _regular_file_or_missing(old_receipt, "legacy installation receipt")
    if old_receipt.is_file():
        value = read_json(old_receipt)
        workflows = value.get("workflows")
        if (
            value.get("schema") == "cannbot.scriptor-install/1"
            and isinstance(workflows, list)
            and PLUGIN in workflows
        ):
            reject(old_receipt)
    brand = config / "cannbot-manifest.json"
    _regular_file_or_missing(brand, "brand manifest")
    if brand.is_file():
        value = read_json(brand)
        if (
            value.get("team") == PLUGIN
            and value.get("schema") != "cannbot.kernel-install/1"
        ):
            reject(brand)
    for path in (root_target, config / "agents" / f"{PLUGIN}.md"):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        old_primary = (
            text.lstrip().startswith("---")
            and f"name: {PLUGIN}" in text.split("---", 2)[1].splitlines()
        )
        if OLD_MARKER in text or (
            old_primary
            and any(
                name in text for name in ("pypto-pro-docs-search", "pypto-pro-scriptor")
            )
        ):
            reject(path)
    if tool != "opencode":
        return
    for name in (
        "pypto-pro-op-lint.ts",
        "pypto-pro-state-transition.ts",
        "pypto-pro-scriptor.ts",
    ):
        relative = f"plugins/{name}"
        if name == "pypto-pro-op-lint.ts" and relative in previous.get(
            "owned_files", {}
        ):
            continue  # Receipt retirement below verifies its hash before backup.
        path = config / relative
        if path.exists() or is_link(path):
            reject(path)


def _managed_prompt(content):
    return f"{BEGIN}\n{content.rstrip()}\n{END}"


def _root_prompt(target, content, previous):
    if not target.exists() and not is_link(target):
        return f"{BEGIN}\n{content.rstrip()}\n{END}\n".encode("utf-8")
    if is_link(target) or not target.is_file():
        raise InstallError(f"root instructions must be a regular file: {target}")
    existing = target.read_bytes().decode("utf-8")
    block = _managed_prompt(content)
    if BEGIN in existing or END in existing:
        if (
            existing.count(BEGIN) != 1
            or existing.count(END) != 1
            or existing.index(BEGIN) > existing.index(END)
        ):
            raise InstallError(f"ambiguous managed instruction block: {target}")
        managed = existing[existing.index(BEGIN) : existing.index(END) + len(END)]
        expected = previous.get("managed_prompt_sha256")
        if expected is None or digest(managed.encode("utf-8")) != expected:
            raise InstallError(
                "managed root instructions were modified; save edits outside the managed block, "
                f"restore or remove the block, then retry: {target}"
            )
        return (
            existing[: existing.index(BEGIN)]
            + block
            + existing[existing.index(END) + len(END) :]
        ).encode("utf-8")
    return (existing.rstrip() + "\n\n" + block + "\n").encode("utf-8")


def _jsonc_tokens(source):
    """Return significant token spans, retaining all original text for editing."""
    tokens = []
    index = 0
    while index < len(source):
        char = source[index]
        if char.isspace() or (index == 0 and char == "\ufeff"):
            index += 1
            continue
        if source.startswith("//", index):
            end = source.find("\n", index)
            index = len(source) if end < 0 else end
            continue
        if source.startswith("/*", index):
            end = source.find("*/", index + 2)
            if end < 0:
                raise InstallError("unterminated JSONC comment")
            index = end + 2
            continue
        start = index
        if char == '"':
            index += 1
            escaped = False
            while index < len(source):
                current = source[index]
                index += 1
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == '"':
                    break
        elif char in "{}[]:,":
            index += 1
        else:
            while (
                index < len(source)
                and not source[index].isspace()
                and source[index] not in "{}[]:,/"
            ):
                index += 1
            if index == start:
                raise InstallError(f"unexpected JSONC syntax at character {index}")
        tokens.append((source[start:index], start, index))
    return tokens


def _select_primary_in_jsonc(source):
    """Edit only the top-level default_agent; preserve comments and formatting."""
    tokens = _jsonc_tokens(source)
    members = []
    index = 1  # read_jsonc has already validated the root object.
    while tokens[index][0] != "}":
        key, key_start, _ = tokens[index]
        index += 2  # key and colon
        value_start = tokens[index][1]
        depth = 0
        while True:
            token, _, value_end = tokens[index]
            if token in {"{", "["}:
                depth += 1
            elif token in {"}", "]"}:
                depth -= 1
            index += 1
            if depth == 0:
                break
        members.append((json.loads(key), key_start, value_start, value_end))
        if tokens[index][0] == ",":
            index += 1
    defaults = [member for member in members if member[0] == "default_agent"]
    if len(defaults) > 1:
        raise InstallError(
            "OpenCode configuration contains duplicate default_agent keys"
        )
    encoded = json.dumps(PLUGIN)
    if defaults:
        _, _, start, end = defaults[0]
        return source[:start] + encoded + source[end:]
    close = tokens[index][1]
    closing_line = source.rfind("\n", 0, close) + 1
    closing_indent = source[closing_line:close]
    trailing_comma = tokens[index - 1][0] == ","
    if not closing_indent.strip():
        newline = "\r\n" if "\r\n" in source else "\n"
        indent = closing_indent + "  "
        if members:
            key_start = members[0][1]
            key_line = source.rfind("\n", 0, key_start) + 1
            candidate = source[key_line:key_start]
            if not candidate.strip():
                indent = candidate
        position = closing_line
        addition = (
            indent
            + '"default_agent": '
            + encoded
            + ("," if trailing_comma else "")
            + newline
        )
    else:
        position = close
        addition = (
            (" " if members else "")
            + '"default_agent": '
            + encoded
            + ("," if trailing_comma else "")
        )
    edits = [(position, addition)]
    if members and not trailing_comma:
        edits.append((members[-1][3], ","))
    # At equal offsets insert the member before its separating comma.
    for position, addition in sorted(edits, key=lambda item: item[0], reverse=True):
        source = source[:position] + addition + source[position:]
    return source


def _opencode_configuration(config):
    paths = [config / "opencode.json", config / "opencode.jsonc"]
    existing = [path for path in paths if path.exists() or is_link(path)]
    if len(existing) > 1:
        raise InstallError(
            f"both opencode.json and opencode.jsonc exist in {config}; select one configuration"
        )
    path = existing[0] if existing else paths[0]
    if is_link(path):
        raise InstallError(f"OpenCode configuration is a symlink: {path}")
    if path.exists():
        value = read_jsonc(path)
    else:
        value = {"$schema": "https://opencode.ai/config.json"}
    previous_default = value.get("default_agent")
    if previous_default is not None and not isinstance(previous_default, str):
        raise InstallError(f"OpenCode default_agent must be a string or null: {path}")
    if path.exists():
        source = path.read_bytes().decode("utf-8")
        if previous_default is not None:
            return path.name, source.encode("utf-8")
        return path.name, _select_primary_in_jsonc(source).encode("utf-8")
    value["default_agent"] = PLUGIN
    return path.name, json_bytes(value)


def _atomic_write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".cannbot-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class _InstallTransaction:
    """Undo completed writes and moves if an installation operation fails."""

    def __init__(self):
        self.undo = []
        self.directories = []

    def mkdir(self, path):
        missing = []
        while not path.exists():
            missing.append(path)
            path = path.parent
        for directory in reversed(missing):
            directory.mkdir()
            self.directories.append(directory)

    def write(self, path, content):
        self.mkdir(path.parent)
        original = path.read_bytes() if path.is_file() else None
        mode = stat.S_IMODE(path.stat().st_mode) if original is not None else None
        self.undo.append(("write", path, original, mode))
        _atomic_write(path, content)

    def move(self, path, destination):
        self.mkdir(destination.parent)
        # Move links themselves, including Windows junctions. Neither the
        # transaction nor its rollback traverses their external targets.
        if is_link(path):
            os.replace(path, destination)
        else:
            shutil.move(str(path), str(destination))
        self.undo.append(("move", path, destination))

    def copymode(self, source, target):
        previous = stat.S_IMODE(target.stat().st_mode)
        if previous != stat.S_IMODE(source.stat().st_mode):
            self.undo.append(("mode", target, previous))
            shutil.copymode(source, target)

    def rollback(self):
        errors = []
        for action in reversed(self.undo):
            try:
                kind, path, *values = action
                if kind == "write":
                    content, mode = values
                    if content is None:
                        if path.exists() or is_link(path):
                            path.unlink()
                    else:
                        _atomic_write(path, content)
                        path.chmod(mode)
                elif kind == "move":
                    destination = values[0]
                    if path.exists() or is_link(path):
                        path.unlink()
                    if is_link(destination):
                        os.replace(destination, path)
                    else:
                        shutil.move(str(destination), str(path))
                else:
                    path.chmod(values[0])
            except OSError as exc:
                errors.append(str(exc))
        for directory in reversed(self.directories):
            try:
                directory.rmdir()
            except OSError as exc:
                errors.append(str(exc))
        return errors


def install(project, *, tool="opencode", global_level=False):
    project = filesystem_path(Path(project).absolute())
    if global_level and is_link(project):
        raise InstallError(f"config root must be a regular directory: {project}")
    project = filesystem_path(project.resolve())
    config = project if global_level else config_directory(project, tool)
    if is_link(config):
        raise InstallError(f"config root must be a regular directory: {config}")
    if config.exists() and not config.is_dir():
        raise InstallError(f"config root must be a regular directory: {config}")
    previous = _load_previous(config)
    root_name = "CLAUDE.md" if tool == "claude" else "AGENTS.md"
    root_target = (config if global_level else project) / root_name
    _reject_legacy_installation(config, root_target, previous, tool)
    retire = set()
    sources = {}
    desired = {}
    for skill in SKILLS:
        source = ORCHESTRATOR.parents[1] / "ops" / skill
        if not (source / "SKILL.md").is_file():
            raise InstallError(f"missing required Skill: {source / 'SKILL.md'}")
        sources[skill] = source
        desired.update(_resources(source, f"skills/{skill}"))
    license_path = ORCHESTRATOR.parents[2] / "LICENSE"
    if not license_path.is_file():
        raise InstallError(f"missing repository license: {license_path}")
    for skill in SKILLS:
        desired[f"skills/{skill}/LICENSE"] = license_path.read_bytes()
    primary = render((ORCHESTRATOR / "AGENTS.md").read_text(encoding="utf-8"), config)
    desired[f"agents/{PLUGIN}.md"] = primary.encode("utf-8")
    for name in AGENTS:
        source_agent = ORCHESTRATOR / "agents" / f"{name}.md"
        if not source_agent.is_file():
            raise InstallError(f"missing required subagent: {source_agent}")
        relative = f"agents/{name}.md"
        desired[relative] = render(
            source_agent.read_text(encoding="utf-8"), config
        ).encode("utf-8")
    if (ORCHESTRATOR / "references").is_dir():
        desired.update(_resources(ORCHESTRATOR / "references", "references", config))
    if tool == "opencode":
        name, content = _opencode_configuration(config)
        desired[name] = content
    # Initializing from this source directory must not rewrite tracked AGENTS.md.
    source_is_target = filesystem_path(root_target.resolve()) == filesystem_path(
        (ORCHESTRATOR / "AGENTS.md").resolve()
    )
    root_content = (
        None if source_is_target else _root_prompt(root_target, primary, previous)
    )
    # Preflight every conflict before creating directories or backing up files.
    for name, content in desired.items():
        target = config / _safe_relative(name)
        _no_link_parents(target.parent, config)
        if not target.exists() and not is_link(target):
            continue
        if is_link(target) or not target.is_file():
            raise InstallError(
                f"installation resource conflicts with a non-regular file: {target}"
            )
        if (
            name in {"opencode.json", "opencode.jsonc"}
            or target.read_bytes() == content
        ):
            continue
        if digest(target.read_bytes()) == previous.get("owned_files", {}).get(name):
            continue
        raise InstallError(
            f"unmanaged or modified installation resource; preserve/move it before retrying: {target}"
        )
    for name, fingerprint in previous.get("owned_files", {}).items():
        if name not in desired:
            target = config / name
            _no_link_parents(target.parent, config)
            if not target.exists() and not is_link(target):
                continue
            if (
                is_link(target)
                or not target.is_file()
                or digest(target.read_bytes()) != fingerprint
            ):
                raise InstallError(
                    f"unmanaged or modified retired installation resource; preserve/move it before retrying: {target}"
                )
            retire.add(name)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = config / f".{PLUGIN}-backups" / f"{timestamp}-{uuid.uuid4().hex[:8]}"
    _no_link_parents(backup, config)
    backed_up = []
    transaction = _InstallTransaction()

    def preserve(path, relative):
        if not path.exists() and not is_link(path):
            return
        destination = backup / relative
        _no_link_parents(destination.parent, config)
        transaction.move(path, destination)
        backed_up.append(relative)

    try:
        for name in sorted(retire):
            preserve(config / name, name)
        for name, content in desired.items():
            target = config / name
            mode_source = None
            if name.startswith("skills/"):
                skill, _, relative = name.removeprefix("skills/").partition("/")
                if skill in sources and relative:
                    mode_source = (
                        license_path
                        if relative == "LICENSE"
                        else sources[skill] / relative
                    )
            if (
                not is_link(target)
                and target.is_file()
                and target.read_bytes() == content
            ):
                if mode_source is not None:
                    transaction.copymode(mode_source, target)
                continue
            preserve(target, name)
            transaction.write(target, content)
            if mode_source is not None:
                transaction.copymode(mode_source, target)
        if root_content is not None and (
            not root_target.is_file() or root_target.read_bytes() != root_content
        ):
            preserve(root_target, f"root/{root_name}")
            transaction.write(root_target, root_content)
        receipt = {
            "schema": "cannbot.kernel-install/1",
            "plugin": PLUGIN,
            "version": "3.0.0",
            "level": "global" if global_level else "project",
            "tool": tool,
            "config_root": display_path(config),
            "installed_skills": list(SKILLS),
            "installed_agents": [PLUGIN, *AGENTS],
            "install_time": timestamp,
            "managed_prompt_sha256": digest(_managed_prompt(primary).encode("utf-8"))
            if root_content is not None
            else None,
            "owned_files": {
                name: digest(content) for name, content in sorted(desired.items())
            },
        }
        transaction.write(config / RECEIPT, json_bytes(receipt))
        # Leave another plugin's brand manifest alone; our named receipt exists.
        manifest = config / "cannbot-manifest.json"
        if not manifest.exists() or read_json(manifest).get("team") == PLUGIN:
            transaction.write(
                manifest, json_bytes({**receipt, "brand": "CANNBot", "team": PLUGIN})
            )
        for name, content in desired.items():
            if digest((config / name).read_bytes()) != digest(content):
                raise InstallError(
                    f"installed resource verification failed: {config / name}"
                )
    except (OSError, InstallError, ValueError) as exc:
        errors = transaction.rollback()
        if errors:
            raise InstallError(
                f"{exc}; rollback incomplete; recover originals from {display_path(backup)}: {'; '.join(errors)}"
            ) from exc
        raise
    # Only prune ancestors of retired files. Never enumerate a user-added child
    # tree, follow a junction, or remove an unrelated empty user directory.
    prune = set()
    for name in retire:
        for parent in (config / name).parents:
            if parent == config:
                break
            prune.add(parent)
    for path in sorted(prune, key=lambda item: len(item.parts), reverse=True):
        if path.is_dir() and not is_link(path):
            _no_link_parents(path.parent, config)
            try:
                path.rmdir()
            except OSError:
                pass
    return {
        "config_root": display_path(config),
        "skills": list(SKILLS),
        "primary_agent": PLUGIN,
        "subagents": list(AGENTS),
        "backup": display_path(backup) if backed_up else None,
        "runtime": "user-supplied PyPTO-Pro/CANN environment",
        "next": f"opencode --agent {PLUGIN}"
        if tool == "opencode"
        else f"Start {tool} in the initialized workspace",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "level", nargs="?", choices=("project", "global"), default="project"
    )
    parser.add_argument("tool", nargs="?", choices=TOOLS, default="opencode")
    parser.add_argument(
        "install_path",
        nargs="?",
        type=Path,
        help="project directory; in global mode an explicit config directory",
    )
    args = parser.parse_args()
    if sys.version_info < (3, 10):
        parser.exit(1, "Python 3.10+ is required\n")
    project = args.install_path or (
        config_directory(Path.cwd(), args.tool, True)
        if args.level == "global"
        else Path.cwd()
    )
    try:
        result = install(project, tool=args.tool, global_level=args.level == "global")
    except (InstallError, OSError, ValueError) as exc:
        parser.exit(1, f"installation failed: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
