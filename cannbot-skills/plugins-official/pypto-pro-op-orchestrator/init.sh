#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

set -e

# --- Color & output helpers ---
if [ -t 1 ]; then
  GREEN='\033[0;32m'; YELLOW='\033[0;33m'; RED='\033[0;31m'
  CYAN='\033[0;36m'; BOLD='\033[1m'; DIM='\033[2m'; NC='\033[0m'
else
  GREEN=''; YELLOW=''; RED=''; CYAN=''; BOLD=''; DIM=''; NC=''
fi

ok()   { echo -e "  ${DIM}${GREEN}✓${NC}${DIM} $*${NC}"; }
warn() { echo -e "  ${YELLOW}⚠${NC}${DIM} $*${NC}"; }
err()  { echo -e "  ${RED}✗${NC}${DIM} $*${NC}"; }
info() { echo -e "  ${DIM}${CYAN}→${NC}${DIM} $*${NC}"; }

# Reject an AGENTS.md that this installer's rendering cannot make correct.
#
# $1 = substituted temp file, $2 = the source AGENTS.md.
assert_config_root_substitutable() {
    local substituted="$1" source_file="$2"
    # The substitution only rewrites the bare `$CANNBOT_CONFIG_ROOT` spelling. An
    # `os.environ['CANNBOT_CONFIG_ROOT']` read or a single-tool `$(pwd)/.opencode` default
    # passes through untouched, and on a non-opencode install both resolve to a directory
    # that does not exist -- Stage-0 bootstrap dies and pro_ops is never prepared.
    if grep -q "os\.environ\[['\"]CANNBOT_CONFIG_ROOT['\"]\]" "$substituted" \
       || grep -q '\${CANNBOT_CONFIG_ROOT:-\$(pwd)' "$substituted"; then
        rm -f "$substituted"
        err "AGENTS.md still resolves CANNBOT_CONFIG_ROOT at runtime (os.environ or a \$(pwd) default). Use the bare \$CANNBOT_CONFIG_ROOT literal so this installer can substitute it."
        exit 1
    fi
    # Checked against the source because substitution would turn this shell expansion into
    # invalid syntax instead of a concrete path.
    if grep -q '\${CANNBOT_CONFIG_ROOT:-\$CANNBOT_CONFIG_ROOT}' "$source_file"; then
        rm -f "$substituted"
        err "AGENTS.md uses a self-referential CANNBOT_CONFIG_ROOT default. Use the bare installer placeholder instead."
        exit 1
    fi
}
step() { echo -e "${DIM}$*${NC}"; }

list_entry_names() (
    local entry
    shopt -s dotglob nullglob
    for entry in "$1"/*; do
        printf '%s\n' "${entry##*/}"
    done
)

# Safe install config file with backup and conflict handling.
# $1 = generated temp file path
# $2 = target file path
# $3 = display name
# $4 = install level (global/project)
safe_install_file() {
    local tmpfile="$1"
    local target="$2"
    local name="$3"
    local level="$4"

    # Idempotency: skip if identical
    if [ -e "$target" ] && diff -q "$tmpfile" "$target" > /dev/null 2>&1; then
        info "$name already up to date"
        rm -f "$tmpfile"
        return 0
    fi

    # Backup existing file before overwriting
    if [ -e "$target" ] || [ -L "$target" ]; then
        local backup
        backup="${target}.bak.$(date +%Y%m%d_%H%M%S)"
        cp -a "$target" "$backup"
        warn "$name already exists, backed up to $(basename "$backup")"

        # Interactive prompt for global mode
        if [ "$level" = "global" ] && [ -t 0 ] && [ -t 1 ]; then
            echo ""
            echo -e "  ${BOLD}${YELLOW}⚠  $name 存在自定义内容，请选择操作：${NC}"
            echo -e "    ${BOLD}[O]${NC} 覆盖      - 用插件内容替换（原内容已备份）"
            echo -e "    ${BOLD}[M]${NC} 合并      - 插件内容置顶，保留原自定义内容"
            echo -e "    ${BOLD}[S]${NC} 跳过      - 保持现有文件不变"
            printf "  %b→%b %b请输入选择 [O/M/S]:%b " "${BOLD}${CYAN}" "${NC}" "${BOLD}" "${NC}"
            read -r choice < /dev/tty
            case "$choice" in
                [Mm]*)
                    cat "$tmpfile" > "${target}.new"
                    echo "" >> "${target}.new"
                    echo "<!-- === User custom content below === -->" >> "${target}.new"
                    echo "" >> "${target}.new"
                    cat "$target" >> "${target}.new"
                    mv "${target}.new" "$target"
                    ok "$name (merged with backup)"
                    rm -f "$tmpfile"
                    return 0
                    ;;
                [Ss]*)
                    info "$name skipped (backup preserved)"
                    rm -f "$tmpfile"
                    return 0
                    ;;
                *) ;; # default: overwrite
            esac
        fi
    fi

    # Overwrite (default for project mode or non-interactive)
    mv "$tmpfile" "$target"
    if [ "$level" = "global" ]; then
        ok "$name (absolute paths for global mode)"
    else
        ok "$name (absolute paths for project mode)"
    fi
}

# Install an owned agent prompt as a rendered copy. Agent discovery files used to be
# symlinks, so `$CANNBOT_CONFIG_ROOT` remained a runtime shell expression and could
# collapse to `/references/...` inside a subagent. Rendering here keeps prompt content
# host-neutral while init.sh owns the concrete resource mapping.
install_rendered_agent() {
    local source="$1" target="$2" tmpfile escaped_config_root
    escaped_config_root="${CONFIG_ROOT//#/\\#}"
    tmpfile=$(mktemp)
    sed "s#\$CANNBOT_CONFIG_ROOT#${escaped_config_root}#g" "$source" > "$tmpfile"
    if grep -q '\$CANNBOT_CONFIG_ROOT' "$tmpfile"; then
        rm -f "$tmpfile"
        err "Agent prompt still contains an unresolved \$CANNBOT_CONFIG_ROOT: $(basename "$source")"
        exit 1
    fi
    [ -e "$target" ] || [ -L "$target" ] && rm -rf "$target"
    mv "$tmpfile" "$target"
}

render_config_content() {
    local source="$1" output="$2"
    local plugin_root_abs escaped_root escaped_config_root
    plugin_root_abs="$PLUGIN_ROOT"
    escaped_root="${plugin_root_abs//#/\\#}"
    escaped_config_root="${CONFIG_ROOT//#/\\#}"
    sed \
      -e "s#\$CANNBOT_CONFIG_ROOT#${escaped_config_root}#g" \
      -e "s#\`workflows/#\`${escaped_root}/workflows/#g" \
      -e "s#pypto/docs/#${escaped_root}/pypto/docs/#g" \
      -e "s#pypto/examples/#${escaped_root}/pypto/examples/#g" \
      "$source" > "$output"
    assert_config_root_substitutable "$output" "$source"
}

install_rendered_config() {
    local source="$1" target="$2" display_name="$3" level="$4" tmpfile
    tmpfile=$(mktemp)
    render_config_content "$source" "$tmpfile"
    safe_install_file "$tmpfile" "$target" "$display_name" "$level"
}

# Pull the remote skills into a config skills directory ($1). The skills are
# maintained in the shared CANNBot Skills git repository and are not vendored in
# this checkout, so a sparse checkout fetches only the needed subtrees. A failed
# refresh keeps an already installed copy; a missing first-time install is fatal.
provision_remote_skills() {
    local skills_dir="$1" repo ref work name missing=""
    [ -n "$REMOTE_SKILLS" ] || return 0
    repo="$CANNBOT_REMOTE_SKILL_REPO"
    ref="$CANNBOT_REMOTE_SKILL_REF"
    if ! command -v git >/dev/null 2>&1; then
        err "未找到 git，无法拉取远程 Skill：$REMOTE_SKILLS"
        return 1
    fi
    mkdir -p "$skills_dir"
    work="$(mktemp -d)"
    if ! git clone --depth 1 --filter=blob:none --no-checkout --branch "$ref" \
            "$repo" "$work/repo" >/dev/null 2>&1; then
        for name in $REMOTE_SKILLS; do
            [ -f "$skills_dir/$name/SKILL.md" ] || missing="$missing $name"
        done
        rm -rf "$work"
        if [ -n "$missing" ]; then
            err "无法从 $repo 拉取远程 Skill：$missing"
            return 1
        fi
        warn "远程 Skill 拉取失败，保留已有副本（$repo@$ref）"
        return 0
    fi
    # Sparse checkout keeps only the requested skill subtrees; create .git/info
    # explicitly because old git does not always populate it.
    mkdir -p "$work/repo/.git/info"
    git -C "$work/repo" config core.sparseCheckout true
    git -C "$work/repo" config core.sparseCheckoutCone false
    if ! { for name in $REMOTE_SKILLS; do printf '/ops/%s/\n' "$name"; done \
             | git -C "$work/repo" sparse-checkout set --stdin >/dev/null 2>&1; } ||
       ! git -C "$work/repo" checkout --detach HEAD >/dev/null 2>&1; then
        rm -rf "$work"
        err "无法从 $repo 拉取远程 Skill：$REMOTE_SKILLS"
        return 1
    fi
    for name in $REMOTE_SKILLS; do
        if [ -f "$work/repo/ops/$name/SKILL.md" ]; then
            rm -rf "${skills_dir:?}/$name"
            cp -a "$work/repo/ops/$name" "$skills_dir/$name"
            ok "远程 Skill: $name (来自 $repo@$ref)"
        else
            rm -rf "$work"
            err "远程仓库缺少 Skill：ops/$name"
            return 1
        fi
    done
    rm -rf "$work"
}

# Detect TRAE variant and return appropriate config root.
# Detect TRAE variant by scanning global config directories.
# Sets global: TRAE_VARIANT=(ide|plugin|cli|unknown)
detect_trae_variant() {
    if [ -d "$HOME/.trae-cn" ]; then
        TRAE_VARIANT="ide"
    elif [ -d "$HOME/.marscode" ]; then
        TRAE_VARIANT="plugin"
    elif [ -d "$HOME/.traecli" ]; then
        TRAE_VARIANT="cli"
    else
        TRAE_VARIANT="unknown"
    fi
}

BRAND="cannbot"
VERSION="1.0.0"

# --- Plugin-specific filters ---
EXCLUDED_SKILL=""
# Skill whitelist (space-separated list) - references shared ops
INCLUDED_SKILLS="pypto-pro-docs-search pypto-pro-environment-check pypto-pro-golden-generate pypto-pro-intent-understand pypto-pro-material-explore pypto-pro-op-design pypto-pro-op-develop pypto-pro-op-perf-tune pypto-pro-op-plan pypto-pro-precision-debug pypto-pro-cann-delivery"
# Remote skills (space-separated list) - maintained in the shared CANNBot Skills
# repo and not vendored in this checkout, so they are pulled from git at install
# time into the config skills directory.
REMOTE_SKILLS="ops-profiling"
CANNBOT_REMOTE_SKILL_REPO="${CANNBOT_REMOTE_SKILL_REPO:-https://gitcode.com/cann/cannbot-skills.git}"
CANNBOT_REMOTE_SKILL_REF="${CANNBOT_REMOTE_SKILL_REF:-master}"
# Agent whitelist (shell pattern) - uses local agents/
INCLUDED_AGENT_PATTERN="pypto-pro-op-*"
SKILL_CATEGORY="ops"
INSTALL_OPENCODE_HOOKS=true
DISPLAY_NAME="PyPTO-Pro Operator Dev Team"
SAMPLE_PROMPT="使用 PyPTO-Pro 开发一个 softmax 算子，支持 float16 数据类型"

agent_is_included() {
    # The variable intentionally contains the plugin-specific agent glob.
    # shellcheck disable=SC2254
    case "$1" in
        $INCLUDED_AGENT_PATTERN) return 0 ;;
        *) return 1 ;;
    esac
}

show_banner() {
  echo ""
  echo -e "${CYAN}"
  cat << 'BANNER'
   ____    _    _   _ _   _ ____        _
  / ___|  / \  | \ | | \ | | __ )  ___ | |_
 | |     / _ \ |  \| |  \| |  _ \ / _ \| __|
 | |___ / ___ \| |\  | |\  | |_) | (_) | |_
  \____/_/   \_\_| \_|_| \_|____/ \___/ \__|
BANNER
  echo -e "${NC}"
  echo -e "  ${BOLD}${DISPLAY_NAME}${NC}"
  echo ""
}

show_help() {
    cat << EOF
CANNBot - ${DISPLAY_NAME} Installer

Usage: init.sh [level] [tool] [install_path]

Arguments:
  level        - Installation level: "project" (default) or "global"
  tool         - Target tool: "opencode" (default), "claude", "trae", "cursor", "copilot", or "codearts"
  install_path - Project-level installation directory (default: current working directory)

Options:
  --help  - Show this help message

Examples:
  init.sh                              # Project-level, OpenCode
  init.sh project opencode             # Project-level, OpenCode
  init.sh global claude                # Global-level, Claude Code
  init.sh project claude               # Project-level, Claude Code
  init.sh project trae                 # Project-level, Trae
  init.sh project cursor               # Project-level, Cursor
  init.sh project codearts             # Project-level, CodeArts
  init.sh project opencode /path/to/proj  # Project-level, OpenCode, custom path
  init.sh project trae /path/to/proj      # Project-level, Trae, custom path
  init.sh project cursor /path/to/proj    # Project-level, Cursor, custom path

Installation paths (CANNBot brand):
  OpenCode: .opencode/{skills,agents}/     + AGENTS.md in project root
  Claude:   .claude/{skills,agents}/ + CLAUDE.md in project root
  Trae IDE:     .trae/{skills,agents}/       + AGENTS.md in project root
  Trae Plugin:  .marscode/{skills,agents}/   + AGENTS.md in project root
  Trae CLI:     .traecli/{skills,agents}/    + AGENTS.md in project root
  Cursor:       .cursor/{skills,agents}/     + AGENTS.md in project root
  Copilot:      .github/{skills,agents}/      + AGENTS.md in project root (project)
                ~/.copilot/{skills,agents}/   + AGENTS.md (global)
  CodeArts:     .codeartsdoer/{skills,agents}/ + AGENTS.md in project root (project)
                ~/.codeartsdoer/{skills,agents}/ + AGENTS.md (global)

After installation, launch directly:
  OpenCode: opencode
  Claude:   claude
  Trae:     通过 CLI 或 IDE 启动
  Cursor:   通过 Cursor IDE 启动
  Copilot:  通过 GitHub Copilot CLI / IDE 启动
  CodeArts: 通过 CodeArts CLI / IDE 启动
EOF
}

LEVEL="project"
TOOL="opencode"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_ROOT="$SCRIPT_DIR"
# Agents: use local agents/ directory (migrated with plugin)
LOCAL_AGENT_ROOT="$PLUGIN_ROOT/agents"
# Skills: reference the matching shared domain directory
if [ -d "$PLUGIN_ROOT/../../$SKILL_CATEGORY" ]; then
    SHARED_SKILL_ROOT="$(cd "$PLUGIN_ROOT/../../$SKILL_CATEGORY" && pwd)"
else
    SHARED_SKILL_ROOT=""
fi
# Knowledge base: reference the shared pypto-pro-op-kb directory (pypto-pro knowledge layer)
# Skills reference it via ../../pypto-pro-op-kb/ relative paths, so it must be symlinked
# into the config root (e.g. .opencode/pypto-pro-op-kb) for those paths to resolve.
if [ -d "$PLUGIN_ROOT/../../ops/pypto-pro-op-kb" ]; then
    SHARED_KB_ROOT="$(cd "$PLUGIN_ROOT/../../ops/pypto-pro-op-kb" && pwd)"
else
    SHARED_KB_ROOT=""
fi

for arg in "$@"; do
    case "$arg" in
        --help)            show_help; exit 0 ;;
        global|project)    LEVEL="$arg" ;;
        opencode|claude|trae|cursor|copilot|codearts)   TOOL="$arg" ;;
    esac
done

# KB is a required part of the PyPTO-Pro Stage 1-4 contract. Failing here is
# clearer and safer than installing a workflow that cannot produce or verify
# KB_SELECTION.json later.
if [ -z "$SHARED_KB_ROOT" ]; then
    err "共享 pypto-pro-op-kb 目录未找到 ($PLUGIN_ROOT/../../ops/pypto-pro-op-kb)，无法安装 PyPTO-Pro 编排器"
    exit 1
fi

# If last argument is not a known keyword, treat it as install_path
if [ $# -gt 0 ]; then
    last_arg="${!#}"
    case "$last_arg" in
        --help|global|project|opencode|claude|trae|cursor|copilot|codearts) ;;
        *) INSTALL_PATH="$last_arg" ;;
    esac
fi

# Determine config root directory
if [ "$LEVEL" = "global" ]; then
    if [ "$TOOL" = "opencode" ]; then
        CONFIG_ROOT="$HOME/.config/opencode"
    elif [ "$TOOL" = "trae" ]; then
        detect_trae_variant
        case "$TRAE_VARIANT" in
            plugin) CONFIG_ROOT="$HOME/.marscode" ;;
            cli)    CONFIG_ROOT="$HOME/.traecli" ;;
            *)      CONFIG_ROOT="$HOME/.trae-cn" ;;
        esac
    elif [ "$TOOL" = "cursor" ]; then
        CONFIG_ROOT="$HOME/.cursor"
    elif [ "$TOOL" = "copilot" ]; then
        CONFIG_ROOT="$HOME/.copilot"
    elif [ "$TOOL" = "codearts" ]; then
        CONFIG_ROOT="$HOME/.codeartsdoer"
    else
        CONFIG_ROOT="$HOME/.claude"
    fi
else
    # Project-level: default to current directory, allow override via install_path arg
    if [ -n "$INSTALL_PATH" ]; then
        INSTALL_BASE="$(cd "$INSTALL_PATH" && pwd)"
        CONFIG_ROOT_BASE="$INSTALL_BASE"
    else
        INSTALL_BASE="$PWD"
        CONFIG_ROOT_BASE="$INSTALL_BASE"
    fi

    if [ "$TOOL" = "opencode" ]; then
        CONFIG_ROOT="$CONFIG_ROOT_BASE/.opencode"
    elif [ "$TOOL" = "trae" ]; then
        detect_trae_variant
        case "$TRAE_VARIANT" in
            plugin) CONFIG_ROOT="$CONFIG_ROOT_BASE/.marscode" ;;
            cli)    CONFIG_ROOT="$CONFIG_ROOT_BASE/.traecli" ;;
            *)      CONFIG_ROOT="$CONFIG_ROOT_BASE/.trae" ;;
        esac
    elif [ "$TOOL" = "cursor" ]; then
        CONFIG_ROOT="$CONFIG_ROOT_BASE/.cursor"
    elif [ "$TOOL" = "copilot" ]; then
        CONFIG_ROOT="$CONFIG_ROOT_BASE/.github"
    elif [ "$TOOL" = "codearts" ]; then
        CONFIG_ROOT="$CONFIG_ROOT_BASE/.codeartsdoer"
    else
        CONFIG_ROOT="$CONFIG_ROOT_BASE/.claude"
    fi
fi

CANNBOT_DIR="$CONFIG_ROOT"

# OpenCode always installs the Pro entry point plus its Scriptor mode, both on a
# fresh project and on refresh. The Python installer preserves unmanaged roots.
if [ "$TOOL" = "opencode" ]; then
    # Print the remote-skills plan before provisioning: this branch execs the
    # Python installer and never reaches the shared display block below.
    if [ -n "$REMOTE_SKILLS" ]; then
        echo ""
        echo -e "${CYAN}远程 Skills (安装时从 git 拉取)：${NC}"
        for name in $REMOTE_SKILLS; do
            echo -e "  ${GREEN}$name${NC} → 将从 ${CANNBOT_REMOTE_SKILL_REPO}@${CANNBOT_REMOTE_SKILL_REF} 拉取"
            echo -e "    ${DIM}目标路径: $CANNBOT_DIR/skills/$name${NC}"
        done
    fi
    # Fetch remote skills before the Python installer runs; it only manages the
    # declared Pro/Scriptor skills and leaves this directory in place.
    provision_remote_skills "$CANNBOT_DIR/skills" || exit 1
    shared_install_base="$CONFIG_ROOT"
    if [ "$LEVEL" = "project" ]; then shared_install_base="$INSTALL_BASE"; fi
    exec "${CANNBOT_PYTHON:-python3}" "$PLUGIN_ROOT/scripts/install_opencode.py" \
        "$LEVEL" opencode "$shared_install_base"
fi

# Keep primary-Agent selection isolated: refuse a different existing primary
# before creating or replacing any installation resource.
config_src="$PLUGIN_ROOT/AGENTS.md"
if [ "$TOOL" = "claude" ]; then
    if [ "$LEVEL" = "project" ]; then
        preflight_config_target="$INSTALL_BASE/CLAUDE.md"
    else
        preflight_config_target="$CONFIG_ROOT/CLAUDE.md"
    fi
elif [ "$LEVEL" = "project" ]; then
    preflight_config_target="$INSTALL_BASE/AGENTS.md"
else
    preflight_config_target="$CONFIG_ROOT/AGENTS.md"
fi
if { [ -e "$preflight_config_target" ] || [ -L "$preflight_config_target" ]; } &&
   [ "$config_src" != "$preflight_config_target" ]; then
    preflight_rendered=$(mktemp)
    render_config_content "$config_src" "$preflight_rendered"
    if ! diff -q "$config_src" "$preflight_config_target" >/dev/null 2>&1 &&
       ! diff -q "$preflight_rendered" "$preflight_config_target" >/dev/null 2>&1; then
        rm -f "$preflight_rendered"
        err "$(basename "$preflight_config_target") already belongs to another configuration"
        exit 1
    fi
    rm -f "$preflight_rendered"
fi

# Clean up legacy cannbot subdirectory from previous installations
if [ -e "$CONFIG_ROOT/$BRAND" ] || [ -L "$CONFIG_ROOT/$BRAND" ]; then
    rm -rf "${CONFIG_ROOT:?}/$BRAND"
fi
# OpenCode: also clean legacy teams link
if [ "$TOOL" = "opencode" ] && [ -L "$CONFIG_ROOT/teams" ]; then
    rm -f "$CONFIG_ROOT/teams"
fi

show_banner
echo "  Tool:      $TOOL"
echo "  Level:     $LEVEL"
echo "  Path:      $CONFIG_ROOT"
echo ""

if [ "$TOOL" = "trae" ] && [ "$LEVEL" = "project" ]; then
    case "$TRAE_VARIANT" in
        ide)
            info "Detected: TRAE IDE (.trae-cn / .trae)"
            ;;
        plugin)
            info "Detected: TRAE Plugin (.marscode)"
            ;;
        cli)
            info "Detected: TRAE CLI (.traecli)"
            ;;
        unknown)
            warn "TRAE variant not detected; defaulting to IDE path"
            warn "If you use TRAE Plugin, ensure ~/.marscode exists before re-running"
            warn "If you use TRAE CLI, ensure ~/.traecli exists before re-running"
            ;;
    esac
    echo ""
fi

# --- Step 0: Confirmation before installation ---
step "[0/5] Checking items to be installed..."

# Collect skills to install (from shared ops)
SKILLS_TO_INSTALL=""
SKILL_COUNT=0
for skill_dir in "$SHARED_SKILL_ROOT"/*/; do
    [ -d "$skill_dir" ] || continue
    name=$(basename "$skill_dir")
    echo "$INCLUDED_SKILLS" | grep -qw "$name" || continue
    [ -n "$EXCLUDED_SKILL" ] && [ "$name" = "$EXCLUDED_SKILL" ] && continue
    SKILLS_TO_INSTALL="$SKILLS_TO_INSTALL $name"
    SKILL_COUNT=$((SKILL_COUNT + 1))
done

# Collect agents to install (from local agents/)
AGENTS_TO_INSTALL=""
AGENT_COUNT=0
for agent_entry in "$LOCAL_AGENT_ROOT"/*; do
    [ -e "$agent_entry" ] || continue
    name=$(basename "$agent_entry")
    base="${name%.md}"
    agent_is_included "$base" || continue
    AGENTS_TO_INSTALL="$AGENTS_TO_INSTALL $name"
    AGENT_COUNT=$((AGENT_COUNT + 1))
done

# Display installation plan
echo ""
echo -e "${BOLD}以下内容将被安装/替换：${NC}"
echo ""
echo -e "${CYAN}Skills (${SKILL_COUNT} 项，来自共享 ${SKILL_CATEGORY} 目录)：${NC}"
for name in $SKILLS_TO_INSTALL; do
    target="$CANNBOT_DIR/skills/$name"
    src="$SHARED_SKILL_ROOT/$name"
    if [ -e "$target" ] || [ -L "$target" ]; then
        echo -e "  ${YELLOW}$name${NC} → 将被替换为软连接到 ${src}"
    else
        echo -e "  ${GREEN}$name${NC} → 将创建软连接到 ${src}"
    fi
    echo -e "    ${DIM}目标路径: $target${NC}"
done

if [ -n "$REMOTE_SKILLS" ]; then
    echo ""
    echo -e "${CYAN}远程 Skills (安装时从 git 拉取)：${NC}"
    for name in $REMOTE_SKILLS; do
        echo -e "  ${GREEN}$name${NC} → 将从 ${CANNBOT_REMOTE_SKILL_REPO}@${CANNBOT_REMOTE_SKILL_REF} 拉取"
        echo -e "    ${DIM}目标路径: $CANNBOT_DIR/skills/$name${NC}"
    done
fi

echo ""
echo -e "${CYAN}Agents (${AGENT_COUNT} 项，来自本地 agents/)：${NC}"
for name in $AGENTS_TO_INSTALL; do
    target="$CANNBOT_DIR/agents/$name"
    if [ -e "$target" ] || [ -L "$target" ]; then
        echo -e "  ${YELLOW}$name${NC} → 将被替换为已渲染资源路径的副本"
    else
        echo -e "  ${GREEN}$name${NC} → 将创建已渲染资源路径的副本"
    fi
    echo -e "    ${DIM}目标路径: $target${NC}"
done

echo ""
if [ -n "$SHARED_KB_ROOT" ]; then
    echo -e "${CYAN}知识库 (pypto-pro-op-kb, 来自共享 pypto-pro-op-kb 目录)：${NC}"
    target="$CANNBOT_DIR/pypto-pro-op-kb"
    if [ -e "$target" ] || [ -L "$target" ]; then
        echo -e "  ${YELLOW}pypto-pro-op-kb${NC} → 将被替换为软连接到 ${SHARED_KB_ROOT}"
    else
        echo -e "  ${GREEN}pypto-pro-op-kb${NC} → 将创建软连接到 ${SHARED_KB_ROOT}"
    fi
    echo -e "    ${DIM}目标路径: $target${NC}"
fi

echo ""
echo -e "${CYAN}配置文件：${NC}"
config_src="$PLUGIN_ROOT/AGENTS.md"
if [ "$TOOL" = "opencode" ]; then
    if [ "$LEVEL" = "project" ]; then
        config_target="$INSTALL_BASE/AGENTS.md"
    else
        config_target="$CONFIG_ROOT/AGENTS.md"
    fi
    if [ "$LEVEL" = "project" ] && [ "$PLUGIN_ROOT" = "$INSTALL_BASE" ]; then
        echo -e "  ${GREEN}AGENTS.md${NC} → 已在目标位置"
    elif [ -e "$config_target" ] || [ -L "$config_target" ]; then
        echo -e "  ${YELLOW}AGENTS.md${NC} → 将被替换为已渲染配置"
    else
        echo -e "  ${GREEN}AGENTS.md${NC} → 将创建已渲染配置"
    fi
    echo -e "    ${DIM}目标路径: $config_target${NC}"
elif [ "$TOOL" = "claude" ]; then
    if [ "$LEVEL" = "project" ]; then
        config_target="$INSTALL_BASE/CLAUDE.md"
    else
        config_target="$CONFIG_ROOT/CLAUDE.md"
    fi
    if [ -e "$config_target" ] || [ -L "$config_target" ]; then
        echo -e "  ${YELLOW}CLAUDE.md${NC} (将被替换)"
    else
        echo -e "  ${GREEN}CLAUDE.md${NC} (将创建)"
    fi
else
    if [ "$LEVEL" = "project" ]; then
        config_target="$INSTALL_BASE/AGENTS.md"
    else
        config_target="$CONFIG_ROOT/AGENTS.md"
    fi
    if [ -e "$config_target" ] || [ -L "$config_target" ]; then
        echo -e "  ${YELLOW}AGENTS.md${NC} (将被替换)"
    else
        echo -e "  ${GREEN}AGENTS.md${NC} (将创建)"
    fi
fi

echo ""
echo -e "${BOLD}${YELLOW}注意：仅替换上述白名单内的内容，不影响其他已存在的 skills/agents${NC}"
echo ""
ok "开始安装..."
echo ""

# --- Step 1: Create directory symlinks ---
step "[1/5] Setting up CANNBot directory..."
mkdir -p "$CANNBOT_DIR"

step1_summary=""
step1_warns=""
if [ "$TOOL" = "opencode" ]; then
    # OpenCode: per-item symlinks for skills (from shared ops, whitelist filtered)
    mkdir -p "$CANNBOT_DIR/skills"
    # Pre-clean existing skill symlinks (only whitelist items)
    for skill_dir in "$SHARED_SKILL_ROOT"/*/; do
        [ -d "$skill_dir" ] || continue
        name=$(basename "$skill_dir")
        # Only clean skills that are in whitelist
        echo "$INCLUDED_SKILLS" | grep -qw "$name" || continue
        target="$CANNBOT_DIR/skills/$name"
        [ -e "$target" ] || [ -L "$target" ] && rm -rf "$target"
    done
    skill_count=0
    for skill_dir in "$SHARED_SKILL_ROOT"/*/; do
        [ -d "$skill_dir" ] || continue
        name=$(basename "$skill_dir")
        # Check if skill is in whitelist (space-separated list)
        echo "$INCLUDED_SKILLS" | grep -qw "$name" || continue
        [ -n "$EXCLUDED_SKILL" ] && [ "$name" = "$EXCLUDED_SKILL" ] && continue
        ln -sfn "$skill_dir" "$CANNBOT_DIR/skills/$name"
        skill_count=$((skill_count + 1))
    done
    step1_summary="skills(${skill_count}) "

    # Render per-agent discovery files so concrete resource roots stay in the adapter layer.
    mkdir -p "$CANNBOT_DIR/agents"
    # Pre-clean existing agent symlinks (only whitelist items)
    for agent_entry in "$LOCAL_AGENT_ROOT"/*; do
        [ -e "$agent_entry" ] || continue
        name=$(basename "$agent_entry")
        base_name="${name%.md}"
        # Only clean agents that match whitelist pattern
        agent_is_included "$base_name" || continue
        target="$CANNBOT_DIR/agents/$name"
        [ -e "$target" ] || [ -L "$target" ] && rm -rf "$target"
    done
    agent_count=0
    for agent_entry in "$LOCAL_AGENT_ROOT"/*; do
        [ -e "$agent_entry" ] || continue
        name=$(basename "$agent_entry")
        base_name="${name%.md}"
        agent_is_included "$base_name" || continue
        install_rendered_agent "$agent_entry" "$CANNBOT_DIR/agents/$name"
        agent_count=$((agent_count + 1))
    done
    step1_summary="${step1_summary}agents(${agent_count})"
    # OpenCode: symlink plugin-level references/ (shared by multiple skills)
    if [ -d "$PLUGIN_ROOT/references" ]; then
        mkdir -p "$CANNBOT_DIR/references"
        for ref_entry in "$PLUGIN_ROOT/references"/*; do
            [ -e "$ref_entry" ] || continue
            ref_name=$(basename "$ref_entry")
            ref_target="$CANNBOT_DIR/references/$ref_name"
            [ -e "$ref_target" ] || [ -L "$ref_target" ] && rm -rf "$ref_target"
            ln -sfn "$ref_entry" "$ref_target"
        done
        step1_summary="${step1_summary} references"
    fi

    # Knowledge base symlink (shared pypto-pro-op-kb, if present)
    # Skills reference it via ../../pypto-pro-op-kb/ relative paths; the symlink at
    # $CANNBOT_DIR/pypto-pro-op-kb makes those paths resolve from the config root.
    if [ -n "$SHARED_KB_ROOT" ]; then
        target="$CANNBOT_DIR/pypto-pro-op-kb"
        [ -e "$target" ] || [ -L "$target" ] && rm -rf "$target"
        ln -sfn "$SHARED_KB_ROOT" "$target"
        step1_summary="${step1_summary} pypto-pro-op-kb"
    fi

    if [ "$INSTALL_OPENCODE_HOOKS" = true ]; then
        mkdir -p "$CANNBOT_DIR/plugins" "$CANNBOT_DIR/hooks"
        # Remove only files owned by this orchestrator before copying. This
        # prevents renamed/deleted hooks from surviving an upgrade while
        # preserving unrelated OpenCode plugins in the same directory.
        for managed_plugin in \
            "$CANNBOT_DIR/plugins"/pypto-pro-op-*.ts \
            "$CANNBOT_DIR/plugins/pypto-pro-state-transition.ts"; do
            [ -e "$managed_plugin" ] || continue
            rm -f -- "$managed_plugin"
        done
        rm -rf -- "${CANNBOT_DIR:?}/hooks/pypto-pro-op-lint"
        mkdir -p "$CANNBOT_DIR/hooks/pypto-pro-op-lint"
        cp -a "$PLUGIN_ROOT/hooks/opencode/." "$CANNBOT_DIR/plugins/"
        cp -a "$PLUGIN_ROOT/hooks/pypto-pro-op-lint/." "$CANNBOT_DIR/hooks/pypto-pro-op-lint/"
        step1_summary="${step1_summary} hooks"
    fi
    ok "Linked: $step1_summary"
else
    # Claude/Trae/Cursor/Copilot: create directories (per-item symlinks handled in Step 3)
    mkdir -p "$CONFIG_ROOT/skills" "$CONFIG_ROOT/agents"
    ok "Prepared: skills/, agents/, rules/"
    warn "Automated state_transition and lint hooks are OpenCode-only; this target installs prompt resources without automatic hard gates"
fi
[ -n "$step1_warns" ] && echo -e "$step1_warns"
echo ""

# --- Step 2: Install rendered config file (AGENTS.md / CLAUDE.md) ---
step "[2/5] Installing configuration..."

config_src="$PLUGIN_ROOT/AGENTS.md"

if [ "$TOOL" = "claude" ]; then
    config_name="CLAUDE.md"
else
    config_name="AGENTS.md"
fi
if [ "$LEVEL" = "project" ]; then
    config_target="$INSTALL_BASE/$config_name"
else
    config_target="$CONFIG_ROOT/$config_name"
fi

if [ "$config_src" = "$config_target" ]; then
    info "$config_name already at target location"
else
    install_rendered_config "$config_src" "$config_target" "$config_name" "$LEVEL"
fi
echo ""

# --- Step 3: Configure tool discovery ---
step "[3/5] Configuring tool discovery..."

if [ "$TOOL" = "opencode" ]; then
    # OpenCode: skills/ agents already at auto-scan paths, no extra discovery needed
    ok "Auto-scan: skills/, agents/"
else
    # Claude/Trae/Cursor/Copilot: create per-skill discovery symlinks (with filter, from shared ops)
    DISCOVERY="$CONFIG_ROOT/skills"

    # Pre-clean existing skills (only whitelist items)
    for skill_dir in "$SHARED_SKILL_ROOT"/*/; do
        [ -d "$skill_dir" ] || continue
        name=$(basename "$skill_dir")
        # Only clean skills that are in whitelist
        echo "$INCLUDED_SKILLS" | grep -qw "$name" || continue
        target="$DISCOVERY/$name"
        [ -e "$target" ] || [ -L "$target" ] && rm -rf "$target"
    done

    link_count=0
    for skill_dir in "$SHARED_SKILL_ROOT"/*/; do
        [ -d "$skill_dir" ] || continue
        name=$(basename "$skill_dir")
        # Check if skill is in whitelist (space-separated list)
        echo "$INCLUDED_SKILLS" | grep -qw "$name" || continue
        [ -n "$EXCLUDED_SKILL" ] && [ "$name" = "$EXCLUDED_SKILL" ] && continue
        target="$DISCOVERY/$name"
        ln -sfn "$skill_dir" "$target"
        link_count=$((link_count + 1))
    done

    # Clean broken symlinks.
    # `*/` only matches paths the shell can resolve to a directory, which a DANGLING
    # symlink is not -- so this loop never cleaned the thing it is named for. Observed
    # after a skill was deleted: its discovery symlink survived every reinstall. Glob
    # both forms and test with -L, which is true for a dangling link.
    for link in "$DISCOVERY"/* "$DISCOVERY"/*/; do
        link="${link%/}"
        [ -L "$link" ] || continue
        [ -e "$link" ] || rm -f "$link"
    done

    ok "Skills: $link_count discovery symlinks"

    # Remote skills maintained in the shared CANNBot Skills git repository.
    provision_remote_skills "$DISCOVERY" || exit 1

    # Render agent discovery files instead of exposing raw prompts through symlinks.
    AGENT_DISCOVERY="$CONFIG_ROOT/agents"

    # Pre-clean existing agents (only whitelist items)
    for agent_entry in "$LOCAL_AGENT_ROOT"/*; do
        [ -e "$agent_entry" ] || continue
        name=$(basename "$agent_entry")
        base="${name%.md}"
        # Only clean agents that match whitelist pattern
        agent_is_included "$base" || continue
        target="$AGENT_DISCOVERY/$name"
        [ -e "$target" ] || [ -L "$target" ] && rm -rf "$target"
    done

    agent_count=0
    for agent_entry in "$LOCAL_AGENT_ROOT"/*; do
        [ -e "$agent_entry" ] || continue
        name=$(basename "$agent_entry")
        base="${name%.md}"
        agent_is_included "$base" || continue
        target="$AGENT_DISCOVERY/$name"
        install_rendered_agent "$agent_entry" "$target"
        agent_count=$((agent_count + 1))
    done

    # Clean broken symlinks
    for link in "$AGENT_DISCOVERY"/*; do
        [ -L "$link" ] && [ ! -e "$link" ] && rm "$link"
    done

    ok "Agents: $agent_count rendered discovery files"

    # Claude/Trae/Cursor: also create references discovery symlinks (plugin-level shared refs)
    if [ -d "$PLUGIN_ROOT/references" ]; then
        REF_DISCOVERY="$CONFIG_ROOT/references"
        mkdir -p "$REF_DISCOVERY"
        for stale_ref in "$REF_DISCOVERY"/scriptor-*; do
            [ -L "$stale_ref" ] || continue
            case "$(readlink "$stale_ref")" in
                "$PLUGIN_ROOT/references/"*) rm -f "$stale_ref" ;;
            esac
        done
        ref_link_count=0
        for ref_entry in "$PLUGIN_ROOT/references"/*; do
            [ -e "$ref_entry" ] || continue
            ref_name=$(basename "$ref_entry")
            case "$ref_name" in scriptor-*) continue ;; esac
            ref_target="$REF_DISCOVERY/$ref_name"
            [ -e "$ref_target" ] || [ -L "$ref_target" ] && rm -rf "$ref_target"
            ln -sfn "$ref_entry" "$ref_target"
            ref_link_count=$((ref_link_count + 1))
        done
        # Clean broken symlinks
        for link in "$REF_DISCOVERY"/*; do
            [ -L "$link" ] && [ ! -e "$link" ] && rm "$link"
        done
        ok "References: $ref_link_count discovery symlinks"
    fi

    # Knowledge base symlink (shared pypto-pro-op-kb, if present) for non-opencode tools
    if [ -n "$SHARED_KB_ROOT" ]; then
        target="$CONFIG_ROOT/pypto-pro-op-kb"
        [ -e "$target" ] || [ -L "$target" ] && rm -rf "$target"
        ln -sfn "$SHARED_KB_ROOT" "$target"
        ok "pypto-pro-op-kb: linked to $SHARED_KB_ROOT"
    fi
fi
echo ""

# --- Step 4: Confirm resource provisioning boundary ---
step "[4/5] Checking resource provisioning..."
info "Development resources are provisioned on demand by the installed Skills."
if [ "$LEVEL" = "project" ] && [ "$TOOL" = "opencode" ] &&
   git -C "$INSTALL_BASE" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git_common_dir="$(git -C "$INSTALL_BASE" rev-parse --git-common-dir)"
    git_common_dir="$(cd "$INSTALL_BASE" && cd "$git_common_dir" && pwd -P)"
    mkdir -p "$git_common_dir/info"
    touch "$git_common_dir/info/exclude"
    grep -Fqx "/$CONFIG_ROOT/" "$git_common_dir/info/exclude" ||
        printf "/$CONFIG_ROOT/\n" >> "$git_common_dir/info/exclude"
    grep -Fqx '/AGENTS.md' "$git_common_dir/info/exclude" ||
        printf '/AGENTS.md\n' >> "$git_common_dir/info/exclude"
fi
echo ""

# --- Step 5: Health check ---
step "[5/5] Running health check..."
health_ok=true
health_errors=""

# Check directory symlinks
for sub in skills agents; do
  target="$CANNBOT_DIR/$sub"
  if [ -d "$target" ]; then
    if ! list_entry_names "$target" | grep -q .; then
      health_errors="${health_errors}\n  ${YELLOW}⚠${NC} $sub/ is empty"
    fi
  else
    health_errors="${health_errors}\n  ${RED}✗${NC} $sub/ missing"
    health_ok=false
  fi
done

# Check required knowledge base symlink.
if [ -n "$SHARED_KB_ROOT" ]; then
    kb_target="$CANNBOT_DIR/pypto-pro-op-kb"
    if [ ! -e "$kb_target" ] && [ ! -L "$kb_target" ]; then
        health_errors="${health_errors}\n  ${RED}✗${NC} pypto-pro-op-kb/ symlink missing (required knowledge base unavailable)"
        health_ok=false
    elif [ -L "$kb_target" ] && [ ! -e "$kb_target" ]; then
        health_errors="${health_errors}\n  ${RED}✗${NC} pypto-pro-op-kb/ symlink broken"
        health_ok=false
    fi
fi

# Check config file
if [ "$TOOL" = "opencode" ]; then
    if [ "$LEVEL" = "project" ]; then
        [ -f "$INSTALL_BASE/AGENTS.md" ] || { health_errors="${health_errors}\n  ${RED}✗${NC} AGENTS.md missing in project directory"; health_ok=false; }
    else
        [ -f "$CONFIG_ROOT/AGENTS.md" ] || { health_errors="${health_errors}\n  ${RED}✗${NC} AGENTS.md missing"; health_ok=false; }
    fi
elif [ "$TOOL" = "claude" ]; then
    if [ "$LEVEL" = "project" ]; then
        [ -f "$INSTALL_BASE/CLAUDE.md" ] || { health_errors="${health_errors}\n  ${RED}✗${NC} CLAUDE.md missing in project directory"; health_ok=false; }
    else
        [ -f "$CONFIG_ROOT/CLAUDE.md" ] || { health_errors="${health_errors}\n  ${RED}✗${NC} CLAUDE.md missing"; health_ok=false; }
    fi
else
    if [ "$LEVEL" = "project" ]; then
        [ -f "$INSTALL_BASE/AGENTS.md" ] || { health_errors="${health_errors}\n  ${RED}✗${NC} AGENTS.md missing in project directory"; health_ok=false; }
    else
        [ -f "$CONFIG_ROOT/AGENTS.md" ] || { health_errors="${health_errors}\n  ${RED}✗${NC} AGENTS.md missing"; health_ok=false; }
    fi
fi

# OpenCode lint/state-transition plugins are required for the automated hard gate.
if [ "$TOOL" = "opencode" ] && [ "$INSTALL_OPENCODE_HOOKS" = true ]; then
    for hook_file in \
        "$CANNBOT_DIR/plugins/pypto-pro-op-lint.ts" \
        "$CANNBOT_DIR/plugins/pypto-pro-state-transition.ts" \
        "$CANNBOT_DIR/plugins/lib/lint-output.ts" \
        "$CANNBOT_DIR/hooks/pypto-pro-op-lint/pypto_pro_op_lint.py" \
        "$CANNBOT_DIR/hooks/pypto-pro-op-lint/rules.json"; do
        if [ ! -f "$hook_file" ]; then
            health_errors="${health_errors}\n  ${RED}✗${NC} required lint component missing: $hook_file"
            health_ok=false
        fi
    done
fi

# Generate brand manifest
MANIFEST="$CONFIG_ROOT/cannbot-manifest.json"

SKILLS_JSON="[]"
if [ -d "$CANNBOT_DIR/skills" ]; then
  SKILLS_JSON=$(list_entry_names "$CANNBOT_DIR/skills" |
    LC_ALL=C sort |
    python3 -c "import sys,json; print(json.dumps([l.strip() for l in sys.stdin if l.strip()]))" 2>/dev/null || echo "[]")
fi

AGENTS_JSON="[]"
if [ -d "$CANNBOT_DIR/agents" ]; then
  AGENTS_JSON=$(list_entry_names "$CANNBOT_DIR/agents" |
    LC_ALL=C sort |
    python3 -c "import sys,json; print(json.dumps([l.strip() for l in sys.stdin if l.strip()]))" 2>/dev/null || echo "[]")
fi

KB_INSTALLED=false
if [ -n "$SHARED_KB_ROOT" ] && { [ -e "$CANNBOT_DIR/pypto-pro-op-kb" ] || [ -L "$CANNBOT_DIR/pypto-pro-op-kb" ]; }; then
    KB_INSTALLED=true
fi

cat > "$MANIFEST" << MANIFEST_EOF
{
  "brand": "CANNBot",
  "version": "$VERSION",
  "team": "$(basename "$SCRIPT_DIR")",
  "level": "$LEVEL",
  "tool": "$TOOL",
  "installed_skills": $SKILLS_JSON,
  "installed_agents": $AGENTS_JSON,
  "kb_installed": $KB_INSTALLED,
  "brand_dir": "$CONFIG_ROOT",
  "install_time": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
MANIFEST_EOF

[ -f "$MANIFEST" ] || { health_errors="${health_errors}\n  ${RED}✗${NC} Manifest generation failed"; health_ok=false; }

if [ "$health_ok" = true ] && [ -z "$health_errors" ]; then
  ok "All checks passed"
else
  echo -e "$health_errors"
  if [ "$health_ok" = true ]; then
    warn "Some warnings, see above"
  else
    err "Some checks failed, see above"
    exit 1
  fi
fi

# --- Summary & Quick Start ---
echo ""
echo -e "  ${GREEN}${BOLD}✓ CANNBot installed successfully!${NC}"
echo ""
echo -e "  ${BOLD}Quick Start:${NC}"
if [ "$TOOL" = "opencode" ]; then
  echo -e "  ${CYAN}1.${NC} 启动 CLI: ${GREEN}opencode${NC}"
  echo -e "  ${CYAN}2.${NC} 告诉 CANNBot: ${GREEN}${BOLD}${SAMPLE_PROMPT}${NC}"
elif [ "$TOOL" = "trae" ]; then
  echo -e "  ${CYAN}1.${NC} 通过 CLI/IDE 启动${NC}"
  echo -e "  ${CYAN}2.${NC} 告诉 CANNBot: ${GREEN}${BOLD}${SAMPLE_PROMPT}${NC}"
elif [ "$TOOL" = "cursor" ]; then
  echo -e "  ${CYAN}1.${NC} 通过 Cursor IDE 启动${NC}"
  echo -e "  ${CYAN}2.${NC} 告诉 CANNBot: ${GREEN}${BOLD}${SAMPLE_PROMPT}${NC}"
elif [ "$TOOL" = "copilot" ]; then
  echo -e "  ${CYAN}1.${NC} 通过 GitHub Copilot CLI / IDE 启动${NC}"
  echo -e "  ${CYAN}2.${NC} 告诉 CANNBot: ${GREEN}${BOLD}${SAMPLE_PROMPT}${NC}"
elif [ "$TOOL" = "codearts" ]; then
  echo -e "  ${CYAN}1.${NC} 通过 CodeArts CLI / IDE 启动${NC}"
  echo -e "  ${CYAN}2.${NC} 告诉 CANNBot: ${GREEN}${BOLD}${SAMPLE_PROMPT}${NC}"
else
  echo -e "  ${CYAN}1.${NC} 启动 CLI: ${GREEN}claude${NC}"
  echo -e "  ${CYAN}2.${NC} 告诉 CANNBot: ${GREEN}${BOLD}${SAMPLE_PROMPT}${NC}"
fi
echo ""
