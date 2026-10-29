#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# pypto_gym run 包安装入口(makeself 解压后执行).
# 入参契约(补丁版 makeself): install.sh --<run包名> --"<执行时cwd>" <用户参数>
# 安装语义: 落位 opp/vendors/<vendor>/ 并维护 config.ini 的 load_priority.

########################################################################################################################
# 包内路径(makeself 解压根)
########################################################################################################################
tmp_cfd="$(dirname "$(readlink -f "$0")")"
readonly TMP_PKG_SCRIPTS_PATH="$tmp_cfd"
readonly TMP_PKG_INFO_PATH="${TMP_PKG_SCRIPTS_PATH}/.."          # share/info/pypto_gym
readonly TMP_PKG_ROOT_PATH="${TMP_PKG_SCRIPTS_PATH}/../../../.."   # makeself 解压根(含 packages/ 与 share/)
readonly TMP_PKG_VERSION_FILE="${TMP_PKG_INFO_PATH}/version.info"
unset tmp_cfd

readonly VENDOR_NAME="pypto_gym"

########################################################################################################################
# 日志
########################################################################################################################
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

exit_log() {
    log "[ERROR] $2"
    exit "$1"
}

########################################################################################################################
# 参数解析
########################################################################################################################
ARG_RUN_FILE_NAME="${1#--}"
ARG_CWD="${2#--}"
shift 2

ARG_MODE="install"
ARG_INSTALL_PATH=""
ARG_QUIET=n
ARG_INSTALL_FOR_ALL=n

while [ $# -gt 0 ]; do
    case "$1" in
    --version)
        [ -f "${TMP_PKG_VERSION_FILE}" ] && grep '^Version=' "${TMP_PKG_VERSION_FILE}" | cut -d= -f2 || echo "none"
        exit 0
        ;;
    --help | -h)
        cat "${TMP_PKG_SCRIPTS_PATH}/help.info"
        exit 0
        ;;
    --quiet)
        ARG_QUIET=y
        shift
        ;;
    --install | --full | --run)
        ARG_MODE="install"
        shift
        ;;
    --uninstall)
        ARG_MODE="uninstall"
        shift
        ;;
    --install-path=*)
        install_path="${1#--install-path=}"
        [ -z "${install_path}" ] && exit_log 1 "invalid --install-path (empty)"
        install_path="${install_path%/}"
        ppath="$(dirname "${install_path}")"
        [ -d "${ppath}" ] || exit_log 1 "parent path of --install-path doesn't exist: ${ppath}, create it first"
        ARG_INSTALL_PATH="${install_path}"
        shift
        ;;
    --install-for-all)
        ARG_INSTALL_FOR_ALL=y
        shift
        ;;
    --*)
        exit_log 1 "unsupported parameter: $1 (try --help)"
        ;;
    *)
        break
        ;;
    esac
done

########################################################################################################################
# 安装目标解析: --install-path > ASCEND_HOME_PATH/opp > /usr/local/Ascend/opp
########################################################################################################################
if [ -n "${ARG_INSTALL_PATH}" ]; then
    targetdir="${ARG_INSTALL_PATH}"
elif [ -n "${ASCEND_HOME_PATH}" ]; then
    targetdir="${ASCEND_HOME_PATH%/}/opp"
else
    targetdir="/usr/local/Ascend/opp"
fi
vendordir="vendors/${VENDOR_NAME}"
config_file="${targetdir}/vendors/config.ini"

[ -d "${targetdir}" ] || exit_log 1 "install target dir not exist: ${targetdir} (source CANN set_env.sh or use --install-path=)"
[ -w "${targetdir}" ] || exit_log 1 "install target dir not writable: ${targetdir} (current user: $(whoami))"

########################################################################################################################
# config.ini load_priority 维护
########################################################################################################################
# 按逗号切分重组 load_priority 值(剔除空段), 避免 sed 手术残留 ",," / ",xxx" 类脏值
rebuild_load_priority() {
    local input="$1"
    local rebuilt="" item old_ifs="$IFS"
    IFS=','
    for item in ${input}; do
        [ -z "${item}" ] && continue
        [ "${item}" = "${VENDOR_NAME}" ] && continue
        rebuilt="${rebuilt:+${rebuilt},}${item}"
    done
    IFS="${old_ifs}"
    echo "${rebuilt}"
}

update_load_priority() {
    if [ ! -f "${config_file}" ]; then
        mkdir -p "${targetdir}/vendors" || exit_log 1 "mkdir ${targetdir}/vendors failed"
        echo "load_priority=${VENDOR_NAME}" > "${config_file}" || exit_log 1 "write load_priority to ${config_file} failed"
        if [ "${ARG_INSTALL_FOR_ALL}" = "y" ]; then
            chmod 755 "${config_file}"
        else
            chmod 640 "${config_file}"
        fi
        return 0
    fi
    local existing rebuilt
    existing="$(grep -w "load_priority" "${config_file}" | cut --only-delimited -d"=" -f2-)"
    rebuilt="$(rebuild_load_priority "${existing}")"
    if [ -n "${rebuilt}" ]; then
        rebuilt="${VENDOR_NAME},${rebuilt}"
    else
        rebuilt="${VENDOR_NAME}"
    fi
    sed -i "s@^load_priority=.*@load_priority=${rebuilt}@" "${config_file}" \
        || exit_log 1 "sed update load_priority in ${config_file} failed, please check write permission (current user: $(whoami))"
}

remove_load_priority() {
    [ -f "${config_file}" ] || return 0
    local existing rebuilt
    existing="$(grep -w "load_priority" "${config_file}" | cut --only-delimited -d"=" -f2-)"
    rebuilt="$(rebuild_load_priority "${existing}")"
    if [ -z "${rebuilt}" ]; then
        rm -f "${config_file}"
    else
        sed -i "s@^load_priority=.*@load_priority=${rebuilt}@" "${config_file}" \
            || exit_log 1 "sed remove vendor from load_priority failed"
    fi
}

########################################################################################################################
# 安装 / 卸载
########################################################################################################################
do_install() {
    [ -d "${TMP_PKG_ROOT_PATH}/packages/vendors/${VENDOR_NAME}" ] \
        || exit_log 1 "vendor payload missing in package: ${TMP_PKG_ROOT_PATH}/packages/vendors/${VENDOR_NAME}"

    if [ -d "${targetdir}/${vendordir}" ]; then
        log "[INFO] ${targetdir}/${vendordir} already exists, old version will be replaced"
        chmod -R +w "${targetdir}/${vendordir}" >/dev/null 2>&1
        rm -rf "${targetdir}/${vendordir}"
    fi
    mkdir -p "${targetdir}/${vendordir}" || exit_log 1 "mkdir ${targetdir}/${vendordir} failed"

    log "[INFO] copying vendor files to ${targetdir}/${vendordir} ..."
    if ! cp -rf "${TMP_PKG_ROOT_PATH}/packages/vendors/${VENDOR_NAME}/." "${targetdir}/${vendordir}/"; then
        exit_log 1 "copy vendor files failed"
    fi
    chmod -R 550 "${targetdir}/${vendordir}" >/dev/null 2>&1
    if [ "${ARG_INSTALL_FOR_ALL}" = "y" ]; then
        # multi-user scenario: world-readable vendor tree + config.ini
        chmod -R 555 "${targetdir}/${vendordir}" >/dev/null 2>&1
    fi

    update_load_priority
    if [ "${ARG_INSTALL_FOR_ALL}" = "y" ] && [ -f "${config_file}" ]; then
        chmod 755 "${config_file}" 2>/dev/null
    fi

    # version.info 落到 vendor 目录
    if [ -f "${TMP_PKG_VERSION_FILE}" ]; then
        cp -f "${TMP_PKG_VERSION_FILE}" "${targetdir}/${vendordir}/version.info" 2>/dev/null
    fi

    # set_env.bash: 导出 ASCEND_CUSTOM_OPP_PATH + LD_LIBRARY_PATH
    mkdir -p "${targetdir}/${vendordir}/bin"
    cat > "${targetdir}/${vendordir}/bin/set_env.bash" <<EOF
#!/bin/bash
export ASCEND_CUSTOM_OPP_PATH=${targetdir}/${vendordir}:\${ASCEND_CUSTOM_OPP_PATH}
export LD_LIBRARY_PATH=${targetdir}/${vendordir}/op_api/lib/:\${LD_LIBRARY_PATH}
EOF
    chmod 550 "${targetdir}/${vendordir}/bin/set_env.bash"

    # uninstall.sh 落到 vendor scripts/ 目录(供 .run 包已删除、仅存安装结果时直接 bash uninstall.sh)
    mkdir -p "${targetdir}/${vendordir}/scripts"
    if [ -f "${TMP_PKG_SCRIPTS_PATH}/uninstall.sh" ]; then
        cp -f "${TMP_PKG_SCRIPTS_PATH}/uninstall.sh" "${targetdir}/${vendordir}/scripts/uninstall.sh"
        chmod 550 "${targetdir}/${vendordir}/scripts/uninstall.sh"
    fi

    log "[INFO] install success. vendor: ${VENDOR_NAME} -> ${targetdir}/${vendordir}"
    log "[INFO] verify: grep load_priority ${config_file}"
    log "[INFO] using requirements: source ${targetdir}/${vendordir}/bin/set_env.bash"
    log "[INFO] uninstall: bash ${targetdir}/${vendordir}/scripts/uninstall.sh"
}

do_uninstall() {
    if [ ! -d "${targetdir}/${vendordir}" ]; then
        log "[INFO] ${targetdir}/${vendordir} not exist, no need to uninstall"
        remove_load_priority
        echo "SUCCESS"
        exit 0
    fi
    chmod -R +w "${targetdir}/${vendordir}" >/dev/null 2>&1
    rm -rf "${targetdir}/${vendordir}" || exit_log 1 "remove ${targetdir}/${vendordir} failed"
    remove_load_priority
    log "[INFO] uninstall success. vendor removed: ${VENDOR_NAME}"
}

if [ "${ARG_MODE}" = "uninstall" ]; then
    do_uninstall
else
    do_install
fi

echo "SUCCESS"
exit 0
