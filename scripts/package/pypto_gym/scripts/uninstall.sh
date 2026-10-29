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
# pypto_gym 独立卸载脚本
#
# 安装后位于 <opp>/vendors/<vendor>/scripts/uninstall.sh, 供用户在已安装目录下直接执行卸载.
# 适用于 .run 包已删除、仅存安装结果的场景(与 install.sh --uninstall 等价).

vendor_name=pypto_gym

# 从本脚本位置反推 vendors 根目录与 config.ini
# 本脚本位于 <opp>/vendors/<vendor>/scripts/uninstall.sh, 向上 2 级 = <opp>/vendors
curr_path=$(dirname "$(readlink -f "$0")")
vendordir=$(realpath "${curr_path}/../../")
config_file="${vendordir}/config.ini"

log() {
    cur_date=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[pypto_gym_uninstall] [$cur_date] $1"
}

# 删除 vendor 目录
if [ ! -d "${vendordir}/${vendor_name}" ]; then
    log "[INFO] ${vendordir}/${vendor_name} not exist, no need to uninstall"
else
    log "[INFO] removing ${vendor_name} ..."
    chmod -R +w "${vendordir}/${vendor_name}" 2>/dev/null
    rm -rf "${vendordir}/${vendor_name}"
    if [ $? -ne 0 ]; then
        log "[ERROR] failed to remove ${vendordir}/${vendor_name}"
        exit 1
    fi
fi

# 从 config.ini 剔除本 vendor(与 install.sh 的 remove_load_priority 逻辑一致)
if [ ! -f "${config_file}" ]; then
    log "[INFO] ${config_file} not exist, done"
    exit 0
fi

line=$(grep -E '^\s*load_priority\s*=' "${config_file}" | head -n1)
if [ -z "$line" ]; then
    log "[INFO] no 'load_priority=' found in ${config_file}, done"
    exit 0
fi

# 按逗号切分, 剔除本 vendor 后重组
value=$(echo "$line" | sed -E 's/^[^=]*=\s*(.*)/\1/' | tr -d ' \t')
new_value=""
old_ifs="$IFS"
IFS=','
for item in ${value}; do
    [ -z "$item" ] && continue
    [ "$item" = "${vendor_name}" ] && continue
    new_value="${new_value:+${new_value},}${item}"
done
IFS="$old_ifs"

chmod +w "${config_file}" 2>/dev/null
if [ -z "$new_value" ]; then
    rm -f "${config_file}"
else
    sed -i "s|^\s*load_priority\s*=.*|load_priority=${new_value}|" "${config_file}"
    if [ $? -ne 0 ]; then
        log "[ERROR] failed to update load_priority in ${config_file}"
        exit 1
    fi
fi

log "[INFO] ${vendor_name} uninstalled successfully"
exit 0
