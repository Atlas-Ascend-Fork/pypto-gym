#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# Keep the payload vendor-local and expose only its binding entries through the
# Python path already set by CANN. Never replace an existing wheel's ops tree.
gym_python_public_dir() {
    local opp_root cann_root
    opp_root=$(readlink -f "$1") || return 1
    cann_root=$(dirname "${opp_root}")
    if [ "${opp_root}" = "${cann_root}/opp" ] &&
       [ -f "${cann_root}/set_env.sh" ] &&
       [ -d "${cann_root}/python/site-packages" ]; then
        printf '%s\n' "${cann_root}/python/site-packages/pypto_gym"
    fi
}

gym_python_link_owned() {
    [ -L "$1" ] && [ "$(readlink -m "$1")" = "$(readlink -m "$2")" ]
}

# Return codes:
#   0 - registration is possible, or the install is not a standard CANN layout
#   1 - an existing entry belongs to another package (fatal: never overwrite)
#   2 - the CANN site-packages is not writable (degradable: install vendor only)
gym_python_check_links() {
    local public_dir vendor_python entry path
    public_dir=$(gym_python_public_dir "$1") || return 1
    [ -n "${public_dir}" ] || return 0
    vendor_python="$1/vendors/pypto_gym/python/pypto_gym"
    if [ -L "${public_dir}" ] || { [ -e "${public_dir}" ] && [ ! -d "${public_dir}" ]; }; then
        printf 'Python binding path is already occupied: %s\n' "${public_dir}" >&2
        return 1
    fi
    # Ownership is checked before writability: another package must never be
    # overwritten even when the current user could technically write the entry.
    for entry in __init__.py _loader.py _native _frontends operators.json; do
        path="${public_dir}/${entry}"
        if { [ -e "${path}" ] || [ -L "${path}" ]; } &&
           ! gym_python_link_owned "${path}" "${vendor_python}/${entry}"; then
            printf 'Python binding entry is already owned by another package: %s\n' "${path}" >&2
            return 1
        fi
    done
    if [ ! -w "$(dirname "${public_dir}")" ] ||
       { [ -d "${public_dir}" ] && [ ! -w "${public_dir}" ]; }; then
        printf 'Python binding path is not writable: %s\n' "${public_dir}" >&2
        return 2
    fi
    return 0
}

# Non-fatal registration failure: keep the vendor install usable through its own
# set_env.bash instead of aborting the whole installation.
gym_python_warn_degraded() {
    local targetdir="$1" reason="$2"
    printf 'warning: %s; Python bindings were not registered\n' "${reason}" >&2
    printf 'warning: run "source %s/vendors/pypto_gym/bin/set_env.bash" to use pypto_gym in this shell\n' \
        "${targetdir}" >&2
}

gym_python_remove_links() {
    local public_dir vendor_python entry path removed_any=n
    public_dir=$(gym_python_public_dir "$1") || return 1
    [ -n "${public_dir}" ] || return 0
    vendor_python="$1/vendors/pypto_gym/python/pypto_gym"
    for entry in __init__.py _loader.py _native _frontends operators.json; do
        path="${public_dir}/${entry}"
        if gym_python_link_owned "${path}" "${vendor_python}/${entry}"; then
            rm -f "${path}" || return 1
            removed_any=y
            case "${entry}" in
                __init__.py|_loader.py)
                    rm -f "${public_dir}/__pycache__/${entry%.py}."*.pyc || return 1
                    ;;
            esac
        fi
    done
    [ "${removed_any}" = y ] || return 0
    rmdir "${public_dir}/__pycache__" "${public_dir}" 2>/dev/null || true
}

gym_python_install_links() {
    local public_dir vendor_python entry link_target
    public_dir=$(gym_python_public_dir "$1") || return 1
    [ -n "${public_dir}" ] || return 0
    vendor_python="$1/vendors/pypto_gym/python/pypto_gym"
    [ -f "${vendor_python}/operators.json" ] || return 0
    mkdir -p -m 755 "${public_dir}" || return 1
    for entry in __init__.py _loader.py _native _frontends operators.json; do
        link_target=$(realpath --relative-to="${public_dir}" "${vendor_python}/${entry}") || return 1
        ln -s "${link_target}" "${public_dir}/${entry}" || return 1
    done
}
