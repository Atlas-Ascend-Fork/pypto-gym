#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
# Python shell-wheel registration (decision 29/30).
#
# The run payload embeds a python shell wheel (pypto_gym-*.whl, py3-none-any:
# entry package + _csrc binding sources, no vendor binaries). For a standard
# CANN layout the installer pip-installs it into <cann_root>/python/site-packages
# (the path CANN's own set_env.sh already puts on PYTHONPATH), mirroring the
# pypto / ops-transformer convention. Failures degrade to a warning: the vendor
# binaries stay usable through pure C++ aclnn, and the wheel remains inside the
# vendor tree for a manual pip install.

# Derive the public site-packages entry from the opp install root: only the
# standard CANN layout qualifies (<cann_root>/opp + set_env.sh + site-packages).
# gym's --install-path points at the opp dir, hence the dirname hop (decision 30).
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

gym_python_warn_degraded() {
    local targetdir="$1" reason="$2"
    printf 'warning: %s; the Python layer was not registered\n' "${reason}" >&2
    printf 'warning: pip install the embedded wheel manually to enable pypto_gym:\n' >&2
    printf 'warning:   python3 -m pip install %s/vendors/pypto_gym/python/pypto_gym-*.whl\n' \
        "${targetdir}" >&2
}

# pip-install the embedded shell wheel. Prints the registered package dir on
# success; always returns 0 (non-fatal by design), warnings go to stderr.
# Namespace ownership: pypto_gym is owned solely by this distribution (the
# shared-namespace kernel wheel is retired, decisions 24/25), so pip's
# whole-directory replacement (--upgrade) is the intended upgrade semantics;
# uninstall remains RECORD-precise and preserves foreign files.
gym_install_shell_wheel() {
    local targetdir="$1" public_dir site_dir whl
    public_dir=$(gym_python_public_dir "${targetdir}") || return 0
    [ -n "${public_dir}" ] || return 0
    site_dir=$(dirname "${public_dir}")
    whl=$(find "${targetdir}/vendors/pypto_gym/python" -maxdepth 1 -name "pypto_gym-*.whl" 2>/dev/null | head -1)
    if [ -z "${whl}" ]; then
        gym_python_warn_degraded "${targetdir}" "no shell wheel in the payload"
        return 0
    fi
    if ! command -v python3 >/dev/null 2>&1; then
        gym_python_warn_degraded "${targetdir}" "python3 is not available"
        return 0
    fi
    # Migration guard: drop legacy symlink trees left by pre-wheel versions.
    if [ -L "${public_dir}" ] || { [ -d "${public_dir}" ] && [ -n "$(find "${public_dir}" -maxdepth 1 -type l 2>/dev/null | head -1)" ]; }; then
        chmod -R +w "${public_dir}" 2>/dev/null
        rm -rf "${public_dir}"
    fi
    # Ownership guard: a foreign real __init__.py without our dist-info means
    # another package squats the name; never clobber it (pip --force-reinstall
    # would). A namespace-style directory (e.g. a kernel wheel's pypto_gym/ops)
    # is mergeable and passes.
    if [ -f "${public_dir}/__init__.py" ] && [ ! -L "${public_dir}/__init__.py" ] && \
       [ -z "$(find "${site_dir}" -maxdepth 1 -type d -name "pypto_gym-*.dist-info" 2>/dev/null | head -1)" ]; then
        gym_python_warn_degraded "${targetdir}" \
            "Python binding path is already occupied by another package: ${public_dir}"
        return 0
    fi
    if [ ! -w "${site_dir}" ]; then
        gym_python_warn_degraded "${targetdir}" "CANN site-packages is not writable"
        return 0
    fi
    if ! python3 -m pip install --disable-pip-version-check --no-deps --no-compile \
            --force-reinstall --upgrade -t "${site_dir}" "${whl}" >/dev/null 2>&1; then
        gym_python_warn_degraded "${targetdir}" "pip install of the shell wheel failed"
        return 0
    fi
    if [ "${ARG_INSTALL_FOR_ALL:-n}" = "y" ]; then
        # Multi-user scenario: pip -t honors umask, so widen readability the
        # same way the vendor tree does under --install-for-all.
        chmod -R a+rX "${public_dir}" 2>/dev/null
        find "${site_dir}" -maxdepth 1 -type d -name "pypto_gym-*.dist-info" \
            -exec chmod -R a+rX {} + 2>/dev/null
    fi
    printf '%s\n' "${public_dir}"
}

# Remove the pip-registered package via its RECORD (files owned by other
# distributions under pypto_gym/, e.g. a kernel wheel's ops/ tree, survive);
# legacy symlink trees are dropped wholesale.
gym_remove_shell_wheel() {
    local targetdir="$1" public_dir site_dir distinfo entry
    public_dir=$(gym_python_public_dir "${targetdir}") || return 0
    [ -n "${public_dir}" ] || return 0
    site_dir=$(dirname "${public_dir}")
    if [ -d "${public_dir}" ] && [ -n "$(find "${public_dir}" -maxdepth 1 -type l 2>/dev/null | head -1)" ]; then
        chmod -R +w "${public_dir}" 2>/dev/null
        rm -rf "${public_dir}" || return 1
    fi
    distinfo=$(find "${site_dir}" -maxdepth 1 -type d -name "pypto_gym-*.dist-info" 2>/dev/null | head -1)
    if [ -n "${distinfo}" ] && [ -f "${distinfo}/RECORD" ]; then
        while IFS= read -r entry; do
            [ -n "${entry}" ] || continue
            rm -f "${site_dir}/${entry%%,*}" 2>/dev/null
        done < "${distinfo}/RECORD"
        rm -rf "${distinfo}"
        # Bytecode caches are not in RECORD but are regenerable and ours
        # (nested too: importing _jit/_frontends creates per-subpackage caches).
        find "${public_dir}" -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null
        # Drop now-empty package subdirs bottom-up, then the package dir itself
        # only when empty: foreign files (shared namespace) survive.
        find "${public_dir}" -mindepth 1 -type d -empty -delete 2>/dev/null
        rmdir "${public_dir}" 2>/dev/null
    fi
    return 0
}
