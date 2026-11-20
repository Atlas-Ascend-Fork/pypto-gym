# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#[[
Runtime-JIT delivery (decision 28): no binding compilation here and no
torch/torch_npu build dependency. The payload ships the binding C++ sources
(_csrc/) plus the Python entry package; pypto_gym._jit compiles the binding
on first call against the user's actual torch/torch_npu (no rpath/RUNPATH
involved, decision 29).

Install destinations come from the root-scope layout variables (config.cmake):
  run layout: GYM_PY_ENTRY_DEST = packages/vendors/<v>/python/pypto_gym
  whl layout: GYM_PY_ENTRY_DEST = pypto_gym (wheel root)
The gym_shell component additionally installs the entry tree rooted at
pypto_gym/ in ANY layout, so `cmake --install --component gym_shell` alone
materializes the shell-wheel source tree (cmake stays the single source of
layout truth; build_ci does no source assembly).
]]

option(GYM_PACKAGE_SHELL_WHL "Embed the Python shell wheel into the run payload" OFF)

# The repo tree under pypto_gym/ mirrors the installed package layout 1:1
# (_loader/, _bindings/csrc/gym_aclnn.h, ...), so the directory install carries
# everything except the per-op materials aggregated below and the configure-time
# operators.json.
install(DIRECTORY pypto_gym/
    DESTINATION ${GYM_PY_ENTRY_DEST}
    COMPONENT ${VENDOR_NAME} PATTERN "__pycache__" EXCLUDE)

# Shell-tree component (py3-none-any shell wheel source, layout independent).
install(DIRECTORY pypto_gym/
    DESTINATION pypto_gym
    COMPONENT gym_shell PATTERN "__pycache__" EXCLUDE)

# Called from ops/<paradigm>/<network>/<op>/torch_extension/CMakeLists.txt.
# Per-op materials aggregate into _bindings/{csrc,frontends}/ at install time
# (decentralized ownership, centralized layout); the JIT compile resolves
# `#include "gym_aclnn.h"` from _bindings/csrc/ (same directory as the sources).
function(gym_add_torch_binding op_name)
    install(FILES "${op_name}.cpp"
        DESTINATION ${GYM_PY_ENTRY_DEST}/_bindings/csrc
        COMPONENT ${VENDOR_NAME})
    install(FILES "${op_name}.py"
        DESTINATION ${GYM_PY_ENTRY_DEST}/_bindings/frontends
        COMPONENT ${VENDOR_NAME})
    install(FILES "${op_name}.cpp"
        DESTINATION pypto_gym/_bindings/csrc
        COMPONENT gym_shell)
    install(FILES "${op_name}.py"
        DESTINATION pypto_gym/_bindings/frontends
        COMPONENT gym_shell)
    set_property(GLOBAL APPEND PROPERTY GYM_TORCH_OPS ${op_name})
endfunction()

function(gym_finalize_torch_bindings)
    get_property(_ops GLOBAL PROPERTY GYM_TORCH_OPS)
    if(NOT _ops)
        message(FATAL_ERROR "None of the selected operators provide a Torch binding")
    endif()
    list(REMOVE_DUPLICATES _ops)
    list(JOIN _ops "\", \"" _ops_json)
    file(WRITE "${CMAKE_BINARY_DIR}/torch_extension/operators.json" "[\"${_ops_json}\"]\n")
    install(FILES "${CMAKE_BINARY_DIR}/torch_extension/operators.json"
        DESTINATION ${GYM_PY_ENTRY_DEST}/_bindings
        COMPONENT ${VENDOR_NAME})
    install(FILES "${CMAKE_BINARY_DIR}/torch_extension/operators.json"
        DESTINATION pypto_gym/_bindings
        COMPONENT gym_shell)
    message(STATUS "Torch binding sources (runtime JIT): ${_ops}")

    if(GYM_PACKAGE_SHELL_WHL)
        # Embed the shell wheel built by build_ci.py between `cmake --build` and
        # `package`; the run installer pip-installs it into the CANN
        # site-packages (decision 29). Globbed at install time and absence is
        # tolerated so manual cmake flows keep working.
        install(CODE "
            file(GLOB _gym_shell_whls \"${CMAKE_BINARY_DIR}/shell_whl/*.whl\")
            if(_gym_shell_whls)
                file(INSTALL \${_gym_shell_whls}
                     DESTINATION \"\${CMAKE_INSTALL_PREFIX}/packages/vendors/${VENDOR_NAME}/python\")
                message(STATUS \"Embedded shell wheel(s): \${_gym_shell_whls}\")
            else()
                message(WARNING \"No shell wheel under ${CMAKE_BINARY_DIR}/shell_whl; \"
                                \"the run package ships without the Python layer\")
            endif()"
            COMPONENT ${VENDOR_NAME})
    endif()
endfunction()
