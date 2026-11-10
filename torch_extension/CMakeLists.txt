# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import pathlib, torch_npu; print(pathlib.Path(torch_npu.__file__).parent)"
    OUTPUT_VARIABLE GYM_TORCH_NPU_ROOT OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _torch_npu_rc ERROR_QUIET)
if(NOT _torch_npu_rc EQUAL 0)
    message(FATAL_ERROR "ENABLE_TORCH_BINDINGS requires torch_npu in ${Python3_EXECUTABLE}")
endif()
find_library(GYM_TORCH_NPU_LIB torch_npu PATHS "${GYM_TORCH_NPU_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
find_library(GYM_ASCENDCL_LIB ascendcl PATHS "${ASCEND_CANN_PACKAGE_PATH}/lib64" NO_DEFAULT_PATH REQUIRED)

add_library(gym_torch_common INTERFACE)
target_include_directories(gym_torch_common INTERFACE
    ${TORCH_INCLUDE_DIRS}
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    ${Python3_INCLUDE_DIRS}
    "${GYM_TORCH_NPU_ROOT}/include"
    "${ASCEND_CANN_PACKAGE_PATH}/include")
target_link_libraries(gym_torch_common INTERFACE
    ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIB} ${GYM_TORCH_NPU_LIB} ${GYM_ASCENDCL_LIB} dl)
separate_arguments(_gym_torch_flags NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
target_compile_options(gym_torch_common INTERFACE ${_gym_torch_flags})
target_compile_features(gym_torch_common INTERFACE cxx_std_17)

install(DIRECTORY pypto_gym/
    DESTINATION packages/vendors/${VENDOR_NAME}/python/pypto_gym
    COMPONENT ${VENDOR_NAME} PATTERN "__pycache__" EXCLUDE)

# Called from ops/<paradigm>/<network>/<op>/torch_extension/CMakeLists.txt.
function(gym_add_torch_binding op_name)
    add_library(torch_binding_${op_name} SHARED "${op_name}.cpp")
    target_link_libraries(torch_binding_${op_name} PRIVATE gym_torch_common)
    set_target_properties(torch_binding_${op_name} PROPERTIES
        OUTPUT_NAME pypto_gym_${op_name}
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/torch_extension"
        INSTALL_RPATH "$ORIGIN/../../../op_api/lib"
        INSTALL_RPATH_USE_LINK_PATH FALSE)
    install(TARGETS torch_binding_${op_name}
        LIBRARY DESTINATION packages/vendors/${VENDOR_NAME}/python/pypto_gym/_native
        COMPONENT ${VENDOR_NAME})
    install(FILES "${op_name}.py"
        DESTINATION packages/vendors/${VENDOR_NAME}/python/pypto_gym/_frontends
        COMPONENT ${VENDOR_NAME})
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
        DESTINATION packages/vendors/${VENDOR_NAME}/python/pypto_gym
        COMPONENT ${VENDOR_NAME})
    message(STATUS "Precompiled Torch bindings: ${_ops}")
endfunction()
