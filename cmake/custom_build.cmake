# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# gym custom 包编排: 算子接入后的 host 侧目标创建(三 .so/核对程序) + kernel 编译链 + install + 打包支撑.
# 依赖顺序: op stub(obj_func.cmake) -> 算子接入(CMakeLists 顶层) -> 本文件(读回登记, 创建目标).

get_property(_aclnn_default_ops GLOBAL PROPERTY GYM_ACLNN_TYPE_aclnn)
get_property(_aclnn_inner_ops   GLOBAL PROPERTY GYM_ACLNN_TYPE_aclnn_inner)
get_property(_aclnn_exclude_ops GLOBAL PROPERTY GYM_ACLNN_TYPE_aclnn_exclude)
list(REMOVE_DUPLICATES _aclnn_default_ops)
list(REMOVE_DUPLICATES _aclnn_inner_ops)
list(REMOVE_DUPLICATES _aclnn_exclude_ops)

# inner/exc 无算子时用 stub, 保证 op_build 三份 ops-info.ini 齐套
if(NOT _aclnn_inner_ops)
    set(_inner_stub ${CMAKE_BINARY_DIR}/op_host_aclnn_inner_stub.cpp)
    file(WRITE ${_inner_stub} "// stub for empty aclnnInner op set\n")
    target_sources(op_host_aclnnInner PRIVATE ${_inner_stub})
endif()
if(NOT _aclnn_exclude_ops)
    set(_exc_stub ${CMAKE_BINARY_DIR}/op_host_aclnn_exc_stub.cpp)
    file(WRITE ${_exc_stub} "// stub for empty aclnnExc op set\n")
    target_sources(op_host_aclnnExc PRIVATE ${_exc_stub})
endif()

# ---------------------------------------- op_build 反射目标 ----------------------------------------
set(base_aclnn_binary_dir ${ASCEND_AUTOGEN_DIR})
set(generate_aclnn_srcs)
set(generate_aclnn_headers)
foreach(_op_name ${_aclnn_default_ops})
    list(APPEND generate_aclnn_srcs    ${base_aclnn_binary_dir}/aclnn_${_op_name}.cpp)
    list(APPEND generate_aclnn_headers ${base_aclnn_binary_dir}/aclnn_${_op_name}.h)
endforeach()

set(generate_aclnn_inner_srcs)
foreach(_op_name ${_aclnn_inner_ops})
    list(APPEND generate_aclnn_inner_srcs ${base_aclnn_binary_dir}/inner/aclnnInner_${_op_name}.cpp)
endforeach()

set(generate_exclude_proto_srcs)
foreach(_op_name ${_aclnn_exclude_ops})
    list(APPEND generate_exclude_proto_srcs ${base_aclnn_binary_dir}/exc/${_op_name}_proto.cpp)
endforeach()

string(REPLACE ";" "\;" OPS_PRODUCT_NAME "${ASCEND_COMPUTE_UNIT}")
if (generate_aclnn_srcs)
    add_custom_command(OUTPUT ${generate_aclnn_srcs} ${generate_aclnn_headers}
            COMMAND mkdir -p ${base_aclnn_binary_dir}
            COMMAND OPS_PROTO_SEPARATE=1
            OPS_ACLNN_GEN=1
            OPS_PROJECT_NAME=aclnn
            OPS_PRODUCT_NAME=\"${OPS_PRODUCT_NAME}\"
            env LD_PRELOAD=
            ${OP_BUILD_TOOL}
            $<TARGET_FILE:op_host_aclnn>
            ${base_aclnn_binary_dir}
    )
endif ()
add_custom_target(opbuild_gen_default
        DEPENDS ${generate_aclnn_srcs} ${generate_aclnn_headers} op_host_aclnn)

if (generate_aclnn_inner_srcs)
    add_custom_command(OUTPUT ${generate_aclnn_inner_srcs}
            COMMAND mkdir -p ${base_aclnn_binary_dir}/inner
            COMMAND OPS_PROTO_SEPARATE=1
            OPS_ACLNN_GEN=1
            OPS_PROJECT_NAME=aclnnInner
            OPS_PRODUCT_NAME=\"${OPS_PRODUCT_NAME}\"
            env LD_PRELOAD=
            ${OP_BUILD_TOOL}
            $<TARGET_FILE:op_host_aclnnInner>
            ${base_aclnn_binary_dir}/inner
    )
endif ()
add_custom_target(opbuild_gen_inner
        DEPENDS ${generate_aclnn_inner_srcs} op_host_aclnnInner)

if (generate_exclude_proto_srcs)
    add_custom_command(OUTPUT ${generate_exclude_proto_srcs}
            COMMAND mkdir -p ${base_aclnn_binary_dir}/exc
            COMMAND OPS_PROTO_SEPARATE=1
            OPS_ACLNN_GEN=0
            OPS_PROJECT_NAME=aclnnExc
            OPS_PRODUCT_NAME=\"${OPS_PRODUCT_NAME}\"
            env LD_PRELOAD=
            ${OP_BUILD_TOOL}
            $<TARGET_FILE:op_host_aclnnExc>
            ${base_aclnn_binary_dir}/exc
    )
endif ()
add_custom_target(opbuild_gen_exc
        DEPENDS ${generate_exclude_proto_srcs} op_host_aclnnExc)

# ---------------------------------------- adapt py 生成 ----------------------------------------
get_property(_pypto_adapt_ops GLOBAL PROPERTY PYPTO_ENABLED_OPS)
set(PYPTO_ADAPT_OPS_OPTION "")
if (_pypto_adapt_ops)
    list(REMOVE_DUPLICATES _pypto_adapt_ops)
    list(JOIN _pypto_adapt_ops "," _pypto_adapt_ops_str)
    set(PYPTO_ADAPT_OPS_OPTION --pypto-ops ${_pypto_adapt_ops_str})
endif ()

add_custom_target(generate_gym_adapt_py
        COMMAND ${Python3_EXECUTABLE} ${PYPTO_GYM_SRC_DIR}/cmake/scripts/util/ascendc_impl_build.py
        \"\"
        \"\"
        \"\"
        \"\"
        ${ASCEND_IMPL_OUT_DIR}
        ${ASCEND_AUTOGEN_DIR}
        --opsinfo-dir ${base_aclnn_binary_dir} ${base_aclnn_binary_dir}/inner ${base_aclnn_binary_dir}/exc
        --compute-unit ${ASCEND_COMPUTE_UNIT}
        ${PYPTO_ADAPT_OPS_OPTION}
)
add_dependencies(generate_gym_adapt_py opbuild_gen_default opbuild_gen_inner opbuild_gen_exc)

# ---------------------------------------- 聚合目标 ----------------------------------------
add_custom_target(prepare_build ALL)
add_custom_target(generate_compile_cmd ALL)
add_dependencies(prepare_build generate_gym_adapt_py generate_compile_cmd)

foreach (compute_unit ${ASCEND_COMPUTE_UNIT})
    add_compile_cmd_target(COMPUTE_UNIT ${compute_unit})
endforeach ()

if (ENABLE_OPS_KERNEL)
    add_custom_target(ops_gym_kernel ALL)
    add_custom_target(ops_gym_config ALL)
    add_dependencies(ops_gym_kernel ops_gym_config)
    foreach (compute_unit ${ASCEND_COMPUTE_UNIT})
        add_bin_compile_target(
                COMPUTE_UNIT ${compute_unit}
                OP_INFO ${OP_DIR_LIST})
    endforeach ()
endif ()

# ---------------------------------------- host 侧: 三 .so + vendors staging ----------------------------------------
# pass-2(PREPARE_BUILD) 跳过: autogen 源尚不存在, 且 host 产物不属于 prepare 阶段
if (ENABLE_OPS_HOST AND NOT PREPARE_BUILD)

    set(_vendor_dir packages/vendors/${VENDOR_NAME})
    set(_arch_lib_dir ${_vendor_dir}/op_proto/lib/linux/${CMAKE_SYSTEM_PROCESSOR})

    get_property(_infer_srcs GLOBAL PROPERTY GYM_INFER_SRCS)
    get_property(_tiling_srcs GLOBAL PROPERTY GYM_TILING_SRCS)
    get_property(_opapi_srcs GLOBAL PROPERTY GYM_OPAPI_SRCS)
    get_property(_pypto_gen_dirs GLOBAL PROPERTY GYM_PYPTO_GEN_DIRS)
    get_property(_tiling_check_srcs GLOBAL PROPERTY GYM_TILING_CHECK_SRCS)
    list(REMOVE_DUPLICATES _infer_srcs)
    list(REMOVE_DUPLICATES _tiling_srcs)
    list(REMOVE_DUPLICATES _opapi_srcs)
    list(REMOVE_DUPLICATES _pypto_gen_dirs)
    list(REMOVE_DUPLICATES _tiling_check_srcs)

    # autogen 源清单
    set(_autogen_aclnn_srcs)
    set(_autogen_proto_srcs)
    set(_autogen_aclnn_headers)
    foreach(_op_name ${_aclnn_default_ops})
        list(APPEND _autogen_aclnn_srcs ${base_aclnn_binary_dir}/aclnn_${_op_name}.cpp)
        list(APPEND _autogen_aclnn_headers ${base_aclnn_binary_dir}/aclnn_${_op_name}.h)
        list(APPEND _autogen_proto_srcs ${base_aclnn_binary_dir}/${_op_name}_proto.cpp)
    endforeach()

    # host 编译公共包含头(以构建期实际产物为准的最小集合)
    set(GYM_HOST_INCLUDES
        ${ASCEND_CANN_PACKAGE_PATH}/include
        ${ASCEND_CANN_PACKAGE_PATH}/include/exe_graph
        ${ASCEND_CANN_PACKAGE_PATH}/include/external
        ${ASCEND_CANN_PACKAGE_PATH}/include/platform
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime
        ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/include
        ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/pkg_inc
        ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/include/exe_graph
        ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/include/aclnn
        ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/ascendc/include
        ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/ascendc/include/highlevel_api
        ${base_aclnn_binary_dir})
    set(GYM_HOST_LINK_DIRS
        ${ASCEND_CANN_PACKAGE_PATH}/lib64
        ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/lib64)
    set(GYM_CANN_LIBS_DIR ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/lib64)

    # ---------- infer/proto obj → cust_opsproto_rt2.0.so ----------
    add_library(ophost_gym_infer_obj OBJECT EXCLUDE_FROM_ALL)
    set_target_properties(ophost_gym_infer_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(ophost_gym_infer_obj PRIVATE ${GYM_HOST_INCLUDES})
    if (_autogen_proto_srcs)
        target_sources(ophost_gym_infer_obj PRIVATE ${_autogen_proto_srcs})
    endif()
    if (_infer_srcs)
        target_sources(ophost_gym_infer_obj PRIVATE ${_infer_srcs})
    endif()
    target_link_libraries(ophost_gym_infer_obj PRIVATE intf_pub)
    add_dependencies(ophost_gym_infer_obj opbuild_gen_default)

    add_library(cust_proto SHARED EXCLUDE_FROM_ALL $<TARGET_OBJECTS:ophost_gym_infer_obj>)
    set_target_properties(cust_proto PROPERTIES OUTPUT_NAME cust_opsproto_rt2.0)
    target_include_directories(cust_proto PRIVATE ${GYM_HOST_INCLUDES})
    target_link_directories(cust_proto PRIVATE ${GYM_HOST_LINK_DIRS})
    target_link_libraries(cust_proto PRIVATE
        ${GYM_CANN_LIBS_DIR}/libexe_graph.so
        ${GYM_CANN_LIBS_DIR}/libgraph.so
        ${GYM_CANN_LIBS_DIR}/libgraph_base.so
        ${GYM_CANN_LIBS_DIR}/libregister.so
        ${GYM_CANN_LIBS_DIR}/liberror_manager.so
        ${GYM_CANN_LIBS_DIR}/libplatform.so
        -Wl,--whole-archive ${GYM_CANN_LIBS_DIR}/librt2_registry.a -Wl,--no-whole-archive
        ${ASCEND_CANN_PACKAGE_PATH}/lib64/libc_sec.so
        ascendalog pthread)
    target_link_libraries(cust_proto PRIVATE intf_pub)
    add_dependencies(cust_proto ophost_gym_infer_obj)

    # ---------- tiling obj → cust_opmaster_rt2.0.so + liboptiling.so 软链 ----------
    add_library(ophost_gym_tiling_obj OBJECT EXCLUDE_FROM_ALL)
    set_target_properties(ophost_gym_tiling_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(ophost_gym_tiling_obj PRIVATE ${GYM_HOST_INCLUDES})
    if (_tiling_srcs)
        # force-include pypto 生成头进 tiling 源(与 kernel 侧 tiling 定义一致性的权威参照)
        set(_force_opts "")
        foreach(_gen_dir ${_pypto_gen_dirs})
            file(GLOB _gen_headers ${_gen_dir}/*_tilingkey.h ${_gen_dir}/*_tiling.h)
            foreach(_h ${_gen_headers})
                list(APPEND _force_opts -include ${_h})
            endforeach()
        endforeach()
        if (_force_opts)
            set_source_files_properties(${_tiling_srcs} PROPERTIES COMPILE_OPTIONS "${_force_opts}")
        endif()
        target_sources(ophost_gym_tiling_obj PRIVATE ${_tiling_srcs})
    target_link_libraries(ophost_gym_tiling_obj PRIVATE intf_pub)
    endif()

    # tiling 布局核对程序(tiling_layout_check.py 生成, 运行时布局 vs 编译期真值
    # 交叉验证). 构建并在 tiling 编译前执行, 失败即拒(阻断后续编译).
    foreach(_chk_src ${_tiling_check_srcs})
        get_filename_component(_chk_name ${_chk_src} NAME_WE)
        add_executable(${_chk_name} EXCLUDE_FROM_ALL ${_chk_src})
        target_include_directories(${_chk_name} PRIVATE ${GYM_HOST_INCLUDES})
        target_link_directories(${_chk_name} PRIVATE ${GYM_HOST_LINK_DIRS})
        target_link_libraries(${_chk_name} PRIVATE
                ${GYM_CANN_LIBS_DIR}/libopp_registry.so
                ${GYM_CANN_LIBS_DIR}/libregister.so
                ${ASCEND_CANN_PACKAGE_PATH}/lib64/libc_sec.so
                pthread)
        add_custom_target(run_${_chk_name}
                COMMAND $<TARGET_FILE:${_chk_name}>
                DEPENDS ${_chk_name})
        target_link_libraries(${_chk_name} PRIVATE intf_pub)
        add_dependencies(ophost_gym_tiling_obj run_${_chk_name})
    endforeach()

    add_library(cust_opmaster SHARED EXCLUDE_FROM_ALL $<TARGET_OBJECTS:ophost_gym_tiling_obj>)
    set_target_properties(cust_opmaster PROPERTIES OUTPUT_NAME cust_opmaster_rt2.0)
    target_include_directories(cust_opmaster PRIVATE ${GYM_HOST_INCLUDES})
    target_link_directories(cust_opmaster PRIVATE ${GYM_HOST_LINK_DIRS})
    target_link_libraries(cust_opmaster PRIVATE
        ${GYM_CANN_LIBS_DIR}/libgraph.so
        ${GYM_CANN_LIBS_DIR}/libgraph_base.so
        ${GYM_CANN_LIBS_DIR}/libexe_graph.so
        ${GYM_CANN_LIBS_DIR}/libplatform.so
        ${GYM_CANN_LIBS_DIR}/libregister.so
        ${GYM_CANN_LIBS_DIR}/liberror_manager.so
        -Wl,--whole-archive ${GYM_CANN_LIBS_DIR}/librt2_registry.a -Wl,--no-whole-archive
        ${GYM_CANN_LIBS_DIR}/libtiling_api.a
        ${ASCEND_CANN_PACKAGE_PATH}/lib64/libc_sec.so
        ${ASCEND_CANN_PACKAGE_PATH}/lib64/libascendalog.so
        pthread)
    target_link_libraries(cust_opmaster PRIVATE intf_pub)
    add_dependencies(cust_opmaster ophost_gym_tiling_obj)

    # liboptiling.so 兼容软链(运行时按旧名查找 tiling 库)
    # 目标为安装态相对路径: 软链 install 后位于 op_tiling/ 下, .so 实际在 op_tiling/lib/linux/<arch>/
    # (ALL 目标每次构建以 ln -sf 强制刷新, 无陈旧软链问题;
    #  build/ 内该链悬空属预期, 仅安装态需要解析)
    add_custom_target(gym_optiling_compat ALL
            COMMAND ln -sf lib/linux/${CMAKE_SYSTEM_PROCESSOR}/$<TARGET_FILE_NAME:cust_opmaster>
                    ${CMAKE_BINARY_DIR}/liboptiling.so)

    # ---------- aclnn 静态库 → whole-archive 进 cust_opapi ----------
    if (_autogen_aclnn_srcs)
        add_library(ops_aclnn STATIC EXCLUDE_FROM_ALL ${_autogen_aclnn_srcs})
        set_target_properties(ops_aclnn PROPERTIES POSITION_INDEPENDENT_CODE ON)
        target_link_libraries(ops_aclnn PRIVATE intf_pub)
        target_include_directories(ops_aclnn PRIVATE ${GYM_HOST_INCLUDES})
        add_dependencies(ops_aclnn opbuild_gen_default)
        # 手写 aclnn 通道: 手写源 whole-archive 一并链入
        if (_opapi_srcs)
            add_library(gym_opapi_manual STATIC EXCLUDE_FROM_ALL ${_opapi_srcs})
            set_target_properties(gym_opapi_manual PROPERTIES POSITION_INDEPENDENT_CODE ON)
            target_link_libraries(gym_opapi_manual PRIVATE intf_pub)
            target_include_directories(gym_opapi_manual PRIVATE ${GYM_HOST_INCLUDES})
        endif()
    endif()

    set(_cust_opapi_stub ${CMAKE_BINARY_DIR}/cust_opapi_stub.cpp)
    file(WRITE ${_cust_opapi_stub} "// stub translation unit for libcust_opapi.so\n")
    add_library(cust_opapi SHARED EXCLUDE_FROM_ALL ${_cust_opapi_stub})
    target_include_directories(cust_opapi PRIVATE ${GYM_HOST_INCLUDES})
    target_link_directories(cust_opapi PRIVATE ${GYM_HOST_LINK_DIRS})
    if (TARGET gym_opapi_manual)
        target_link_libraries(cust_opapi PRIVATE
            -Wl,--whole-archive
            $<TARGET_FILE:ops_aclnn>
            $<TARGET_FILE:gym_opapi_manual>
            -Wl,--no-whole-archive)
    else ()
        target_link_libraries(cust_opapi PRIVATE
            -Wl,--whole-archive
            $<TARGET_FILE:ops_aclnn>
            -Wl,--no-whole-archive)
    endif ()
    target_link_libraries(cust_opapi PRIVATE
        ${GYM_CANN_LIBS_DIR}/libnnopbase.so
        ${GYM_CANN_LIBS_DIR}/libprofapi.so
        ${GYM_CANN_LIBS_DIR}/libge_common_base.so
        ${GYM_CANN_LIBS_DIR}/libascend_dump.so
        ascendalog dl)
    target_link_libraries(cust_opapi PRIVATE intf_pub)
    add_dependencies(cust_opapi ops_aclnn)
    add_custom_target(ops_gym_host ALL)
    add_dependencies(ops_gym_host cust_opapi cust_proto cust_opmaster gym_optiling_compat)

    if (ENABLE_ASAN OR ENABLE_UBSAN)
        message(STATUS "gym sanitizers enabled via intf_pub: ASAN=${ENABLE_ASAN} UBSAN=${ENABLE_UBSAN} (host side)")
    endif()

    # ---------- ops-info.json (三 ini 合成) ----------
    foreach (compute_unit ${ASCEND_COMPUTE_UNIT})
        set(_ops_info_json ${ASCEND_AUTOGEN_DIR}/aic-${compute_unit}-ops-info.json)
        add_custom_command(OUTPUT ${_ops_info_json}
                COMMAND ${Python3_EXECUTABLE} ${GYM_UTIL_SCRIPT_DIR}/parse_ini_to_json.py
                ${base_aclnn_binary_dir}/aic-${compute_unit}-ops-info.ini
                ${base_aclnn_binary_dir}/inner/aic-${compute_unit}-ops-info.ini
                ${base_aclnn_binary_dir}/exc/aic-${compute_unit}-ops-info.ini
                ${_ops_info_json}
                DEPENDS opbuild_gen_default opbuild_gen_inner opbuild_gen_exc)
        add_custom_target(generate_ops_info_${compute_unit} ALL DEPENDS ${_ops_info_json})
        add_dependencies(prepare_build generate_ops_info_${compute_unit})
    endforeach ()

    # ---------- opc 编译期 custom opp staging ----------
    # opc 通过 ASCEND_CUSTOM_OPP_PATH 读取 op store(输入/输出数校验)与 tiling 结构
    # (liboptiling.so, 决定 .json 的 opParaSize/parameters). 构建期用本仓产物组一个
    # 临时 opp 树, 既不依赖已安装 vendor, 也避免 "do not registe tiling struct" 导致的
    # opParaSize=8(运行期 TilingDef::SaveToBuffer capacity 不足).
    set(_opstore_dir ${GYM_OPSTORE_DIR})
    set(_opstore_stamp ${_opstore_dir}/.stamp)
    add_custom_command(OUTPUT ${_opstore_stamp}
            COMMAND ${CMAKE_COMMAND} -E make_directory
            ${_opstore_dir}/op_impl/ai_core/tbe/config/${ASCEND_COMPUTE_UNIT}
            COMMAND ${CMAKE_COMMAND} -E make_directory
            ${_opstore_dir}/op_impl/ai_core/tbe/op_tiling
            COMMAND ${CMAKE_COMMAND} -E create_symlink
            ${ASCEND_AUTOGEN_DIR}/aic-${ASCEND_COMPUTE_UNIT}-ops-info.json
            ${_opstore_dir}/op_impl/ai_core/tbe/config/${ASCEND_COMPUTE_UNIT}/aic-${ASCEND_COMPUTE_UNIT}-ops-info.json
            COMMAND ${CMAKE_COMMAND} -E create_symlink
            $<TARGET_FILE:cust_opmaster>
            ${_opstore_dir}/op_impl/ai_core/tbe/op_tiling/liboptiling.so
            COMMAND ${CMAKE_COMMAND} -E touch ${_opstore_stamp}
            DEPENDS generate_ops_info_${ASCEND_COMPUTE_UNIT} gym_optiling_compat cust_opmaster
            VERBATIM)
    add_custom_target(gym_opstore_stage ALL DEPENDS ${_opstore_stamp})
    get_property(_gym_kernel_targets GLOBAL PROPERTY GYM_KERNEL_TARGETS)
    foreach(_kernel_target ${_gym_kernel_targets})
        add_dependencies(${_kernel_target} gym_opstore_stage)
    endforeach ()

    # ---------- vendors staging install 规则(COMPONENT = 包组件, 供 CPack staging 收集) ----------
    install(TARGETS cust_opapi LIBRARY DESTINATION ${_vendor_dir}/op_api/lib COMPONENT ${VENDOR_NAME})
    install(TARGETS cust_proto LIBRARY DESTINATION ${_arch_lib_dir} COMPONENT ${VENDOR_NAME})
    install(TARGETS cust_opmaster
            LIBRARY DESTINATION ${_vendor_dir}/op_impl/ai_core/tbe/op_tiling/lib/linux/${CMAKE_SYSTEM_PROCESSOR}
            COMPONENT ${VENDOR_NAME})
    install(FILES ${CMAKE_BINARY_DIR}/liboptiling.so
            DESTINATION ${_vendor_dir}/op_impl/ai_core/tbe/op_tiling COMPONENT ${VENDOR_NAME})
    if (_autogen_aclnn_headers)
        install(FILES ${_autogen_aclnn_headers} DESTINATION ${_vendor_dir}/op_api/include/aclnnop
                COMPONENT ${VENDOR_NAME})
        # aclnn 汇总头: 汇总各算子生成的 aclnn_<op>.h, 提供单入口头;
        # include 全部 aclnn_<op>.h, 使用方 #include "aclnn_ops_<vendor>.h" 即可
        set(_gym_aclnn_master ${CMAKE_BINARY_DIR}/aclnn_ops_${VENDOR_NAME}.h)
        file(WRITE ${_gym_aclnn_master}
                "#ifndef ACLNN_OPS_${VENDOR_NAME}_H_\n"
                "#define ACLNN_OPS_${VENDOR_NAME}_H_\n\n")
        foreach(_aclnn_h ${_autogen_aclnn_headers})
            get_filename_component(_aclnn_name ${_aclnn_h} NAME)
            file(APPEND ${_gym_aclnn_master} "#include \"${_aclnn_name}\"\n")
        endforeach()
        file(APPEND ${_gym_aclnn_master} "\n#endif // ACLNN_OPS_${VENDOR_NAME}_H_\n")
        install(FILES ${_gym_aclnn_master} DESTINATION ${_vendor_dir}/op_api/include/aclnnop
                COMPONENT ${VENDOR_NAME})
    endif()
    # proto 头 install: autogen 下全部 *_proto.h
    file(GLOB _autogen_proto_headers ${ASCEND_AUTOGEN_DIR}/*_proto.h)
    if (_autogen_proto_headers)
        install(FILES ${_autogen_proto_headers}
                DESTINATION ${_vendor_dir}/op_proto/inc OPTIONAL
                COMPONENT ${VENDOR_NAME})
    endif()
    foreach (compute_unit ${ASCEND_COMPUTE_UNIT})
        install(FILES ${ASCEND_AUTOGEN_DIR}/aic-${compute_unit}-ops-info.json
                DESTINATION ${_vendor_dir}/op_impl/ai_core/tbe/config/${compute_unit}
                COMPONENT ${VENDOR_NAME})
    endforeach ()
    foreach (_op_dir ${OP_DIR_LIST})
        get_filename_component(_op_name ${_op_dir} NAME)
        install(FILES ${ASCEND_IMPL_OUT_DIR}/dynamic/${_op_name}.py
                DESTINATION ${_vendor_dir}/op_impl/ai_core/tbe/${VENDOR_NAME}_impl/dynamic OPTIONAL
                COMPONENT ${VENDOR_NAME})
        install(DIRECTORY ${_op_dir}/op_kernel
                DESTINATION ${_vendor_dir}/op_impl/ai_core/tbe/${VENDOR_NAME}_impl/ascendc/${_op_name}
                OPTIONAL COMPONENT ${VENDOR_NAME}
                PATTERN "__pycache__" EXCLUDE)
    endforeach ()
endif ()
