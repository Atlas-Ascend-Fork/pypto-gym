if(PREPARE_BUILD)
    return()
endif()
set(_metadata quant_lightning_indexer_metadata)
set(_codegen ${CMAKE_CURRENT_SOURCE_DIR}/build_native.py)
set(_native_dir ${CMAKE_BINARY_DIR}/aicpu_native/${_metadata})
set(_native_so ${_native_dir}/lib${_metadata}.so)
set(_native_header ${_native_dir}/${_metadata}_native.h)
add_custom_command(OUTPUT ${_native_so} ${_native_header}
    COMMAND ${CMAKE_COMMAND} -E env PYTHONDONTWRITEBYTECODE=1
        ${Python3_EXECUTABLE} ${_codegen} --out-dir ${_native_dir}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${_metadata}.py ${_codegen}
    COMMENT "Building offline native AICPU metadata (beta fast-launch)"
    VERBATIM)
add_custom_target(qli_metadata_native ALL DEPENDS ${_native_so} ${_native_header})
install(FILES ${_native_so} DESTINATION ${GYM_VENDOR_DEST}/op_api/lib COMPONENT ${VENDOR_NAME})
install(FILES ${_native_header} DESTINATION ${GYM_VENDOR_DEST}/op_api/include/aclnnop COMPONENT ${VENDOR_NAME})

# The installed JIT entry loads its planner relative to op_kernel in both DSL layouts.
set(_qli_impl ${GYM_VENDOR_DEST}/op_impl/ai_core/tbe/${VENDOR_NAME}_impl)
foreach(_layout IN ITEMS dynamic ascendc/quant_lightning_indexer)
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/${_metadata}.py
        DESTINATION ${_qli_impl}/${_layout}/op_kernel_aicpu COMPONENT ${VENDOR_NAME})
endforeach()
