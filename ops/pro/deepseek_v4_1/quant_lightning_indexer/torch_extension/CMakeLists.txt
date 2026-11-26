if(ENABLE_TORCH_BINDINGS AND NOT PREPARE_BUILD)
    gym_add_torch_binding(quant_lightning_indexer)
    # The runtime Torch compiler resolves quoted includes alongside the installed .cpp.
    set(qli_binding_headers
        "${CMAKE_CURRENT_SOURCE_DIR}/../op_host/quant_lightning_indexer_config.h"
        "${CMAKE_BINARY_DIR}/aicpu_native/quant_lightning_indexer_metadata/quant_lightning_indexer_metadata_native.h")
    install(FILES ${qli_binding_headers}
        DESTINATION ${GYM_PY_ENTRY_DEST}/_bindings/csrc COMPONENT ${VENDOR_NAME})
    install(FILES ${qli_binding_headers}
        DESTINATION pypto_gym/_bindings/csrc COMPONENT gym_shell)
endif()
