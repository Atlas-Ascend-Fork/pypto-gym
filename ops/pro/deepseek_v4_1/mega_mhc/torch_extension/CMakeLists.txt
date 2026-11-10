if(ENABLE_TORCH_BINDINGS AND NOT PREPARE_BUILD)
    gym_add_torch_binding(mega_mhc)
endif()
