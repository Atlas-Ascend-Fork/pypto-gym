if(ENABLE_TORCH_BINDINGS AND NOT PREPARE_BUILD)
    gym_add_torch_binding(compressor_v2 compressor_v2_plan.h)
endif()
