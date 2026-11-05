# 官方指定算子样例清单

本清单由 `pypto-pro-docs-search` 维护，是 Pro 资料装配与缓存检索共用的文件清单。`sync_devkit.py` 通过 `--samples` 读取本文件，按清单准备 `$PYPTO_DEVKIT_DIR/pro_ops/`。

维护规则：增删或重命名条目时只维护本清单，并核对引用相应文件名的使用方。路径使用相对 `$PYPTO_DEVKIT_DIR` 的 `pro_ops/` 前缀。各条目的接口与行为仍须在匹配目标版本的原文件中核实。

| # | 算子名称 | 缓存相对路径 | 类型 | 描述 |
|---|---------|-------------|------|------|
| 1 | add | `pro_ops/element_wise/test_add.py` | elementwise | |
| 2 | matmul_8k_example | `pro_ops/matmul/test_matmul_8k_example.py` | matmul 入门 | |
| 3 | matmul_perf_asw_4k | `pro_ops/matmul/test_matmul_perf_asw_4k_dn_move_offset.py` | matmul 性能 | |
| 4 | matmul_perf_asw_8k_k128 | `pro_ops/matmul/test_matmul_perf_asw_8k_k128_dn_move_offset.py` | matmul 性能 | |
| 5 | matmul_perf_asw_4k_dynamic | `pro_ops/matmul/test_matmul_perf_asw_4k_dn_move_offset_dynamic.py` | matmul 性能（动态轴） | |
| 6 | matmul_perf_asw_8k_k128_dynamic | `pro_ops/matmul/test_matmul_perf_asw_8k_k128_dn_move_offset_dynamic.py` | matmul 性能（动态轴） | |
| 7 | fa_perf_tkv_preload | `pro_ops/fa/test_fa_perf_tkv_preload_dn_vf_bufid_dynrank.py` | FlashAttention 生产级 | |
| 8 | fa_tilingkey_attn_mask | `pro_ops/fa/test_fa_tilingkey_attn_mask.py` | FlashAttention 教学 | |
| 9 | fa_with_mask | `pro_ops/fa/test_fa_with_mask.py` | FlashAttention 性能（mask + NBuffer + auto_mutex） | |
| 10 | flex_attention | `pro_ops/fa/test_flex_attention.py` | FlexAttention 性能 | |
| 11 | quant_lightning_indexer_vf | `pro_ops/lightning_indexer/test_quant_lightning_indexer_vf.py` | VF TopK | |
| 12 | layernorm_tile_group_vf | `pro_ops/vf_api/test_layernorm_tile_group_vf.py` | VF LayerNorm | |
| 13 | softmax_tile_group_vf | `pro_ops/vf_api/test_softmax_tile_group_vf.py` | VF Softmax | |
