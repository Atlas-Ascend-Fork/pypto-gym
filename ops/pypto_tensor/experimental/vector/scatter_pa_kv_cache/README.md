# scatter_pa_kv_cache 算子实现


## 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 算子概述

scatter_pa_kv_cache 算子用于在 Paged Attention 推理场景中更新 KV cache。将当前 step 生成的多个 token 的 key 和 value 数据，按照 slot_mapping 指定的位置，散布到对应的 cache block 中。

### 数学公式

```
key_cache[block_idx, block_offset, :, :] = key[i, :, :]
value_cache[block_idx, block_offset, :, :] = value[i, :, :]

其中：
block_idx = slot_mapping[i] // block_size
block_offset = slot_mapping[i] % block_size
```

### 核心特性

- **动态 shape**：num_tokens 可动态变化（1 ~ 16384）
- **in-place 更新**：key_cache 和 value_cache 原地更新，减少内存拷贝
- **scatter 索引**：按 slot_mapping 索引写入，支持块内偏移计算
- **BFLOAT16 dtype**：仅支持 BF16 数据类型

## 目录结构

```
custom/scatter_pa_kv_cache/
├── SPEC.md                        # 算子规格文档
├── API_REPORT.md                  # API 探索报告
├── DESIGN.md                      # 设计方案文档
├── scatter_pa_kv_cache_golden.py  # Golden 参考实现
├── scatter_pa_kv_cache_impl.py    # PyPTO kernel 实现
├── test_scatter_pa_kv_cache.py    # 测试入口
├── test_cases.json                # 测试用例配置
└── README.md                      # 本文件
```

## 实现说明

### API 映射

使用 `pypto.scatter_update(cache_2d, -2, index_2d, src_2d)` API 进行 KV cache 更新。

- **API 选择理由**：scatter_update 是专为 Paged Attention KV cache 优化的 API
- **dim 参数**：保持默认值 `-2`（沿倒数第二维更新）
- **不支持 broadcast**：src 和 index shape 必须严格匹配

### Tiling 策略

- **算子类型**：Vector（scatter_update 是索引更新操作）
- **TileShape 配置**：`pypto.set_vec_tile_shapes(tile_tokens, kv_dim)`
  - tile_tokens = 8（每次 scatter 更新 8 个 token）
  - kv_dim = 512（num_heads * head_size = 2 * 256）
- **尾轴约束**：kv_dim = 512 > 16（BF16 对齐要求）

### Loop 结构

- **动态轴处理**：num_tokens 标为 `pypto.DYNAMIC`
- **Loop 模式**：使用 `pypto.loop(num_tokens_loop)` 遍历 num_tokens
- **尾块处理**：使用 `valid_shape` 处理最后一个 tile 的动态边界

### 数据流

```
4D key_cache [num_blocks, block_size, num_heads, head_size]
    ↓ reshape(inplace=True)
2D key_cache_2d [num_blocks * block_size, num_heads * head_size]
    ↓ scatter_update(dim=-2, index, src)
2D key_cache_2d_result [num_blocks * block_size, num_heads * head_size]
    ↓ .move()
4D key_cache [num_blocks, block_size, num_heads, head_size]（原地更新）
```

类似地，value 和 value_cache 进行相同操作。

## 运行方式

### 环境准备

```bash
# 设置空闲 NPU device ID
export TILE_FWK_DEVICE_ID=0

# 编译 PyPTO（如未安装）
python3 build_ci.py -f python3 --disable_auto_execute
```

### 执行测试

```bash
# 运行所有测试用例（默认 NPU 模式）
python test_scatter_pa_kv_cache.py

# 运行单个测试用例
python test_scatter_pa_kv_cache.py config1_performance_p0

# 列出所有测试用例
python test_scatter_pa_kv_cache.py --list

# 使用 sim 模式（无 NPU 环境时）
python test_scatter_pa_kv_cache.py --run_mode sim
```

### 测试用例

| 用例 ID | 描述 | num_tokens | block_size | num_heads | head_size |
|---------|------|------------|------------|-----------|-----------|
| config1_performance_p0 | 适中序列长度性能测试 | 2633 | 128 | 2 | 256 |
| config2_function_p0 | 较长序列功能验证 | 7902 | 128 | 2 | 256 |
| config3_boundary_p0 | 最大序列长度边界测试 | 16384 | 128 | 2 | 256 |

## 精度验证

- **精度标准**：atol = 0.0001, rtol = 0.0078125（BFLOAT16）
- **验证方式**：使用 `scatter_pa_kv_cache_golden.py` 作为参考实现
- **对比方法**：numpy.testing.assert_allclose
- **三态标记**：
  - `[PRECISION_PASS]`：精度验证通过
  - `[PRECISION_FAIL]`：精度验证失败（数值不匹配）

## 已知限制

1. **不支持压缩特性**：P0 版本暂不支持 compress_lens_optional、compress_seq_offset_optional、seq_lens_optional 参数
2. **单一 dtype**：仅支持 BFLOAT16
3. **固定 shape**：num_heads=2, head_size=256, block_size=128 为编译期常量（kernel 注解中）
4. **动态轴限制**：num_blocks 在 kernel 注解中为静态值，但实际支持动态输入

## 参考资源

- **Golden 参考**：`examples/scatter_pa_kv_cache.py`（100% 匹配）
- **生产级实现**：`models/glm_v4_5/glm_attention_fusion.py`（Line 461-462）
- **API 文档**：`docs/api/operation/pypto-scatter_update.md`
- **执行约束**：`../../../../../../../cannbot-skills/ops/pypto-op-develop/references/execution-constraints.md`

## 性能目标

- **目标**：首跑精度成功性能的 2 倍
- **优化点**：
  - Tiling 配置优化（tile_tokens 可调整）
  - Loop unroll 优化（可添加 unroll_list）
  - UB 内存利用率优化

---

**生成时间**: 2026-05-13  
**实现状态**: Stage 5 首次实现  
**置信度**: 高（基于详细设计方案和多个生产级参考实现）