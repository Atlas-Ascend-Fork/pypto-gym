# IndexerPrologueK 结果与性能（ACLNN 交付）

> 实现：`op_kernel/indexer_prologue_k.py`、`indexer_prologue_k_projection.py`、
> `indexer_prologue_k_postprocess.py`；测试：`tests/test_indexer_prologue_k.py`。
> 平台：Ascend950PR（28 Cube / 56 Vector）；CANN 9.3.0。

## 1. 交付验证

### 1.1 测试

| 测试 | 结果 |
|---|---|
| ACLNN | `60 passed` |
| JIT | `60 passed` |
| 融合 / 两阶段一致性 | `test_fused_matches_two_kernel_dispatch`（T=4096）通过 |

测试覆盖：

- T 边界（1/15/16/17）、投影 M 尾块与 `base_n`/`base_k` 回退；
- RoPE 短尾/全尾、小 head dim；
- 重复 index、乱序 index、全 `-1` 跳过、未命中哨兵；
- mode 0 / mode 1 组合行、`k_scale_cache` 有无；
- fused 与 Projection + Postprocess 两条路径逐字节一致。

### 1.2 D 覆盖

- `D=512` 在 ACLNN/JIT 交付测试中覆盖；
- `D=1024/2048` 为本地手测（T=16）：ACLNN 与 CPU golden 逐位一致，不进入 CI；
- `DimCap` 三档：`512/1024/2048`，`D` 落在档内时按档位 UB 布局执行。

### 1.3 下发形态

- `T <= 2048`：`aclnnIndexerPrologueK` 融合；
- `T > 2048`：`aclnnIndexerPrologueKProjection` + `aclnnIndexerPrologueKPostprocess`；
- 测试桥按 T 自动路由，两条路径输出逐字节一致。

## 2. 性能采集

口径：ACLNN，`msprof op_summary` 的 kernel Task Duration；3 次 warmup + 20 次采集，取中位。
“修复前/后”指 D>1024 的 `DimCap` UB 布局修复；D512 布局未变。
D512 用例为模型规格 `H=7168, Dr=510`；D1024/2048 用例 `Dr=64`。

| 用例 | 修复前 µs | 修复后 µs | 变化 |
|---|---:|---:|---:|
| D512 T64 | 22.744 | 22.584 | -0.7% |
| D512 T512 | 104.013 | 107.683 | +3.5% |
| D512 T2048 | 428.232 | 396.319 | -7.5% |
| D512 T4096 Projection | 612.744 | 613.300 | +0.1% |
| D512 T4096 Postprocess | 185.219 | 184.364 | -0.5% |
| D1024 T512 | 181.899 | 180.984 | -0.5% |
| D2048 T512 | 445.470 | 436.309 | -2.1% |

说明：

- D512 的 UB 地址布局和 `DYN_UB_SIZE` 未改，T512 +3.5% 与 T2048 -7.5% 方向相反，
  属于采集波动；
- D1024/2048 使用独立布局，修复后没有性能回退，D2048 约快 2%；
- T4096 两阶段路径的 Projection/Postprocess 均稳定。

## 3. 构建与测试命令

```bash
python3 build_ci.py --pkg-type run --ops indexer_prologue_k --soc ascend950 --no-torch-bindings
```

ACLNN：

```bash
export ASCEND_CUSTOM_OPP_PATH=$PWD/build/ascend950/_CPack_Packages/makeself_staging/packages/vendors/pypto_gym
export LD_LIBRARY_PATH=$ASCEND_CUSTOM_OPP_PATH/op_api/lib:$LD_LIBRARY_PATH
python3 -m pytest ops/pro/deepseek_v4_1/indexer_prologue_k/tests \
  --mode aclnn --aclnn-lib-dir build/ascend950/tests -m 'not jit_only' -q
```

JIT：

```bash
python3 -m pytest ops/pro/deepseek_v4_1/indexer_prologue_k/tests --mode jit -q
```

## 4. JIT 参考

原始 JIT 实现保留在 `op_kernel/indexer_prologue_k_jit.py`，其 `T <= 2048` 融合 /
`T > 2048` 两阶段的分核和数值语义与 ACLNN 交付一致；JIT 60 例全部通过。详细 JIT
性能参考见原 `pypto_pro` 的 `RESULT.md`。
