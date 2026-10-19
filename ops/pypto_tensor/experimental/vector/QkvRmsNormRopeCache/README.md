# qkv_rms_norm_rope_cache


## 产品支持情况

- Ascend 950PR：不支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 算子说明

`qkv_rms_norm_rope_cache` 是 PyPTO 自定义融合算子，完成：

```text
SplitVD -> RMSNorm(Q/K) -> RoPE(Q/K) -> Q 输出
                               └-> K/V PA_NZ cache 更新
```

当前实现包含两类路径：

- TP4/TP1 INT8 quant cache 快路径：覆盖既有网络性能用例，K/V 在 kernel 内完成对称量化后写回 INT8 PA_NZ cache。
- Generic fallback：覆盖超出 TP4/TP1 head 规模的功能验证规格，Q/K/V 分组计算，K/V 通过 `index` 写入 INT8 PA_NZ cache。

## 接口

```python
from qkv_rms_norm_rope_cache_impl import qkv_rms_norm_rope_cache_wrapper

q_out, k_cache, v_cache = qkv_rms_norm_rope_cache_wrapper(
    qkv,
    q_gamma,
    k_gamma,
    cos,
    sin,
    index,
    q_out,
    k_cache,
    v_cache,
    k_scale=None,
    v_scale=None,
    k_offset=None,
    v_offset=None,
    qkv_size=[B, S, Nqkv, D],
    head_nums=[Nq, Nk, Nv],
    epsilon=1e-6,
    cache_mode="PA_NZ",
    is_output_qkv=False,
)
```

## 输入输出规格

| 名称 | shape | dtype | 说明 |
| --- | --- | --- | --- |
| `qkv` | `[T, Nqkv * D]` | `torch.bfloat16` | QKV 融合输入 |
| `q_gamma` | `[D]` | `torch.bfloat16` | Q RMSNorm gamma |
| `k_gamma` | `[D]` | `torch.bfloat16` | K RMSNorm gamma |
| `cos` | `[T, D]` | `torch.bfloat16` | RoPE cos |
| `sin` | `[T, D]` | `torch.bfloat16` | RoPE sin |
| `index` | `[T]` | `torch.int64` | cache slot |
| `q_out` | `[T, Nq * D]` | `torch.bfloat16` | Q 输出 buffer |
| `k_cache` | `[BlockNum, Nk * D / C0, BlockSize, C0]` | `torch.int8` | K PA_NZ cache，当前目标路径 |
| `v_cache` | `[BlockNum, Nv * D / C0, BlockSize, C0]` | `torch.int8` | V PA_NZ cache，当前目标路径 |
| `k_scale` | `[Nk, D]` | `torch.float32` | K cache 量化 scale |
| `v_scale` | `[Nv, D]` | `torch.float32` | V cache 量化 scale |

`k_offset/v_offset` 当前只接受 `None`。非 `None` 会报 `NotImplementedError`，因为当前只实现对称量化。

## 动态与静态轴

当前 JIT 签名只把 token 轴设为动态：

```text
qkv:   [DYNAMIC, STATIC]
cos:   [DYNAMIC, STATIC]
sin:   [DYNAMIC, STATIC]
index: [DYNAMIC]
q_out: [DYNAMIC, STATIC]
```

其他维度静态编译，包括 `D`、head 数、cache layout、scale shape。

## 当前验证用例

| Case | qkv | qkv_size | head_nums | q_out | k/v_cache | k/v_scale |
| --- | --- | --- | --- | --- | --- | --- |
| `mtp2_tp4_network_quant` | `[48, 2304]` | `[16, 3, 18, 128]` | `[16, 1, 1]` | `[48, 2048]` | `[11898, 4, 128, 32]` | `[1, 128]` |
| `mtp2_tp1_network_quant` | `[12, 9216]` | `[4, 3, 72, 128]` | `[64, 4, 4]` | `[12, 8192]` | `[11898, 16, 128, 32]` | `[4, 128]` |
| `pypto_qkv_rms_norm_rope_cache_id15` | `[2, 49152]` | `[1, 2, 384, 128]` | `[128, 128, 128]` | `[2, 16384]` | `[1, 512, 128, 32]` | `[128, 128]` |
| `pypto_qkv_rms_norm_rope_cache_id15_indexed` | `[2, 49152]` | `[1, 2, 384, 128]` | `[128, 128, 128]` | `[2, 16384]` | `[2, 512, 128, 32]` | `[128, 128]` |

共同属性：

```text
gamma_q=[128]
gamma_k=[128]
cos/sin=[T,128]
index=[T]
epsilon=1e-6
cache_mode=PA_NZ
is_output_qkv=False
k_offset=None
v_offset=None
```

## INT8 量化语义

INT8 cache 分支在 kernel 内完成：

```text
K: Split -> RMSNorm -> RoPE -> round(K / k_scale) -> saturate int8 -> PA_NZ cache
V: Split -> round(V / v_scale) -> saturate int8 -> PA_NZ cache
```

当前实现使用 `CAST_RINT` 做四舍五入，并开启饱和写入到 int8。

## 当前限制

- 仅支持 `cache_mode="PA_NZ"`。
- `is_output_qkv=True` 未实现。
- `k_offset/v_offset` 非 `None` 未实现。
- 既有 TP4/TP1 快路径仍针对当前网络性能用例采用 page0 连续写入：测试入口生成 `index=torch.arange(T)`，且 `T <= block_size`。
- Generic fallback 支持基于 `index` 的 INT8 PA_NZ cache 写入，已用非连续 `index=[129, 3]` 验证。该路径以功能兜底为目标，不作为既有性能 case 的默认路径。
- 已验证过 AscendC-like 动态 offset `assemble` 方案：小规模 proof 可编译并上板正确，但迁入 QKV generic 后泳道图显示前三条用例劣化，仅 indexed fallback 受益。因此当前不替换既有路径，只保留为后续非连续 index 专用兜底模板候选。
- 当前重点覆盖 `D=128`、`C0=32`、`BlockSize=128`。

## 验证

直接使用 Python 测试入口，不依赖 `run_npu.sh`：

```bash
source /mnt/workspace/gitCode/cann/pypto/env_setup.sh
cd /mnt/workspace/zhangsr/pypto/custom/qkv_rms_norm_rope_cache
env -u ASCEND_VISIBLE_DEVICES -u NPU_VISIBLE_DEVICES -u NPU-VISIBLE-DEVICES \
  HOME=/tmp/pypto-home \
  ASCEND_PROCESS_LOG_PATH=/tmp/ascend_plog \
  ASCEND_GLOBAL_LOG_LEVEL=3 \
  PYTHONPATH=/tmp/pypto-wheel:${PYTHONPATH} \
  TILE_FWK_DEVICE_ID=0 \
  /opt/buildtools/Python-3.11.4/bin/python3 test_qkv_rms_norm_rope_cache.py --run-mode npu
```

性能 benchmark：

```bash
env -u ASCEND_VISIBLE_DEVICES -u NPU_VISIBLE_DEVICES -u NPU-VISIBLE-DEVICES \
  HOME=/tmp/pypto-home \
  ASCEND_PROCESS_LOG_PATH=/tmp/ascend_plog \
  ASCEND_GLOBAL_LOG_LEVEL=3 \
  PYTHONPATH=/tmp/pypto-wheel:${PYTHONPATH} \
  TILE_FWK_DEVICE_ID=0 \
  /opt/buildtools/Python-3.11.4/bin/python3 test_qkv_rms_norm_rope_cache.py --run-mode npu --benchmark --warmup 3 --repeat 30
```
