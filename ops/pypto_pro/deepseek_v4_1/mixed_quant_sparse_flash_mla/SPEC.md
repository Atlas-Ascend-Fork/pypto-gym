# SPEC — mixed_quant_sparse_flash_mla (PyPTO-Pro, Scriptor)

## 0. 任务身份

- op：`mixed_quant_sparse_flash_mla`（MQSMLA，混合量化稀疏 Flash MLA）
- 目标硬件：Ascend 950PR（Ascend950PR_9579，28 Cube / 56 Vector；UB/L1/L0 容量以安装平台文件为准）
- 参考实现：同功能的外部稀疏 MLA 实现（移植版）
- 模式：Scriptor；`delivery_sync_mode=auto_mutex`；单 `@pl.jit(auto_mutex=True)` kernel，CV 融合。
- **用户指令**：metadata（分核规划）在 host 侧 Python 实现，作为独立公开入口
  `mixed_quant_sparse_flash_mla_metadata`（对应参考实现的独立 host metadata 规划器；
  本机 28 核下参考实现同样全部走 uniform 整行规划，FD 字段为 0）。主 kernel wrapper
  只接收 `metadata` 张量（int32[1024]），不读取任何输入张量值。
- kernel 始终计算并输出 LSE（[1,T1,64]）；公开 wrapper 的 `return_softmax_lse=False`
  语义 = 内部分配并丢弃 lse（与参考"不返回"语义一致；偏差：kernel 多一次小 GM 写）。
- 索引/长度张量在 kernel 合同中必填；公开 wrapper 允许省略 topk length（省略 = 全 K
  列有效，数学等价，wrapper 传存在性标志 0 与占位张量，kernel 不读其值）。

## 1. 公开接口

```python
def mixed_quant_sparse_flash_mla_metadata(ori_topk_length, cmp_topk_length, *,
        cu_seqlens_q=None, num_heads_q=64, num_heads_kv=1, head_dim=512,
        quant_mode=1, has_cmp_kv=True, block_num=None) -> torch.Tensor  # int32[1024]
def mixed_quant_sparse_flash_mla(q, *, ori_kv, ori_sparse_indices,
        ori_block_table, cu_seqlens_q, sinks, metadata, quant_mode=1,
        softmax_scale=None, cmp_kv=None, cmp_sparse_indices=None,
        cmp_block_table=None, ori_topk_length=None, cmp_topk_length=None,
        seqused_q=None, seqused_ori_kv=None, seqused_cmp_kv=None,
        layout_q="TND", layout_kv="PA_BBND", return_softmax_lse=False,
        out=None, lse=None) -> None
```

- 主 wrapper host 行为仅限：参数校验（shape/dtype/stride 元数据）、`torch.empty`
  分配合同输出（out/lse 未预分配时）、派生 Python 标量（softmax_scale 默认
  1/sqrt(512)、kv 页距 stride(0)、cmp/长度存在标志）并启动一次 kernel。
- `seqused_*` 仅校验类型/形状（与参考一致）。quant_mode 固定 1；layout 固定。
- host metadata 入口读取 cu_q/length 张量值（D2H）做 uniform 整行规划（移植参考
  `_md_uniform`），不属于主 kernel wrapper 边界，REPORT 如实说明。

## 2. 数学定义

对每个 query token t、query head h：按稀疏索引经 block table 映射收集两侧有效列的
反量化 KV（ORI 列在前、CMP 列在后，保持索引顺序）合并为 X，令
z_j = softmax_scale·q_t^T X_j，则

Z = exp(sinks_h) + Σ_j exp(z_j)；O = Σ_j exp(z_j)·X_j / Z；LSE = log Z。

- sink 只进分母不贡献 Value；两侧有效长度均为 0 时输出严格为 0、LSE 严格等于 sink。
- 反量化：ORI = FP8 E4M3FN group32（每 token 512 数据字节 + 16 个 BF16 scale =
  544B）；CMP = FP4 E2M1 group16（256 打包字节 + 32 个 BF16 scale = 320B，偶数
  位置在低 nibble）。反量化值 = 量化值(FP32)×scale(BF16→FP32) 后舍入 BF16。
- 在线 softmax 按 128-token tile、跨 ORI/CMP 连续推进；P 舍入 BF16 后 PV，
  分子分母 FP32 累加；每 tile 先以 alpha=exp(m_old−m_new) 缩放历史 O。
- 每 tile 无效列（≥ 有效前缀）地址 clamp 到该行最后有效索引；softmax 排除无效列、
  P 无效列写 0；PV 的 K 维扩展到 16 的倍数。
- KV 页地址：idx→page=idx//bs, off=idx%bs, phys=block_table[b,page]，字节地址 =
  phys×kv.stride(0)+off×row_bytes（支持页间 padding 的 axis-0 非连续）。

```json machine-contract
{
  "schema_version": 1,
  "op_name": "mixed_quant_sparse_flash_mla",
  "formula": "O[t,h,:] = sum_j exp(softmax_scale*q[t,h,:]·X_j - m_h)*BF16(X_j) / (exp(sinks_h-m_h) + sum_j exp(z_j-m_h)), LSE[t,h] = log(exp(sinks_h)+sum_j exp(z_j)), X = dequant(ori_kv[indices]) ++ dequant(cmp_kv[indices]) over valid prefix columns, m_h = running max incl. sinks_h; empty rows give O=0 and LSE=sinks_h",
  "supported_dtypes": ["bfloat16", "uint8", "int32", "float32"],
  "inputs": [
    {"name": "q", "shape": ["T1", 64, 512], "dtype": "bfloat16", "value_range": [-10, 10]},
    {"name": "ori_kv", "shape": ["ORI_PAGES", "ORI_BS", 1, 544], "dtype": "uint8", "value_range": [0, 255]},
    {"name": "ori_sparse_indices", "shape": ["T1", 1, "K1"], "dtype": "int32", "value_range": [-1, 131072]},
    {"name": "ori_block_table", "shape": ["B", "ORI_BT"], "dtype": "int32", "value_range": [0, 1000000]},
    {"name": "ori_topk_length", "shape": ["T1", 1], "dtype": "int32", "value_range": [0, 4096]},
    {"name": "sinks", "shape": [64], "dtype": "float32", "value_range": [-10, 10]},
    {"name": "metadata", "shape": [1024], "dtype": "int32", "value_range": [0, 2147483647]},
    {"name": "cmp_kv", "shape": ["CMP_PAGES", "CMP_BS", 1, 320], "dtype": "uint8", "value_range": [0, 255]},
    {"name": "cmp_sparse_indices", "shape": ["T1", 1, "K2"], "dtype": "int32", "value_range": [-1, 65536]},
    {"name": "cmp_block_table", "shape": ["B", "CMP_BT"], "dtype": "int32", "value_range": [0, 1000000]},
    {"name": "cmp_topk_length", "shape": ["T1", 1], "dtype": "int32", "value_range": [0, 4096]},
    {"name": "cu_seqlens_q", "shape": ["B + 1"], "dtype": "int32", "value_range": [0, 2147483647]}
  ],
  "outputs": [
    {"name": "out", "shape": ["T1", 64, 512], "dtype": "bfloat16", "value_range": [-100, 100]},
    {"name": "lse", "shape": [1, "T1", 64], "dtype": "float32", "value_range": [-200, 200]}
  ],
  "default_params": {
    "s_ori": 257, "s_cmp": 193, "k_ori": 128, "k_cmp": 129,
    "bs_ori": 128, "bs_cmp": 128, "has_cmp": true, "vary_lengths": false,
    "zero_ori": false, "full_lengths": false, "kv_axis0_noncontiguous": false,
    "softmax_scale": 0.04419417382415922
  },
  "tolerance": {"policy": "pro_scheme_a"},
  "dynamic_axes_ranges": {
    "T1": [1, 4096], "B": [1, 1024], "K1": [1, 4096], "K2": [1, 4096],
    "ORI_PAGES": [1, 65536], "ORI_BS": [1, 1024], "ORI_BT": [1, 1024],
    "CMP_PAGES": [1, 65536], "CMP_BS": [1, 1024], "CMP_BT": [1, 1024]
  },
  "shape_constraints": [
    "cu_seqlens_q is a valid int32 prefix sum: cu[0]==0, non-decreasing, cu[-1]==T1",
    "per-row ori_topk_length in [0, K1] and cmp_topk_length in [0, K2]",
    "valid sparse indices in [0, s_side) of the owning batch; indices outside the valid per-row prefix are ignored (may be -1)",
    "ori_kv/cmp_kv axis-0 stride >= ORI_BS*544 / CMP_BS*320 (page padding allowed), inner dims contiguous",
    "block_table maps logical pages to physical pages in [0, ORI_PAGES) / [0, CMP_PAGES)",
    "metadata is the int32[1024] output of mixed_quant_sparse_flash_mla_metadata with matching block count"
  ],
  "p0_cases": [
    {
      "name": "mixed_decode",
      "params": {"s_ori": 257, "s_cmp": 193, "k_ori": 128, "k_cmp": 129, "bs_ori": 128, "bs_cmp": 128, "has_cmp": true, "vary_lengths": false, "zero_ori": false, "full_lengths": false, "kv_axis0_noncontiguous": false, "softmax_scale": 0.04419417382415922},
      "input_shapes": {"q": [2, 64, 512], "ori_kv": [6, 128, 1, 544], "ori_sparse_indices": [2, 1, 128], "ori_block_table": [2, 3], "ori_topk_length": [2, 1], "sinks": [64], "metadata": [1024], "cmp_kv": [4, 128, 1, 320], "cmp_sparse_indices": [2, 1, 129], "cmp_block_table": [2, 2], "cmp_topk_length": [2, 1], "cu_seqlens_q": [3]},
      "output_shapes": {"out": [2, 64, 512], "lse": [1, 2, 64]}
    },
    {
      "name": "ori_only",
      "params": {"s_ori": 257, "s_cmp": 193, "k_ori": 128, "k_cmp": 129, "bs_ori": 128, "bs_cmp": 128, "has_cmp": false, "vary_lengths": false, "zero_ori": false, "full_lengths": false, "kv_axis0_noncontiguous": false, "softmax_scale": 0.04419417382415922},
      "input_shapes": {"q": [2, 64, 512], "ori_kv": [6, 128, 1, 544], "ori_sparse_indices": [2, 1, 128], "ori_block_table": [2, 3], "ori_topk_length": [2, 1], "sinks": [64], "metadata": [1024], "cmp_kv": [1, 1, 1, 320], "cmp_sparse_indices": [2, 1, 1], "cmp_block_table": [2, 1], "cmp_topk_length": [2, 1], "cu_seqlens_q": [3]},
      "output_shapes": {"out": [2, 64, 512], "lse": [1, 2, 64]}
    },
    {
      "name": "empty_and_tail",
      "params": {"s_ori": 257, "s_cmp": 193, "k_ori": 128, "k_cmp": 129, "bs_ori": 128, "bs_cmp": 128, "has_cmp": true, "vary_lengths": true, "zero_ori": false, "full_lengths": false, "kv_axis0_noncontiguous": false, "softmax_scale": 0.04419417382415922},
      "input_shapes": {"q": [8, 64, 512], "ori_kv": [9, 128, 1, 544], "ori_sparse_indices": [8, 1, 128], "ori_block_table": [3, 3], "ori_topk_length": [8, 1], "sinks": [64], "metadata": [1024], "cmp_kv": [6, 128, 1, 320], "cmp_sparse_indices": [8, 1, 129], "cmp_block_table": [3, 2], "cmp_topk_length": [8, 1], "cu_seqlens_q": [4]},
      "output_shapes": {"out": [8, 64, 512], "lse": [1, 8, 64]}
    },
    {
      "name": "padded_pages",
      "params": {"s_ori": 257, "s_cmp": 193, "k_ori": 128, "k_cmp": 129, "bs_ori": 64, "bs_cmp": 32, "has_cmp": true, "vary_lengths": false, "zero_ori": false, "full_lengths": false, "kv_axis0_noncontiguous": true, "softmax_scale": 0.04419417382415922},
      "input_shapes": {"q": [5, 64, 512], "ori_kv": [10, 64, 1, 544], "ori_sparse_indices": [5, 1, 128], "ori_block_table": [2, 5], "ori_topk_length": [5, 1], "sinks": [64], "metadata": [1024], "cmp_kv": [14, 32, 1, 320], "cmp_sparse_indices": [5, 1, 129], "cmp_block_table": [2, 7], "cmp_topk_length": [5, 1], "cu_seqlens_q": [3]},
      "output_shapes": {"out": [5, 64, 512], "lse": [1, 5, 64]}
    },
    {
      "name": "single_token_k",
      "params": {"s_ori": 257, "s_cmp": 193, "k_ori": 1, "k_cmp": 1, "bs_ori": 128, "bs_cmp": 128, "has_cmp": true, "vary_lengths": true, "zero_ori": false, "full_lengths": false, "kv_axis0_noncontiguous": false, "softmax_scale": 0.04419417382415922},
      "input_shapes": {"q": [4, 64, 512], "ori_kv": [3, 128, 1, 544], "ori_sparse_indices": [4, 1, 1], "ori_block_table": [1, 3], "ori_topk_length": [4, 1], "sinks": [64], "metadata": [1024], "cmp_kv": [2, 128, 1, 320], "cmp_sparse_indices": [4, 1, 1], "cmp_block_table": [1, 2], "cmp_topk_length": [4, 1], "cu_seqlens_q": [2]},
      "output_shapes": {"out": [4, 64, 512], "lse": [1, 4, 64]}
    },
    {
      "name": "narrow_k_tail",
      "params": {"s_ori": 257, "s_cmp": 193, "k_ori": 17, "k_cmp": 65, "bs_ori": 128, "bs_cmp": 128, "has_cmp": true, "vary_lengths": true, "zero_ori": false, "full_lengths": false, "kv_axis0_noncontiguous": false, "softmax_scale": 0.04419417382415922},
      "input_shapes": {"q": [4, 64, 512], "ori_kv": [3, 128, 1, 544], "ori_sparse_indices": [4, 1, 17], "ori_block_table": [1, 3], "ori_topk_length": [4, 1], "sinks": [64], "metadata": [1024], "cmp_kv": [2, 128, 1, 320], "cmp_sparse_indices": [4, 1, 65], "cmp_block_table": [1, 2], "cmp_topk_length": [4, 1], "cu_seqlens_q": [2]},
      "output_shapes": {"out": [4, 64, 512], "lse": [1, 4, 64]}
    },
    {
      "name": "cmp_only_rows",
      "params": {"s_ori": 257, "s_cmp": 193, "k_ori": 128, "k_cmp": 129, "bs_ori": 128, "bs_cmp": 128, "has_cmp": true, "vary_lengths": true, "zero_ori": true, "full_lengths": false, "kv_axis0_noncontiguous": false, "softmax_scale": 0.04419417382415922},
      "input_shapes": {"q": [3, 64, 512], "ori_kv": [3, 128, 1, 544], "ori_sparse_indices": [3, 1, 128], "ori_block_table": [1, 3], "ori_topk_length": [3, 1], "sinks": [64], "metadata": [1024], "cmp_kv": [2, 128, 1, 320], "cmp_sparse_indices": [3, 1, 129], "cmp_block_table": [1, 2], "cmp_topk_length": [3, 1], "cu_seqlens_q": [2]},
      "output_shapes": {"out": [3, 64, 512], "lse": [1, 3, 64]}
    },
    {
      "name": "multirow_42",
      "params": {"s_ori": 257, "s_cmp": 193, "k_ori": 128, "k_cmp": 129, "bs_ori": 128, "bs_cmp": 128, "has_cmp": true, "vary_lengths": true, "zero_ori": false, "full_lengths": false, "kv_axis0_noncontiguous": false, "softmax_scale": 0.04419417382415922},
      "input_shapes": {"q": [42, 64, 512], "ori_kv": [3, 128, 1, 544], "ori_sparse_indices": [42, 1, 128], "ori_block_table": [1, 3], "ori_topk_length": [42, 1], "sinks": [64], "metadata": [1024], "cmp_kv": [2, 128, 1, 320], "cmp_sparse_indices": [42, 1, 129], "cmp_block_table": [1, 2], "cmp_topk_length": [42, 1], "cu_seqlens_q": [2]},
      "output_shapes": {"out": [42, 64, 512], "lse": [1, 42, 64]}
    },
    {
      "name": "selective_fd_72",
      "params": {"s_ori": 257, "s_cmp": 769, "k_ori": 128, "k_cmp": 512, "bs_ori": 128, "bs_cmp": 128, "has_cmp": true, "vary_lengths": false, "zero_ori": false, "full_lengths": false, "kv_axis0_noncontiguous": false, "softmax_scale": 0.04419417382415922},
      "input_shapes": {"q": [72, 64, 512], "ori_kv": [36, 128, 1, 544], "ori_sparse_indices": [72, 1, 128], "ori_block_table": [12, 3], "ori_topk_length": [72, 1], "sinks": [64], "metadata": [1024], "cmp_kv": [84, 128, 1, 320], "cmp_sparse_indices": [72, 1, 512], "cmp_block_table": [12, 7], "cmp_topk_length": [72, 1], "cu_seqlens_q": [13]},
      "output_shapes": {"out": [72, 64, 512], "lse": [1, 72, 64]}
    },
    {
      "name": "omit_lengths",
      "params": {"s_ori": 257, "s_cmp": 193, "k_ori": 128, "k_cmp": 129, "bs_ori": 128, "bs_cmp": 128, "has_cmp": true, "vary_lengths": false, "zero_ori": false, "full_lengths": true, "kv_axis0_noncontiguous": false, "softmax_scale": 0.04419417382415922},
      "input_shapes": {"q": [2, 64, 512], "ori_kv": [6, 128, 1, 544], "ori_sparse_indices": [2, 1, 128], "ori_block_table": [2, 3], "ori_topk_length": [2, 1], "sinks": [64], "metadata": [1024], "cmp_kv": [4, 128, 1, 320], "cmp_sparse_indices": [2, 1, 129], "cmp_block_table": [2, 2], "cmp_topk_length": [2, 1], "cu_seqlens_q": [3]},
      "output_shapes": {"out": [2, 64, 512], "lse": [1, 2, 64]}
    },
    {
      "name": "decode_1",
      "params": {"s_ori": 32768, "s_cmp": 16384, "k_ori": 128, "k_cmp": 512, "bs_ori": 128, "bs_cmp": 128, "has_cmp": true, "vary_lengths": false, "zero_ori": false, "full_lengths": false, "kv_axis0_noncontiguous": false, "softmax_scale": 0.04419417382415922},
      "input_shapes": {"q": [32, 64, 512], "ori_kv": [8192, 128, 1, 544], "ori_sparse_indices": [32, 1, 128], "ori_block_table": [32, 256], "ori_topk_length": [32, 1], "sinks": [64], "metadata": [1024], "cmp_kv": [4096, 128, 1, 320], "cmp_sparse_indices": [32, 1, 512], "cmp_block_table": [32, 128], "cmp_topk_length": [32, 1], "cu_seqlens_q": [33]},
      "output_shapes": {"out": [32, 64, 512], "lse": [1, 32, 64]}
    },
    {
      "name": "decode_6",
      "params": {"s_ori": 32768, "s_cmp": 16384, "k_ori": 128, "k_cmp": 512, "bs_ori": 128, "bs_cmp": 128, "has_cmp": true, "vary_lengths": false, "zero_ori": false, "full_lengths": false, "kv_axis0_noncontiguous": false, "softmax_scale": 0.04419417382415922},
      "input_shapes": {"q": [192, 64, 512], "ori_kv": [8192, 128, 1, 544], "ori_sparse_indices": [192, 1, 128], "ori_block_table": [32, 256], "ori_topk_length": [192, 1], "sinks": [64], "metadata": [1024], "cmp_kv": [4096, 128, 1, 320], "cmp_sparse_indices": [192, 1, 512], "cmp_block_table": [32, 128], "cmp_topk_length": [192, 1], "cu_seqlens_q": [33]},
      "output_shapes": {"out": [192, 64, 512], "lse": [1, 192, 64]}
    },
    {
      "name": "decode_6_long_s2",
      "params": {"s_ori": 131072, "s_cmp": 65536, "k_ori": 128, "k_cmp": 512, "bs_ori": 128, "bs_cmp": 128, "has_cmp": true, "vary_lengths": false, "zero_ori": false, "full_lengths": false, "kv_axis0_noncontiguous": false, "softmax_scale": 0.04419417382415922},
      "input_shapes": {"q": [72, 64, 512], "ori_kv": [12288, 128, 1, 544], "ori_sparse_indices": [72, 1, 128], "ori_block_table": [12, 1024], "ori_topk_length": [72, 1], "sinks": [64], "metadata": [1024], "cmp_kv": [6144, 128, 1, 320], "cmp_sparse_indices": [72, 1, 512], "cmp_block_table": [12, 512], "cmp_topk_length": [72, 1], "cu_seqlens_q": [13]},
      "output_shapes": {"out": [72, 64, 512], "lse": [1, 72, 64]}
    }
  ],
  "perf_target": null,
  "exit_criteria": {
    "all": [
      {
        "id": "golden_ratio_all_p0",
        "metric": "golden_reference_ratio",
        "operator": ">=",
        "threshold": 1.0,
        "unit": "x",
        "basis": "hardware",
        "cases": ["mixed_decode", "ori_only", "empty_and_tail", "padded_pages", "single_token_k", "narrow_k_tail", "cmp_only_rows", "multirow_42", "selective_fd_72", "omit_lengths", "decode_1", "decode_6", "decode_6_long_s2"],
        "aggregation": "each",
        "source": "formal golden compare"
      }
    ]
  }
}
```

## 3. 精度判据与附加准出

- `tolerance.policy = pro_scheme_a`：对独立 CPU golden（`*_golden_cpu.py`，从合同
  字节池独立解码反量化并按第 2 节公式计算）做方案 A 比较。
- 附加准出（golden 自验证与逐 case 检查）：
  - LSE `rtol=atol=1e-5`（FP32 直接比较）。
  - 空行（该行两侧有效长度均 0）输出严格为 0，LSE 严格等于 sinks。
  - 输出必须有限。
- 输入无 Inf/NaN（input_special_values 均不适用；无效列以 -1 索引表达，属合法域内）。

## 4. exit_criteria 说明

`golden_reference_ratio >= 1.0×`（逐 P0，硬件口径：冻结基线时延 / PyPTO target-kernel
单次时延）。基线为同机冻结的 median µs。`decode_1`/`decode_6`/`decode_6_long_s2` 为长序列
场景，基线在完全相同 P0 输入与计时协议下取得。这 3 条使用**完整块表**
（`ori_block_table` 宽 256/256/1024，`cmp_block_table` 宽 128/128/512；索引覆盖整个
KV 窗口）。B 项能力差异已补齐：page size `bs` 1..1024、TopK `K` 1..4096、块表宽度
`bt` 1..1024 均由单 tilingkey 内核承载（`bs/k` 为按值特化的 tilingkey 字段，`bt` 为
运行时标量 + 1024 宽 UB 表区），真机证据见
`reports/diagnostics/bcap/{FINDINGS.md,probe_bcap_arb.py,probe_b3.py}`。
优化默认至少 10 有效轮、`stop_on_criteria=false`、无轮数上限；`perf_target=null`。

## 5. 验收范围声明

- 通过标准：13 个 P0 全部真机精度 PASS（pro_scheme_a + 第 3 节附加准出）+
  exit_criteria 达标 + delivery-check PASS。
- 本 SPEC 不声明：任意 T1/B/K/页尺寸组合覆盖（仅 P0 具体 shape）、quant_mode≠1、
  其他 layout、FD 归约执行路径（metadata 契约字段保留，uniform 规划下为 0）、
  确定性 level-3 逐行拷贝模式。
