# flash_kda_cp_preprocess 结果说明

## 用例

全部输入使用 D=128，激活和 beta 为 BF16，a_log 和 dt_bias 为 FP32。表中的 S/T 表示稠密序列长度或打包 token 总数。除标明两个序列的打包用例外，B=1。21 项单次测试覆盖三种布局、分组头、短尾、打包序列、宽头及长序。8 项一致性测试各运行一次作为对照，再使用同一输入和调度表追加 162 次复跑，共调用 170 次。

| 测试 | 布局 | Hq/Hv | S/T | cu_seqlens | 调用数 |
|---|---|---:|---:|---|---:|
| `test_aligned` | BNSD | 1/1 | 64 | — | 1 |
| `test_tail` | BNSD | 1/3 | 17 | — | 1 |
| `test_tail_gqa_two_chunks` | BNSD | 2/4 | 65 | — | 1 |
| `test_bsnd` | BSND | 1/5 | 49 | — | 1 |
| `test_bsnd_wide_heads` | BSND | 1/48 | 65 | — | 1 |
| `test_single_token` | BNSD | 1/2 | 1 | — | 1 |
| `test_tnd` | TND | 3/6 | 193 | [0, 193] | 1 |
| `test_tnd_wide_heads` | TND | 2/40 | 193 | [0, 193] | 1 |
| `test_wide96_heads` | BNSD | 3/96 | 65 | — | 1 |
| `test_wide96_single_chunk_bnsd` | BNSD | 3/96 | 64 | — | 1 |
| `test_wide96_single_chunk_bsnd` | BSND | 3/96 | 64 | — | 1 |
| `test_wide96_single_chunk_tnd` | TND | 3/96 | 64 | [0, 64] | 1 |
| `test_packed_tnd` | TND | 1/3 | 82 | [0, 17, 82] | 1 |
| `test_perf_p0` | BNSD | 32/32 | 8192 | — | 1 |
| `test_perf_bsnd_p0` | BSND | 32/32 | 8192 | — | 1 |
| `test_perf_tnd_p0` | TND | 32/32 | 8192 | [0, 8192] | 1 |
| `test_perf_bsnd_6_16384` | BSND | 6/6 | 16384 | — | 1 |
| `test_perf_bsnd_12_16384` | BSND | 12/12 | 16384 | — | 1 |
| `test_perf_bsnd_96_1024` | BSND | 96/96 | 1024 | — | 1 |
| `test_readme_bnsd_5_577` | BNSD | 5/5 | 577 | — | 1 |
| `test_readme_bnsd_1_1024` | BNSD | 1/1 | 1024 | — | 1 |
| `test_repeat_bnsd_5_5_577` | BNSD | 5/5 | 577 | — | 13 |
| `test_repeat_bnsd_2_4_65` | BNSD | 2/4 | 65 | — | 13 |
| `test_repeat_bnsd_8_16_193` | BNSD | 8/16 | 193 | — | 13 |
| `test_repeat_tnd_3_6_193` | TND | 3/6 | 193 | [0, 193] | 13 |
| `test_repeat_bsnd_6_6_16384` | BSND | 6/6 | 16384 | — | 61 |
| `test_repeat_bsnd_12_12_16384` | BSND | 12/12 | 16384 | — | 31 |
| `test_repeat_bnsd_32_32_8192` | BNSD | 32/32 | 8192 | — | 13 |
| `test_repeat_tnd_32_32_8192` | TND | 32/32 | 8192 | [0, 8192] | 13 |

在算子目录下，两种入口均通过 29 项测试。

```bash
python -m pytest -q tests/test_flash_kda_cp_preprocess.py
python tests/test_flash_kda_cp_preprocess.py
```

## 精度标准

单次测试将八路输出与独立 CPU golden 对照，检查形状、dtype、连续存储和有限性。有效元素的归一化误差 `|actual - golden| / (0.005 + 0.005 × |golden|)` 要求 99.9% 分位不超过 1。单路元素超过 1000 万时，分位数用固定步长抽样。逐块输出还使用仓内 precision 工具的匹配率和最大绝对误差判据。复跑测试逐次对照两项摘要的 golden，并要求各次摘要逐元素完全一致。打包用例另校验未用 decay 槽为零。

## 性能结果

### 绝对性能结果

下表列出 12 个规格在 Ascend 950PR（32 AIC）、CANN 9.3.0、PyPTO 72695091 的实测。每次测量前用 1 GiB 缓冲驱逐 L2 数据。两组反序采样，每组预热 3 次、采样 12 次，取主 kernel 的 24 个 `msprof` Task Duration 中位数。所有输入 B=1，D=128，TND 的 cu_seqlens 为 [0,T]，scale=128⁻⁰·⁵，lower_bound=-0.1。BNSD、BSND、TND 的冷、热入口共跟踪 9 次，每次只记录到一个 MIX 任务。

| 布局 | Hq/Hv | S/T | 设备时间 (µs) |
|---|---:|---:|---:|
| BNSD | 1/1 | 1024 | 50.96 |
| BNSD | 32/32 | 8192 | 2395.56 |
| BNSD | 5/5 | 577 | 64.32 |
| BSND | 1/48 | 65 | 66.33 |
| BSND | 1/5 | 49 | 24.18 |
| BSND | 12/12 | 16384 | 2097.33 |
| BSND | 32/32 | 8192 | 2400.66 |
| BSND | 6/6 | 16384 | 1087.87 |
| BSND | 96/96 | 1024 | 916.50 |
| TND | 2/40 | 193 | 125.07 |
| TND | 3/96 | 577 | 552.48 |
| TND | 32/32 | 8192 | 2420.57 |

### 理论评估对比

摘要行拆增加核数利用率，同时增加 φ 的读流量和核间交接。单块少头路径保留每头完整摘要，省掉重复的摘要准备。TND 单行片尾块用实际 K 直接读入 Cube，省掉 Vector 中转和相关等待。
