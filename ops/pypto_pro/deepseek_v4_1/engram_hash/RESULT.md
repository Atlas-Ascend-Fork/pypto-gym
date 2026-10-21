# engram_hash 结果说明（RESULT）

> 实现：`op_kernel/engram_hash.py`
> 用例：`tests/test_engram_hash.py`
> 精度基准（golden）：`tests/assets/engram_hash_golden.py`

## 1. 交付状态

可交付。正确性用例全部逐位通过；性能以 kernel 采集耗时为准，数据与现状见第 4、5 节。

## 2. 用例与覆盖

`tests/test_engram_hash.py` 采用 pytest，标记 `@pytest.mark.soc("950")`；无 NPU 环境下
NPU 用例自动跳过（`make_offsets` 的 CPU 用例始终运行）。用例矩阵与规格如下：

| 用例 | 规格 | 覆盖点 |
|---|---|---|
| `test_make_offsets` | CPU | exclusive prefix-sum，int32 |
| `test_contract_errors` | 8 tokens | rank / dtype / shape / device 契约失败路径 |
| `test_correctness`（24 例） | T∈{1,127,128,129,300,1024} × G∈{3,4} × mask∈{否,是}，L=2、Tb=8 | 尾块、整块、多块、mask |
| `test_correctness_signed_multipliers` | T=300/G=4（mask）、T=128/G=3（无 mask） | 负乘子（向下取整取模分支） |
| `test_mask_dtype_states` | 256 tokens | mask 的 bool / uint8 / None 三种形态 |
| `test_zero_tokens` | T=0 | 空输入 |
| `test_benchmark_spec` | 由 `ENGRAM_HASH_BENCH=1` 启用 | 性能采集（第 4 节） |

## 3. 精度标准

- 输出为整型，与 `tests/assets/engram_hash_golden.py` 使用 `torch.equal` 逐位比较，
  不设容差。
- 同一输入下，输出与 torch 参考实现逐位一致。
- 取模按向下取整语义，对任意 int64 输入成立。
- T=0、尾块、mask 三态、负乘子与非法输入均有独立用例覆盖。

## 4. 性能

### 4.1 采集环境与口径

采集环境（本机规格）：

| 项目 | 规格 |
|---|---|
| 设备 | Ascend950PR_9579，单卡，HBM 128 GB |
| SoC | DAV_3510（arch=a5）；core_num=28，cube_core_num=28，vector_core_num=56 |
| CANN | 9.2.0 |
| PyPTO | pypto_pro 0.2.1（随 CANN 9.2 发布包提供） |
| torch / torch_npu | 2.12.0 / 2.12.0.post2 |
| 主机 | AMD EPYC 9355（128 线程），754 GB 内存 |

采集口径：性能以 **kernel 采集耗时（msprof `Task Duration`）** 为准，条件如下。

- 每次测量前使用 256 MB 缓冲刷新 L2；每个用例 warmup 5 次、测量 10 次，取均值；
  同一配置独立采集两次，表中并列给出。
- 采集用例规格：L=2、Tb=8、mask=False、T∈{512, 8001}、G∈{3,4}。

### 4.2 kernel 耗时

| 用例 | bytes | kernel mean（两次） |
|---|---:|---:|
| T=512, G=3 | 0.072 MB | 12.93 / 13.27 us |
| T=512, G=4 | 0.107 MB | 19.17 / 19.36 us |
| T=8001, G=3 | 1.12 MB | 26.69 / 26.41 us |
| T=8001, G=4 | 1.66 MB | 38.24 / 38.81 us |

两次采集的极差不超过 4%。

### 4.3 有效带宽

按“输入 + 输出字节数 / 内核耗时”计：

| 用例 | 有效带宽 |
|---|---:|
| T=512, G=3 | ≈5.5 GB/s |
| T=512, G=4 | ≈5.6 GB/s |
| T=8001, G=3 | ≈42 GB/s |
| T=8001, G=4 | ≈43 GB/s |

T=512 用例的数据量不足 0.11 MB，固定开销占比高，因此有效带宽偏低；T=8001 用例的
数据量约 1.1-1.7 MB，接近该实现在此形态下的稳态吞吐。

## 5. 管道利用率与访存现状（MFU / MTE）

本算子为整型查表计算，不含矩阵乘与浮点运算，因此不存在 FLOPs 意义的 MFU。可用的
计算侧口径为 profiler 的 SIMT ALU（`aiv_vec` / `aiv_scalar`）占用率，以及整型操作
速率；访存侧口径为 MTE 占用率与带宽。数据来自 4.1 的同口径采集（刷 L2）：

| 用例 | Block Num | aiv_vec | aiv_scalar | aiv_mte2 | aiv_mte3 | cube |
|---|---:|---:|---:|---:|---:|---:|
| T=512, G=3 | 4 | 0.962 | 0.024 | 0.000 | 0.000 | 0.000 |
| T=512, G=4 | 4 | 0.989 | 0.010 | 0.000 | 0.000 | 0.000 |
| T=8001, G=3 | 63 | 0.950 | 0.049 | 0.000 | 0.000 | 0.000 |
| T=8001, G=4 | 63 | 0.925 | 0.075 | 0.000 | 0.000 | 0.000 |

- 计算侧：SIMT ALU 占用率为 0.93-0.99，为该算子当前的计算瓶颈；cube 为 0。
  按标量操作统计，每 token 约 74 次（G=3）/110 次（G=4）整型操作，其中 int64 取模
  16/48 次；对应的有效速率为 **≈22-23 Gops/s**，其中取模 **≈9.6-10.0 Gmod/s**
  （T=8001）。G=3 与 G=4 的 op/s 一致，说明吞吐由取模决定。
- 访存侧：`aiv_mte2` 与 `aiv_mte3` 均为 0.000，未使用 DMA 搬运；SIMT 线程直接访问
  GM，由 D-cache 服务。主存读 0.81-2.81 GB/s、写 16.1-29.0 GB/s（输出写回）；
  UB vector 读 4.8-8.4 GB/s、写 4.2-7.5 GB/s；L2 读命中率 92.4-98.2%、
  写命中率 97.5-98.5%。
- 带宽侧：T=8001 的有效带宽约 42-43 GB/s；同机 d2d 拷贝的实测吞吐约 3.1 TB/s
  （25.2 MB 读写 / 8 us），对应利用率约 1.4%。该算子不受带宽限制。

## 6. 结论

- 计算量集中在 int64 取模：同一次采集会话内的消融显示，移除每线程 48 次取模后
  T=65536 用例的耗时由 0.175 ms 降至 0.070 ms，取模占比约 58%。
- 上限由 SIMT ALU 吞吐决定（占用率 0.93-0.99），MTE 与带宽均非瓶颈。

## 7. 工具链相关说明

当前工具链版本下，SIMT 读取 UB tile 会触发设备异常（vector core exception 507035；
写 UB tile 正常），因此实现选择 GM 直访的数据通路。若后续工具链版本支持该通路，
可在相同接口下评估 UB staging 的收益；当前观测表明瓶颈在 SIMT ALU 而非访存。

## 8. 数据来源

- 实现与设计：`op_kernel/engram_hash.py`、`DESIGN.md`。
- 精度：`tests/assets/engram_hash_golden.py`、`tests/test_engram_hash.py`（30 passed，
  1 skipped：未设置 `ENGRAM_HASH_BENCH` 时跳过采集用例）。
- kernel 时间（刷 L2 口径）：
  `ENGRAM_HASH_BENCH=1 ENGRAM_HASH_FLUSH_MB=256 msprof_profile_run.sh --warm-up=3
  --output=<dir> -- python3 -m pytest tests/test_engram_hash.py -q -k benchmark -s`；
  解析 `op_summary` 时过滤 flush kernel，按 `Task Start Time` 排序后每 15 行一组，
  对应用例顺序为 T=512/G=3、T=512/G=4、T=8001/G=3、T=8001/G=4（其后为 mask 开启的
  同序用例），取 10 次的均值与最小值。
- 消融数据来自本机测量脚本（会话产物，未入库）。
