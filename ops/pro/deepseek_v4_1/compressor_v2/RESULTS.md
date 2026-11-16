# CompressorV2 结果说明

## 用例

`tests/test_compressor_v2.py` 包含 25 个默认用例，覆盖 TH、BSH 两种布局，以及长序列预填充、单 token 解码、多序列短输入、历史前缀拼接和有效输出为空的场景。默认输入和两路权重为 BF16，输入特征宽度 `H=4096`，输出特征宽度 `D` 为 `128` 或 `512`，压缩率 `r` 为 `4` 或 `128`；缓存为 FP32。

激活、权重和初始缓存按固定种子在 `[-10, 10]` 范围均匀采样，激活和权重随后转换为 BF16。默认用例均显式提供 `start_pos`，`seqused=None`；TH 的序列边界由 `cu_seqlens` 给出。缓存配置包含连续存储、块间填充、非零存储偏移和打乱的物理块映射。有效输出为空的用例仍执行投影、缓存更新和缓存检查。

`Prefill1` 对同一组输入连续调用三次，每次恢复初始缓存并校验结果。

## 精度标准

参考计算位于 `tests/assets/compressor_v2_golden_cpu.py`。它读取实际类型的输入和权重，在 CPU 上以 FP64 计算投影、历史拼接和分组加权；有效输出转换为输入类型，缓存更新结果转换为 FP32。

`tests/assets/compressor_v2_precision.py` 对每个用例分别检查以下三个区域，各区域的选中元素展平后独立判定：

| 区域 | 检查器名称 | 比较范围 |
| :--- | :--- | :--- |
| 有效压缩输出 | `cmp_kv` | 参考输出掩码选中的所有行及其全部 D 个通道。 |
| KV 缓存更新区 | `kv_state_update` | 参考更新掩码选中的缓存前 D 个通道。 |
| 分数缓存更新区 | `score_state_update` | 参考更新掩码选中的缓存后 D 个通道。 |

三个区域的实际值和参考值均先转换为 FP32，再按算子输出类型选择下表容差。FP32 缓存更新区也使用输出类型对应的容差。

| 输出类型 | 相对容差 `rtol` | 绝对容差 `atol` |
| :--- | ---: | ---: |
| BF16 | 0.0078125 | 0.0001 |
| FP16 | 0.005 | 0.000025 |

对有限值，满足 `abs(actual - expected) <= atol + rtol * abs(expected)` 时记为匹配。实际使用 `np.isclose(..., equal_nan=True)`。每个非空区域至少 99.5% 的元素匹配，即 `check_result` 的默认 `pct_thd=0.005`。

对不匹配元素，继续按原函数计算归一化误差：

```text
absolute_diff = abs(actual - expected)
base = max(abs(actual), abs(expected), 2^(-14) / 0.005) + 1e-9
relative_error = absolute_diff / (base + 1e-9)
```

最大值取自展平顺序中的前一千万个不匹配元素；该最大值 `>= 10` 时判失败。分母中的 `0.005` 来自 `diff_thd`，与控制匹配率的 `pct_thd` 分别参与判定。两次小量相加、原函数的 `max` 及非有限值处理均保持不变，不以 `allclose` 或其他新增误差指标替代。实际值和参考值均为空的区域按检查器规则通过，缓存及保护区仍须检查。

输出还需满足参考形状、类型和连续布局，缓存形状和 FP32 类型须一致。未更新缓存通过 INT32 视图逐位比较，要求内容不变。缓存视图以外的分配保护区逐元素保持哨兵值 `7717`，包括起始偏移区、块间间隔和末尾保护区；任一保护元素变化即失败。

单个用例必须同时满足三个区域的数值判据、未更新缓存检查、保护区检查及形状和类型检查。匹配率不跨区域或跨用例合并计算；输出分配容量中的无效行不参与数值比较。

## 性能结果

### 绝对性能结果

测试平台为 Ascend 950PR_957b，包含 28 个 Cube 核、56 个 Vector 核和 112 MiB L2；使用 CANN 9.2.0、PyPTO 0.3.0、PyTorch 2.10.0 和 torch_npu 2.10.0.post4，在物理 4 号卡串行采集。

比较对象为已合入版本的原始 JIT 和新安装完整算子包的 ACLNN 调用。两者在同一进程、同一设备及同一 Stream 中配对运行，使用相同输入和初始缓存。每个实现预热 5 次，再采集 20 对正式样本；JIT 先运行和二进制先运行各 10 对，保留全部正式样本。表内 25 例来自同一轮完整采集，未跨轮次拼接单例结果。预热和正式调用均检查精度及缓存保护区。

采集遵循[官方 msprof 指导](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/920beta1/devaids/Profiling/atlasprofiling_16_0011.html)，使用 L1 任务记录、AI Core task-based 模式和 PipeUtilization 指标。每次调用前先恢复初始缓存并同步，再执行 [cann-bench](https://gitcode.com/cann/cann-bench) 的清 L2 流程并同步；参考版本为 `6b0bda120caf615a336a1e8221cd1609e46ab46d`。清理工作集为 192 MiB，大于测试设备的 112 MiB L2。JIT 与二进制分别清理，同一 Stream 中清理结束与首个目标任务之间没有其他设备算子。

默认 25 例均使用两个计算阶段。单次耗时为该调用两条目标任务的 `Task Duration(us)` 之和，按设备、Stream ID、Task ID 和算子标识核对归属与顺序。该字段按[官方定义](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/920beta1/devaids/Profiling/atlasprofiling_16_0067.html)包含设备调度、加速器执行和结束响应；本表不包含主机 API、阶段间隙、预热、缓存恢复或清 L2 任务耗时。

表中耗时为 20 次正式采样的“算术均值 ± 总体标准差”，单位为 μs。性能占比为“原始 JIT 均值 / ACLNN 二进制均值”，使用未四舍五入的数据计算，以数值表示。

`B` 为序列数，`S` 为每条序列本次新增 token 数，`T=B×S`，`H` 为输入特征宽度，`D` 为投影及输出宽度。`cmp_ratio` 是一个压缩组的 token 数。`block_count`、`block_size` 分别为缓存物理块数和每块槽位数；`cache_stride` 是相邻块起点之间的 FP32 元素数，最后两轴步长为 `[2D,1]`；`cache_offset` 是缓存视图起点的 FP32 元素偏移。`start_pos` 为各序列的绝对起点，单位为 token；`seed` 为固定输入种子。`dtype` 适用于激活、两路权重和输出。

所有用例均省略 `seqused`，显式提供 `start_pos`；表内各序列起点相同。TH 用例为等长输入，`cu_seqlens[b]=b×S`；BSH 省略该参数。`state_block_table` 的形状为 `[B]`，指定序列到物理块的映射，具体内容见测试定义。

| 用例及参数 | 原始 JIT（μs） | ACLNN 二进制（μs） | 性能占比 |
| :--- | ---: | ---: | ---: |
| `Prefill0: B=1, S=8192, T=8192, H=4096, D=512, cmp_ratio=4, layout=TH, dtype=BF16, cache_dtype=FP32, index_dtype=INT32, block_count=1, block_size=128, cache_stride=131086, cache_offset=6, start_pos=0, seqused=None, seed=2026102801` | 239.619 ± 1.331 | 238.270 ± 0.894 | 1.005663 |
| `Prefill1: B=1, S=8192, T=8192, H=4096, D=128, cmp_ratio=4, layout=TH, dtype=BF16, cache_dtype=FP32, index_dtype=INT32, block_count=1, block_size=128, cache_stride=32784, cache_offset=6, start_pos=0, seqused=None, seed=2026102802` | 98.295 ± 1.451 | 90.936 ± 1.079 | 1.080927 |
| `Prefill2: B=1, S=8192, T=8192, H=4096, D=512, cmp_ratio=128, layout=TH, dtype=BF16, cache_dtype=FP32, index_dtype=INT32, block_count=1, block_size=128, cache_stride=131077, cache_offset=1, start_pos=0, seqused=None, seed=2026102803` | 240.180 ± 1.091 | 238.942 ± 1.015 | 1.005182 |
| `decode0_start8192: B=1, S=1, T=1, H=4096, D=512, cmp_ratio=4, layout=TH, dtype=BF16, cache_dtype=FP32, index_dtype=INT32, block_count=1, block_size=128, cache_stride=131110, cache_offset=20, start_pos=8192, seqused=None, seed=2026102804` | 13.701 ± 0.399 | 12.315 ± 0.270 | 1.112531 |
| `decode0_start8195: B=1, S=1, T=1, H=4096, D=512, cmp_ratio=4, layout=TH, dtype=BF16, cache_dtype=FP32, index_dtype=INT32, block_count=1, block_size=128, cache_stride=131088, cache_offset=6, start_pos=8195, seqused=None, seed=2026102805` | 14.329 ± 0.462 | 12.489 ± 0.225 | 1.147315 |
| `decode1_start8192: B=1, S=1, T=1, H=4096, D=128, cmp_ratio=4, layout=TH, dtype=BF16, cache_dtype=FP32, index_dtype=INT32, block_count=1, block_size=128, cache_stride=32791, cache_offset=0, start_pos=8192, seqused=None, seed=2026102806` | 9.974 ± 0.274 | 9.182 ± 0.312 | 1.086269 |
| `decode1_start8195: B=1, S=1, T=1, H=4096, D=128, cmp_ratio=4, layout=TH, dtype=BF16, cache_dtype=FP32, index_dtype=INT32, block_count=1, block_size=128, cache_stride=32817, cache_offset=24, start_pos=8195, seqused=None, seed=2026102807` | 10.113 ± 0.292 | 9.181 ± 0.365 | 1.101564 |
| `decode2_start8192: B=1, S=1, T=1, H=4096, D=512, cmp_ratio=128, layout=TH, dtype=BF16, cache_dtype=FP32, index_dtype=INT32, block_count=1, block_size=128, cache_stride=131096, cache_offset=17, start_pos=8192, seqused=None, seed=2026102808` | 13.991 ± 0.303 | 13.443 ± 0.374 | 1.040739 |
| `decode2_start8195: B=1, S=1, T=1, H=4096, D=512, cmp_ratio=128, layout=TH, dtype=BF16, cache_dtype=FP32, index_dtype=INT32, block_count=1, block_size=128, cache_stride=131098, cache_offset=18, start_pos=8195, seqused=None, seed=2026102809` | 14.213 ± 0.213 | 13.344 ± 0.363 | 1.065150 |
| `decode3_start8193: B=8, S=3, T=24, H=4096, D=512, cmp_ratio=4, layout=TH, dtype=BF16, cache_dtype=FP32, index_dtype=INT32, block_count=8, block_size=128, cache_stride=131082, cache_offset=0, start_pos=8193, seqused=None, seed=2026102810` | 19.589 ± 0.605 | 16.449 ± 0.557 | 1.190898 |
| `decode4_start8192: B=8, S=3, T=24, H=4096, D=128, cmp_ratio=4, layout=TH, dtype=BF16, cache_dtype=FP32, index_dtype=INT32, block_count=8, block_size=128, cache_stride=32803, cache_offset=25, start_pos=8192, seqused=None, seed=2026102811` | 13.562 ± 0.171 | 9.797 ± 0.310 | 1.384340 |
| `decode4_start8193: B=8, S=3, T=24, H=4096, D=128, cmp_ratio=4, layout=TH, dtype=BF16, cache_dtype=FP32, index_dtype=INT32, block_count=8, block_size=128, cache_stride=32803, cache_offset=24, start_pos=8193, seqused=None, seed=2026102812` | 13.639 ± 0.190 | 9.747 ± 0.321 | 1.399367 |
| `decode4_start8191: B=8, S=3, T=24, H=4096, D=128, cmp_ratio=4, layout=TH, dtype=BF16, cache_dtype=FP32, index_dtype=INT32, block_count=8, block_size=128, cache_stride=32779, cache_offset=6, start_pos=8191, seqused=None, seed=2026102813` | 15.743 ± 0.227 | 10.000 ± 0.216 | 1.574338 |
| `decode5_start8193: B=8, S=3, T=24, H=4096, D=512, cmp_ratio=128, layout=TH, dtype=BF16, cache_dtype=FP32, index_dtype=INT32, block_count=8, block_size=128, cache_stride=131077, cache_offset=1, start_pos=8193, seqused=None, seed=2026102814` | 18.622 ± 0.534 | 15.177 ± 0.259 | 1.226939 |
| `FullLoad0: B=1, S=4, T=4, H=4096, D=512, cmp_ratio=4, layout=BSH, dtype=BF16, cache_dtype=FP32, index_dtype=INT32, block_count=1, block_size=128, cache_stride=131103, cache_offset=8, start_pos=0, seqused=None, seed=2026102815` | 14.007 ± 0.209 | 12.720 ± 0.286 | 1.101242 |
| `FullLoad1: B=1, S=4, T=4, H=4096, D=128, cmp_ratio=4, layout=BSH, dtype=BF16, cache_dtype=FP32, index_dtype=INT32, block_count=1, block_size=128, cache_stride=32797, cache_offset=0, start_pos=0, seqused=None, seed=2026102816` | 10.175 ± 0.395 | 9.096 ± 0.264 | 1.118542 |
| `FullLoad2: B=1, S=4, T=4, H=4096, D=512, cmp_ratio=128, layout=BSH, dtype=BF16, cache_dtype=FP32, index_dtype=INT32, block_count=1, block_size=131, cache_stride=134147, cache_offset=0, start_pos=0, seqused=None, seed=2026102817` | 14.053 ± 0.472 | 12.792 ± 0.358 | 1.098533 |
| `FullLoad3: B=64, S=4, T=256, H=4096, D=512, cmp_ratio=4, layout=BSH, dtype=BF16, cache_dtype=FP32, index_dtype=INT32, block_count=64, block_size=128, cache_stride=131090, cache_offset=11, start_pos=0, seqused=None, seed=2026102818` | 31.695 ± 0.325 | 27.640 ± 0.453 | 1.146709 |
| `FullLoad4: B=64, S=4, T=256, H=4096, D=128, cmp_ratio=4, layout=BSH, dtype=BF16, cache_dtype=FP32, index_dtype=INT32, block_count=64, block_size=128, cache_stride=32806, cache_offset=31, start_pos=0, seqused=None, seed=2026102819` | 17.045 ± 0.249 | 14.587 ± 0.700 | 1.168553 |
| `FullLoad5: B=8, S=2, T=16, H=4096, D=512, cmp_ratio=4, layout=BSH, dtype=BF16, cache_dtype=FP32, index_dtype=INT32, block_count=8, block_size=128, cache_stride=131119, cache_offset=11, start_pos=0, seqused=None, seed=2026102820` | 15.907 ± 0.230 | 14.208 ± 0.343 | 1.119559 |
| `FullLoad6: B=1, S=1, T=1, H=4096, D=512, cmp_ratio=4, layout=BSH, dtype=BF16, cache_dtype=FP32, index_dtype=INT32, block_count=1, block_size=128, cache_stride=131095, cache_offset=2, start_pos=100, seqused=None, seed=2026102821` | 13.843 ± 0.259 | 12.014 ± 0.235 | 1.152209 |
| `FullLoad7: B=1, S=1, T=1, H=4096, D=128, cmp_ratio=4, layout=BSH, dtype=BF16, cache_dtype=FP32, index_dtype=INT32, block_count=1, block_size=128, cache_stride=32789, cache_offset=12, start_pos=100, seqused=None, seed=2026102822` | 9.893 ± 0.195 | 8.792 ± 0.246 | 1.125189 |
| `FullLoad8: B=1, S=1, T=1, H=4096, D=512, cmp_ratio=128, layout=BSH, dtype=BF16, cache_dtype=FP32, index_dtype=INT32, block_count=1, block_size=128, cache_stride=131094, cache_offset=14, start_pos=100, seqused=None, seed=2026102823` | 13.433 ± 0.356 | 12.730 ± 0.248 | 1.055235 |
| `Contiguous0: B=2, S=3, T=6, H=4096, D=512, cmp_ratio=4, layout=TH, dtype=BF16, cache_dtype=FP32, index_dtype=INT32, block_count=2, block_size=128, cache_stride=131072, cache_offset=0, start_pos=8193, seqused=None, seed=2026102824` | 14.335 ± 0.416 | 12.667 ± 0.218 | 1.131713 |
| `NonContiguous0: B=2, S=3, T=6, H=4096, D=512, cmp_ratio=4, layout=TH, dtype=BF16, cache_dtype=FP32, index_dtype=INT32, block_count=2, block_size=128, cache_stride=131108, cache_offset=4, start_pos=8193, seqused=None, seed=2026102825` | 14.805 ± 0.276 | 13.784 ± 0.250 | 1.074086 |

### 理论评估对比

按一次乘加计两次运算，两路投影的逻辑运算量为 `4×T×H×D`。该式不包含部分和合并、softmax 加权、类型转换、缓存处理及硬件块填充带来的额外工作。

下表取自同次正式采集的 20 条二进制 Stage0 任务，覆盖长序列、两种压缩率、单 token 解码、历史拼接、多序列 BSH 及带间隔缓存的 TH 输入。按[官方字段说明](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/920beta1/devaids/Profiling/atlasprofiling_16_0067.html)，MAC 和 MTE2 占比分别表示矩阵指令、数据搬入指令的周期数占对应总周期数的比例，`1.0` 表示全部周期；这里报告算术均值及总体标准差。

| 用例 | 两路投影逻辑运算量 | Stage0 MAC 周期占比 | Stage0 MTE2 周期占比 |
| :--- | ---: | ---: | ---: |
| Prefill0 | 68,719,476,736 | 0.9028 ± 0.0028 | 0.7724 ± 0.0104 |
| Prefill2 | 68,719,476,736 | 0.9013 ± 0.0047 | 0.7814 ± 0.0059 |
| decode0_start8192 | 8,388,608 | 0.1256 ± 0.0028 | 0.7596 ± 0.0144 |
| decode0_start8195 | 8,388,608 | 0.1268 ± 0.0027 | 0.7569 ± 0.0105 |
| FullLoad4 | 536,870,912 | 0.2826 ± 0.0144 | 0.4386 ± 0.0240 |
| NonContiguous0 | 50,331,648 | 0.1231 ± 0.0037 | 0.7505 ± 0.0138 |

不同指令流水可重叠，两列占比不能相加为总耗时比例，也不能直接当作峰值算力或带宽利用率。完整调用还包含 Stage1 的向量计算和缓存写回；这些数据描述 Stage0 的 Cube 侧流水，不代表整个算子的耗时构成。
