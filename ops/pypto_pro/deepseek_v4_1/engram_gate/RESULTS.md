# engram_gate 结果说明

本文给出 `engram_gate` 的精度与性能结果：用哪些用例检验、按什么标准判定、实测数据是多少、
以及这些数据说明什么。表中数值均由同一套平台实测得到；由实测值再算一步得到的量（有效带宽、
利用率、浮点吞吐）都在出现的地方写明算法。

## 1 用例

**测试环境**

| 项目   | 取值                                                                                                                        |
| :--- | :------------------------------------------------------------------------------------------------------------------------ |
| 硬件平台 | Ascend 950PR（`get_device_name(0)` = `Ascend950PR_9589`），cube 32 核 / vector 64 核，L2 128 MiB，HBM 123 GiB              |
| 软件环境 | CANN 9.3.0、PyPTO-Pro DSL、Python 3.10.20、torch 2.8.0、torch_npu 2.8.0.post4                                               |
| 平台信息 | `get_platform_info()` = `PlatformInfo(soc_version='DAV_3510', core_num=32, cube_core_num=32, vector_core_num=64)`          |

**用例清单**

`tests/test_engram_gate.py` 里一个用例对应一个 `test_xxx` 函数，共 5 个：

| case                              |  t | hc |  dim | image_mask | 说明                              | 起核数 |
| :-------------------------------- | -: | -: | ---: | :--------- | :------------------------------ | --: |
| `partial_image_mask`              |  5 |  4 | 5120 | `idx%2==1` | 部分 mask 直通（5 个 token 里 2 个）     |  20 |
| `model_token_count_nomask`        | 72 |  4 | 5120 | 全 False    | 能喂满 64 个向量核的用例                  |  64 |
| `chunked_one_chunk_tail_one`      |  3 |  4 | 8449 | 全 False    | 接近 `dim` 上限；`dim` 非 64 倍数（尾段 1） |  12 |
| `model_dim_plus_one_partial_mask` |  5 |  4 | 5121 | `idx%2==1` | `dim` 非 64 倍数（尾段 1）+ 部分 mask    |  20 |
| `compile_default_shape`           |  2 |  4 | 5120 | 全 False    | 小 token 默认形状                    |   8 |

五个用例覆盖了满段路径（5120 是 64 的整数倍，尾段为空）与尾段路径（5121 与 8449 的尾段都是
1），也覆盖了直通分支与计算分支，`dim` 都 ≤ 8512。

`hc` 的其它取值、`dim` 的上限与越界点、全 True 的 mask 不在上表之内，其行为见 2.2 节。

**运行方式**

```bash
pytest ops/pypto_pro/deepseek_v4_1/engram_gate/tests -q
```

精度判定与参考值都在 `tests/assets/` 下，随算子一起维护。

## 2 精度标准

精度判定统一走 `tests/assets/engram_gate_precision.py`（方案 A 混合容差），参考值由
`tests/assets/engram_gate_golden_cpu.py` 在 CPU 上**独立**算出，另有一份 FP64 oracle
在 golden 自检时交叉验证。判定规则：

- 逐元素：`|actual - golden| <= atol + rtol * |golden|`；
- 整体：以下四条**同时**满足
  - `matched_ratio >= 0.99`
  - `max_abs_error <= max(fixed_limit, 32 × ULP)`（ULP 是该浮点格式在 1.0 附近的最小间隔）
  - `mean_abs_error <= atol`
  - `mean_rel_error <= rtol`

`y` 是 BF16，阈值表由比较器硬编码：

| 参数 | 取值 |
| :--- | ---: |
| `rtol` | 1.562e-02（2⁻⁶） |
| `atol` | 1.562e-02（2⁻⁶） |
| `max_abs_error` 上限 `fixed_limit` | 1.0 |
| `matched_ratio` 下限 | 0.99 |

### 2.1 结果

**5 / 5 通过。** 每个用例的判据与实测余量：

| case                              | 实测 `matched_ratio` | 实测 `max_abs_error` | 实测 `mean_abs_error` | 实测 `mean_rel_error` | 逐位不等的元素 |
| :-------------------------------- | -----------------: | -----------------: | -----------------: | -----------------: | --------------: |
| `partial_image_mask`              |             1.0000 |          2.441e-04 |          2.384e-09 |          6.028e-08 |      1 / 102400 |
| `model_token_count_nomask`        |             1.0000 |          3.906e-03 |          1.508e-08 |          1.888e-07 |    50 / 1474560 |
| `chunked_one_chunk_tail_one`      |             1.0000 |          1.907e-06 |          1.881e-11 |          6.323e-08 |      1 / 101388 |
| `model_dim_plus_one_partial_mask` |             1.0000 |          6.104e-05 |          5.959e-10 |          5.917e-08 |      1 / 102420 |
| `compile_default_shape`           |             1.0000 |          7.629e-06 |          1.863e-10 |          1.157e-07 |       1 / 40960 |

三点说明：

- 「实测 `matched_ratio`」就是比较器的判定口径：满足 `|actual - golden| <= atol + rtol·|golden|`
  的元素占比。5 个用例都是 `1.0000`，即**没有任何元素超出容差**。
- 最右列是另一个更严的口径——把两边都落到 BF16 之后**逐位**比较，数出连最低位都不一样的
  元素个数。它不参与判定，只用来量出误差的实际大小；5 个用例合计 54 个，占全部 1,821,728 个
  元素的 0.0030%。
- 「实测 `mean_abs_error`」与「实测 `mean_rel_error`」的上限就是 `atol` 与 `rtol`，
  它们的余量分别是 6 个和 5 个数量级。四个量都按比较器自己的打印格式（`matched_ratio`
  4 位小数，其余 3 位有效数字）原样给出。

`matched_ratio` 与 `max_abs_error` 的余量很大，下面两个事实更能说明误差的实际量级：

1. **误差只出现在 BF16 的最后一位。** 把每个元素的 `|actual - golden|` 除以该元素自身的
   BF16 ULP，5 个用例里出现差异的元素，偏差只取 1 与 2 两个值，最大的一个也只有 **2.00**
   个 BF16 最小单位。具体到绝对量：5 个用例里最大的 `max_abs_error` 是 3.906e-03
   （`model_token_count_nomask`），它出现在 `|y| = 0.8555` 处；BF16 在 `|y| ∈ [0.5, 1)`
   上的 1 个 ULP 正好是 2⁻⁸ = 3.906e-03，所以这个最大的绝对误差**恰好是 1 个 ULP**。
   这是浮点运算次序不同带来的取整差异，不是算法偏差。全部 5 个用例合计只有 54 个元素与
   golden 不同（占 182 万个元素的 0.003%），其余元素逐位相同。
2. **`image_mask == True` 的行与 `x` 逐位相等。** kernel 走直通分支，CPU golden 也直接取
   `x`，两边都是位拷贝，没有取整参与；已用 `torch.equal` 逐位确认。

参考实现（`tests/assets/engram_gate_golden_cpu.py`）自身也带一组自检：4 个 contract
用例 + 4 项性质检查（`dot=0 → gate=0.5`、符号单调性、全 True mask 逐位等于 `x`、非默认
`eps`/`clamp_value`）+ 7 例动态轴泛化（`t=1…8192`、`hc=1…16`、`dim=1…65536`，其中
`dim=65` 那一例带部分 mask）。contract 与泛化这 11 个用例都会把结果与一份独立的 FP64
oracle 交叉对比（容差 0.15），所以参考值不是「自己和自己比」。

### 2.2 形状与边界行为

声明范围之内的形状都能正常输出，声明之外的输入在 host 侧报错、不静默给出结果。逐条实测的
行为如下：

| 情形 | 实测行为 |
| :--- | :--- |
| `hc` 取 1、2、3、5、8、16、64 | 正常输出，精度与 golden 一致 |
| `dim` 取 63、64、65、8511、8512 | 正常输出；覆盖非 64 倍数与 `dim` 上限 |
| `image_mask` 全为 True / 仅一个为 True | 对应行逐位等于 `x`（`torch.equal` 为真） |
| `t = 1`，`hc = 1` 或 4 | 正常输出 |
| `token` 维与 `dim` 维带步长的非连续输入 | 结果与连续输入一致 |
| `dim > 8512`（如 8513） | `ValueError`：超过单趟扫描宽度 8512 |
| `hc = 0` | `ZeroDivisionError`：起核数要按 `hc` 求商 |
| `hc > 向量核数`（如 65）、`t = 0` | `InvalidType`：`block_dim must be a positive integer, got 0` |
| `eps` / `clamp_value` 取默认值以外的值 | `ValueError`：报出 kernel 内编定的取值 |
| dtype 与声明不符 | `InvalidType`：报出期望与实际 dtype |

边界形状的实测精度同样在容差内：`hc` 的 7 组、`dim` 的 5 组、全 True / 单 True 的 mask、
退化的 `t` 与 `hc`、非连续输入全部通过，`matched_ratio` 均为 1.0000；其中 `dim = 8512`
的 `max_abs_error` 是 1.221e-04，为该元素自身 ULP 的 6 倍，其余各项都不超过 1 个 ULP。

## 3 性能结果

本算子不使用 Cube，单次调用只下发一个 kernel，因此本算子的性能即该 kernel 的耗时。下表
给出任务级耗时。

**采集方法**

| 项 | 做法 |
| :--- | :--- |
| 计时 | AscendC 官方工具链 `msprof` 的任务级耗时（`--task-time`，取 `op_summary` 的 `Task Duration(us)`，op 名为 `PYPTO_engram_gate_kernel`） |
| 冷 L2 | 每次被测调用前，先用 cann-bench 的升频算子把频率拉起来，再用其清 L2 算子（一个远大于 L2 容量的缓冲）把片上缓存冲干净，然后才调用被测 kernel |
| 统计 | 固定用单卡、单进程；一个用例一个进程，`op_summary` 里的行全部归属当前用例；连测 21 次、丢弃首次（含 JIT 编译），对剩余 20 次取**最小值**与**中位数** |
| 参考带宽 | 1 GiB 连续 BF16 设备内拷贝（读 1 GiB + 写 1 GiB），3 轮各 30 次，取其中最快的一轮；本平台实测 1450 GB/s |

冷启动用的升频与清缓存算子是独立 kernel，不参与统计。参考带宽是一个**实测代理值**，不是
器件规格书里的峰值带宽，只作为同一平台上的相对参照。

### 3.1 绝对性能结果

| case                              | min (µs) | median (µs) | 起核数 |
| :-------------------------------- | -------: | ----------: | --: |
| `partial_image_mask`              |    2.795 |       2.908 |  20 |
| `model_token_count_nomask`        |    6.973 |       7.360 |  64 |
| `chunked_one_chunk_tail_one`      |    3.523 |       3.622 |  12 |
| `model_dim_plus_one_partial_mask` |    2.982 |       3.162 |  20 |
| `compile_default_shape`           |    2.586 |       2.716 |   8 |

起核数与 `DESIGN.md` 第 2 节的推算一致。

**绝对流量与有效带宽**

「HBM 流量」按冷 L2 下**必须从 HBM 取的唯一字节数**算：`x` 与 `y` 按全部 `t` 个 token 算
（`image_mask == True` 的 token 也要把 `x` 搬进来、把 `y` 写出去），`key` 与 `value` 只按
未 mask 的 token 算，`weight` 只算唯一的一份（`hc × dim`）。
「发出的读写」是算子实际下发的量：`x` 与 `y` 按全部 `t` 个 token 的 `(token, head)` 行算，
`key` 按未 mask 的 `(token, head)` 行算，`value` 按未 mask 的 `(token, head)` 行重复下发
（在 head 维广播），`weight` 按实际起核数重复下发（每个核各搬自己那一行）。

| case                              | HBM 流量 (MB) | 发出的读写 (MB) | 冷 L2 min (µs) | 有效 HBM 带宽 (GB/s) | 占同机参考带宽 |
| :-------------------------------- | ----------: | ----------: | -----------: | ---------------: | ------: |
| `partial_image_mask`              |       0.645 |       1.065 |        2.795 |              231 |   15.9% |
| `model_token_count_nomask`        |       9.667 |      13.107 |        6.973 |          **1386** | **95.6%** |
| `chunked_one_chunk_tail_one`      |       0.794 |       1.217 |        3.523 |              225 |   15.5% |
| `model_dim_plus_one_partial_mask` |       0.645 |       1.065 |        2.982 |              216 |   14.9% |
| `compile_default_shape`           |       0.348 |       0.492 |        2.586 |              135 |    9.3% |

「有效 HBM 带宽」= HBM 流量 ÷ 实测 min 耗时；「占同机参考带宽」= 有效带宽 ÷ 1450 GB/s。

读出来两件事：

1. **只有 `model_token_count_nomask` 真的在跑带宽。** 冷 L2 下它的有效 HBM 带宽是
   1386 GB/s，达到同机拷贝参考值的 95.6%，基本被 HBM 打满；实测 6.973 µs 与「HBM 流量 ÷
   参考带宽」算出的 6.667 µs 只差 4.6%，剩下的差就是片上搬运与标量开销。
2. **另外四个用例没有被带宽卡住。** 它们的有效带宽只有参考值的 9%~16%，起核数也都只有
   8~20 个：每个核只拿到一行，行内 `pass1 → pass2` 又是串行的（`pass2` 要用 `pass1` 搬
   进来的那一行 `x`），没有第二行可供重叠，形不成对 HBM 的持续压力。

### 3.2 理论评估对比

**Bound 判定**

**算术强度**：每个 `(token, head, dim)` 位置做 9 次浮点运算——`Σx²`、`Σkey²` 各是一次乘法
加一次累加（2 + 2），`Σ(x·w·key)` 是一次 `x·w` 乘法加一次融合乘加 `tmp·key + acc`（1 + 2），
残差更新的 `gate·value` 与 `x +` 是 2。`image_mask == True` 的行不做这些运算，所以下面按
**未 mask 的元素数**统计。每个 `(token, head)` 行另有一小段与 `dim` 无关的标量运算（两路 RMS、
`sigmoid`、地址计算），它随行数增长、不随 `dim` 增长。

| 口径（`model_token_count_nomask`） | 算术强度 |
| :--- | ---: |
| 只算 HBM 流量 | 1.37 FLOP / Byte |
| 算发出的全部读写 | 1.01 FLOP / Byte |

1~1.4 FLOP/Byte 是极低的算术强度——每搬 1 字节只做 1 次左右的浮点运算。Roofline 判据把
「算力上限」与「带宽上限」两条线画在一张图上，算术强度低于两条线的交点时，性能由带宽决定；
本算子的 1~1.4 FLOP/Byte 远低于这个交点，所以无论向量核的浮点峰值是多少，它都落在
**带宽受限区**。

**实测利用率**取「有效 HBM 带宽占同机参考带宽的比例」，原始数据在 3.1 节：

| case                              | 有效 HBM 带宽 (GB/s) | 占同机参考带宽 | 判定 |
| :-------------------------------- | ---------------: | ------: | :--- |
| `model_token_count_nomask`        |             1386 |   95.6% | 被 HBM 打满 |
| `partial_image_mask`              |              231 |   15.9% | 未受带宽限制 |
| `chunked_one_chunk_tail_one`      |              225 |   15.5% | 未受带宽限制 |
| `model_dim_plus_one_partial_mask` |              216 |   14.9% | 未受带宽限制 |
| `compile_default_shape`           |              135 |    9.3% | 未受带宽限制 |

按 3.1 的实测时间反推，算子实际达到的浮点吞吐（FP32）：分子是上面「Bound 判定」里那份
「9 次运算 × 未 mask 的元素数」的浮点运算量，分母是实测耗时。

| 口径 | `model_token_count_nomask` | 其余四个用例 |
| :--- | ---: | ---: |
| 冷 L2 | 1.90 TFLOP/s | 0.14 ~ 0.26 TFLOP/s |

三件事：

1. **只有 `model_token_count_nomask` 由 HBM 供数。** 它把 64 个向量核全部铺满，有效带宽到
   同机参考值的 95.6%，HBM 流量对应的时间（6.667 µs）与实测（6.973 µs）只差 4.6%——
   它就是被 HBM 带宽卡住的。
2. **其余四个用例没有打满任何一条管道。** 它们总共只有 8~20 行，起核数也只有 8~20 个
   （见 `DESIGN.md` 第 2 节），每个核只拿到一行；行内 `pass1 → pass2` 又是串行的
   （`pass2` 要用 `pass1` 搬进来的那一行 `x`），没有第二行可供重叠。所以它们的耗时下限
   由行内串行指令数与访存时延决定（实测 2.59~3.52 µs），与带宽无关——这也是它们的有效
   带宽只有参考值 9%~16% 的原因。
3. **每个核只有一行时，行内定长开销摊不开。** 标量链与 `dim` 无关，行数少、核上只有一行
   时它占的比例就上去了；反过来，把行数做多（如 `model_token_count_nomask`）之后，这部分
   就被行间重叠盖住了。
