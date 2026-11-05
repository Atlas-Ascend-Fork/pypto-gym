# 完整调用测量与跨环境证据

在诊断既有多 launch 算子、移动输入预处理、比较分位数或迁移环境时使用本页。它补充[证据协议](perf-analysis/evidence-protocol.md)，执行边界遵循[核心交付规则](../../pypto-pro-kernel/references/workflows/delivery.md#kernel-definitions-and-dispatch)，计时归属遵循[完整调用计时规则](../../pypto-pro-kernel/references/verification/native-timing.md#measure-one-complete-call)，保留项目已经冻结的 seed、预热、测量次数、重复轮数和评分指标。

按一次完整调用实际触发的 kernel/launch 集合归属和统计。证据协议中的单 target 采集器仅适用于单 target-task 调用；多阶段或获批预处理参与比较时，正式比较须覆盖完整调用，单阶段采集只作诊断。实现改动遵守项目的 launch 约束。

## 先冻结语义与执行范围

- 公开接口、输入输出的形状/dtype/布局、参数和可选项语义、支持的规格范围；
- 输入预处理及舍入位置，参数/规格分派、fallback 和实际执行图；
- JIT/launch 数，以及预处理和辅助工作在 host、框架算子还是目标 kernel 内；
- native kernel duration、device event、wrapper wall time 各自包含什么。

同名函数、相同 shape 或 dtype 不足以证明合同一致。移动归一化、激活或 cast 会改变测量范围及
舍入位置；比较前核对执行图，并保持原 Golden 与容差。

## 按调用归属，再统计

1. 通过 profiler 的 correlation/launch 索引或已验证的任务序列归属一次调用内的全部 kernel，
   包括替代、辅助、预处理及 fallback。先核对阶段顺序、完整调用数和每次 kernel 数。
2. 若冻结指标为 kernel duration 合计，先求每次调用的合计，再计算 mean、p50、p90。
   **阶段 p50 之和不等于完整调用 p50**。有并发 launch 时，合计也不等于关键路径或墙钟时间；
   另报设备时间跨度及重叠，保持指标名称明确。
3. 单列冷启动、编译、预热和测量样本。每轮报告 count、mean、p50、p90，再单列 pooled 结果；
   固定分位数算法。mean 为所有测量值的算术平均，p50/p90 为对应分位数，不能互相替代。
4. 不按测量结果删慢样本、挑卡或追加轮次取最好值。重测须记录原因，保留前轮身份与结果；
   profiler 丢失/不完整调用应按冻结的有效性规则处理并披露。
5. collection 成功、原始归档完整、正确性通过、数值性能目标通过分别给结论。
   每轮门槛不可被 pooled 平均或仅部分阶段的结果替代。

包装层 wall/event 与 native duration 分开报告，注明同步和 profiler 开销，不能以 kernel 合计
宣称端到端 host 调用耗时，也不能把不同采集模式的数值直接相减作为预处理开销。

## 核数、pipe 与整卡峰值

- native launch 的 block 数不代表有效参与计算的核数；核对逻辑任务到执行核的映射、
  空闲分支、尾部任务和工作量分布，再判断并行度是否受限。
- pipe busy 是诊断比例，多个 pipe 可重叠，不能求和当作 kernel 时长或整卡利用率。
- 整卡计算利用率须使用匹配 SKU、精度路径、核数的峰值，运算量和测量时间覆盖同一范围；
  带宽利用率同样匹配实际传输字节、时间和 HBM/缓存层级。有效运算量与硬件实际执行量分开标注。
- 先核实 PMU 支持、采集模式和计数器有效性；缺失指标记录 unavailable，全零读数结合采集
  状态与执行路径解释，不能直接推断整个资源没有活动，也不能补造指标。

## 环境与迁移

跨卡异常先按[环境检查 skill](../../pypto-pro-environment-check/SKILL.md)分别核实映射、健康/错误、
占用、目标执行期频率及短 smoke。健康正常或 smoke 通过不证明长目标负载频率正常。
低频运行不能按频率比例换算成验收证据；有错误码的正常频率卡也不能直接宣布合格。

runtime/平台核数不一致时保留所有原始读数及版本，再看实际 launch 与工作覆盖。调整执行核数时，
同步核对任务划分、缓冲区容量与寻址、跨核依赖和结果写入覆盖，不能只替换 block_dim。

迁移时记录实际 import 文件、依赖、模板/生成器、编译器和运行库身份；生成的 Python 变体需
同步可维护生成源，生成 C++ 仅作只读证据。旧 checkout、缓存或运行回执不能替代当前源码身份
和进程状态。跨环境结论只覆盖已完成的验证阶段。
