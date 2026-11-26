# flash_kda_cp_preprocess 设计说明

## 计算阶段划分

Stage 1 按 token 块归一化 Q/K、计算门控与衰减、构造三角系统并求解。两个 AIV 子块各自计算 Q 路和 K 路需要的门控，向 AIC 分别提交乘法操作数。AIC 交错执行块内分数、FP16 Neumann 倍增及 u/w 乘法。输出直接写有效 token 行。

Stage 2 计算块摘要。局部路径在同一核计算 φ 并递推。共享路径的 2a 按块计算 φ 并写到 GM，2b 按行片取用，递推 R 和 ψ。多轮计划从前轮输出继续递推。

## 核间切分策略

metadata ABI v2 记录块偏移、核任务范围和摘要行片数。Stage 1 以（序列、头、chunk）为任务，规划器按有效行估算成本。Stage 2 以（序列、头、行片）分区，每条链按核数选择 1、2、4 或 8 个行片。四片和八片计划共享 φ。两片仅在启用扩展工作区的单序列长 BSND 且调度只有一轮时共享 φ。BSND 单序列单块采用四片调度时，由首片所属核计算整个头的摘要。

## 核内切分策略、基本块选择与核内 Buffer 划分

基本块为 64 token × 128 特征。Cube 乘法按有效 K 收窄。Stage 2 局部路径可用 Cube 直接转置读取稠密整块和单行片 TND，后者也覆盖短尾。另一路经 AIV 转置，短尾由 VF 清零无效位置。

Stage 1 每任务使用 72 KiB GM 工作区，存放 BF16 RIGHT 面板、βV 和 βK̃。LEFT 面板直接经 L1 交接。

| 缓冲区 | 分配及复用 |
|---|---|
| L1（512 KiB） | Stage 1 的 T/seed 与评分面板双槽。Stage 2 的 R/ψ 各一槽，φ、uᵀ、kg 按流水使用双槽 |
| UB（248 KiB） | Stage 1 的 kk/mqk、Q/K、T/seed 暂存区，在 Stage 2 用作 φ 双槽和 ND→NZ 中转 |
| L0B（64 KiB） | [128,128] FP32 φ 占满 64 KiB，BF16 操作数分时使用同址，按 MTE1/M 的事件交接 |

Stage 1 的 `l1_t/l1_seed/l1_id/l1_a16` 与 φ 的 L1 槽 B 分时复用。LEFT 面板与 Stage 2 的 `wt2/vp_wt2/kg2` 共用地址。Stage 1 的 kk/mqk、归一化和 T/seed 暂存区在 Stage 2 用于 φ UB/NZ 暂存。第一道 MIX 全核屏障分隔 Stage 1 与 Stage 2，轮末 MIX 屏障保护下一轮反向复用。Stage 1 内部的 `ub_lnz/ub_lnz1` 与 `ub_tf` 同址部分使用同一 mutex 10。

## 接口选择

公开入口在 Host 校验张量属性和分配缓冲，提交一次 MIX kernel。metadata 入口负责序列边界校验和核调度，并预先准备只读常量。主入口接收 ABI v2 调度表，按设备缓存常量，首次缺失时上传。

## 流水设计

Stage 1 相邻任务交错。EV_READY_OP 等待两路操作数，EV_READY_KM/TS 串起 Cube 分数、AIV 三角数据与 Cube 求解。返回事件保护会被下一任务复用的槽。

共享 φ 路径的 2a 用两个 wᵀ L1 槽支撑每批两项任务，避免后一项覆盖前一项的操作数。AIV 经 MTE3 写回 φ，发出 EV_PHI_RELEASE 后 Cube 才可覆盖 φ 的 UB 双槽。2a/2b 之间用 MIX 屏障。2b 的 R/ψ 分别以 EV_FX_RR/EV_FX_RP 释放下一轮读取。标志首词初始化为 FP32 1.0 的位型，代表 R 存活。AIV 经 MTE3 写入零检查结果并发布 EV_ZERO_READY，AIC 等待事件后对标志所在 GM 缓存行执行 DCCI，再用 getval 读取。读到 0 表示已确认 R 全零，允许在非检查轮且非末轮跳过 R 矩阵乘法。
