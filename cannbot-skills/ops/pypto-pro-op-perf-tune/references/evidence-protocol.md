# 性能调优证据协议

> 本文件定义 PyPTO-Pro 性能调优的采集与证据要求。
> [msprof 采集指南](msprof-guide.md)、[msprof op 补充指南](msprof-op-guide.md)、
> [CSV 字段参考](csv_fields_reference.md)、[A5 Roofline 与杠杆](general-knowledge/a5-roofline-and-levers.md)
> 按需提供分析方法；本协议在显式传入 `--case-manifest`（或使用逐 case Golden
> 采集器/时间线入口）时生效。不传 manifest 时，compare/quick/batch
> 保持既有的 Markdown Golden 流程不变。

下文 `PERF_SKILL_ROOT` 指本 Skill 的实际绝对目录，算子目录可替换为项目实际路径。

## Contents

- [Discovery profile：确认真实 lowering Op Name](#discovery)
- [逐 case 契约与 case manifest 契约](#per-case-manifest)
- [seed 契约](#seed-contract)
- [compare / quick / batch 命令](#compare-quick-batch)
- [设备语义](#device-semantics)
- [逐 case Golden 采集器](#golden-collector)
- [逐核证据作用域](#per-core-scope)
- [Final 的补充指令时间线](#instruction-timeline)
- [Roofline / 流水终态证据](#terminal-evidence)
- [持久归档](#archive-layout)
- [CSV 字段与阈值的调优语义](#csv-semantics)
- [A5 证据状态与使用边界](#a5-boundary)
- [msprof op 证据优先级与补充边界](#msprof-op-boundary)

---

## <a id="discovery"></a>Discovery profile：确认真实 lowering Op Name

正式 baseline 前先做一次 **discovery profile**。discovery runner 必须只选择一个已知 case 并只 launch 一次 target kernel；它用于读取真实 lowering `Op Name`，不进入 baseline/final 统计。完整正确性与 JIT 预热已通过后运行：

```bash
# runner 已有 selector
bash "$PERF_SKILL_ROOT/scripts/msprof_profile_run.sh" \
  --warm-up=3 --output=./msprof_output -- \
  python3 custom/{operator_name}/test_{operator_name}.py --case-id <case-id>

# runner 无 selector：用逐 case 适配器调用 manifest 中的 test_function
bash "$PERF_SKILL_ROOT/scripts/msprof_profile_run.sh" \
  --warm-up=3 --output=./msprof_output -- \
  python3 "$PERF_SKILL_ROOT/scripts/msprof_perf_summary.py" \
    --run-case-function \
    --test-script custom/{operator_name}/test_{operator_name}.py \
    --case-manifest custom/{operator_name}/PERFORMANCE_CASES.json \
    --case-id <case-id>
```

采集完成后列出该次 profile 中的完整名称，不要用模糊 grep 直接选结果：

```bash
GROUP_DIR=$(ls -td <output_dir>/PROF_GROUP_* | head -1)
python3 "$PERF_SKILL_ROOT/scripts/msprof_perf_summary.py" "$GROUP_DIR" --list-op-names
```

结合 `@pl.jit` 函数名、Task Type 和 discovery 中唯一的 target launch 确认完整 mangled `Op Name`。如果仍有多个候选或同名多次 launch，先修正 discovery 的逐 case/单 launch 入口；不能选择最长 Task Duration。确认后的精确字符串才可传给正式 compare。

## <a id="per-case-manifest"></a>逐 case 契约与 case manifest 契约

- 一次 runner 执行多个 case 时，即使 `Op Name` 相同，也不能可靠地把聚合行归属到某个 case。调优闭环的 compare/quick 必须显式传入 `--case-manifest=<path>`，不从 Markdown 报告推断 case。
- runner 支持 `--case-id <id>` 等 CLI 协议时传 `--case-arg=--case-id`；读取环境变量时传 `--case-env=PYPTO_PERF_CASE`。
- 精确 id 分发时必须只执行该既有 case，并精确打印一行 `PYPTO_PERF_SELECTED_CASE=<case-id>`；未知 id 必须非零退出。脚本会在 profiling 前逐 id 运行并校验此 marker，再逐 case 采集；禁止把一次全量运行的同一时延复制给多行。
- 每个 profiling process 只能有一个 measured target launch。若同名目标出现多次，只有 runner marker 与 occurrence/correlation/timestamp/launch index 能唯一、可复核地绑定本次 measured launch 时才允许继续；否则 fail closed。绝不能取最大 Task Duration 偏置结果。

**manifest 格式**：通用 case manifest 是 UTF-8 JSON，至少包含 `{"schema_version":1,"cases":[{"id":"p0","shape":"[1,32]","dtype":"fp16","test_function":"test_p0"}]}`。`cases` 非空；`id`/`shape`/`dtype` 都是非空字符串，id 唯一。完整调优合同还必须包含 `"optimization_target":{"case_ids":["p0"],"selection_mode":"user_selected"}`，且 `cases` 覆盖全部性能 P0。`case_ids` 是非空、无重复的 `cases[].id` 子集，只决定候选排名；`selection_mode` 取 `user_selected`、`single_p0` 或 `all_p0_no_questions`，分别表示用户指定、唯一 P0 自动选定或用户未指定目标且要求不再提问时采用全部 P0。目标 case 必须在任何调优采集或改码前确定；全部 P0 无论是否入选仍须完成正确性、baseline/final、时间线与逐 case 披露。

`test_function` 是可选的逐 case 调用信息：它必须精确指向 `test_{op}.py` 中已通过正确性验证的无参 `test_*` 函数。已有可靠 selector 时可不写该字段；manifest 有多个 case 且调用方没有传 `--case-arg`/`--case-env` 时，每个 case 都必须提供 `test_function`，采集脚本用临时适配入口只调用该既有函数并输出 selector marker。manifest 只有一个 case 时，可省略 selector 与 `test_function` 并直接运行整个 runner，但其 profiling 进程仍必须只有一次可由精确 `Op Name` 唯一归属的 target launch，否则硬失败。

显式 manifest 是执行 case 的真值来源，并以原始字节 SHA256 和绝对路径写入 `performance_cases`、`collection.json`。采集脚本只消费并校验 `cases`；执行者须在采集前对照用户或调用方输入核验 `optimization_target`，验收时用冻结记录复核，完整原始字节身份也会冻结该字段。`collect_golden_reference.py` 以同一份 manifest 生成 `GOLDEN_PERF_REPORT.json` 机器合同和 `GOLDEN_PERF_REPORT.md` 人读报告；机器判断只读取 JSON。compare 要求已存在的 Golden JSON 所记录的 manifest 原始身份与当前文件一致，case id 集完全一致，shape/dtype 规范化后一致，再按 exact id join；多余、缺失、重复或元数据不一致均属于待修复的性能证据错误。Golden 合同不存在时，PyPTO baseline→candidate 测量仍然有效，默认 Golden 1.0 理想参考状态记为 `unavailable`；任何 Markdown 报告都不能用于 `target_met`。

## <a id="seed-contract"></a>seed 契约

manifest 协议固定使用 `torch.manual_seed(42)`，先确认 runner 在该 seed 下通过完整正确性。baseline 与 final 必须保持 seed=42 和同一输入生成方式；`--seed` 仅接受 42（传 `--case-manifest` 时；不传 manifest 的既有模式保持 `--seed=0` 与 `PYPTO_PERF_SEED` 透传语义）。

## <a id="compare-quick-batch"></a>compare / quick / batch 命令（manifest 协议）

```bash
bash "$PERF_SKILL_ROOT/scripts/msprof_profile_run.sh" --compare \
  --output-dir=custom/{operator_name} --warm-up=3 --repeats=3 \
  --op-name='<exact PyPTO-Pro lowering Op Name>' \
  --case-manifest=custom/{operator_name}/PERFORMANCE_CASES.json
```

上例由 manifest 的 `test_function` 适配既有测试；runner 自带 selector 时追加 `--case-arg=--case-id` 或 `--case-env=PYPTO_PERF_CASE`，并可省略 `test_function`。单 case manifest 也可省略三者直接执行整个 runner，但仍必须只有一次可唯一归属的 target launch。正式 baseline 和 final 都必须按 manifest 逐 case 各自完成一整套 compare；discovery profile、quick 结果或一次全量 runner 的聚合行不能替代它们。

`--quick` 仅用于按预先冻结的规则筛选候选，写入 `quick_performance.json`、`quick_performance.log`、`quick_perf_report.md`；它不会覆盖正式 compare 三件套，也不能充当正式交付证据。

采集默认在每次 profiled run 前刷新 L2 cache（cannbench 式大张量归约，`--no-l2-clean` 关闭）；该刷新在独立进程、msprof 窗口之外执行，不进入目标 `op_summary`。baseline 与 final 必须使用同一 L2 清刷设置，否则耗时不可比；该设置记录在 `performance.json` 的 `l2_clean` 字段。

批量采集会先完整预检再启动：同构算子可给全局 `--op-name`，异构目录必须给 `--op-name-map=<tsv>`（每行 `operator-directory<TAB>exact Op Name`）。每个算子目录都必须提供 `PERFORMANCE_CASES.json`，也可用 `--case-manifest-map=<tsv>` 逐算子指定（相对路径按 map 所在目录解析）。batch 不接受一个全局 manifest 静默套到异构算子。每轮 batch 日志、Markdown、JSON 均带 collection id，失败批次不会覆盖上次成功汇总。

## <a id="device-semantics"></a>设备语义

`--device` 与 `TILE_FWK_DEVICE_ID` 使用物理 NPU id。脚本启动 runner 时会移除 `ASCEND_RT_VISIBLE_DEVICES`，避免 visibility mask 把物理卡重编号为逻辑 0 后又按原 id `set_device`。外部环境只有非零 visibility mask、却未设置 `TILE_FWK_DEVICE_ID` 时会拒绝猜测，请显式传 `--device`。不传 manifest 的既有模式保持既有环境变量行为（`ASCEND_RT_VISIBLE_DEVICES`）。

## <a id="golden-collector"></a>逐 case Golden 采集器

使用默认 Golden 理想参考时，在冻结 case manifest 后对现有数学参考运行本 Skill 的采集器：

```bash
python "$PERF_SKILL_ROOT/scripts/collect_golden_reference.py" \
  "custom/<op>/<op>_golden.py" --function "<op>_golden" --factory _make_inputs \
  --case-manifest "custom/<op>/PERFORMANCE_CASES.json" --output-dir "custom/<op>" \
  --device <physical_device_id> --warmup 3 --repeats 3 --seed 42
```

采集器固定每个 repeat 只调用一次 Golden，生成 `GOLDEN_PERF_REPORT.json` 机器合同、`GOLDEN_PERF_REPORT.md` 人读报告，并把原始 profile 写到 `prof/golden-reference/<safe-case-id>/collection-*/repeat-*/`。manifest 只有一个 case 时兼容既有 `_make_inputs(device)` 的单 case 返回格式并继承唯一 id；多个 case 时 factory 必须返回 named case list，且 id 集与 manifest 精确一致，不得为此改写冻结的 Golden。其 E2E 是同 case 一次 Golden 调用产生的全部 device kernel 总和；与 PyPTO 精确 target-kernel Task Duration 执行范围不同，但本 skill 有意定义 `golden_reference_ratio = golden_per_iteration_npu_e2e_us / pypto_target_kernel_us` 作为默认跨实现理想参考。只有 JSON exact-id 完整覆盖 manifest，shape/dtype、device、seed、warm-up/repeats、固定 `iterations=1`、原始样本及聚合值自洽，所有时间均为有限正数，且使用正式 compare 时，`valid_for_target_met=true`；每个 P0 case 都 >=1.0 才 `default_target_met=true`，并据此输出 `default_target_status=met|not_met|unavailable`。这些字段只描述理想参考是否达到，不是 optimization speedup，也不是调优完成门禁；正式 baseline→final 仍必须对同一 PyPTO target kernel 使用完全相同协议各自重采。Golden JSON 不存在时 PyPTO 采集仍可产出，理想参考状态记为 `unavailable` 并如实披露。

## <a id="per-core-scope"></a>逐核证据作用域

sample-based 的 `aicore.db` 是**进程级**逐核 cycle，该表不一定能按精确 `Op Name` 关联目标 kernel；缺失/歧义或无法归属时必须记录 `per_core_status`，不得拿目标 `aicore_time` 换算并冒充目标逐核证据。仅当当前 CANN schema 能可靠把逐核行关联到精确目标 `Op Name` 时，才以目标 aicore_time / max_cycles 反推主频并换算 per-core 时长；否则只作进程级定性诊断（归档 `process_scope_core_cycles.csv`）。"逐核负载均衡"段与三档判定（<10% / 10–30% / >30%）保持原样输出。

## <a id="instruction-timeline"></a>Final 的补充指令时间线

七指标 formal compare 保持 canonical Task Duration；不要在其中开启 instruction profiling。final compare 完成后，对 manifest 的**每个 P0 case**另起一次补充采集：

最终正确性、final compare 和全部 timeline 必须对应同一份未改动的最终源码。正式 compare 只记录
严格选中 runner 的内容摘要 `executable_sha256`；compare 结束、batch 复核以及 timeline 开始和发布前
都会重算，防止拼接改码前后的性能证据。`PERFORMANCE_REPORT.md` 另记可复核的源码 revision 或 diff，
用于关联独立正确性；期间一旦改码，这三类验证全部重做。

```bash
python3 "$PERF_SKILL_ROOT/scripts/msprof_perf_summary.py" --timeline \
  --output-dir=custom/{operator_name} \
  --case-manifest=custom/{operator_name}/PERFORMANCE_CASES.json \
  --case-id=<case-id> \
  --op-name='<exact PyPTO-Pro lowering Op Name>' \
  --device=<与 final compare 相同的物理 device> \
  --warmup=3
```

该入口在同一个采集进程中启用 `--instr-profiling=on`，并以 `{"json_process":{"ascend":true,"biu_perf":true}}` 导出 task 与 BIU 时间线。脚本要求 `performance.json` 指向正式 final compare，且 target `Op Name`、manifest 原始身份、case、device、seed、warm-up 全部一致；目标 task 必须在导出 JSON 中精确且唯一。归档写入 final round 的 `case_<id>/instruction_timeline/`：

- `timeline_evidence.json`：目标 task 时间窗、受监控 Group lane、pipe 事件计数和**同一 `(pid,tid)` lane** 内的区间重叠；
- `msprof_timeline.json`：完整导出 timeline；
- `biu_perf.db` / `ascend_task.db`：原始指令与 task 数据；
- `reports.json`、采集/导出日志：命令与工具返回。

时间戳用十进制定点解析，不能把约 `1e15` 的时间戳先转二进制 float。不同 lane 同时活跃只说明不同核/监控 lane 并行，不能当作单核搬算流水。BIU 事件没有 `Op Name` 和 tile id；脚本只能借唯一 target task 时间窗归属，并输出 `pipeline_evidence_status=requires_tile_dag_correlation`，**不会自动宣布 `proven`**。必须再结合 lowering/源码 Tile DAG、稳态循环、TileGroup slot 轮转，证明该同-lane overlap 对应相邻 Tile 的 load/compute/store。instruction profiling 会扰动耗时，其 task duration 不能覆盖 formal compare 的 baseline/final 数值；只监控到的代表 lane 也不能外推成全部核均流水。工具链不支持、target 不唯一或证据缺失时记 `unverified` 并继续补证，不能用 ratio 代替。

## <a id="terminal-evidence"></a>Roofline / 流水终态证据

对每个 P0 case，完成 compare 归档后还必须补齐三组独立证据：

1. **Roofline 模型**：记录语义必需的有效计算量、分层必要搬运字节、arithmetic intensity、目标平台峰值计算/带宽及来源。由模型先路由为计算候选或搬运候选，再用本次 profiler 与受控 A/B 验证关键路径。未知平台不得套用 A5 常量。
2. **依赖与重叠**：画出 PyPTO-Pro Tile DAG。Vector 区分外层 `GM↔Vec/UB` MTE2/MTE3 与 VF 的 `Vec Tile↔register`；Cube 区分 MTE2 `GM→Mat`、MTE1 `Mat→Left/Right`、Cube/M、Acc 及输出。检查 TileGroup `current()/next()`、slot/mutex、`auto_mutex`、stage/preload 的真实依赖。按上面的 `--timeline` 对 final 的每个 P0 case 补采；只有唯一目标时间窗内的同-lane事件和当前实现的 DAG/slot/stage 映射共同证明相邻 Tile 重叠，才能写 `pipeline_evidence.status=proven`。同一证据证明存在合法重叠边但当前实现未形成重叠时，写 `residual_not_overlapped` 并进入 candidate sweep。无 trace、只有跨 lane 并行或只有区间交叠而无 Tile 映射都是 `unverified`。若 DAG 证明没有两个可流水 work item 或没有任何合法可重叠边，可写 `not_applicable_with_dag` 并附 DAG。
3. **Scalar 分析**：ratio 只做报警，最终须结合 hot region、trace/生成代码及单变量 A/B 判断 Scalar、同步等待、逐 Tile 地址/mask/descriptor/branch 是否为关键路径。Scalar 指令存在是正常的；`scalar_evidence.dominant=true` 作为残留瓶颈进入候选 sweep，`unknown` 表示证据仍不可评估。

理想的 `roofline_terminal.status` 是 `compute_bound`、`data_movement_bound` 或 `balanced_compute_movement`，理想流水状态为 `pipeline_evidence.status ∈ {proven, not_applicable_with_dag}`，理想 Scalar 状态为 `scalar_evidence.dominant=false`。`scalar_bound`、`wait_bound`、`residual_not_overlapped` 或 Scalar `dominant=true` 必须如实记录并进入 final candidate sweep，但在全部相关候选闭合后不单独阻止交付；终验才发现新项时返回优化闭环并在改动后重做最终验收。`insufficient_evidence`、`unverified`、`unknown`、来源过期或仅有 ratio 标签仍属于证据不可评估。最终实现必须另起一次与 baseline 同协议的 formal compare 重采；终态证据路径写入 `PERFORMANCE_REPORT.md`，不得引用已被后续改动淘汰的中间轮次。

## <a id="archive-layout"></a>持久归档

```
custom/{算子名}/docs/perf/
├── round_001/
│   ├── collection.json                 ← collection 状态与逐 case 完成情况
│   ├── case_<case-id>_<hash>/
│   │   ├── measurement.json
│   │   ├── repeat_001/
│   │   │   ├── op_summary_PipeUtilization.csv  ← canonical Task Duration
│   │   │   ├── op_summary_<其余六组 Metric>.csv
│   │   │   ├── op_statistic_<Metric>.csv、task_time_<Metric>.csv
│   │   │   ├── evidence_status.json
│   │   │   ├── per_core_cycles.csv（仅能可靠归属目标 Op Name 时）
│   │   │   ├── process_scope_core_cycles.csv（不能归属时的进程级聚合，不得冒充目标结论）
│   │   │   └── summary.txt
│   │   └── repeat_NNN/...
│   │   └── instruction_timeline/        ← final 每 P0 case 的补充时间线
│   └── case_<下一 case>_<hash>/...
└── ...
```

既有单层 `round_NNN/op_summary_<Metric>.csv + per_core_cycles.csv + summary.txt` 结构在标准模式（不传 manifest）下保持不变。

## <a id="csv-semantics"></a>CSV 字段与阈值的调优语义

- 本表是 A2/A3 来源字段字典，不证明 Ascend 950/A5 导出的 schema 与阈值相同。每次采集都先读取实际 CSV header、工具版本、SoC 与单位；字段缺失/更名时标记 unavailable，禁止补零或按列序猜测。表中所有百分比只保留为来源诊断 prior，不能单独判 `target_met`、接近 roofline 或优化完成；正式结论以 SPEC、同协议 `Task Duration(us)` compare 和当前平台证据为准。
- `Task Duration(us)` 是同协议 baseline/final compare 的核心指标，并作为逐 case Golden 参考比值的 PyPTO target-kernel 分母。
- `block_id` 是 logical block id/index，不是核数；launch 数看 `Block Dim`，实际参与数须统计唯一 id 并结合 sub-block/trace。
- `Current Freq` / `Rated Freq` 作为采样上下文记录；单行差异不能证明性能回退由 DVFS 导致，需重复采样、设备状态与同协议 A/B 判断。
- `aic_fixpipe_ratio >15%` 仅提示 FIXP 活跃；地址/布局原因需 trace 与 A/B 验证。
- bankgroup/bank/resc/mte 冲突：先核对生成地址/trace，再单变量 A/B 布局或流水编排；不粘贴 AscendC 参数名，以 PyPTO-Pro Tile 实际 shape/stride/padding 与 generated CCE/trace 为证。
- 各"瓶颈判定阈值"（VEC >50%、SCALAR >30% 等）只用于安排下一项调优实验，不用于判定完成。

## <a id="a5-boundary"></a>A5 证据状态与使用边界

- A5 参考页中的原语代价表、lever 收益、校准测量与可达带宽均为 **`unverified_external_historical`**：本 skill 不含原始 profiler artifact/命令，因而这些数值不可从仓内独立复算。它们只能用于给同 SKU 候选排序，**不能代替当前算子的 trace/A-B**，也不能直接写成当前算子的已验证结论。
- 参考页提到的私有 `roofline_a5.py` helper 与 `tasks/metadata/950pr.json` 等外部 benchmark 元数据不在本 skill 内，因此没有可执行的 `tools/roofline_a5.py` 命令；如需复用其公式，从当前 checkout 重算并保留 ini 路径、键与算术过程。
- `DAV_3510` 加上核数不唯一确定 SKU；拿不到可验证的设备名/HAL 时，数值 roofline 标记 unavailable。
- 公式中的符号必须先归一化到 SI 单位（例如 `cube_freq=1650` MHz 是 `1.650e9 cycles/s`；`vec_freq` 同理）。`ddr_rate` 在所选 ini/schema 未说明是 bytes/cycle/core、速率系数还是已归一化带宽之前，不得代入带宽公式。算不出单位就宁可不给数字。
- 数据墙（字节不可约减 + 占用率达到设备极限）只用于关闭受证据支配的候选，本身不是调优成功条件；目标差距可提高候选优先级，但全部来源、候选与 final sweep 合法闭合后不阻止交付。
- 停止纪律见 [`pypto-pro-op-kb/references/investigation-discipline.md`](../../pypto-pro-op-kb/references/investigation-discipline.md)：保留反事实证据、如实报告阻断，禁止把历史阈值转写成当前成功。

## <a id="msprof-op-boundary"></a>msprof op 证据优先级与补充边界

- 性能判定以真实 NPU 上板数据为准。当前 PyPTO-Pro 没有已核实的 Execute Graph/pass verify/sim-run 选项；若项目另有 simulator，只能作为诊断辅助，必须记录工具版本和误差，并以上板结果验收。
- 本仓 `perf_summary.py` 只提供 msprof op 归档统计（不校验归属、不做达标判定）；其产物不能替代调优报告和正式 compare 三件套；正式 baseline/final 必须走本文件的 compare 协议。
- compare 的 quick 筛选产物固定为 `quick_performance.json`、`quick_performance.log`、`quick_perf_report.md`，不会覆盖正式三件套且不能交付。
- 若做 msprof op 附加诊断，应先用精确 `Op Name` 筛目标行；每个 profiling process 应只有一个 measured target launch；同名重复只有 occurrence/correlation/timestamp/launch index 可验证绑定时才可使用，否则 fail closed，禁止取最大 Task Duration。
- 仅当当前 CANN 的 `msprof op`/`msopprof` 明确支持 Python runner 时使用该页流程；PyPTO-Pro 的已核实官方入口是 `msprof python3 test_kernel.py`，环境不支持本模式时回到 [`msprof-guide.md`](msprof-guide.md)，不得套用 AscendC 可执行文件构建流程。
- `Current < Rated` 只生成 DVFS 假设；需重复采样或受控频率 A/B 才能归因。来源默认 `--warm-up=10` 可作起点，但当前 case 须先画/查序列确认稳态，再冻结 baseline/final 相同 warm-up 与 repeat；不能把 10 当跨平台常数。
