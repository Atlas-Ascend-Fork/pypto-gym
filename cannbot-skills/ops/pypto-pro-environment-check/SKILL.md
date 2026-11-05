---
name: pypto-pro-environment-check
description: 诊断 PyPTO-Pro 导入、CANN 配置、NPU 可用性及编译/运行超时等疑似环境问题，区分环境故障与算子故障。按症状选择有界检查，返回事实、结论范围和下一步建议。
---

# PyPTO-Pro 环境检测

目标是在当前环境中取得足以指导下一步的证据。根据已有错误、设备状态和独立对照选择检查，复用仍有效的结果。

## 诊断依据

- 使用当前 Python/CANN 配置、目标设备和已有日志；官方 smoke 还需 `PYPTO_DEVKIT_DIR` 指向目标版本 devkit。
- 记录失败命令、原始错误、目标设备和检查范围。配置问题在建议中列出具体组件及官方安装资料。
- 单个 smoke 通过只证明它覆盖的路径在该次运行可用，不能排除其他执行域、shape、资源压力或间歇性环境问题。smoke 失败也不能单独证明设备损坏。

## 选择有信息增量的检查

| 当前证据 | 可选检查及用途 |
|---|---|
| 明确的 import 失败、CANN 路径或版本错误 | 直接核对对应组件，必要时运行下方 `env_preflight.py`；无需先运行必然受同一缺失阻断的 kernel |
| 设备枚举不到卡 | 核对设备可见性配置（如 `ASCEND_RT_VISIBLE_DEVICES`），结合双后端枚举结果与设备状态定位原因 |
| 超时、无响应、设备告警或同步/拷贝异常 | 有界读取设备状态及原始日志，区分编译、运行和同步阶段；状态不足以解释故障时，再选择独立对照 |
| 自定义 kernel 失败，尚不清楚环境是否可用 | 在同卡、同环境运行官方 VF smoke 或能覆盖疑似故障路径的目标版本官方样例 |
| VF smoke 已通过，原 kernel 仍失败 | 结合原始日志定位 kernel，或补查 smoke 未覆盖的执行域和条件；不重复运行相同对照来宣称整个环境正常 |
| 换卡/跨机器耗时异常，或健康正常但目标负载异常慢 | 分别记录物理/逻辑映射、健康/错误、占用及目标负载执行期频率；空闲查询和短矩阵 smoke 不能替代目标负载读数 |
| runtime、编译器或平台核数读数不一致 | 保留每项读数及工具版本，核对实际 native launch 和有效工作数；不覆盖查询值，也不套用另一 SKU 的峰值 |

为本轮检查选择总时间预算和各命令超时，考虑已有 JIT 编译耗时、设备状态及任务时间限制。超时是待解释的现象；短超时、进程退出成功或单次设备查询都不能独立证明永久 hang 或全链路健康。

只在出现新证据、条件变化，或有明确理由认为等待能恢复时重试，并说明下一次检查能排除什么。证据足够、预算耗尽或重复检查已无信息增量时返回；原因未定则使用 `INCONCLUSIVE`，列明缺口与下一项有区分力的检查，不以固定重试次数自动升级为设备故障。

## 官方 VF smoke

devkit 中的官方脚本为：

```text
$PYPTO_DEVKIT_DIR/pro_ops/vf_api/test_softmax_tile_group_vf.py
```

它编译 softmax VF kernel（`@pl.jit(auto_mutex=True)` 与 `@pl.vector_function`），运行 6 组动态 shape，与 `torch.softmax` 对比。使用目标版本 devkit 中的脚本，缺失时报告所需版本与路径。

```bash
TILE_FWK_DEVICE_ID=<id> timeout 600 python "$PYPTO_DEVKIT_DIR/pro_ops/vf_api/test_softmax_tile_group_vf.py"
```

`TILE_FWK_DEVICE_ID` 是本次命令的选卡参数；未指定卡号时默认 0。600 秒是包含 JIT 编译的参考超时，可根据日志与剩余预算调整。预算不足以完成编译时如实记录未完成，不能据此判 hang。

PASS 需要同时满足进程退出码为 0、每个 case 输出 `PASS`、末尾含 `Test completed!`。记录实际设备、版本及覆盖范围：

- PASS：本次 VF smoke 覆盖路径可用；结合原故障是否使用相同路径决定继续调试或补充对照。
- FAIL：保留原始错误，按软件、设备、编译或数值比对证据判断故障类别。
- TIMEOUT：记录最后进度与是否进入设备执行。优先解释停在哪一层，再决定是否值得延长超时。

## 设备异常与精度失败

以逐 case 的比对证据解释 `TOTAL n/N passed`。同步、拷贝或调用在比对前失败时，该 case 记为“未产生精度结论”；已完成比对的 case 保留真实结果。对 0/N 的归因结合比对是否执行及故障定位证据。

以下证据可帮助区分原因，需结合目标型号、驱动版本、运行日志与独立对照判断：

| 观察 | 下一步判别 |
|---|---|
| `npuSynchronizeDevice` / `copy_between_host_and_device_opapi` 等入口抛异常 | 确认是否在比对前失败，并核对设备状态；kernel 越界或同步缺陷也可能触发运行时异常 |
| 未修改的多个 case 同时失败 | 核对共同环境与输入依赖，用独立工作负载对照，不能只凭失败数量归因 |
| 异常 core id 或跨进程重复的设备错误 dump | 核对该 SKU 与版本的编号含义、时间戳及告警，判断是否在重复读取同一设备错误 |
| 自定义 kernel 启动前，纯 torch control 也失败 | 这是独立于本次 kernel 的环境线索；保留其命令、设备和原始错误 |
| `npu-smi` 告警，或无进程时仍持续占用 | 与进程、温度及驱动状态交叉核对；查询工具异常或残缺 stub 时可用 preflight 的 torch_npu 后端补证 |
| 比对实际执行并报告误差指标 | 保留真实误差指标，结合数值/逻辑/同步及环境证据定位根因 |

独立对照、设备状态和日志足以支持结论时，报告设备故障，并可列出其他可用卡及换卡建议。恢复后须记录恢复前后的证据及重新通过的检查。

## 分层诊断脚本

需要补充软件缺失、设备/架构/版本信息或候选卡列表的证据时，可按需运行本 skill 的脚本。将 `<skill-dir>` 替换为实际 skill 路径，并为命令设置与本轮预算相符的超时：

```bash
timeout <seconds> python "<skill-dir>/scripts/env_preflight.py"
```

脚本汇总检测结果，输出人类可读摘要及末行 `PREFLIGHT_JSON: {...}`。它依赖同目录 `scripts/_npu_info.py` 和 `scripts/get_npu_arch.py`。

| 层 | 检测与输出 |
|---|---|
| 设备 | npu-smi 为主，torch_npu 为回退；`devices` 提供卡号、名称与可取得的健康信息，`npu_usable` 表示能否枚举可用卡 |
| 架构 | 经 `libascend_hal.so` 探测，输出 `npu_arch` |
| 运行时 | torch、torch_npu、pypto_pro.language / pl.jit 导入；`torch_npu_ok`、`pypto_pro_ok` |
| CANN | toolkit 路径、版本及 OPP；`cann_version` |

`errors` 是阻断项，`warnings` 是非阻断线索；`passed` 仅表示脚本未发现阻断错误，不代替实际 kernel 验证。若脚本本身超时或未产生完整 JSON，保留已有输出并报告检查未完成。

## 诊断结论

按实际执行情况填写以下字段；未执行 smoke 时写 `NOT_RUN`：

```text
device_assessment: HANG_CONFIRMED / TRANSIENT_RECOVERED / DEVICE_HEALTHY / FAULT_NO_TIMEOUT / INCONCLUSIVE
smoke_script: <实际脚本路径，或 NOT_RUN>
smoke_result: PASS / FAIL / TIMEOUT / NOT_RUN
evidence: <命令、设备、退出码、原始错误/状态、耗时与检查覆盖范围>
recommendation: <下一项检查、需修复的配置或候选卡及建议依据>
```

`HANG_CONFIRMED` 表示证据支持持续设备 hang，`FAULT_NO_TIMEOUT` 表示没有超时但已有设备故障证据；二者都不能由一次异常或重试计数推导。`DEVICE_HEALTHY` 须附本次已验证路径，不代表所有负载健康。`TRANSIENT_RECOVERED` 须有前后对照，原因仍不清楚时报告 `INCONCLUSIVE`。

报告区分事实、推断和建议，附未覆盖范围。配置修复建议可引用 [torch_npu 安装资料](https://gitcode.com/cann/pytorch/releases) 及目标版本 CANN / PyPTO-Pro 官方文档。
