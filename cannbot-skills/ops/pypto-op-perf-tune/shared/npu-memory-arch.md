# Ascend NPU 内存体系参考（调优用）

> 核心真源：《昇腾950 NPU架构白皮书》（表3-1 SKU 规格、表4-2 Buffer 物理容量、§4.3 内存层次）
> 用途：TileShape 设置、UB/L1 容量规划、泳道图搬运分析、Stitch/合图调优时的硬件约束查询。表内为**族系典型值**，LLM 规划阶段可直接使用。kernel 代码中获取精确值见下文「在 kernel 代码中获取 Buffer 容量」。

## 一、AI Core 片上 Buffer 层级（内存金字塔）

| Buffer | 910B2 (DAV_2201) | 950PR (DAV_3510) | 用途 |
|---|---|---|---|
| **L1** | 512 KB | 512 KB | Cube 输入缓存（矩阵左右操作数驻留） |
| **L0A / L0B** | 64 KB / 64 KB | 64 KB / 64 KB | Cube 左/右矩阵操作数 |
| **L0C** | 128 KB | **256 KB（翻倍）** | Cube 输出，决定基本块 Tiling 上限 |
| **UB** | 192 KB | **248 KB（每 AIV 可用）** | Vector 工作区，每 AIV 独立一份，SIMT/SIMD 共享 |

### Cube tile 对齐下限

**`CUBE_ALIGN = 16`**（NZ fractal `C0=16`，即 CANN `BLOCK_CUBE=16`；两平台相同）。
kL0/nL0 必须 `CUBE_ALIGN` 对齐（FC4001 编译检查），mL0 仅 `a_trans=True` 时须对齐。
详见 [tile-shapes.md](../../pypto-general-debug/references/tile-shapes.md)。
S-14 Mix合图场景 mL0 ≥ 64（优先 128），见 tune-swimlane/references/merge-optimization.md §4.2。

### UB 容量口径说明

- 950 白皮书表 4-2 标注 "UB 512 KB per AI Core"，实际按 **AI Core 组（1 AIC + 2 AIV）** 计物理容量：
  - 单 AIV 物理 256 KB × 2 = 组级 512 KB
  - 单 AIV 用户可用 = 256 KB − 8 KB 预留 = **248 KB**（INI `ub_size = 253952`，即 `GetCoreMemSize(UB)` 返回值）
- SIMT 场景还需为 DCache 让位 ≥ 32 KB
- 同一代架构内各 SKU 的 L1/L0/UB/BT 通常一致；L2 与 Memory 可能因子型号/形态而异

## 二、L2 与全局 Memory（因子型号/形态而异）

| 参数 | 910B2 | 950PR PCIE | 950PR Server |
|---|---|---|---|
| Cube 核数 | 24 | 28 | 32 |
| 频率 | 1.8 GHz | 1.65 GHz | 1.65 GHz |
| **L2** | 192 MB | 112 MB | 128 MB |
| **Memory 容量** | 64 GB | 112 GB | 128 GB |
| **Memory 带宽** | 1.6 TB/s（经验值） | 1.4 TB/s（白皮书证实） | 1.6 TB/s（白皮书证实） |

白皮书 §4.3 补充：

- 950DT：Memory 144 GB / 4 TB/s
- L2 为 UMA 架构，512B CacheLine、4×128B Sector，支持 L2 Hint 与 CMO

**调优含义**：带宽是搬运算子（纯 Vector、GM↔UB 密集）的屋顶线；L2 容量决定大权重/激活驻留策略（如 weight-none-l2-cacheable 场景）。

## 在 kernel 代码中获取 Buffer 容量

本表 §一/§二 为族系典型值，供 LLM 规划阶段估算 tile 预算。kernel 代码中须以下列 API 获取精确值，确保跨 SKU 可移植：

### C++ tiling 代码（AscendC）

```cpp
platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
uint64_t ub_size, l0c_size, l1_size, l0a_size, l0b_size;
platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0c_size);
platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1_size);
platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, l0a_size);
platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0b_size);
```

`CoreMemType` 枚举：`L0_A, L0_B, L0_C, L1, L2, UB, HBM, FB, BT`

头文件：`${ASCEND_HOME_PATH}/<arch>/asc/include/utils/tiling/platform/platform_ascendc.h`

### 核数（Python runtime）

```python
from pypto_pro.runtime.platform import get_platform_info
cube_cores = get_platform_info().cube_core_num
vector_cores = get_platform_info().vector_core_num
```

> `pypto.platform.npuarch` 仅返回架构字符串（如 `'DAV_3510'`），用于条件分支，不返回 Buffer 容量。

## 参考信息

- [昇腾950 NPU架构白皮书](https://public-download.obs.cn-east-2.myhuaweicloud.com/ascend/%E6%98%87%E8%85%BE950%20NPU%E6%9E%B6%E6%9E%84%E7%99%BD%E7%9A%AE%E4%B9%A6.pdf)（DAV_3510 官方规格与微架构真源）
- [npu-arch SKILL.md](https://gitcode.com/cann/cannbot-skills/blob/master/ops/npu-arch/SKILL.md)
- [npu-arch-guide.md](https://gitcode.com/cann/cannbot-skills/blob/master/ops/npu-arch/references/npu-arch-guide.md)
- CANN 安装包：`${ASCEND_HOME_PATH}/<arch>/asc/include/utils/tiling/platform/platform_ascendc.h`
