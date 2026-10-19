# Qwen3-VL-8B RMSNorm 算子集成


## 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 概述

将 RMSNorm 算子从原始 torch 实现替换为 PyPTO 融合算子，提升 NPU 上的推理性能。

## 文件结构

```
ops/pypto_tensor/qwen3_vl_8b_instruct_unredacted_max/
└── rms_norm/
    ├── rms_norm_impl.py           # PyPTO 实现
    └── README.md                  # 本文档
```

## 测试验证

```bash
cd tests/ops/qwen3_vl_8b_instruct_unredacted_max

python3 test_rms_norm.py
```

### 测试用例来源

从 Qwen3-VL-8B 模型打点采集的真实 shape/dtype：
- prefill 阶段：[1, 11, 2048]
- decode 阶段：[1, 1, 2048]

## 技术说明

### 场景判断

原始实现使用纯 torch 基础算子（pow、mean、rsqrt），属于**场景A**：
- Golden 直接复制原始代码
- 无需 torch_npu 验证

### 实现选择

使用 PyPTO 内置 `pypto.rms_norm` 融合算子实现。

## 性能数据

> 测试环境: Ascend 910B, CANN 8.5.0
>
> 测试方法: Swimlane (泳道图) — `debug_options={"runtime_debug_mode": 1}`, 解析 `merged_swimlane.json` X 事件 span

| 算子 | 输入 shape | 输入 dtype | kernel 耗时 (μs) | 数据来源 |
|------|-----------|-----------|-----------------|---------|
| rms_norm (prefill 主 norm) | [1, 11, 2048] | float16 | 25.4 | Swimlane |
| rms_norm (prefill q_norm) | [1, 11, 16, 128] | float16 | 30.9 | Swimlane |
| rms_norm (prefill k_norm) | [1, 11, 8, 128] | float16 | 19.9 | Swimlane |
| rms_norm (decode 主 norm) | [1, 1, 2048] | float16 | 19.8 | Swimlane |
| rms_norm (decode q_norm) | [1, 1, 16, 128] | float16 | 5.3 | Swimlane |
| rms_norm (decode k_norm) | [1, 1, 8, 128] | float16 | 4.9 | Swimlane |

## 状态

✅ 环境验证
✅ 网络基线验证
✅ 打点采集
✅ Golden 编写
✅ 单算子精度验证
✅ 单算子性能采集
✅ 模型集成
✅ 端到端验证
⏳ 性能调优
