# RMSNorm 算子集成 (Phi-3-mini-4k-instruct)


## 产品支持情况

- Ascend 950PR：不支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 概述

将 Phi3RMSNorm 替换为 PyPTO 融合算子。D=3072，FP16。

## 测试

```bash
export TILE_FWK_DEVICE_ID=4
python3 tests/ops/phi_3_mini_4k_instruct/test_rms_norm_phi_3_mini_4k_instruct.py
```

## 测试用例来源

从模型打点采集的真实 shape/dtype：decode [1,1,3072]，prefill [1,5,3072] / [1,128,3072] / [1,4096,3072]。

## 技术说明

| 项目 | 说明 |
|------|------|
| 场景 | A — 原始实现只用 torch 基础算子，Golden 直接复制原始代码 |
| 实现 | PyPTO 手动 tiling，TILE_M=4，DYNAMIC 序列长度 |
| ACLGraph | 已注册 `torch.library` — `pypto::rms_norm_phi3`，支持 `--use-acl-graph` |

## 状态

✅ 单算子精度 | ✅ 整网集成 | ✅ ACLGraph | ⏳ 性能调优
