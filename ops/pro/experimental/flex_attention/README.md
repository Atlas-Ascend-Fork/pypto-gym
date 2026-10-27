# FlexAttention (flex_attention)

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |

## 功能说明

- 算子功能：mask_info 契约驱动的 FlashAttention。输入 TND 连续的 q/k/v，按 `compile_mask.md`
  的 mask_info 描述符生成逐 tile 的可见性位图，执行带掩码的 online-softmax flash-attention，
  输出 `[tq, n, d]` 的注意力结果。

  - mask_info 契约由上游（接口层）数据派生出三种 schedule，经属性 `schedule_type` 传入 kernel：
    - `DENSE (0)`：`tile_range1/tile_range2` 区间调度（前缀全可见 + 稀疏尾）。
    - `BITMASK (1)`：`block_range` + `block_compute/block_mask` 位图调度。
    - `CALLABLE (2)`：位图 + `bounds` 驱动的可调用行求值（kernel 内绑定固定 `mask_mod`）。
  - 描述符 `maskr/mask_start/mask_len`（`mask_num` 个掩码间隙，从可见前缀 `[0, maskr)` 中扣除）在 kernel 内由 `decode_mask` 生成逐 tile 位图。
  - 全可见块（`mask_bit==0`）原经 GM `zero_mask` 张量 load；本交付形态已将 zero_mask 退役，
    改为向量单元就地 `fill_zero_mask` 合成全零位图（消除 GM 辅助张量，无用户 workspace）。

- 实现为单次 launch 的 PyPTO delivery-form kernel（`op_kernel/...py`），core 间按**代价均衡**的
  `work_starts` 切分（host tiling 经数据依赖读 `actual_seq_q` + schedule 代价张量做二分均衡）。

- 计算公式：

$$
O = \mathrm{softmax}\!\left(\frac{Q K^{\mathrm{T}}}{\sqrt{d}} \odot \mathrm{mask}\right) V
$$

其中 mask 由 mask_info 描述符在 kernel 内逐 tile 生成。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| ------ | -------------- | ---- | -------- | -------- |
| query | 输入 | query，`[tq, n, d]`。 | FLOAT16、BFLOAT16 | ND |
| key | 输入 | key，`[tkv, n, d]`。 | FLOAT16、BFLOAT16 | ND |
| value | 输入 | value，`[tkv, n, d]`。 | FLOAT16、BFLOAT16 | ND |
| maskr | 输入 | 描述符右界，`[B, Pf]`，`Pf = Rf*128`。始终存在。 | INT32 | ND |
| mask_start | 可选输入 | 描述符掩码间隙左界，`[B, mask_num, Pf]`，`mask_num>0` 时必传。 | INT32 | ND |
| mask_len | 可选输入 | 描述符掩码间隙宽度，`[B, mask_num, Pf]`，`mask_num>0` 时必传。 | INT32 | ND |
| block_range | 可选输入 | 每 Q tile 上界 KV tile 数（kv_loop），`[B, Rf]`（`BITMASK/CALLABLE`）。 | INT32 | ND |
| tile_range1 | 可选输入 | DENSE 前缀全可见块数，`[B, Rf]`（`DENSE` 时必传）。 | INT32 | ND |
| tile_range2 | 可选输入 | DENSE 末可见块索引+1，`[B, Rf]`（`DENSE` 时必传）。 | INT32 | ND |
| block_compute | 可选输入 | 计算位图，`[B, Rf, Wf]`（`BITMASK/CALLABLE` 时必传）。 | UINT32 | ND |
| block_mask | 可选输入 | 掩码位图，`[B, Rf, Wf]`（`BITMASK/CALLABLE` 时必传）。 | UINT32 | ND |
| bounds | 可选输入 | 可调用行批级右界，`[B, num_bounds]`（`CALLABLE` 且 `num_bounds>0` 时必传）。 | INT32 | ND |
| actual_seq_q | 输入 | per-batch q 前缀和，`[B]`。 | INT32 | ND |
| actual_seq_kv | 输入 | per-batch kv 前缀和，`[B]`。 | INT32 | ND |
| attn_out | 输出 | 注意力输出，shape 同 `query`。 | FLOAT16、BFLOAT16 | ND |
| schedule_type | 必选属性 | 调度类型：0=DENSE，1=BITMASK，2=CALLABLE。 | INT | - |
| num_bounds | 可选属性 | `bounds` 维度（`CALLABLE`），默认 0。 | INT | - |

## 约束说明

- `d`（head_dim）仅支持 128。
- `mask_num` 非独立输入：host tiling 从 `mask_start` 的 shape（dim 1）推导，
  `mask_start` 缺省即 `mask_num=0`；上限 256（`mask_num > 16` 走 HoleChunk 分块解码）。
- `schedule_type` 与所传可选输入须匹配：DENSE 需 `tile_range1/tile_range2`；
  BITMASK/CALLABLE 需 `block_range/block_compute/block_mask`；CALLABLE 另需 `bounds`。
- `actual_seq_q` 为 **tiling 数据依赖输入**：host tiling 需其 Host 数据派生 `total_work` 与
  per-batch 块数；框架须将其标记为 tiling 数据依赖（`gert::TilingContext` 在配置后才返回
  Host 地址，见 `tiling_context.h`）。schedule 的代价张量（`block_compute` 或
  `tile_range1/tile_range2`）为可选数据依赖：数据可用则按代价均衡切分，不可用则退化为
  均匀连续切分（仍正确，kernel 内 `compute_bit==0` 跳过空块）。
- 无用户 workspace（zero_mask 已退役为向量单元就地合成）。

## 调用说明

| 调用方式 | 说明 |
| -------- | ---- |
| Kernel 直调（PyPTO） | 参考 `python/tests/st/pypto_pro/frontend/fa/test_flex_attention.py`：经 host 入口 `flex_attention(q, k, v, seq_q_list, seq_kv_list, maskr, mask_start, mask_len, block_range, block_compute, block_mask, attn_out, softmax_max, softmax_sum)` 构造 `FlexAttnTiling` 并启动 `flex_attention_kernel[None, blockDim, None, datatype](...)`（`mask_num` 从 `mask_start.shape[1]` 推导）。 |
| aclnn API | 待补齐。 |

## 目录结构

```
flex_attention/
├── op_kernel/
│   └── flex_attention.py   # PyPTO delivery-form kernel（raw Ptr + OpTiling）
└── op_host/
    ├── flex_attention_def.cpp       # 算子注册
    ├── flex_attention_infershape.cpp
    ├── flex_attention_tiling.cpp/.h # host tiling（与 op_kernel FlexAttnTiling 字段对齐）
    ├── CMakeLists.txt
    └── config/ascend950/
        ├── flex_attention_binary.json
        └── flex_attention_simplified_key.ini
```

验证：`python3 python/tests/st/pypto_pro/frontend/fa/test_flex_attention.py`（覆盖 f16/bf16、bitmask/dense/callable、mask_num=0、契约校验）。
