# CSA2 Compressor 算子说明文档（训练用）

## 1. 算子概述

CSA2 Compressor 是 DeepSeek V4.1-Flash 架构中的 **token 压缩器（latent compressor）**：将连续
`ratio` 个 token 压缩为 1 个 pre-RoPE latent，供 Attention 侧使用（RoPE / Hadamard 由 Attention
侧施加，压缩器本身不做）。KV 池化权重由 gate 投影经 softmax 产生。

**本实现为训练算子**，算子实现只有三个文件，职责与依赖分层如下：

### 1.1 文件组成

| 文件 | 职责 | 对外接口 |
|---|---|---|
| `csa2_compressor_fwd_impl.py` | **前向实现**。包含三个 `@pypto.frontend.jit` kernel（`csa2_compressor_ratio1_kernel` 纯投影分支、`csa2_compressor_kernel` ratio=2 BSH 压缩分支、`csa2_compressor_th_kernel` ratio=2 TH (BS 合轴) 分支（含 prefill + decode）），并将算子注册到 `torch.library`（`pypto::csa2_compressor`，Meta shape 推导 + NPU dispatch） | `CSA2CompressorArgs`（namedtuple 参数打包）、`csa2_compressor_pypto(args)`（走 `torch.ops` 派发）、`npu_csa2_compressor(args)`（`@allow_in_graph` 直调，绕过派发，图捕获友好；内部按 `cu_seqlens` 是否为 None 分发到 `_csa2_compressor_bsh` / `_csa2_compressor_th`） |
| `csa2_compressor_bwd_impl.py` | **反向实现**。包含三个 kernel（`csa2_compressor_bwd_ratio1_kernel`、`csa2_compressor_bwd_kernel`（BSH）、`csa2_compressor_bwd_th_kernel`（TH），静态块对齐 + 权值单次 bf16 量化），注册为 `pypto::csa2_compressor_bwd`；wrapper 侧完成余数行零填、scratch 分配、权值预备与 state 切片 | `CSA2CompressorBwdArgs`（namedtuple 参数打包）、`csa2_compressor_bwd_pypto(args)`、`npu_csa2_compressor_bwd(args)`（按 `cu_seqlens` 分发到 `_csa2_compressor_bwd_bsh` / `_csa2_compressor_bwd_th`），返回 `(dx, dwkv, dwgate, dweight)` |
| `csa2_compressor.py` | **公共入口（autograd 组合层）**。不含 kernel，只做两件事：前向调 `npu_csa2_compressor` 并保存反向所需上下文（含前向**写后**的 state 切片）；backward 调 `npu_csa2_compressor_bwd` 并按 forward 入参顺序回传梯度。同时处理 state 预分配（dim0 > bsz）的精确切片 | `CSA2CompressorFunction`（`torch.autograd.Function`）、`csa2_compressor(...)`（functional）、`CSA2Compressor`（`nn.Module`，持有 ratio / eps） |

### 1.2 依赖分层

```
csa2_compressor.py（autograd 入口，训练推荐用法）
    ├── csa2_compressor_fwd_impl.py   →  npu_csa2_compressor        （前向, BSH/TH 分发）
    └── csa2_compressor_bwd_impl.py   →  npu_csa2_compressor_bwd    （反向, BSH/TH 分发）
```

- 入口文件的依赖边界（除 pip 包外）只有两个 impl 文件的 eager wrapper（本目录内相对引用），
  三个文件可整体移植到任意包目录；
- 前向 / 反向 wrapper 供公共入口（§5）内部调用，**不对外直接调用**；训练统一经公共入口使用。

测试位于 `tests/ops/deepseek_v4_1/`（精度 golden + 泛化用例 + 性能基准例），不属于算子实现。

运行环境：昇腾 NPU（910B3 与 Ascend910_9392 均验证通过），基于 PyPTO 框架
（`pypto.frontend.jit`）开发，依赖 `torch` / `torch_npu`。测试经 `deepseek_v4_1.` 包式
导入加载本目录实现。TH 扩展（本 README 所载）另于 Ascend910（soc 251）/ CANN 9.1.0
上全量复测通过。

---

## 2. 正向算法

记 `x: (bsz, seq, h)`（BSH；TH 见 §2.3），`h = 5120`（静态维，JIT 特化维度，主验证维；反向功能例 / 反向泛化例
另覆盖 `h = 4096`）；`d = head_dim`（如 512 / 128）；`ratio ∈ {1, 2}`。

### 2.1 ratio == 1（纯投影分支）

```
out = rms_norm(wkv(x))          # bf16 投影（不转 fp32），无 gate / state / 池化
```

- 每 token 产出一个 latent，输出 `(bsz, seq, d)`，**永不返回 None**；
- `wkv` 保持 bf16，`wgate / kv_state / score_state` 不存在（传 `None`）。

### 2.2 ratio == 2（压缩分支）

公共投影（`wkv/wgate` 统一 bf16，乘积 fp32 累加；`x` 先 `float()`）：

```
kv    = x_fp32 @ bf16(wkv)^T    # (bsz, seq, d)
score = x_fp32 @ bf16(wgate)^T  # (bsz, seq, d)
```

**整段输入（start_pos == 0，训练常规用法，seq 可为任意长度）**：

```
should_compress = seq >= 2
remainder = seq % 2             # 尾部余数 token 数
完整组 g（[2g, 2g+2)）:
    alpha_g = softmax(score_g, dim=组内)      # 2 个 token 的池化权重
    latent_g = Σ_i alpha_g[i] * kv_g[i]
余数 token:
    kv_state[:bsz, :remainder], score_state[:bsz, :remainder] = 投影结果   # 原地写入
```

**单 token 输入（start_pos != 0，每次输入位置 start_pos 处的 1 个 token）**：

```
slot = start_pos % 2
kv_state[:bsz, slot], score_state[:bsz, slot] = 当前 token 投影结果          # 原地写入
should_compress = (start_pos + 1) % 2 == 0
组满时: latent = Σ_slot softmax(score_state, dim=槽位) * kv_state            # 输出 1 行
```

**输出**：`should_compress` 为真时

```
out = rms_norm(latent.to(bf16))              # 池化结果先落回 bf16 再 RMSNorm → pre-RoPE latent
```

形状 `(bsz, max(seq // ratio, 1), d)` bf16。**非压缩步**（整段输入时 `seq < 2`，或单 token 输入时
组未满）返回 `None`；kernel wrapper 仍返回同形状的**占位 tensor（未写入，值无定义）**，
调用方应以 `seq >= ratio` / `(start_pos+1) % ratio == 0` 判断是否消费输出。盲反传占位输出的语义
见 §4.2（整段输入 `seq < ratio` 为全零梯度；单 token 组未满会按当前 state 池化真实回传当前槽位
梯度——规范用法是不反传占位输出）。

### 2.3 TH (BS 合轴) 模式

x 为 `(t, h)` 2D 时按 `cu_seqlens` 分段（batch b 的 token 为 `x[cu[b]:cu[b+1]]`）：

- 每个 batch 段**独立**分组池化（组网格在段起点重启，不跨段配对）；段尾余数 token 写
  `kv_state[b, 0]`；ratio==1 对分段不敏感（逐 token 投影）；
- ratio==2 prefill 输出为**紧凑拼接**：batch b 的压缩行写到 `out_base[b]` 起
  （`out_base[b] = Σ_{i<b} L_i // ratio`，由 wrapper 设备侧计算后传入 kernel：B=1 短路
  为常量 0，B>1 走 fp32 AiCore cumsum，均无 host 同步），
  有效行数 `Σ_b L_b // ratio`，其后为 pad；总行数 `Sr = min(t, t//ratio + bsz)`；
- TH decode（start_pos != 0）要求每段恰 1 token（`cu_seqlens == [0, 1, ..., bsz]`），
  `slot = start_pos % 2` 全 batch 一致，跨 batch 批量处理，输出行按 batch 序；
- 长度 0 的段跳过：不写 state、不产生输出。

### 2.4 数值约定

- `kv_state` 初值 0，`score_state` 初值 `-inf`（`register_buffer` 语义）；**前向原地更新**二者；
- softmax / rsqrt 使用 `PrecisionType.HIGH_PRECISION`；RMSNorm 内部 fp32，输入先 cast bf16
  （对齐 `kv.to(dtype)` 后再 norm）；
- matmul fp32 累加。

---

## 3. 正向 wrapper 接口（`csa2_compressor_fwd_impl.py`）

### 3.1 参数打包与注册

```python
CSA2CompressorArgs = namedtuple('CSA2CompressorArgs',
    ['x', 'wkv', 'wgate', 'weight', 'kv_state', 'score_state', 'start_pos', 'cu_seqlens',
     'ratio', 'eps'])   # defaults=(None, 2, 1e-20); cu_seqlens 位于标量参数之前, BSH 时传 None
```

算子已通过 `torch.library` 注册为 `pypto::csa2_compressor`（含 Meta shape 推导；NPU dispatch
注册依赖 torchair，未安装时自动跳过）。文件对外提供 `csa2_compressor_pypto(args)`（走
`torch.ops` 派发）与 `npu_csa2_compressor(args)`（`@allow_in_graph` 直调，绕过派发，图捕获
友好）；后者先 `check_args` 校验，再按 `cu_seqlens` 是否为 None 分发到 `_csa2_compressor_bsh` /
`_csa2_compressor_th`（各自分配 out 并调 kernel；fwd 的 state 预分配切片在公共入口完成）。
二者供公共入口（§5）内部使用，不对外直接调用。

### 3.2 tensor 契约

| 参数 | 形状 | dtype | 说明 |
|---|---|---|---|
| `x` | BSH: `(bsz, seq, 5120)`；TH: `(t, 5120)` | bf16 | BSH 的 seq 为动态轴，h 为静态特化维（实测 5120） |
| `wkv` | `(d, 5120)` | bf16（接受 bf16/fp32，内部归一） | kv 投影权值；ratio1/2 统一 bf16 |
| `wgate` | `(d, 5120)` 或 `None` | bf16（接受 bf16/fp32，内部归一） | gate 投影权值；ratio==1 必须为 None |
| `weight` | `(d,)` | fp32 | RMSNorm 权重（`kv_norm.weight`） |
| `kv_state` | `(bsz, 2, d)` 或 `None` | fp32 | 不可微 cache，**前向原地更新**；ratio==1 必须为 None |
| `score_state` | `(bsz, 2, d)` 或 `None` | fp32 | 初值 `-inf`；同上 |
| `start_pos` | `(1,)` | int32（NPU 设备张量） | token 序列起始位置：`0` = 整段输入（从位置 0 开始分组），`>0` = 单 token 输入（当前位置 start_pos）；需在 aclgraph 捕获前创建，勿在捕获内新建 |
| `cu_seqlens` | TH: `(bsz+1,)` 或 BSH: `None` | int32 | **TH (BS 合轴) 前缀和**：`cu[0]==0`、`cu[-1]==t`、非降；batch b 的 token 为 `x[cu[b]:cu[b+1]]`。x 为 3D (BSH) 时必须为 None |
| `ratio` | — | int | 1 或 2 |
| `eps` | — | float | RMSNorm epsilon（常用 `1e-20`） |
| 返回 `out` | BSH: ratio==1 `(bsz, seq, d)`；ratio==2 `(bsz, max(seq//2, 1), d)`。TH: ratio==1 `(t, d)`；ratio==2 prefill `(min(t, t//2+bsz), d)`（compact 压缩行 + pad），decode `(bsz, d)` | bf16 | 非压缩步为占位（见 §2.2） |

**完整约束（模式与形状、dtype、start_pos 与 decode 单 token、state、反向输入、图捕获）见 §6**；
TH (BS 合轴) 语义与 `out_base` 说明见 §2.3。

---

## 4. 反向算法（`csa2_compressor_bwd_impl.py`）

### 4.1 梯度语义

- `kv_state / score_state` 视为**不可微 cache**（`register_buffer(persistent=False)`），不产生
  梯度（返回 `None`）；梯度回传到 `x / wkv / wgate / weight`。
- 单 token 输入时只回传**当前调用**产生的梯度，不回传历史 token 的梯度。
- 反向需要读**前向更新后**的 state（单 token 输入分支的池化来源）。

### 4.2 三分支反向

**ratio == 1**（`out = norm(wkv(x))`）：

```
kv = x @ wkv^T                                  # 前向投影重算（bf16）
dkv, dweight = rms_norm_backward(grad_out, kv, weight)
dkv = dkv.to(bf16)                              # 对齐 kernel cube matmul 边界的量化点
dx   = dkv @ wkv                                # (bsz, seq, h)
dwkv = Σ_{batch,seq} dkv^T @ x                  # (d, h)，展平 batch/seq 求和
```

**ratio == 2 整段输入**：完整组可微；余数 token 无输出 → **梯度为 0**。

```
dlatent, dweight = rms_norm_backward(grad_out, bf16(latent), weight)
d_alpha = dlatent * kv_g
d_kv    = dlatent * alpha                       # 完整组 token 的 kv 梯度
d_score = softmax_backward(d_alpha, alpha)      # softmax 解析反向
```

**ratio == 2 单 token 输入**：latent 由 state 池化，**仅当前 token 写入的槽位可微**：

```
alpha   = softmax(score_state)                  # 前向后的 state
dlatent, dweight = rms_norm_backward(grad_out, bf16(latent), weight)
d_kv[slot]  = dlatent * alpha[slot]             # slot = start_pos % 2，其余槽位梯度 0
d_sc[slot]  = softmax_backward(dlatent * kv_state, alpha)[slot]
```

**非压缩步的"盲反传"**（对占位输出直接 backward）：整段输入 `seq < ratio` 时该段无组/槽贡献
→ 梯度全零；单 token 组未满时会按当前 state 池化**真实回传当前槽位的梯度（非零）**（与 golden
同口径）——规范用法是不消费、不反传该占位输出。

### 4.3 公共尾段与数值策略

所有分支汇合到公共尾段：

```
dx    = d_kv @ wkv + d_score @ wgate
dwkv  = Σ d_kv^T @ x ;  dwgate = Σ d_score^T @ x
```

数值策略（对齐 kernel 的 bf16-MAC 形态）：

- `dx / dwkv / dwgate` 的 cube matmul 操作数在边界量化 bf16、fp32 累加，`dx` 单次最终舍入；
- 前向 `kv/score` 重算对齐 kernel gb 段语义：**权值单次量化 bf16**（`wkv/wgate.to(bf16)`，与
  golden 逐位一致），x 原生 bf16，乘积 fp32 精确、fp32 累加（残余差异仅有累加顺序，容差内覆盖）；
- wrapper 侧完成 `d_kv/d_sc` scratch 分配与权值 bf16 量化预备，余数行零填（BSH 由 wrapper 批量
  零填、TH 由 kernel 内按段零填）；kernel 端各组块按 `valid_g` / 尾块 `valid_shape` 裁剪，无越界写。

### 4.4 反向 wrapper 接口

```python
CSA2CompressorBwdArgs = namedtuple('CSA2CompressorBwdArgs',
    ['grad_out', 'x', 'wkv', 'wgate', 'weight', 'kv_state', 'score_state',
     'start_pos', 'cu_seqlens', 'ratio', 'eps'])      # cu_seqlens 位于标量参数之前
# 返回: (dx, dwkv, dwgate, dweight)
```

与正向同构：`csa2_compressor_bwd_pypto(args)` 走 `torch.ops` 派发，`npu_csa2_compressor_bwd(args)`
为 `@allow_in_graph` 直调，供公共入口（§5）内部使用，不对外直接调用。返回值：

| 返回值 | 形状 | dtype | 说明 |
|---|---|---|---|
| `dx` | BSH `(bsz, seq, 5120)`；TH `(t, 5120)` | bf16 | 与 x 同形 |
| `dwkv` | `(d, 5120)` | 同 `wkv`（bf16） | |
| `dwgate` | `(d, 5120)` 或 `None` | 同 `wgate`（bf16） | ratio==1 时为 `None` |
| `dweight` | `(d,)` | fp32 | |

入参契约同 §3.2（`grad_out` 形状与前向输出一致，bf16），另注意：

- `kv_state / score_state` 必须传**前向更新后**的值（单 token 输入分支从中读取）；
- state dim0 预分配（`> bsz`）同样支持：反向只读 state，wrapper 精确切片（TH 按段数 bsz），
  无写回语义；
- BSH `grad_out` 按前向输出形状 `(bsz, max(seq//2,1), d)` 直传（无 padding）；余数行梯度由
  wrapper 零填（`s1 % ratio != 0` 时批量清零各 batch 尾行）；
- TH `grad_out` 为前向输出 2D `(Sr, d)`，仅每段 compact 前缀
  `[out_base[b], out_base[b]+ng_b)` 有效（pad 行不参与）；段尾余数行梯度由 kernel 内按段零填。

---

## 5. 正反向联合接口（公共入口，`csa2_compressor.py`）

对外暴露三种形式，内部串联前向 kernel 与反向 kernel：

```python
CSA2CompressorFunction        # torch.autograd.Function：forward / backward 入口
csa2_compressor(...)          # functional 形式
CSA2Compressor(ratio, eps)    # nn.Module 形式（持有 ratio / eps，权重与 state 由调用方传入）
```

```python
# functional (cu_seqlens 位于标量参数 ratio/eps 之前)
out = csa2_compressor(x, wkv, wgate, weight, kv_state, score_state,
                      start_pos, None, ratio=2, eps=1e-20)          # BSH
out = csa2_compressor(x, wkv, wgate, weight, kv_state, score_state,
                      start_pos, cu, ratio=2, eps=1e-20)            # TH (x 2D)

# nn.Module
model = CSA2Compressor(ratio=2, eps=1e-20)
out = model(x, wkv, wgate, weight, kv_state, score_state, start_pos, cu_seqlens=None)
```

autograd 语义：

- 前向调用 `npu_csa2_compressor`，**原地更新** `kv_state/score_state`，并经 `ctx.save_for_backward`
  保存全部输入与前向**写后的 state 切片**（TH 下同时保存 `cu_seqlens`）；
- `out.backward(grad_out)` 触发 `npu_csa2_compressor_bwd`，梯度按 forward 入参顺序返回
  （`dx, dwkv, dwgate, dweight, None, None, None, None, None, None`），落入各输入的 `.grad`；
- `kv_state/score_state/start_pos/ratio/eps/cu_seqlens` 无梯度；
- state 预分配（dim0 > bsz）时入口内精确切片（TH 按段数 bsz），前向原地写落回大 buffer
  `[0, bsz)` 行，反向经 ctx 保存的切片读到前向后的值。

---

## 6. 约束汇总（调用方必读）

以下为算子**全部约束**。`check_args` 仅做 rank / shape / dtype / 设备一致性等**静态校验**；
标注「不校验」的约束不会报错，违反时结果未定义，须由调用方保证（算子内部不做任何 D2H 同步，
故无法在运行时校验设备标量/数组的**值**）。

### 6.1 模式、形状与标量参数

| 约束 | 校验 |
|---|---|
| `ratio ∈ {1, 2}` | ✅ |
| `start_pos` 必须为 `(1,)` int32 设备张量 | ✅ |
| `eps ≥ 0`（非负 float） | Module 构造与反向 wrapper 校验；前向 wrapper 不校验 |
| x 布局二选一（wrapper 按 `cu_seqlens` 分发）：`cu_seqlens=None` → BSH，x 必须 rank 3 `(bsz, seq, h)`；`cu_seqlens` 给定 → TH，x 必须 rank 2 `(t, h)` 且 `cu_seqlens` 为 1D int32 `(bsz+1,)`；两者不一致报错 | ✅ |
| `cu_seqlens[0]==0`、`cu_seqlens[-1]==t`、非降（允许相等 = 空段）；TH 下 batch 数 `bsz = numel-1`（不是 `t`） | ❌ 不校验 |
| `wkv.shape[1] == x.shape[-1]`；`weight.shape[0] == wkv.shape[0] == d`；`h`、`d` 为 JIT 静态特化维（不同取值触发重新编译；常用 `h` 4096/5120、`d` 512/128） | ✅ |
| BSH 整段输入 `seq < ratio` 无有效输出（返回占位，见 §2.2）；TH 段长可为 0（空段跳过：不写 state、不产生输出） | — |
| `bsz / seq / t` 取 0 的极端形状未纳入当前验证范围 | — |
| 输出形状：BSH ratio1 `(bsz, seq, d)` / ratio2 `(bsz, max(seq//ratio,1), d)`；TH ratio1 `(t, d)` / ratio2 prefill `(min(t, t//ratio+bsz), d)`（compact 压缩行 + pad，**pad 行未写入、值未定义**）/ decode `(bsz, d)` | ✅（Meta 推导） |

### 6.2 dtype 契约

| 张量 | ratio==1 | ratio==2 |
|---|---|---|
| `x` / `grad_out` | bf16 | bf16 |
| `wkv` | 前向 bf16（接受 bf16/fp32，内部归一 bf16）；反向同 | 前向 bf16（接受 bf16/fp32，内部归一 bf16）；反向同 |
| `wgate` | 必须为 `None` | bf16（接受 bf16/fp32，内部归一 bf16） |
| `weight` | fp32（前向强制；反向 bf16/fp32） | 同左 |
| `kv_state` / `score_state` | 必须为 `None` | 必须提供，`(bsz, 2, d)` fp32 |
| `start_pos` | int32 `(1,)` | 同左 |
| `cu_seqlens` | TH 时 int32 `(bsz+1,)`；BSH 必须为 `None` | 同左 |
| 反向输出 dtype | `dwkv` 跟随 `wkv`、`dweight` 跟随 `weight`、`dwgate=None` | `dwkv/dwgate` 跟随 `wkv/wgate`、`dweight` 跟随 `weight` |

注：`wkv/wgate` 在 ratio1/2 下统一 bf16。前向 wrapper 接受 bf16/fp32 并在 host 侧归一为 bf16，kernel 以 bf16 cube 执行、
fp32 累加；golden 同步按 bf16 权值对齐。反向 wrapper 同样放宽到 bf16/fp32 并内部统一量化 bf16。

### 6.3 start_pos 与 decode 单 token（值不校验）

- `start_pos` 为 `(1,)` int32 **设备张量**：`0` = 整段输入（prefill，从位置 0 分组）；`>0` = 单 token
  （decode，当前位置）；须在 aclgraph 捕获前创建、勿在捕获内新建。ratio==1 忽略该值（无 state）。
- **decode 仅支持每 batch 恰 1 token**（多 token decode 暂未实现）：BSH 要求 `x seq == 1`；TH 要求
  `T == bsz`（即 `cu_seqlens == [0,1,...,bsz]`）。**不做运行时校验**（读 `start_pos` 需 D2H 同步）；
  违反时不报错，kernel decode 分支会按"行序 = batch 序"静默错算（BSH 忽略每 batch 第 0 行以外的
  token；TH 忽略实际段长）。
- 单 token 组未满时输出为占位（未写入）；是否消费输出由调用方按 `(start_pos+1) % ratio == 0`
  判断；盲反传语义见 §4.2。

### 6.4 state 约束

- ratio==2 必须提供 `kv_state/score_state`，形状 `(bsz, 2, d)` fp32；**前向原地更新**、不可微
  （梯度返回 `None`）；反向必须传**前向更新后**的值；decode 分支仅当前 token 槽位可微。
- dim0 允许 `> bsz`（`max_batch_size` 预分配）：前向经公共入口、反向经 wrapper 精确切片到
  `(bsz, 2, d)`（**TH 按段数 bsz，而非 t**）；切片共享存储，原地写落回大 buffer `[0, bsz)` 行，
  `[bsz:)` 行不受影响。
- 初值惯例（`register_buffer`）：`kv_state` 全 0、`score_state` 全 `-inf`。

### 6.5 反向输入约束

- `grad_out` 形状必须与前向输出一致（BSH `(bsz, n_out, d)`，n_out = `seq` / `max(seq//ratio,1)`；
  TH `(Sr, d)`，Sr = `t` / `min(t, t//ratio+bsz)`），dtype 与 `x` 相同（bf16）。
- **设备一致性**：反向显式校验 `grad_out / wkv / wgate / weight / kv_state / score_state / start_pos`
  与 `x` 同设备（`cu_seqlens` 未纳入该校验）；前向均未显式校验，须调用方保证全部输入同设备。
- 余数行零填：BSH 由 wrapper（`s1 % ratio != 0` 时批量清零各 batch 尾行）；TH 由 kernel 内按段零填。
- 反向 wrapper 内部完成 `grad_out.contiguous()`、权值 `.to(bf16)` 量化与 scratch 分配
  （`torch.empty`，不引入额外算子约束）。

### 6.6 图捕获（aclgraph）

- `start_pos` / `cu_seqlens` 及各输入输出 buffer 须在捕获前创建；算子内部不做任何 D2H 同步
  （decode 单 token 校验因此不做，见 §6.3）。
- 形状动态性由 JIT 承担：`h` 静态特化，`bsz / seq / t` 动态；TH 的 `out_base` 前缀和由 wrapper
  内设备侧小算子计算（B=1 短路为单 mul 归零；B>1 走 fp32 AiCore cumsum，规避 int32 cumsum
  落 AICPU 的 ~39us 开销），图捕获下进入图。

---

## 7. 调用案例

```python
import torch
import torch_npu  # noqa: F401 —— NPU 设备初始化必需
from deepseek_v4_1.csa2_compressor import csa2_compressor, CSA2Compressor

device = "npu:0"
bsz, seq, h, d = 8, 3, 5120, 512

x      = torch.rand((bsz, seq, h), dtype=torch.bfloat16, device=device).requires_grad_(True)
wkv    = torch.rand((d, h), dtype=torch.bfloat16, device=device).requires_grad_(True)
wgate  = torch.rand((d, h), dtype=torch.bfloat16, device=device).requires_grad_(True)
weight = (torch.rand((d,), dtype=torch.float32, device=device) + 0.5).requires_grad_(True)
kv_state    = torch.zeros((bsz, 2, d), dtype=torch.float32, device=device)
score_state = torch.full((bsz, 2, d), float("-inf"), dtype=torch.float32, device=device)
start_pos   = torch.zeros(1, dtype=torch.int32, device=device)

# 方式一：functional
out = csa2_compressor(x, wkv, wgate, weight, kv_state, score_state,
                      start_pos, ratio=2, eps=1e-20)          # (bsz, seq//2, d) bf16
loss = out.float().mean()
loss.backward()      # 梯度落入 x.grad / wkv.grad / wgate.grad / weight.grad

# 方式二：nn.Module（持有 ratio / eps）
model = CSA2Compressor(ratio=2, eps=1e-20)
out = model(x, wkv, wgate, weight, kv_state, score_state, start_pos)
```

其他场景调用形式不变，只换参数：

| 场景 | 差异 |
|---|---|
| ratio=1 | `CSA2Compressor(ratio=1)`；`wkv` 用 bf16；`wgate / kv_state / score_state` 传 `None` |
| 单 token 输入 | BSH：`seq=1`；TH：`cu_seqlens == [0,1,...,bsz]`（每段 1 token）；`start_pos` 传当前位置；组满（`(pos+1) % ratio == 0`）才有有效输出 |
| TH (BS 合轴) | `x` 传 2D `(t, h)`、`cu_seqlens` 传 `(bsz+1,)` int32 前缀和（BSH 时 `cu_seqlens` 必须传 `None`）；ratio2 输出 2D compact + pad，ratio1 输出 `(t, d)` |
| state 预分配 | `kv_state / score_state` 的 dim0 可大于 bsz（按 max_batch_size），入口自动切片（TH 按段数） |

---


## 8. 精度基线与测试

精度基线（dtype 键控，与 golden `torch` 实现对比）：

| 输出 | dtype | atol | rtol | 备注 |
|---|---|---|---|---|
| `out` / `dx` | bf16 | 1e-4 | 0.0078125 | 舍入边界翻转类另加**越限元素比例 ≤ 1e-3** |
| ratio1 `dwkv` | bf16 | 1e-4 | 0.0078125 | **逐元素全量**（无越限比例豁免） |
| `kv_state` / `score_state` / `dweight` | fp32 | 2.5e-5 | 0.005 | 累积类 按 `n_tok = bsz*seq` 档放宽 atol |
| ratio2 `dwkv` / `dwgate` | bf16 | 1e-3（`n_tok≥65536` → 4e-3） | 0.0078125（bf16 1 ulp） | **逐元素全量**（无越限比例豁免）；fp32 累加后单次舍入 |

测试（`tests/ops/deepseek_v4_1/` 9 个文件、217 个用例，已全量通过）：

```bash
# 在 pypto-gym 仓库根目录执行
# 全量（实测约 12 分钟，含 128K 长序列泛化与性能基准例）
TILE_FWK_DEVICE_ID=0 python3 -m pytest tests/ops/deepseek_v4_1/ -v -s

# 单文件 / 单用例
TILE_FWK_DEVICE_ID=0 python3 -m pytest tests/ops/deepseek_v4_1/test_csa2_compressor_fwd.py -v
TILE_FWK_DEVICE_ID=0 python3 -m pytest "tests/ops/deepseek_v4_1/test_CSA2_compressor_entry.py::test_csa2_entry_ratio1" -v
```

| 测试文件 | 覆盖 |
|---|---|
| `test_csa2_compressor_fwd.py` | 前向：整段输入 / 余数 / d128 / ratio1 / 单 token / 单 token 组满压缩（含 TH golden） |
| `test_CSA2_compressor_fwd_gen.py` | 前向泛化：B 1~256、S 1~131072、对齐/不对齐 |
| `test_csa2_compressor_bwd.py` | 反向：ratio1/2、单 token、预分配 state + 17 组泛化（seq 128~128K）+ 3 性能基准例（含 bsz=1 seq=8K 调优目标例，含 TH golden） |
| `test_CSA2_compressor_bwd_gen.py` | 反向泛化：B/S/d 矩阵 + 单 token 双槽预填 |
| `test_CSA2_compressor_entry.py` | 公共入口 fwd+bwd 连跑（Module / functional / autograd.Function / 预分配 state） |
| `test_CSA2_compressor_entry_gen.py` | 入口泛化：大 B×大 S（128K token）、奇偶对齐档、随机泛化 |
| `test_CSA2_compressor_th_fwd.py` | **TH 前向**（18 例）：cu_seqlens 分段、compact+pad 输出、ratio1/2、TH decode；泛化含 8 对齐边缘簇、B=32 varlen（T=4K/8K/16K/64K/128K，lens 为随机递增序列一次性生成后**固化字面量**，每次运行一致；同一 cu_seqlens 内对齐/不对齐混合），每条用例标注测试点 |
| `test_CSA2_compressor_th_bwd.py` | **TH 反向**（17 例）：逐段组块反向、段尾余数零梯度、TH decode 盲反传（与 golden 同口径）；同结构泛化（bwd 独立固化数据，含 T=64K/128K） |
| `test_CSA2_compressor_th_entry.py` | **TH 入口正反连跑**（17 例）：Module/functional、max_batch_size 预分配（按段数切片）、同结构泛化（entry 独立固化数据，T=4K/8K/16K/128K + ratio1 大 varlen） |

测试产物（2026-10-09 全量复测（TH 扩展前的 165 例），Ascend910_9392 / CANN 9.1.0；均位于 `tests/ops/deepseek_v4_1/perf/`）：

| 产物 | 路径 |
|---|---|
| 精度测试报告（165/165 通过，含 workspace/IO 采集） | `tests/ops/deepseek_v4_1/perf/CSA2_compressor_precision_report.md` |
| 性能测试报告（msprof 采集，golden 小算子 vs pypto kernel） | `tests/ops/deepseek_v4_1/perf/CSA2_compressor_perf_report.md` |
| 交付报告 | `tests/ops/deepseek_v4_1/perf/CSA2_compressor_交付报告.md` |
| 性能比对 / msprof 采集 / 解析 / 报告生成脚本 | `perf/perf_cmp_compressor.py` / `perf/run_msprof_perf.sh` / `perf/parse_msprof_perf.py` / `perf/gen_perf_report.py`、`perf/gen_precision_report.py` |
| 调优记录与根因分析 | `perf/logs/tune_record.md`、`perf/logs/msprof_analyze/ANALYSIS_2100_vs_4800.md` |
| 全量测试日志 / msprof 原始数据 | `logs/` |
