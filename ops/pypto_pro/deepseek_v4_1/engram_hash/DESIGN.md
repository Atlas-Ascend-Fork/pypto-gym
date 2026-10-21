# engram_hash 设计说明（DESIGN）

> 实现：`op_kernel/engram_hash.py`
> 用例：`tests/test_engram_hash.py`
> golden：`tests/assets/engram_hash_golden.py`

## 1. 计算定义

对每个 token、每个 n-gram layer：

```text
hash_0 = int64(id_0) * mult[layer, 0]
hash_i = hash_{i-1} XOR (int64(id_i) * mult[layer, i]),           i = 1..G-1
out[layer, t, (i-1)*Tb + tb] = hash_i % vocab[layer, i-1, tb] + offsets[layer, (i-1)*Tb + tb]
out[layer, t, :] = -1                                             当 image_token_mask[t] 为真
```

DeepSeek V4.1 使用 `L=2`、`Tb=8`、`G=4`，输出 `[2, T, 24]`；层数、头数与 `G` 均可配置。

## 2. 线程映射与数据通路

- grid 为 `ceil(T / 128)`，每个 block 使用 128 个 SIMT 线程，一个线程对应一个 token；
  `layer` 循环为编译期展开，每个线程独立完成自身 token 的全部输出列。
- SIMT 线程直接从 GM 读取 `ngram_token_ids`、`multipliers`、`vocab_sizes`、`offsets`、
  `image_token_mask`，并将结果直接写回 `output`。实现不使用 UB tile 作为 SIMT 实参，
  也不使用 MTE2/MTE3 搬运；访存由 D-cache 服务。
- 尾块由运行时上界 `valid = min(128, T - row)` 保护，越界位置不写出。

## 3. 编译期专门化

当前工具链的 `pl.simt.launch` 要求 Tensor 实参具有静态 shape，因此 kernel 以
`(L, G, Tb, T, has_mask)` 为键进行专门化；wrapper 内部缓存已编译变体，相同键的重复
调用不重复编译。`has_mask` 对应两个 SIMT 变体：不传 `image_token_mask` 时不生成 mask
参数与 mask 分支。

## 4. 取模语义

参考实现使用 torch 的 `%`（向下取整）。设备侧整数取模为截断语义，当被除数为负时两者
相差一个除数，因此实现中对负余数做一次加法修正：

```python
r = h % d
if r < 0:
    r = r + d
```

该修正对任意 int64 输入与参考实现逐位一致。热态下的分项对照显示其开销约占
0.007 ms / 13.6 MB（T=65536），故始终开启。

## 5. mask 处理

`image_token_mask=None` 时不传入 mask 张量，使用不含 mask 读与分支的 kernel 变体；
传入 mask 时使用带 mask 的变体，`True` 的 token 输出整行写 -1。两种形态在测试中均覆盖。

## 6. Host 接口与下发

- 参数校验按 rank、dtype、shape、device 的顺序执行，不符合契约时抛出 `ValueError`，
  混设备输入在 launch 前拦截。
- `make_offsets` 在 host 侧计算 exclusive prefix-sum（int32），与参考实现一致。
- 调用路径开销（参数校验、形状整理与编译产物下发）与内核执行相互独立，不随内核规模
  变化；重复调用命中已编译变体。

## 7. 已知限制

- `T` 变化会引入一次新的编译；编译在 JIT 缓存内完成，重复调用不受影响。
- 调用路径开销与内核规模无关；单次调用规模较小时，其占比相对更高。
- 调用方需保证 `token_id × multiplier` 不溢出 int64（与参考实现契约一致）。
