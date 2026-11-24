# GM 物理行号表消费：原 MTE2→S 卡点与 fused prologue

状态：**已解决（native row table + fused prologue）** ｜ 更新：2026-11-23

## 原卡点

旧 far 实现把地址表放 GM 后，行首 `pl.load`（MTE2）到 UB bank，再由 gather 的
`pl.getval`（标量 S 管线）直接读 bank。编译器不跟踪 MTE2→S 依赖，标量读到上一行的
旧表；当上一行与本行 ori/cmp 拍数分布不同时，会把 ori 池地址当 cmp 地址使用，
触发 error 95（MTE 越界）或死锁。

## 现在的 fused prologue

attention kernel 的 vector 段先由本 block 构建自己后续消费的行：

1. 读取原始 sparse indices / block table（8192 宽块表按 batch 缓存在 UB），
   VF 解析 `row = block_table[page] * bs + off`；
2. 每行按 128 列 tile 写入 GM workspace（FD partials 之后）；ori/cmp 两遍复用块表 tile；
3. `bar_all` 后进入主循环；主循环每 tile 把 64 个物理行号用 MTE2 搬进 UB，
   VF 用 `mull/addc` 把 int32 page/off 转成 int64 地址并写入 int64 pair 表（V→S），
   主循环标量取 `(min, gap)` 后用 int64 stride 发 KV load。

只保留 `MTE2→V` 和 `V→S` 两条可跟踪依赖；旧 1024 项 UB 块表窗口、`far_row_vf`、
`_load_kv_group_far*`、`BT_GM` 和独立 builder launch 均已删除；GM 行号表解析 VF
现在只在 fused prologue 内。

## 当前边界

- fused prologue 的块表 UB tile 固定 8192 项，因此 host/wrapper 限制 block-table 宽度 ≤ 8192。
- per-tile pair 表改为 int64 地址，不再有 int32 池跨度限制；2 GiB gap 合成用例已通过。
- `spread_8192`（8192 个有效页，`bs_ori=128` / `bs_cmp=64`）约 570 MB。
