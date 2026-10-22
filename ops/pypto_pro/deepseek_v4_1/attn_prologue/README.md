# attn_prologue（PyPTO-Pro）

`attn_prologue` 是用 PyPTO-Pro 实现的 MXFP8 attention 前置处理（attention prologue）融合算子。
一个 kernel 完成 QA 投影、KV 投影、RMSNorm、QR 动态 MXFP8 量化、第二次 Q 投影、尾部 RoPE 以及
KV cache 的分页写回；公开输出 `(q, qr, descale_qr)`，`kv_cache` 就地更新。生产维度固定为
`(H,R,N,D,Dr)=(5120,1280,64,512,64)`，唯一动态轴是 token 数 `T=batch*seq`，支持范围
`[1, 262144]`，覆盖 decode 与 prefill 两类形态。接口契约、数值语义与支持范围由本目录的实现与
测试定义。

## 产品支持情况

| 项 | 取值 |
| :--- | :--- |
| 芯片 | Ascend 950PR / 950DT（`NpuArch.DAV_3510`，arch `3510`） |
| 能力依赖 | `matmul_mx` / `matmul_mx_acc`（MXFP8 矩阵乘：E4M3FN×E4M3FN，FP32 累加，E8M0 分组 scale）、FRACTAL_NZ 权重及 NZ/ZN 同址双视图、`MemorySpace.ScaleLeft/ScaleRight`（scale 地址 = 数据地址 >>4 的硬件耦合）、`AccToVecMode.DualModeSplitM`（L0C→UB 按 M 维对半分发）、VF 寄存器级动态量化（`bit_cast`/位移构造 `2^-e`/`astype` 到 E4M3FN 的 `CAST_RINT` 舍入/`StoreDist.PACK4` 字节密排）、VF 尾部 RoPE（寄存器 `gather` 去交错）、`sync_all(MIX)` 与 `set/wait_cross_core` 的 Cube/Vector 握手、分页 KV 写回（`pl.getval` 取索引 + 运行期偏移 `pl.store`）。以上能力在目标工具链中仅按 950 系列登记支持 |
| 分核数 | 运行期由 `pypto_pro.runtime.platform.get_platform_info()` 查询，`block_dim = min(cube_core_num, vector_core_num//2)`，代码内不硬编码核数 |
| 实测环境 | DAV_3510，32 Cube / 64 Vector；CANN 9.3.0、PyPTO-Pro、PyTorch 2.12.0+cpu、torch_npu、Python 3.12 |
| 测试标注 | `pytestmark = pytest.mark.soc("950")` + `@pytest.mark.npu`；不满足 SoC 标注的用例由仓库 conftest 统一跳过 |
| 依赖 | pypto_pro、torch、torch_npu、NumPy、pytest |

## 功能说明

一个公开 host 函数加一个模块级 `@pl.jit(auto_mutex=True)` 的 MIX kernel 服务全部输入；每次公开
调用在设备侧只发射一个 kernel。三个矩阵乘在 Cube 上执行：`matmul_mx` 直接消费 E4M3 payload 与
E8M0 分组 scale、以 FP32 累加，全程不物化反量化后的 X；归约、归一化、量化、RoPE 与写回在
Vector 上执行。kernel 编译为单一二进制（不使用 tiling key），运行期按 token tile 数选择工作
分配方式（见 `DESIGN.md`）。

计算全流程（`Deq` 表示结合 E8M0 scale 的逻辑反量化，`Quant32` 表示沿特征轴每 32 个元素共享
一个 scale 的动态量化，`@` 表示矩阵乘）：

```text
X        = Deq(x, descale_x)                                    # [T,H]
qa       = X @ Deq(wqa).T                                       # [T,R]  FP32 累加
kv       = X @ Deq(wkv).T                                       # [T,D]  FP32 累加

qr, descale_qr = Quant32(RMSNorm(qa, norm_weight_qr, norm_eps)) # [T,R] e4m3 + [T,R/64,2] int8
q_proj   = Deq(qr, descale_qr) @ Deq(wqb).T                     # [T,N*D] 消费的是量化后的 qr
q        = BF16(reshape(q_proj,[T,N,D]) → TailRoPE)             # [T,N,D] bf16

kv_rope  = TailRoPE(RMSNorm(kv, norm_weight_kv, norm_eps))      # 先归一化和 RoPE，再量化
kv_bytes = ByteView_uint8(E4M3FN(clamp(kv_rope, -448, 448)))    # scale 固定为 1
kv_cache[cache_index[t]//BS, cache_index[t]%BS, 0, :] = kv_bytes[t, :]   # 就地写回
```

**RMSNorm 与动态量化**：RMSNorm 对每个 token 的完整特征行独立计算，不减均值：
`RMSNorm(v, gamma)[j] = v[j] * gamma[j] / sqrt(mean(v^2) + norm_eps)`；QA 支路沿 R 归约，KV 支路
沿 D 归约。QR 每 32 个元素组成一个量化组，对非零组：

```text
amax  = max(abs(group))
e     = clamp(floor(log2(amax)) - 8, -127, 127)      # floor(log2) 直接取 FP32 指数字段
scale = 2^e
payload    = E4M3FN(clamp(group / scale, -448, 448)) # 最近偶数舍入
scale 字节 = e + 127                                 # E8M0 偏置编码
```

全零组的 payload 与 scale 字节都写 0。相邻两个组的 scale 字节成对存储，因此每行 R 个元素对应
`[R/64,2]` 的 scale 形状。注意 E8M0 字节 0 表示 `2^-127`，不是数值 0。

**尾部 RoPE**：Q 的每个 head 与 KV 向量都保留前 `D-Dr` 维逐位透传，只变换末尾 `Dr=64` 维。
令 z 为尾部向量、`h=Dr/2=32`，对 `i=0..h-1`（INTERLEAVE_HALF 约定，旋转符号由 sin/cos 表自身
承载，公式内不出现负号）：

```text
out_tail[i]     = z[2i] * rope_cos[t,i]   + z[2i+1] * rope_sin[t,i]
out_tail[h + i] = z[2i] * rope_sin[t,h+i] + z[2i+1] * rope_cos[t,h+i]
```

同一 token 的位置表在全部 N 个 Q head 上广播。

## 函数原型

```python
# op_kernel 加入 sys.path 后：
from attn_prologue import attn_prologue, to_nz

q, qr, descale_qr = attn_prologue(
    x, wqa, wqb, wkv,
    descale_x, descale_wqa, descale_wqb, descale_wkv,
    norm_weight_qr, norm_weight_kv,
    rope_sin, rope_cos, cache_index, kv_cache,
    norm_eps=1e-6,
)
```

`to_nz(weight)` 把 NPU 上的权重（`wqa/wqb/wkv`）转换为 FRACTAL_NZ 布局
（`torch_npu.npu_format_cast(w, 29)`），应在权重加载期一次性调用；公开函数只校验权重已经是
NZ（`get_npu_format == 29`），自身不做任何转换。全部张量须位于同一 NPU。

### 参数说明

`T=batch*seq` 为展平后的 token 数；`H=5120`、`R=1280`、`N=64`、`D=512`、`Dr=64`；`P` 为
cache 页数，`BS` 为每页 slot 数，均由调用方给定。下表均为逻辑形状；NZ 只改变物理布局，不改变
权重的逻辑矩阵方向。

| 参数 | dtype | shape / 布局 | 说明 |
| :--- | :--- | :--- | :--- |
| `x` | E4M3FN | `[T,H]` / ND | token 输入特征 payload |
| `wqa` | E4M3FN | `[R,H]` / **NZ** | Q 第一次投影权重 |
| `wqb` | E4M3FN | `[N*D,R]` / **NZ** | Q 第二次投影权重 |
| `wkv` | E4M3FN | `[D,H]` / **NZ** | KV 投影权重 |
| `descale_x` | E8M0 | `[T,H/64,2]` / ND | 输入的分组 scale；成对存储，末维是长度为 2 的物理 phase 轴 |
| `descale_wqa` | E8M0 | `[R,H/64,2]` / ND | QA 权重的分组 scale |
| `descale_wqb` | E8M0 | `[N*D,R/64,2]` / ND | QB 权重的分组 scale |
| `descale_wkv` | E8M0 | `[D,H/64,2]` / ND | KV 权重的分组 scale |
| `norm_weight_qr` | FP32 | `[R]` / ND | QR 支路的 RMSNorm 权重 |
| `norm_weight_kv` | FP32 | `[D]` / ND | KV 支路的 RMSNorm 权重 |
| `rope_sin` / `rope_cos` | FP32 | 各 `[T,Dr]` / ND | 每 token 的位置变换系数；旋转符号约定由表承载 |
| `cache_index` | INT64 | `[T]` / ND | 要更新的扁平 cache slot，`0 <= idx < P*BS`，一次调用内不重复 |
| `kv_cache` | UINT8 | `[P,BS,1,D]` / ND | 就地更新的分页缓存 |
| `norm_eps` | float | 标量 | 有限正数，默认 `1e-6`，两条支路共用；作为运行期 FP32 参数传入，数值变化不触发重编译 |

E4M3FN / E8M0 分别对应 `torch.float8_e4m3fn` 与 `torch.float8_e8m0fnu`。输入与权重沿最后一维
每 32 个 payload 元素共享一个 scale；有限 E8M0 编码字节 b 对应 `2^(b-127)`。以 `int8`/`uint8`
承载的历史 FP8 数据应通过 `.view(...)` 恢复 FP8 dtype，不能用数值类型转换替代字节重解释。

### 返回值说明

返回 `(q, qr, descale_qr)` 三元组；`kv_cache` 就地更新，不额外返回。

| 输出 | dtype | shape | 说明 |
| :--- | :--- | :--- | :--- |
| `q` | `torch.bfloat16` | `[T,N,D]` | 第二次 Q 投影并完成尾部 RoPE 后的最终 Q |
| `qr` | `torch.float8_e4m3fn` | `[T,R]` | RMSNorm 后动态量化的中间结果 |
| `descale_qr` | `torch.int8` | `[T,R/64,2]` | QR 的 E8M0 编码字节；内部按 e8m0 分配，返回 int8 视图 |
| `kv_cache`（就地） | `torch.uint8` | `[P,BS,1,D]` | E4M3FN 编码字节，仅改写 `cache_index` 命中的 slot |

三个输出都由 host 用 `torch.empty` 分配、不做任何设备侧预置；kernel 在每次调用内先写后读其
使用的全部输出与中间缓冲区域。KV 支路不返回独立 scale，也不返回独立的 K、V 张量。

## 约束说明

**输入与布局**

1. 全部张量须逻辑连续、位于同一 NPU，dtype 与 shape 如上表；`wqa/wqb/wkv` 必须已是
   FRACTAL_NZ（format 29）。公开函数显式校验 NZ 标记，未转换时直接报错，避免 kernel 按 ND
   静默读出错乱布局。
2. `H`、`R`、`D` 须为 64 的倍数；`Dr` 为正且是 16 的倍数，满足 `Dr <= 64`、`Dr <= D/2`。
   完整验收范围为上述固定生产维度下的 T 泛化（`1 <= T <= 262144`）。
3. `x[t]`、位置表第 t 行与 `cache_index[t]` 须对应同一个 token；算子不从 T 推断位置或所属序列。
4. 调用方须保证 `0 <= cache_index[t] < P*BS` 且一次调用内没有重复 slot（重复值的语义未定义）；
   未命中的 slot 保持原始字节不变。host 侧只校验元数据，不把设备索引值读回 CPU。

**数值与执行**

5. 第二次 Q 投影的 A 操作数是量化后的 `qr`（不是 FP32 归一化值）；KV 支路先 RMSNorm 和 RoPE
   再量化、且 scale 固定为 1。这两条顺序都是数值契约的组成部分，见「功能说明」的公式。
6. E4M3FN 转换采用最近偶数舍入（`CAST_RINT`），量化前显式饱和到 ±448；全零量化组的 payload
   与 scale 字节均为 0。
7. `norm_eps` 是运行期 FP32 标量参数，改变其数值不触发重编译。
8. host 侧公开函数只做元数据校验、派生整数量、`torch.empty` 分配输出与中间缓冲、单次 launch；
   不包含任何设备侧计算或数据搬运（`to_nz` 属于权重加载期，不在公开函数内）。
9. kernel 的模块级不变量（L1/L0A/L0B/L0_MX/L0C/UB 地址图容量、MX scale 的 >>4 地址耦合、
   跨核事件 id 预留、mutex id 预算、几何整除性）在导入期由 `_check_constant` 校验，违反时抛
   `RuntimeError`；`python -O` 下也不会被静默跳过。
10. 尾块（T 不是 128 的整数倍）全部在 kernel 内处理：逻辑 T 决定输出行数、`cache_index` 读取
    与公开形状，物理行 `ceil(T/128)*128` 只服务中间缓冲与 NZ 分形对齐；调用者无需补齐。

## 调用示例

权重一次性转 NZ 后直接调用（测试目录的 CPU fixture 可用于构造输入）：

```python
import sys

sys.path.insert(0, "ops/pypto_pro/deepseek_v4_1/attn_prologue/op_kernel")
from attn_prologue import attn_prologue, to_nz

wqa, wqb, wkv = to_nz(wqa), to_nz(wqb), to_nz(wkv)   # 权重加载期一次性转换
q, qr, descale_qr = attn_prologue(
    x, wqa, wqb, wkv,
    descale_x, descale_wqa, descale_wqb, descale_wkv,
    norm_weight_qr, norm_weight_kv,
    rope_sin, rope_cos, cache_index, kv_cache,       # kv_cache 就地更新
    norm_eps=1e-6,
)
```

运行测试（均在仓库根目录执行，需先加载 CANN 与 PyPTO 环境；设备由 `TILE_FWK_DEVICE_ID` 选择，
未设置时为逻辑设备 0）：

```bash
D=ops/pypto_pro/deepseek_v4_1/attn_prologue

# CPU golden 自检（形状/dtype/有限性，不碰 NPU）
python $D/tests/assets/attn_prologue_golden_cpu.py
# 精度判据自检（CPU）
python $D/tests/assets/attn_prologue_precision.py

# 全量精度矩阵（32 case）；或子集
TILE_FWK_DEVICE_ID=1 python $D/tests/test_attn_prologue.py
TILE_FWK_DEVICE_ID=1 python $D/tests/test_attn_prologue.py --t 1 --t 72 --t 1024
python $D/tests/test_attn_prologue.py --list          # 列出 case（不需 NPU）
pytest $D/tests -q                                    # 冒烟 case + 判据自检
```

直接运行测试文件时，逐 case 打印一行判定，并把结果写入 `--out` 指定的 JSON（默认
`tests/reports/pro_pypto/validation.json`，目录自动创建）；`--list` 向 stdout 输出与用例矩阵
一致的 JSON。精度判据、用例清单与绝对性能结果见同级目录的 [`RESULTS.md`](RESULTS.md)，实现
结构与流水设计见 [`DESIGN.md`](DESIGN.md)。
