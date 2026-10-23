# attn_prologue（PyPTO-Pro）

`attn_prologue` 是使用 PyPTO-Pro 实现的 Attention 前置处理融合算子。一次设备调用完成两级 Q 投影、KV 投影、RMSNorm、QR 动态 MXFP8 量化、尾部 RoPE 和分页 KV cache 写回，返回 `(q, qr, descale_qr)`，并就地更新 `kv_cache`。

当前 kernel 的特征维度固定为 `(H,R,N,D,Dr)=(5120,1280,64,512,64)`，分别表示输入特征宽度、Q 中间投影宽度、Q head 数、每 head 宽度及 RoPE 处理宽度。唯一动态轴是展平后的 token 数 `T=batch*seq`，host 接受 `1 <= T <= 262144`。它不计算 QK、softmax 或 Attention 加权输出，也不读取已有 cache 做注意力计算。接口与数值行为以本目录的实现和测试为依据。

## 产品支持情况

| 项 | 取值 |
| :--- | :--- |
| 芯片 | Ascend 950PR / 950DT（目标架构 `NpuArch.DAV_3510`） |
| 能力依赖 | `matmul_mx/matmul_mx_acc` 联合消费 E4M3FN 数据与 E8M0 分组 scale，FP32 累加；FRACTAL_NZ 权重及 NZ/ZN 同址视图；ScaleLeft/ScaleRight 与操作数地址配对；`DualModeSplitM` 将累加结果分发给两个 AIV；VF 量化与 RoPE；MIX 全局同步和跨核事件 |
| 启动核数 | 正常查询路径取 `max(1, min(cube_core_num, vector_core_num//2))`；平台查询异常或返回异常核数时，`_fused_grid()` 回退到 32 Cube / 64 Vector 的配置 |
| 原有实测环境 | DAV_3510，32 Cube / 64 Vector；CANN 9.3.0、PyPTO-Pro、PyTorch 2.12.0+cpu、torch_npu、Python 3.12；本次文档完善未重新采集设备性能 |
| 测试标注 | 测试模块标注 `pytest.mark.soc("950")`，4 个 NPU 冒烟项另标注 `pytest.mark.npu`；通过 pytest 执行时由仓库 conftest 应用这些标记 |
| 依赖 | kernel 使用 pypto_pro、torch、torch_npu；CPU 参考与判据模块只依赖 torch，测试入口另使用 pytest |

host 指 CPU 侧接口代码，AIC 是 Cube 矩阵计算单元，AIV 是向量计算单元；MIX kernel 使用两者协作。一次启动中的一个核组对应 1 个 AIC 与 2 个 AIV。VF 指 `@pl.vector_function` 定义的向量函数。GM 是设备全局内存，L1/L0 用于 Cube 操作数与累加结果，UB 是 AIV 的本地缓冲区。

## 功能说明

公开函数 `attn_prologue` 只检查输入元数据、分配输出及工作区，再调用一次 `attn_prologue_kernel`。三个矩阵乘都由 Cube 使用 MXFP8 数据和分组 scale 直接完成，结果以 FP32 累加；设备上不会先构造完整的反量化 X 和权重。Vector 完成归一化、量化、RoPE 和输出处理。kernel 不使用 tiling key，T 由动态张量形状提供，内部根据 `ceil(T/128)` 选择工作分配，详见 [DESIGN.md](DESIGN.md)。

以下使用 CPU 参考实现中已有的函数名表示数学过程：`dequantize_mxfp8` 将 FP8 数据乘以对应分组 scale，`quantize_mxfp8` 做每 32 元素一组的动态量化，`rms_norm` 和 `apply_tail_rope` 分别执行整行归一化及尾部位置变换。它们用于说明数值顺序，不表示 kernel 实际调用 CPU 函数。

```text
x_deq = dequantize_mxfp8(x, descale_x)                         # [T,5120]
qa    = x_deq @ dequantize_mxfp8(wqa, descale_wqa).T           # [T,1280]
kv    = x_deq @ dequantize_mxfp8(wkv, descale_wkv).T           # [T,512]

qr_norm       = rms_norm(qa, norm_weight_qr, norm_eps)
qr, descale_qr = quantize_mxfp8(qr_norm)                      # 另有参考诊断量，此处省略
q_proj        = dequantize_mxfp8(qr, descale_qr) @ dequantize_mxfp8(wqb, descale_wqb).T
q             = BF16(apply_tail_rope(reshape(q_proj, [T,64,512]), rope_sin, rope_cos))

kv_norm  = rms_norm(kv, norm_weight_kv, norm_eps)
kv_rope  = apply_tail_rope(kv_norm, rope_sin, rope_cos)
kv_bytes = E4M3FN(clamp(kv_rope, -448, 448)).view(uint8)
kv_cache[cache_index[t]//BS, cache_index[t]%BS, 0, :] = kv_bytes[t, :]
```

QA 与 QB 是代码对 Q 第一次、第二次投影的称呼；`qr` 是两次投影之间经过 RMSNorm 和量化的低秩表示。QB 必须消费量化后的 `qr/descale_qr`，不能直接使用 FP32 的 `qr_norm`。KV 支路先 RMSNorm、再 RoPE、最后转 E4M3FN，scale 固定为 1；返回的 QR scale 不用于解释 KV cache。

**RMSNorm**：对每个 token 的整行计算，不减均值。QR 沿 R=1280 归约，KV 沿 D=512 归约。实现先求 `rstd = 1/sqrt(sum(v*v)/width + norm_eps)`，再依次计算 `(v*rstd)*gamma`。QR VF 按 20 段、KV VF 按 8 段的 64 个 FP32 元素累加平方和；Cube 的分块归约和 VF 的求和顺序可能与 CPU torch 不同，精度判据对此单独处理。

**QR 动态量化**：每行含 40 个连续的 32 元素组。对通常的非零有限输入组，计算：

```text
amax        = max(abs(group))
e           = max(floor(log2(amax)) - 8, -127)
scale       = 2^e
payload     = E4M3FN(clamp(group / scale, -448, 448))
scale_byte  = e + 127
```

kernel 通过 FP32 指数字段取得共享指数，并构造 `2^-e` 与数据相乘。CPU 参考使用 `frexp` 和 `ldexp`，另对指数上限 127 做钳位；对有限 FP32 正常数，上限不会起作用。E4M3FN 采用最近偶数舍入（`CAST_RINT`），在转换前显式饱和到 ±448。该指数规则不等于用 `amax/448` 生成连续 scale，组内部分值仍可能饱和。

全零组写零 payload 和编码字节 0；E8M0 字节 0 本身表示 `2^-127`，不是数值 0。最后两维 `[R/64,2]` 展平后，40 个 scale 依次对应特征 `0..31`、`32..63`、…、`1248..1279`；末维长度 2 是框架加载 scale 分形时使用的物理 phase 轴。这里的 scale 是乘法因子，名字中的 `descale` 不表示取倒数。

**尾部 RoPE**：沿用 CPU 参考的 `RopeConvention.INTERLEAVE_HALF`。对每个 Q head 和 KV 向量，只变换末尾 64 维；前 448 维不做位置变换，Q 最后仍需转 BF16、KV 仍需转 E4M3FN。设尾部输入为 z，对 `i=0..31`：

```text
out_tail[i]      = z[2*i] * rope_cos[t,i]    + z[2*i+1] * rope_sin[t,i]
out_tail[32 + i] = z[2*i] * rope_sin[t,32+i] + z[2*i+1] * rope_cos[t,32+i]
```

输入按相邻元素配对，输出分别放入两个连续半区。同一 token 的表在全部 64 个 Q head 上共享。代码没有额外添加负号，不能直接替换为常见的带负号 RoPE 公式；是否构成保范数的旋转取决于调用方提供的系数表。kernel 只实现上述变换，不生成或校验角度表。

## 函数原型

```python
from attn_prologue import attn_prologue, to_nz

q, qr, descale_qr = attn_prologue(
    x, wqa, wqb, wkv,
    descale_x, descale_wqa, descale_wqb, descale_wkv,
    norm_weight_qr, norm_weight_kv,
    rope_sin, rope_cos, cache_index, kv_cache,
    norm_eps=1e-6,
)
```

`to_nz(weight)` 要求 NPU 张量，调用 `torch_npu.npu_format_cast(weight.contiguous(), 29)` 转为 FRACTAL_NZ。应在权重加载期一次性执行；公开计算函数只通过 `get_npu_format` 检查三份权重的 format 29，不自动转换。

### 参数说明

`T` 是 token 总数，batch 和 seq 不单独传入；`P` 是 cache 页数，`BS` 是每页 slot 数。下表权重形状均为逻辑形状，NZ 改变物理排布，不改变 `[输出宽度,输入宽度]` 的矩阵方向。

| 参数 | dtype | shape / 布局 | 说明 |
| :--- | :--- | :--- | :--- |
| `x` | E4M3FN | `[T,5120]` / ND | 输入的 FP8 编码数据，每字节一个元素 |
| `wqa` | E4M3FN | `[1280,5120]` / NZ | Q 第一次投影权重 |
| `wqb` | E4M3FN | `[32768,1280]` / NZ | Q 第二次投影权重，输出列按 head 连续组织 |
| `wkv` | E4M3FN | `[512,5120]` / NZ | KV 投影权重 |
| `descale_x` | E8M0 | `[T,80,2]` / ND | 每行 160 个 scale，各覆盖 32 个输入元素 |
| `descale_wqa` | E8M0 | `[1280,80,2]` / ND | QA 权重沿输入特征维的 scale |
| `descale_wqb` | E8M0 | `[32768,20,2]` / ND | QB 权重每行 40 个 scale |
| `descale_wkv` | E8M0 | `[512,80,2]` / ND | KV 权重每行 160 个 scale |
| `norm_weight_qr` | FP32 | `[1280]` / ND | QR RMSNorm 的逐特征权重，不要求全 1 |
| `norm_weight_kv` | FP32 | `[512]` / ND | KV RMSNorm 的逐特征权重 |
| `rope_sin` / `rope_cos` | FP32 | 各 `[T,64]` / ND | 每 token 的位置变换系数，不从 `cache_index` 推导 |
| `cache_index` | INT64 | `[T]` / ND | 扁平 cache slot，约定 `0 <= idx < P*BS`，一次调用内不重复 |
| `kv_cache` | UINT8 | `[P,BS,1,512]` / ND | 就地写回的 E4M3FN 编码字节，读取时按 scale=1 解释 |
| `norm_eps` | float | 默认 `1e-6` | 两支路共用，作为运行期 FP32 参数传入；调用方应提供有限正数 |

E4M3FN 和 E8M0 对应 `torch.float8_e4m3fn`、`torch.float8_e8m0fnu`。host 的 dtype 检查也接受 `int8/uint8` 形式的编码字节；测试入口 `_device_input` 会先搬运原始字节，再 `.view` 为 kernel 声明的 FP8 dtype。为保持接口与 JIT 类型一致，建议按上表传入 FP8 视图，不能用 `.to(float8)` 对编码整数做数值转换。

有限 E8M0 编码 b（0–254）表示 `2^(b-127)`。每组 scale 为一个字节，因此 x 的 scale 开销是每 token 160 B，而不是 80 B。测试主要使用非单位 scale 和有限随机输入；NaN、无穷或极端动态范围不属于本目录已完整验证的输入范围。

### 返回值说明

返回 `(q, qr, descale_qr)`；`kv_cache` 就地更新，不额外返回。

| 输出 | dtype | shape | 说明 |
| :--- | :--- | :--- | :--- |
| `q` | BF16 | `[T,64,512]` | QB 投影与尾部 RoPE 后的最终 Q |
| `qr` | E4M3FN | `[T,1280]` | QA 经 RMSNorm、动态量化后的中间表示，同时作为 QB 输入 |
| `descale_qr` | INT8 | `[T,20,2]` | E8M0 编码字节；内部按 E8M0 分配，返回 int8 视图 |
| `kv_cache`（就地） | UINT8 | `[P,BS,1,512]` | 仅覆盖 `cache_index` 命中的整行，每 slot 512 B |

解释 `descale_qr` 时应先 `.view(torch.uint8)` 或 `.view(torch.float8_e8m0fnu)`，不能直接把 int8 的负数当作有符号指数。三个返回张量和两个 FP32 工作区均由 host 用 `torch.empty` 分配；`kv_cache` 使用调用方提供的存储，未命中 slot 保留原始字节。

最终 Q 每 token 为 65536 B；T=262144 时单个 q 输出为 16 GiB。测试通过分块回传与比较来限制 CPU 参考内存，设备上的公开输出仍按完整 T 分配。

## 约束说明

**输入与布局**

1. 当前只支持固定的 H/R/N/D/Dr，上述维度不是可调接口参数。CPU 参考支持小几何自检，不代表设备 kernel 同时支持这些形状。T 必须在 `[1,262144]`。
2. 调用方应提供同一 NPU 上的张量，ND 输入按连续布局准备，三份权重预先转 NZ。`_validate_metadata` 显式检查 shape、dtype 集合和权重 NZ 标记，但没有统一检查所有输入的设备一致性或连续性，也不自动修正布局。
3. `x[t]`、位置表第 t 行与 `cache_index[t]` 必须对应同一个 token。算子不根据 T、batch 或 cache 页号推断位置。
4. cache 页数和页大小应为正，索引须在 `[0,P*BS)` 且唯一。kernel 对负索引有跳过写入的分支，但 CPU golden 拒绝负索引，当前公开约定不将其作为有效输入；kernel 不检查索引上界或重复项，host 也不把设备索引读回 CPU。

**数值与执行**

5. QB 消费量化后的 QR，KV 先归一化和位置变换再量化。不能交换这些步骤，也不能将 QR 的动态 scale 用在 KV 上。
6. E4M3FN 转换为最近偶数舍入，并先显式饱和到 ±448；QR 全零组的编码为零 payload 与 scale 字节 0。Q 的 BF16 转换在尾部 RoPE 之后执行。
7. host 只对 `norm_eps` 做 `float(...)` 转换，没有有限性和正数检查；调用方应保证其有效。它以运行期 FP32 标量传入，不作为 tiling key。
8. 一次公开调用只启动一个 kernel；CPU 侧的输入构造、NZ 权重转换、参考计算和输出比较均不在 kernel 内，也不包含于设备 task duration。
9. `_check_constants()` 在导入期检查列块整除、归约步数、若干地址区域的容量及相邻区域关系、MX 地址配对、跨核事件编号和预载 mutex 编号。检查通过不表示运行期输入内容已被逐项验证；这些检查通过 `RuntimeError` 保持在 `python -O` 下有效。
10. 调用方不需把 T 补为 128 的倍数。普通路径的 `ws_rows=T`；启用归约维二分时工作区为两个 `ceil(T/128)*128` 行带。Cube 在片上使用固定矩阵窗口，GM 读取、工作区写入、Q 存储与 cache 索引访问分别按逻辑行数限制；Q 的无效 AIV 半区仍参与必要事件握手。

## 调用示例

以下假设全部输入已按参数表准备并位于同一 NPU，FP8 字节已恢复为相应 dtype：

```python
import sys

sys.path.insert(0, "ops/pypto_pro/deepseek_v4_1/attn_prologue/op_kernel")
from attn_prologue import attn_prologue, to_nz

# 权重加载期执行一次，后续调用复用 NZ 权重。
wqa, wqb, wkv = to_nz(wqa), to_nz(wqb), to_nz(wkv)
q, qr, descale_qr = attn_prologue(
    x, wqa, wqb, wkv,
    descale_x, descale_wqa, descale_wqb, descale_wkv,
    norm_weight_qr, norm_weight_kv,
    rope_sin, rope_cos, cache_index, kv_cache,
    norm_eps=1e-6,
)
```

运行测试时在仓库根目录加载 CANN 与 PyPTO 环境。CLI 的 `--device` 优先于 `TILE_FWK_DEVICE_ID`，环境变量未设置时使用逻辑设备 0：

```bash
D=ops/pypto_pro/deepseek_v4_1/attn_prologue

# 两种几何的 CPU 参考实现自检，不使用 NPU。
python $D/tests/assets/attn_prologue_golden_cpu.py
# 两组 CPU 比较器自检，内部共 12 个检查场景。
python $D/tests/assets/attn_prologue_precision.py

# 完整的 32 个设备场景，或按 T 选取子集；--t 可重复，不能传逗号列表。
TILE_FWK_DEVICE_ID=1 python $D/tests/test_attn_prologue.py
python $D/tests/test_attn_prologue.py --device 1 --t 1 --t 72 --t 1024
python $D/tests/test_attn_prologue.py --list

# pytest 收集 4 个设备冒烟项和 2 个 CPU 判据项，并非完整的 32 场景。
pytest $D/tests -q
```

`--list` 在加载 NPU wrapper 前输出所选场景 JSON；也可使用 `--phase`、`--case-id` 和 `--unique-t` 筛选。`expected_template` 是参考用例记录的历史标签，不是 kernel 的实际分支判定，实际调度需看 T 和平台核数。

精度入口每完成一个 case 就更新 `--out` 指定的 JSON，默认 `tests/reports/pro_pypto/validation.json`；`--row-chunk` 控制 CPU 分块比较，默认 256，`--norm-eps` 同时用于设备调用与参考计算。当前入口没有性能采样选项，历史 msprof 口径与结果见 [RESULTS.md](RESULTS.md)。
