# AttenEpilogue 设计说明

## 计算阶段划分

一个 MIX 设备任务内完成逆 RoPE 与第一次激活量化、第一层 MX 矩阵乘、第二次激活量化和第二层 MX 矩阵乘。Vector 核处理旋转、BF16 舍入、块缩放和 FP8 转换；Cube 核处理两层矩阵乘。每个输出元素保持 K 方向的累加次序，第一层 FP32 累加结果转换为 BF16 后再量化，第二层直接写出 BF16。

内部令 `M=ceil(T/16)*16`。框架提供一块 `42240*M` 字节的用户工作区，四个载荷按下表顺序连续划分；各起始地址均为 512 字节的整数倍。工作区中的补齐行与公开输出分开处理，输出始终只有 T 行。

| 载荷 | 元素内容 | 字节数 | 相对工作区的起始字节偏移 |
| --- | --- | --- | --- |
| AQ | 第一层输入 FP8，按 8 组排列 | `32768*M` | 0 |
| ASC | 第一层输入 E8M0 缩放 | `1024*M` | `32768*M` |
| YQ | 第二层输入 FP8 | `8192*M` | `33792*M` |
| YS | 第二层输入 E8M0 缩放 | `256*M` | `41984*M` |

## 核间切分策略

host 根据实际 Cube 核数与 T 选择有限配置，并以实际 Cube 核数作为 blockDim，保持每个 block 的 1 Cube : 2 Vector 配比。28 核和 32 核使用同一个 Python kernel 入口、同一份 TilingData 及同一个公开接口。配置选择只读取形状和平台信息。

第一层按组、M 块和 N 块分配任务，第二层按 M 块和输出 N 块分配任务。32 核且 T≤128 时，第一层有 8×4=32 个 N 方向任务，每个 Cube 核负责一个任务；该范围只需一个 Cube→Vector 交接缓冲槽。28 核使用 128 或 160 列的第一层 N 块和 192 列的第二层 N 块，以覆盖不同核数下的工作分配；尾列只写实际有效范围。

| Cube 核数 | T 范围 | 主要路径 |
| --- | --- | --- |
| 32 | 1..64 | 小 M；物理行容量为 `ceil(T/16)*16`，有效行数保留运行时 T |
| 32 | 65..256 | 中等 M；第一层 N=256，第二层 N=160 |
| 32 | 257..384 | 配对尾块路径 |
| 32 | 385..1024 | 大 M 流水路径 |
| 28 | 1..64 | 小 M；第一层 N=128，第二层 N=192 |
| 28 | 65..1024 | N=160 或 256 的第一层与 N=192 的第二层流水路径 |
| 其他核数，或更大的 T | 公开分派的其余范围 | 通用容量路径 |

这些路径对应 19 个可达编译配置。16 核只做配置选择与 host tiling 校验；本轮设备数值实测、当前离线二进制和完整性能采集覆盖 28/32 核。

## 核内切分策略、基本块选择与核内 Buffer 划分

L1 分别保存激活、权重及各自的缩放；L0A、L0B 及其 Scale 区分别保存当前矩阵乘基本块；L0C 保存 FP32 累加结果。数据与缩放各有独立的互斥组。Vector UB 保存逆 RoPE 输入、Cube 交接结果、BF16 转换结果、量化数据及缩放打包临时量；这些缓冲的地址范围和交接生命周期由各编译路径固定。

小 M 第一层以 K=512、N=64 的矩阵乘更新四个独立的累加片段，再以原有复合视图完成 N=256 的交接。物理 M 容量与 `ceil(T/16)*16` 一致，有效 M 为 T，因而片段间的硬件行间距与复合视图一致。第二层保留 N=64、64、32 的分片，各分片分别写回，避免以不一致的物理行间距拼接尾块。

32 核且 65≤T≤128 时，第一层使用两个 A256 和三个 B512 缓冲，权重采用提前两个面板的环形预取。L1 总高水位为 499712 字节，低于 524288 字节容量。第三个 B 缓冲使用 Cube 互斥 ID16，第一层完成后的 Cube 屏障排空该生命周期，第二层再使用该 ID。第二层使用两个 A512 和三个 B512 缓冲，L1 高水位为 450560 字节；每个 A512 面板仍分成四次有序的 K128 更新，保持原有数值累加次序。

量化以连续 32 个 BF16 值为一个缩放块。缩放计算、BF16 归一化、有限饱和 FP8 转换和交错字节打包使用 VF 指令；缩放字节与载荷分别写出。尾行和尾列的处理采用运行时有效尺寸，物理存储容量不因尾块缩小而低于实际搬运范围。

## 接口选择

公开 Python 接口与 ACLNN 均为 7 个输入和 1 个调用方分配的输出。Python 入口负责检查维度、dtype、连续 ND 存储及同设备约束，并按设备和补齐行数复用内部工作区。

离线 kernel 依次接收 7 个输入裸指针、输出裸指针、用户工作区指针和运行时 tiling。TilingData 仅有顺序固定的两个 int64 字段 `t`、`padded`，共 16 字节。host 填写实际注册类型，通过官方生成的 `GET_TPL_TILING_KEY` 取得合法编译配置键，同时设置 blockDim 和工作区大小。

保留现有 `op_host`、`op_kernel`、`tests/aclnn_bridge` 和 CMake 结构，由工程已有的 PyPTO 离线 codegen 和 CANN 打包入口生成算子包。算子类型保持 `AttenEpilogue`。OpDef 使空输出也进入 host tiling，从而按 T≥1 的契约拒绝空 token 输入；构建依赖显式关联三类 host 注册库，保证 OpDef 改动后重新生成对应 API 元数据。

加载对应环境后，在 gym 仓库根目录进行干净构建；已有构建目录应先另行保留。

```bash
cmake -S . -B build/ascend950 \
  -DASCEND_COMPUTE_UNIT=ascend950 \
  -DASCEND_OP_NAME=atten_epilogue \
  -DENABLE_TESTS=ON -DPACKAGE_TYPE=run \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE="$(command -v python)" \
  -DCMAKE_INSTALL_PREFIX="$PWD/build_out/run/ascend950"
cmake --build build/ascend950 --parallel 2
cmake --build build/ascend950 --target package --parallel 2
```

安装 `build_out/run/ascend950` 中本次生成的 `.run` 包，使用它的 `--install-path` 参数指定独立安装目录。将该目录的 `vendors/pypto_gym` 加入 `ASCEND_CUSTOM_OPP_PATH`，并将该 vendor 的 `op_api/lib` 加入 `LD_LIBRARY_PATH`。测试从设备环境变量 `TILE_FWK_DEVICE_ID` 选择设备：

```bash
python -m pytest ops/pro/deepseek_v4_1/atten_epilogue/tests --mode jit -q
python -m pytest ops/pro/deepseek_v4_1/atten_epilogue/tests \
  --mode aclnn --aclnn-lib-dir build/ascend950/tests -m 'not jit_only' -q
```

ACLNN 测试通过本次构建的桥接库调用安装目录中的二进制；仅检查 Python JIT 结果不足以证明离线包的实际调用链路。

## 流水设计

逆 RoPE 与第一次量化按输入 K 片段发布就绪信号，第一层 Cube 在对应片段就绪后读取 AQ 与 ASC，随后通知 Vector 可以复用相应生产缓冲。该依赖允许生产与消费在不同片段间推进；不能把整个任务理解为四个完全串行的全局阶段。

中等 M 第一层先发出当前 A 的加载，再发出提前两个面板的 B 预取，使未来权重搬运有机会与当前 MTE1 和矩阵乘重叠。三个环形槽保证预取目的槽与当前读取槽分离；启动时只预载 B0、B1，每个权重面板仍加载一次。实际是否缩短任务耗时以完整任务测量为准，缓冲深度本身不构成提速证据。

第一层累加完成后，Cube 按显式交接协议把 FP32 结果移交给两个 Vector 子核，各自转换并量化负责的有效行。Vector 写完 YQ、YS 后先完成跨 block 汇合，再发布第二层就绪事件。第二层可以在此之前预载不依赖 YQ 的只读权重及缩放，但只在等待就绪事件后读取激活。

不同阶段重用 L1 地址和 Cube 互斥 ID 前，保留原有 Cube 屏障以排空先前访问。Vector 控制寄存器的量化策略在任务内保存、设置并恢复。最后一次累加和最后一次写回使用相应 Final 阶段标记，保证累加器生命周期与实际消费者一致。
