# PyPTO-Gym Torch 绑定

Torch 绑定使用已安装的 ACLNN 算子库，在当前 NPU stream 上执行预编译 kernel。
C++ 绑定也在构建期编译并随 `.run` 交付；运行时加载 `.so`，无需编译器或
PyPTO Python 包。当前已接入 `mega_mhc`。

## 构建和安装

构建环境需要 CANN、PyPTO-pro、torch、torch_npu 和 PyYAML（读取发布清单）。执行：

```bash
python3 build_ci.py --pkg-type run --ops mega_mhc --soc ascend950
bash build_out/run/ascend950/cann-950-pypto_gym_*.run --install-path="$ASCEND_HOME_PATH/opp"
```

`.run` 默认附带预编译 Torch 绑定，对应 `-DENABLE_TORCH_BINDINGS=ON`，
无需额外传入 `--torch-bindings`。仅需 ACLNN 接口时，使用 `--no-torch-bindings`
（手工 CMake 使用 `-DENABLE_TORCH_BINDINGS=OFF`）。`whl` 构建不启用此绑定。
部署环境需要与构建环境兼容的 Python、torch、torch_npu 和 CANN。
在已加载目标 CANN 基础环境的终端中，安装到其 `opp` 后即可直接导入，
无需额外 source vendor 的 `set_env.bash`。安装器在 CANN 的 `python/site-packages`
中登记指向 vendor 私有文件的相对链接；绑定通过相对运行时库路径加载同一 vendor 的 ACLNN 库。
升级、卸载只维护本包创建的链接，保留已有 kernel wheel 的 `ops` 目录；
若入口已由其他 Python 包占用，安装会报错而不覆盖。

安装到非 CANN 的独立目录时，仍可 source `<opp>/vendors/pypto_gym/bin/set_env.bash`
配置该自定义位置的搜索路径。新终端需要加载 CANN 基础环境。

```python
import torch
import pypto_gym

# 九个输入 tensor 均位于同一 NPU；形状和类型见算子 README。
outputs = pypto_gym.mega_mhc(*inputs)
# 直接通过 Torch dispatcher 调用前，先显式加载绑定；batch=0 表示自动配置。
pypto_gym.load()
outputs = torch.ops.pypto_gym.mega_mhc(*inputs, batch=0)
```

`import pypto_gym` 是惰性的：不会导入 torch/torch_npu，也不会加载绑定库。访问
`pypto_gym.mega_mhc` 会在首次调用时加载；直接使用 `torch.ops.pypto_gym.*` 前需先
调用 `pypto_gym.load()`（或先访问任一 `pypto_gym.<op>`）。因此缺少 torch 或绑定
ABI 不兼容时，`import pypto_gym` 仍然成功。

`pypto_gym.mega_mhc` 支持 `batch=None/16` 和六个运行时标量，五个输出由绑定
内部申请并以返回值形式返回。功能入口注册了 `PrivateUse1` 和 `Meta`，支持 FakeTensor
和符号 token 维度的输出推导。当前提供 forward 计算，不提供 autograd backward。

## 测试

```bash
# 构建、隔离安装到 build_out，并通过正式 Torch 入口运行测试。
python3 build_ci.py --pkg-type run --ops mega_mhc --stest
# 已安装到当前 CANN 时，无需额外 source vendor 脚本：
pytest ops/pro/deepseek_v4_1/mega_mhc/tests -q
```

`mega_mhc` 的 24 项完整算子用例中，12 项直接调用 `pypto_gym.mega_mhc`，
12 项直接调用 `torch.ops.pypto_gym.mega_mhc`，每项只执行一次算子，输出由绑定内部申请。
测试校验五输出数值精度、只读输入和输入内存 guard；入口不受 `--mode` 参数控制。
用例清单见算子 README。

另有一项存储格式边界用例，验证 NZ 输入会在进入异步队列前被拒绝。
绑定读取真实存储格式元数据，不做隐式格式转换或 NPU 同步。

## 多算子构建到同一个包

`.run` 的 `ALL`（包括省略 `--ops`）由
[ops/pro/build_ops.yaml](../ops/pro/build_ops.yaml) 控制，使用 YAML 列表，路径相对于 `ops/pro`：

```yaml
- deepseek_v4_1/mega_mhc
```

当前清单仅列出 `mega_mhc`。文件中列出的路径参与构建，未列出的路径不参与。
空行、以 `#` 开头的注释及行尾 `#` 注释会被忽略。
新增算子需要把完整相对路径加入清单；不想构建某个算子时，注释掉对应行即可。
例如，接入新算子 `example_op` 后可加入第二行，再按需注释：

```yaml
- deepseek_v4_1/mega_mhc
# - experimental/example_op
```

上述清单的 `ALL` 只构建 `deepseek_v4_1/mega_mhc`；取消第二行的注释后包含
两个算子。`example_op` 是占位名称，启用前需完成实际算子的二进制接入。

列出的目录需满足二进制构建约定：位于规定的目录深度内，并同时具备算子级
`CMakeLists.txt` 和 `op_host/`。路径按完整目录匹配，实际算子名取目录末级名称，
同一个包中的算子名必须唯一。清单需至少列出一个有效目录；文件缺失、路径错误或
全部被注释时构建报错。

显式 `--ops mega_mhc,example_op` 仍按算子目录名选择，绕过此清单，
也可显式选择未列入清单的算子。显式选择的候选范围包括 `ops/pro/` 和
`ops/tensor/` 下已接入二进制构建的算子；清单的相对路径基准固定为 `ops/pro/`。
`whl` 的 `ALL` 保持原有发现范围，不使用这份 `.run` 发布清单。

CMake 构建和 `--stest` 共用同一份解析逻辑；CMake 会跟踪清单变化并重新配置。
`operators.json` 是构建生成的 Torch 绑定名单，不作为发布范围的配置入口。

`--ops` 支持逗号分隔的算子目录名。同一 SoC、同一次构建选择的所有算子会进入
同一个 `.run` 包。例如，完成 `example_op` 接入后，与 `mega_mhc` 一起交付：

```bash
python3 build_ci.py --pkg-type run --soc ascend950 \
  --ops mega_mhc,example_op
```

命令中的逗号两侧不要加空格，整个名单作为一个参数传入。名单填写算子目录名，
例如 `mega_mhc`，而不是 C++ 类型名 `MegaMhc` 或 ACLNN 函数名。

| 选择方式                          | 构建范围                                                |
| --------------------------------- | ------------------------------------------------------- |
| `--ops mega_mhc`                | 单个算子                                                |
| `--ops mega_mhc,example_op`     | 指定的多个算子，共用一个包                              |
| `--ops ALL`，或省略 `--ops`   | `.run` 仅构建 `ops/pro/build_ops.yaml` 中列出的目录 |

`.run` 默认附带预编译 Torch 绑定，`--no-torch-bindings` 可显式关闭。
算子是否进入包和是否提供 Torch 入口是两件事：

- 所选算子的 host、ACLNN 和 kernel 产物一起打包。
- 其中已接入 `torch_extension/` 的算子，额外打包各自的绑定 `.so` 和 Python 前端。
- 当前 `mega_mhc` 已接入绑定，提供 `pypto_gym.mega_mhc` 和对应的 `torch.ops` 入口。
  新增算子仍需完成绑定接入，构建不会从 kernel 源码自动生成 Torch 参数接口。
- `run` 默认尝试启用绑定；若所选算子都没有接入绑定，会自动降级为 ACLNN-only 包并打印
  WARNING，原有 ACLNN 调用/测试方式不受影响。显式传 `--torch-bindings` 时，所选算子中
  至少要有一个提供绑定，否则 CMake 会报错。也可用 `--no-torch-bindings` 强制 ACLNN-only。

输出目录为 `build_out/run/ascend950/`。包名按 SoC、vendor、版本和架构生成，
不包含算子名单。例如，x86_64 环境下生成
`cann-950-pypto_gym_9.2.0_linux-x86_64.run`，它可以同时包含多个算子。
要交付一个包含多个算子的包，应在一次命令中指定完整名单；分次构建单算子不会自动
把前后两次产物合并，且可能使用相同的包名。

按发布清单构建并验证当前列出的算子：

```bash
python3 build_ci.py --pkg-type run --soc ascend950 \
  --ops ALL --stest --jobs 8
```

`--stest` 可用于构建后验证：构建脚本按 `operators.json` 将已有绑定的算子交给
Torch 测试，将其余算子交给 ACLNN 测试桥；后者需要相应的测试桥和用例。
`mega_mhc` 的用例已直接固定使用两个正式接口，不再依赖测试入口分发函数。

## 新算子接入

本章使用 `outputs = torch.ops.pypto_gym.example_op(*inputs)` 的普通函数接口，
输出由 C++ 绑定内部分配。

以下以 `example_op` 为占位名称；实际接入时替换为真实算子目录名。
可对照完整实现 [mega_mhc/torch_extension](../ops/pro/deepseek_v4_1/mega_mhc/torch_extension/)。
公共构建逻辑位于 [CMakeLists.txt](CMakeLists.txt)，ACLNN 调用辅助代码位于
[include/gym_aclnn.h](include/gym_aclnn.h)。

### 1. 先具备算子的二进制构建能力

Torch 绑定建立在可独立交付的 ACLNN 算子之上。算子需要先接入仓库的 host、kernel
构建流程，能通过 `--pkg-type run --ops example_op --no-torch-bindings` 生成
ACLNN 二进制产物；完成下面的 Torch 接入后，构建时使用默认开启的绑定。
算子目录至少应有算子级 `CMakeLists.txt` 和 `op_host/`，以便构建系统发现。
host、kernel 的基础接入流程见 [构建指南](../docs/build-guide.md)。

接入前明确以下接口约定，并在算子 README 中写清楚：

- 输入和输出的顺序、shape、dtype、layout、设备及连续性要求。
- 标量参数的类型、默认值、取值范围，以及空输入的行为。
- ACLNN 函数名、完整函数签名、输出分配规则和是否支持预分配输出。
- 是否提供反向计算；只有 forward 的算子应明确限制需要梯度的调用。

Torch schema、C++ 函数、Python 前端与 ACLNN 的参数映射必须保持一致。
ACLNN 签名以该算子构建生成的头文件为准，不要只替换其他算子的函数名。

### 2. 增加目录和 CMake 登记

在算子目录增加以下三个文件：

```text
ops/pro/experimental/example_op/
├── CMakeLists.txt
├── op_host/
├── op_kernel/
├── tests/
└── torch_extension/
    ├── CMakeLists.txt
    ├── example_op.cpp
    └── example_op.py
```

`torch_extension/CMakeLists.txt` 内容为：

```cmake
if(ENABLE_TORCH_BINDINGS AND NOT PREPARE_BUILD)
    gym_add_torch_binding(example_op)
endif()
```

算子级 `CMakeLists.txt` 必须派发到这个子目录。若像 `mega_mhc` 一样已自动遍历
带 `CMakeLists.txt` 的子目录，无需额外添加；若采用显式派发，则在算子级增加：

```cmake
if(ENABLE_TORCH_BINDINGS AND NOT PREPARE_BUILD)
    add_subdirectory(torch_extension)
endif()
```

两种派发方式选一种，避免重复添加子目录。绑定登记放在
`torch_extension/CMakeLists.txt` 中，文件名须与 `gym_add_torch_binding()` 的参数一致。

公共构建函数会自动完成以下工作：

1. 创建 `torch_binding_example_op` 共享库目标，并链接公共 Torch、torch_npu、CANN 依赖。
2. 生成 `libpypto_gym_example_op.so`，安装到包内的 `_native/`。
3. 将 `example_op.py` 安装到 `_frontends/`。
4. 把 `example_op` 加入本次构建的全局绑定名单。

根构建流程在所有所选算子登记完成后，统一生成 `operators.json`。
新增算子无需修改公共加载器或手动维护这个 JSON。

### 3. 实现 C++ Torch 注册和 ACLNN 调用

`example_op.cpp` 负责将 Torch Tensor 转接到该算子的 ACLNN 接口。
可参考 [mega_mhc.cpp](../ops/pro/deepseek_v4_1/mega_mhc/torch_extension/mega_mhc.cpp)。

下面仅展示一个单输入、单输出算子的注册部分；`ExampleOpNpu` 和 `ExampleOpMeta`
须在同一文件中按真实计算和输出规则实现：

```cpp
#include "gym_aclnn.h"
#include <ATen/ATen.h>
#include <torch/library.h>

// 具体实现略：NPU 函数执行 ACLNN，Meta 函数只推导输出。
at::Tensor ExampleOpNpu(const at::Tensor &x);
at::Tensor ExampleOpMeta(const at::Tensor &x);

TORCH_LIBRARY_FRAGMENT(pypto_gym, m) {
    m.def("example_op(Tensor x) -> Tensor");
}
TORCH_LIBRARY_IMPL(pypto_gym, PrivateUse1, m) {
    m.impl("example_op", TORCH_FN(ExampleOpNpu));
}
TORCH_LIBRARY_IMPL(pypto_gym, Meta, m) {
    m.impl("example_op", TORCH_FN(ExampleOpMeta));
}
```

同一个包中的所有算子共用 `pypto_gym` 命名空间，因此使用
`TORCH_LIBRARY_FRAGMENT` 分别声明各自 schema，避免重复定义命名空间。
schema 中的 Tensor、标量、默认值和返回值必须与 C++ 函数签名匹配。

NPU 实现按以下顺序处理：

1. 校验输入设备、shape、dtype、连续性、真实存储格式和标量范围，设置正确的 device guard。
   ND 算子调用 `gym_torch::CheckNDStorage(tensor, name)`；逻辑连续性不足以判断物理格式。
2. 在调用线程分配输出；若支持预分配输出，校验输出规格、真实存储格式和输入输出之间的重叠。
   支持空输入时按算子约定返回正确形状的空输出。
3. 用 `gym_torch::Symbol<函数指针类型>()` 解析该算子的
   `aclnnExampleOpGetWorkspaceSize` 和 `aclnnExampleOp`；函数指针类型与生成头文件一致。
4. 将参与计算的输入、输出 Tensor 列表交给 `gym_torch::Enqueue()`。
   回调接收 Tensor 元数据，通过 `gym_torch::Tensor()` 构造 ACL Tensor 描述符。
5. 在回调中调用 `GetWorkspaceSize` 获取 workspace 大小和 executor，检查返回码后，
   调用 `gym_torch::ExecuteOp()` 在当前 NPU stream 执行算子。

复用公共辅助函数的 stream 记录和 workspace 管理。队列回调只捕获标量等 host 数据，
不要捕获 `at::Tensor`，也不要在回调中用 `at::empty` 创建 workspace Tensor；
这些操作可能在资源释放时重新进入任务队列。算子正常调用保持异步，数值测试需要时
再执行 `torch.npu.synchronize()`。

`Meta` 实现只根据输入元数据和属性创建输出，不分配 NPU workspace、不执行 ACLNN。
若需要动态 shape，应使用符号维度推导接口。

### 4. 导出 Python 便捷入口

`example_op.py` 中必须提供与算子同名的可调用函数。对于上面的单输入示例：

```python
import torch


def example_op(x):
    return torch.ops.pypto_gym.example_op(x)
```

这里可以补充 Python 参数校验、默认值转换和 `out` 参数展开；C++ 入口仍需保留
必要校验，因为用户也可以直接调用 `torch.ops`。Python 前端负责参数处理和转发，
计算交给预编译绑定；无需导入 `op_kernel` 或使用运行时扩展编译。

`import pypto_gym` 只读取绑定清单，不加载绑定库；`pypto_gym.example_op` 在首次
访问时按需加载对应 `.so` 并完成 Torch 算子注册。两种入口都可使用：

```python
import torch
import pypto_gym

result = pypto_gym.example_op(x)  # 首次访问时加载 example_op 绑定。
# 直接调用 Torch 算子前，先显式加载（或先访问 pypto_gym.example_op）：
pypto_gym.load()
result = torch.ops.pypto_gym.example_op(x)
```

### 5. 与已有算子一起构建、安装和验收

完成上述接入后，在一次构建中选择两个算子：

```bash
# example_op 为占位名称，须先完成实现再执行。
python3 build_ci.py --pkg-type run --soc ascend950 \
  --ops mega_mhc,example_op --stest
```

`--stest` 要求新算子具备可运行的测试。测试可直接使用上述任一正式接口，每个
shape 或边界用例选择一个入口，比较完整输出与 golden。需要看护两种入口时，
将不同用例分配给两个入口，避免为了切换入口重复执行相同的算子场景。

在与构建环境 ABI 兼容、已加载 CANN 基础环境的终端中，将本次生成的 `.run`
安装到 `$ASCEND_HOME_PATH/opp`，即可直接导入。vendor 内的文件结构如下
（省略其他 host、kernel 产物）：

```text
<install-path>/vendors/pypto_gym/
├── op_api/lib/libcust_opapi.so
└── python/pypto_gym/
    ├── __init__.py
    ├── _loader.py
    ├── operators.json                 # 包含 mega_mhc 和 example_op
    ├── _native/
    │   ├── libpypto_gym_mega_mhc.so
    │   └── libpypto_gym_example_op.so
    └── _frontends/
        ├── mega_mhc.py
        └── example_op.py
```

标准 CANN 安装还会在 `$ASCEND_HOME_PATH/python/site-packages/pypto_gym/` 中
创建 `__init__.py`、`_loader.py`、`_native`、`_frontends`、`operators.json`
这五个相对链接，目标均指向上述 vendor 私有目录。

构建期应看到 `Precompiled Torch bindings:` 列出两个算子；安装后可确认：

```bash
cat "$ASCEND_HOME_PATH/opp/vendors/pypto_gym/python/pypto_gym/operators.json"
```

随后在新 Python 进程中 `import pypto_gym`，使用对应输入调用两个算子的正式接口。
仅出现在 `--ops` 中还不足以提供 Torch 入口；还必须完成绑定的 CMake 登记、
C++ 注册和同名 Python 前端导出。
