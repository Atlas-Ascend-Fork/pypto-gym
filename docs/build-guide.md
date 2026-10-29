# PyPTO-Gym 构建与测试指南

## 1. 构建

### 1.1. 概述

PyPTO-Gym 是 PyPTO 实验算子仓，应该提供从算子源码到可安装产物的完整构建链。

重构后对于 Pro 模式算子，每个算子只需编写一次，即可同时交付两种产物形态：

| 产物 | 命令 | 安装方式 | 适用场景 |
| --- | --- | --- | --- |
| **whl**（JIT 直调） | `python3 build_ci.py --pkg-type whl` | `pip install` | 开发调试、快速验证、PIP源生态发布 |
| **.run**（custom run 包） | `python3 build_ci.py --pkg-type run` | `bash *.run` | 生产部署、批量交付 |

### 1.2. 两种产物模式的区别

#### 1.2.1. whl 模式（JIT 直调）

```bash
pip install build_out/whl/ascend950/pypto_gym-*.whl
    → import pypto_gym.ops.pro.experimental.flex_attention
    → 首次调用时 JIT 编译 kernel
```

- 算子以 Python 包形态分发，`op_kernel/*.py` 内的 `@pl.jit` kernel 在运行时按需编译
- 适合开发迭代：改完 kernel 直接跑，无需重新打包

### 1.2.2. run 模式（custom run 包）

```bash
bash cann-950-pypto_gym_9.2.0_linux-aarch64.run
    → 安装到 $ASCEND_HOME_PATH/opp/vendors/pypto_gym/
    → config.ini 注册 load_priority=pypto_gym
    → aclnn 调用时直接使用预编译二进制
```

- 内含预编译产物：kernel `.o`（按 dtype 组合）+ host 侧三个 `.so` + aclnn 头文件 + ops-info.json
- 安装后零 JIT 开销，aclnn 接口直接查找已编译二进制

### 1.3. 快速开始

#### 1.3.1. 前置条件

| 依赖 | 要求 |
| --- | --- |
| CANN 包 | ≥ 9.2.0（含 `op_build` 工具） |
| bisheng / asc_opc | 在 PATH 中（`source CANN set_env.sh`） |
| pypto_pro | Python 包可导入 |
| cmake | ≥ 3.18（系统级，非 pip 版） |

#### 1.3.2 构建算子

```bash
# 构建单个算子的 .run 包
python3 build_ci.py --pkg-type run --ops flex_attention

# 构建所有算子（默认）
python3 build_ci.py --pkg-type run

# 构建 whl
python3 build_ci.py --pkg-type whl
```

> 测试用的 aclnn 桥默认不参与构建，需要 `ENABLE_TESTS=ON`（`build_ci.py` 暂未暴露该开关，手工 cmake 重配方式见 §2.3）。

#### 1.3.3. 安装 .run 包

```bash
# 默认安装到 $ASCEND_HOME_PATH/opp
bash build_out/run/ascend950/cann-950-pypto_gym_9.2.0_linux-aarch64.run

# 指定路径
bash build_out/run/ascend950/cann-950-pypto_gym_9.2.0_linux-aarch64.run --install-path=/your/opp/dir

# 多用户场景（world-readable 权限）
bash build_out/run/ascend950/cann-950-pypto_gym_9.2.0_linux-aarch64.run --install-for-all
```

#### 1.3.4. 卸载

```bash
bash build_out/run/ascend950/cann-950-pypto_gym_9.2.0_linux-aarch64.run --uninstall
```

### 1.3.5. 目录结构

```bash
pypto-gym/
├── build_ci.py              # 构建入口（双模式）
├── CMakeLists.txt           # 顶层 CMake（流程叙事）
├── conftest.py              # pytest 全局入口（--mode / 路径注入）
├── pytest.ini               # pytest 配置（pythonpath = tests/common）
├── setup.py                 # whl 打包配置
├── version.cmake            # 包版本定义
├── cmake/                   # CMake 构建系统
│   ├── config.cmake         #   四阶段: 环境检查/选项缺省/入参校验/预处理
│   ├── func.cmake           #   算子发现 + pypto 桥接 + kernel 编译链
│   ├── obj_func.cmake       #   host 侧源收集宏 + 反射载体
│   ├── custom_build.cmake   #   host 目标创建 + install + 打包支撑
│   ├── prepare_3rd.cmake    #   三方库路径 + cann-cmake fetch
│   └── scripts/             #   codegen / 布局校验 / 二次构建脚本
├── ops/                     # 算子源码树
│   └── pro/experimental/flex_attention/
│       ├── CMakeLists.txt   #   算子级 dispatcher
│       ├── op_host/         #   C++ host 侧（def/tiling/infershape）
│       ├── op_kernel/       #   PyPTO kernel（@pl.jit）
│       └── tests/
│           ├── CMakeLists.txt      # 测试侧构建入口（仅 ENABLE_TESTS）
│           ├── aclnn_bridge/       # pybind11 桥（aclnn 模式）
│           ├── assets/             # golden/常量/期望构造
│           ├── flex_attention_call.py  # 用例共享调用 helper
│           └── test_*.py           # 用例（pytest 收集）
├── tests/                   # 测试支撑 + 历史 pytest 用例
│   ├── common/op_caller.py  #   jit/aclnn 调用适配
│   └── ops/pro/aclnn_bridge_common/   # 测试桥公共层
├── scripts/package/         # .run 安装脚本骨架
└── build_out/
    ├── whl/<soc>/           # .whl + site-packages(测试安装) + ops_registry.json
    └── run/<soc>/           # .run + vendors(测试安装, 下次 stest 前覆盖)
```

### 1.3.6. 算子目录契约

新增算子只需满足以下结构，构建链自动发现并接入：

```bash
ops/<paradigm>/<network>/<op_name>/
├── CMakeLists.txt           # 算子级 dispatcher（模板化，无需修改）
├── op_host/
│   ├── CMakeLists.txt       # 调用 require_pypto_pro / add_modules_sources 等
│   ├── <op>_def.cpp         # 算子定义（输入/输出/属性）
│   ├── <op>_tiling.cpp      # tiling 计算
│   ├── <op>_tiling.h        # TilingData 结构（须与 kernel dataclass 字节对齐）
│   └── <op>_infershape.cpp  # shape 推导
├── op_kernel/
│   └── <op>.py              # PyPTO kernel（唯一 @pl.jit）
└── tests/
    ├── CMakeLists.txt       # 测试侧构建入口（仅 ENABLE_TESTS）
    ├── aclnn_bridge/        # pybind11 桥（可选；aclnn 模式）
    ├── assets/              # golden/常量/期望构造（文件名带算子前缀）
    └── test_*.py            # 用例
```

**关键约束**：

- 每个入口文件仅允许一个 `@pl.jit` kernel
- `@pl.jit` 必须带 `tiling_key=` schema（binary 交付硬要求）
- `TilingData` 字段名/顺序/类型必须与 kernel dataclass 严格一致（构建期自动校验）
- tiling 参数必须是 kernel 的最后一个参数

### 1.3.7. 多 soc 支持

当前白名单仅 `ascend950`。工程侧已支持 `--soc` 逗号分隔多值：每个 soc 独立一轮完整构建（configure → build → package），产出带 chip 名的独立 `.run`：

```bash
# 白名单放开后可直接多 soc 构建
python3 build_ci.py --pkg-type run --soc ascend950,ascend910b
# → 产出 cann-950-*.run + cann-910b-*.run
```

> 说明：多 soc 的算子侧还需要 `@pl.jit(arch=...)` 参数化；当前仅 ascend950 在范围内。

### 1.3.8. 环境变量

| 变量 | 用途 | 默认值 |
| --- | --- | --- |
| `ASCEND_HOME_PATH` | CANN 包路径 | `/usr/local/Ascend/cann` |
| `Python3_EXECUTABLE` | Python 解释器 | 系统 python3 |
| `CANN_3RD_LIB_PATH` | 三方库缓存路径 | `<repo>/third_party/` |
| `GYM_SOC` | pytest aclnn 模式默认桥目录使用的 soc | `ascend950` |

### 1.3.9 FAQ

**Q: 构建报 "bisheng not found"？**
A: 先 `source /usr/local/Ascend/cann/set_env.sh`，确保 bisheng/asc_opc 在 PATH 中。

**Q: .run 安装后 aclnn 找不到算子？**
A: 检查 `grep load_priority $ASCEND_HOME_PATH/opp/vendors/config.ini` 是否含 `pypto_gym`；并确认 `LD_LIBRARY_PATH` 包含 `$ASCEND_HOME_PATH/opp/vendors/pypto_gym/op_api/lib/`。

**Q: whl 安装后 import 报错？**
A: JIT 模式依赖 `torch + torch_npu`，需先 source CANN 环境并安装 torch-npu。

## 2. 测试

### 2.1. 两种执行模式

算子测试支持同一组用例、同一组 golden 在两种模式下执行，通过 `pytest --mode` 切换：

| 模式 | 调用链 | 依赖 | 适用 |
| --- | --- | --- | --- |
| `jit`（默认） | `op_caller.get_op_caller("<op>")` → import `op_kernel/<op>.py` 宿主入口 → 首次调用 JIT 编译 kernel | 源码树 + CANN + pypto_pro | 开发调试、改 kernel 后快速验证 |
| `aclnn` | `op_caller.get_op_caller("<op>")` → import `aclnn_bridge_<op>.so` → `dlopen` `libcust_opapi.so` → 调 C++ aclnn 接口 | 预编译桥 + 已安装的 run 包 | 验证 run 包安装后的二进制复用链路 |

两种模式共用：

- 用例：`ops/<...>/<op>/tests/test_*.py`
- golden/常量/期望构造：`ops/<...>/<op>/tests/assets/`
- 调用 helper：`ops/<...>/<op>/tests/<op>_call.py`
- 调用适配：`tests/common/op_caller.py`（`pytest.ini` 的 `pythonpath` 引入）
- 路径注入：根 `conftest.py` 集中处理（`tests/`、`tests/assets/`、`../op_kernel/`），用例文件不写 `sys.path`

当前 flex_attention 的 `test_flex_attention.py` 支持双模式；`test_flex_attention_matrix.py`、`test_flex_attention_perf.py`、`test_flex_attention_prefill.py` 目前仅 JIT 模式。

### 2.2. 运行 JIT 模式

以下命令均在仓库根目录执行（`pytest.ini` 所在目录）；无需构建测试桥，直接运行：

```bash
# 单文件
pytest ops/pro/experimental/flex_attention/tests/test_flex_attention.py --mode jit -q

# 整个算子测试目录（含 matrix/perf/prefill，均为 JIT）
pytest ops/pro/experimental/flex_attention/tests --mode jit -q
```

### 2.3. 构建 aclnn 测试桥

测试桥只在 `ENABLE_TESTS=ON` 时构建（`build_ci.py` 暂未暴露该开关，使用手工 cmake 重配）：

```bash
cmake -S . -B build/ascend950 \
      -DPython3_EXECUTABLE=$(which python3) \
      -DASCEND_COMPUTE_UNIT=ascend950 \
      -DPACKAGE_TYPE=run \
      -DASCEND_OP_NAME=flex_attention \
      -DCMAKE_INSTALL_PREFIX=$PWD/build_out/run/ascend950 \
      -DENABLE_TESTS=ON

cmake --build build/ascend950 --target aclnn_bridge_flex_attention -j 32
```

产物：`build/<soc>/tests/aclnn_bridge_<op>.so`。

说明：

- `ENABLE_TESTS=ON` 时 `cmake/config.cmake` 才会查找 torch；`OFF` 时 run/kernel 构建不依赖 torch。
- 桥的源文件/target/产物/模块名统一为 `aclnn_bridge_<op>`，由 `gym_add_aclnn_bridge(<op>)` 生成。
- 新增算子时，在算子 `tests/aclnn_bridge/CMakeLists.txt` 调用 `gym_add_aclnn_bridge(<op>)` 并放置 `aclnn_bridge_<op>.cpp` 即可；算子 `tests/CMakeLists.txt` 会自动派发。

### 2.4. 运行 aclnn 模式

aclnn 模式需要两部分同时就绪：

1. 测试桥 `.so`：位于 `build/<soc>/tests/`，或通过 `--aclnn-lib-dir` 指定；
2. 已安装的 run 包：提供 `libcust_opapi.so`，并通过 `set_env.bash` 写入 `ASCEND_CUSTOM_OPP_PATH`。

```bash
# 1) 安装 run 包（已安装可跳过）
bash build_out/run/ascend950/cann-950-pypto_gym_*.run --install-path=$ASCEND_HOME_PATH/opp
source $ASCEND_HOME_PATH/opp/vendors/pypto_gym/bin/set_env.bash

# 2) 运行（默认桥目录 build/<soc>/tests/）
pytest ops/pro/experimental/flex_attention/tests/test_flex_attention.py --mode aclnn -q

# 桥不在默认目录时显式指定
pytest ops/pro/experimental/flex_attention/tests/test_flex_attention.py \
       --mode aclnn --aclnn-lib-dir=build/ascend950/tests -q
```

- 默认桥目录由 `--aclnn-lib-dir` → `$GYM_SOC`（默认 `ascend950`）决定，即 `build/<soc>/tests/`。
- 只跑双模式用例时先 `source set_env.bash`；否则桥 `dlopen libcust_opapi.so` 会失败。
- 未安装 run 包时也可以 `--mode jit` 做功能验证。

### 2.5. 测试目录约定

```text
ops/<...>/<op>/tests/
├── CMakeLists.txt          # 测试侧构建入口(仅 ENABLE_TESTS)
├── aclnn_bridge/
│   ├── CMakeLists.txt      # gym_add_aclnn_bridge(<op>)
│   └── aclnn_bridge_<op>.cpp
├── assets/
│   ├── <op>_golden.py      # 参考实现/期望构造
│   ├── <op>_mask_specs.py  # 数据规范(按需)
│   └── <op>_kernel_consts.py
├── <op>_call.py            # 用例共享调用 helper
└── test_*.py
```

导入约定：`assets/` 目录整体在 `sys.path` 中，使用**带算子前缀的顶层模块名**（如 `from flex_attention_golden import ...`），避免多算子同 session 时模块名冲突。

### 2.6. 一键 stest（build_ci.py）

`build_ci.py` 新增 `-s/--stest`，按 `--pkg-type` 自动选择执行模式：

```bash
# jit：构建 whl → 安装到 build_out/whl/<soc>/site-packages → pytest --mode jit
python3 build_ci.py --pkg-type whl --stest

# aclnn：构建 run → 隔离安装到 build_out/run/<soc> → pytest --mode aclnn
python3 build_ci.py --pkg-type run --ops flex_attention --soc ascend950 --stest

# 用例过滤：--stest 的值作为 pytest -k 表达式（逗号分隔按 or 处理）
python3 build_ci.py --pkg-type whl --stest=test_fa_mask_info
python3 build_ci.py --pkg-type run --ops flex_attention \
        --stest=test_fa_mask_info,test_fa_mask_info_bshd_layout
```

行为说明：

- jit 模式测试依赖**安装后的 whl**：`pip install --target build_out/whl/<soc>/site-packages`，并通过 `ops_registry.json` 让 `op_caller` 从安装包内 `pypto_gym.ops...op_kernel.<op>` 导入；
- run 模式测试安装到 `build_out/run/<soc>`（`--install-path`），只向 pytest 子进程注入 `ASCEND_CUSTOM_OPP_PATH` / `LD_LIBRARY_PATH`，不修改父进程环境、不 source set_env；
- 测试结束不清理 `site-packages` / `vendors`；下次 stest 前会先 `--uninstall` 再安装，`-c` 可整体清理；
- run/aclnn 自动追加 `-m "not jit_only"`，排除 matrix/perf/prefill 等 JIT-only 用例；
- `--stest` 只允许单个 soc，多 soc 直接报错；`--utest` 暂未提供。

产物布局：

```text
build_out/
├── whl/<soc>/{*.whl, site-packages/, ops_registry.json}
└── run/<soc>/{*.run, vendors/}
```

### 2.7. 测试 FAQ

**Q: `--mode aclnn` 报 `pybind11 bridge dir not found`？**
A: 先按 §2.3 构建 `aclnn_bridge_<op>.so`，或用 `--aclnn-lib-dir` 指向实际目录。

**Q: aclnn 模式报 `libcust_opapi.so not found`？**
A: 确认已安装 run 包并 `source .../vendors/pypto_gym/bin/set_env.bash`，检查 `ASCEND_CUSTOM_OPP_PATH`。

**Q: `pytest` 退出码 139 / teardown core dump？**
A: torch_npu 退出阶段噪音，用例结果已输出；以日志中的 passed/failed 为准。

**Q: 只跑某个用例？**
A: `pytest <test_file> -k <case_name> --mode jit|aclnn`；先用 `--collect-only` 确认用例名。
