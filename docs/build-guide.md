# PyPTO-GYM 构建指南

## 概述

pypto-gym 是 PyPTO 算子仓，提供从算子源码到可安装产物的完整构建链。每个算子只需编写一次，即可同时交付两种产物形态：

| 产物 | 命令 | 安装方式 | 适用场景 |
|---|---|---|---|
| **whl**（JIT 直调） | `python3 build_ci.py --pkg-type whl` | `pip install` | 开发调试、快速验证 |
| **.run**（custom run 包） | `python3 build_ci.py --pkg-type run` | `bash *.run` | 生产部署、批量交付 |

## 两种产物模式的区别

### whl 模式（JIT 直调）

```
pip install pypto_gym-*.whl
    → import pypto_gym.ops.pro.experimental.flex_attention
    → 首次调用时 JIT 编译 kernel
```

- 算子以 Python 包形态分发，`op_kernel/*.py` 内的 `@pl.jit` kernel 在运行时按需编译
- 适合开发迭代：改完 kernel 直接跑，无需重新打包

### run 模式（custom run 包）

```
bash cann-950-pypto_gym_9.2.0_linux-aarch64.run
    → 安装到 $ASCEND_HOME_PATH/opp/vendors/pypto_gym/
    → config.ini 注册 load_priority=pypto_gym
    → aclnn 调用时直接使用预编译二进制
```

- 内含预编译产物：kernel `.o`（按 dtype 组合）+ host 侧三个 `.so` + aclnn 头文件 + ops-info.json
- 安装后零 JIT 开销，aclnn 接口直接查找已编译二进制

## 快速开始

### 前置条件

| 依赖 | 要求 |
|---|---|
| CANN 包 | ≥ 9.2.0（含 `op_build` 工具） |
| bisheng / asc_opc | 在 PATH 中（`source CANN set_env.sh`） |
| pypto_pro | Python 包可导入 |
| cmake | ≥ 3.18（系统级，非 pip 版） |

### 构建算子

```bash
# 构建单个算子的 .run 包
python3 build_ci.py --pkg-type run --ops flex_attention

# 构建所有算子（默认）
python3 build_ci.py --pkg-type run

# 构建 whl
python3 build_ci.py --pkg-type whl
```

### 安装 .run 包

```bash
# 默认安装到 $ASCEND_HOME_PATH/opp
bash build_out/cann-950-pypto_gym_9.2.0_linux-aarch64.run

# 指定路径
bash build_out/cann-950-pypto_gym_9.2.0_linux-aarch64.run --install-path=/your/opp/dir

# 多用户场景（world-readable 权限）
bash build_out/cann-950-pypto_gym_9.2.0_linux-aarch64.run --install-for-all
```

### 卸载

```bash
bash build_out/cann-950-pypto_gym_9.2.0_linux-aarch64.run --uninstall
```

## 目录结构

```
pypto-gym/
├── build_ci.py              # 构建入口（双模式）
├── CMakeLists.txt           # 顶层 CMake（流程叙事）
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
│   └── pro/
│       └── experimental/
│           └── flex_attention/
│               ├── CMakeLists.txt   # 算子级 dispatcher
│               ├── op_host/         # C++ host 侧（def/tiling/infershape）
│               ├── op_kernel/       # PyPTO kernel（@pl.jit）
│               └── tests/           # 测试
├── scripts/package/         # .run 安装脚本骨架
└── build_out/               # 构建产物（.run / .whl）
```

## 算子目录契约

新增算子只需满足以下结构，构建链自动发现并接入：

```
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
```

**关键约束**：

- 每个入口文件仅允许一个 `@pl.jit` kernel
- `@pl.jit` 必须带 `tiling_key=` schema（binary 交付硬要求）
- `TilingData` 字段名/顺序/类型必须与 kernel dataclass 严格一致（构建期自动校验）
- tiling 参数必须是 kernel 的最后一个参数

## 多 soc 支持

当前支持 `ascend950`。多 soc 通过外部循环实现——每个 soc 独立一轮完整构建，产出独立 `.run`：

```bash
# 未来扩展（白名单放开后）
python3 build_ci.py --pkg-type run --soc ascend950,ascend910b
# → 产出 cann-950-*.run + cann-910b-*.run 两个包
```

## 环境变量

| 变量 | 用途 | 默认值 |
|---|---|---|
| `ASCEND_HOME_PATH` | CANN 包路径 | `/usr/local/Ascend/cann` |
| `Python3_EXECUTABLE` | Python 解释器 | 系统 python3 |
| `CANN_3RD_LIB_PATH` | 三方库缓存路径 | `<repo>/third_party/` |

## FAQ

**Q: 构建报 "bisheng not found"？**
A: 先 `source /usr/local/Ascend/cann/set_env.sh`，确保 bisheng/asc_opc 在 PATH 中。

**Q: .run 安装后 aclnn 找不到算子？**
A: 检查 `grep load_priority $ASCEND_HOME_PATH/opp/vendors/config.ini` 是否含 `pypto_gym`；并确认 `LD_LIBRARY_PATH` 包含 `$ASCEND_HOME_PATH/opp/vendors/pypto_gym/op_api/lib/`。

**Q: whl 安装后 import 报错？**
A: JIT 模式依赖 `torch + torch_npu`，需先 source CANN 环境并安装 torch-npu。
