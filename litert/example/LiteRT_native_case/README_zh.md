# aml_compiled_model_case

[English](README.md)

AML NPU 上的精简 LiteRT JIT 示例：加载 `.tflite`，经 AML compiler plugin 编译，再通过 AML dispatch 跑一遍推理。

和 LiteRT 树里的 `benchmark_online_aml` 不同，本 case 编译时只链接 `libLiteRt.so`；compiler plugin 与 dispatch 库在运行时从当前目录 `dlopen`。

## 功能

1. 创建 LiteRT Environment，配置：
   - dispatch 库目录：`.`
   - compiler plugin 库目录：`.`
   - 编译缓存目录：`./cache`
2. 打开 NPU，并设置 AML 选项：
   - 模型输出路径：`/data/vendor/nn/`
   - 模型名：`.tflite` 文件名去掉扩展名
3. 创建 `CompiledModel`（经 `libLiteRtCompilerPlugin_Aml.so` 做 JIT 编译）
4. 输入来自可选的二进制文件；未提供则用随机字节填充
5. 推理一次，打印每个输出 buffer 的大小

## 目录结构

```
Aml_CompiledModel_Case/
  aml_case.cc              # 主程序
  case_runtime_shims.cc    # LrtAmlOptions / 运行时 shim（不依赖 Abseil）
  include/                 # case 使用的 LiteRT 头文件
  CMakeLists.txt           # Linux / Yocto 编译
  Android.mk               # Android ndk-build
  Application.mk
  build_aml_case.sh        # 多平台编译并打包
  out/<platform>/          # 可执行文件 + 所需 .so
```

各平台产物：

| 目录 | 依赖的预编译库 |
|---|---|
| `out/android-arm64/` | `../compiled_litert/arm64` |
| `out/android-arm/` | `../compiled_litert/arm` |
| `out/linux-aarch64/` | `../compiled_litert/linux-aarch64` |
| `out/yocto-aarch64/` | `../compiled_litert/yocto-aarch64` |

## 前置条件

先编译 LiteRT AML 产物，例如：

```bash
cd /path/to/LiteRT
./build_all_aml.sh
# 或只编某一平台，例如：
# SKIP_ANDROID_LITERT=1 SKIP_YOCTO_LITERT=1 ./build_all_aml.sh
```

`Aml_LiteRT_Out/compiled_litert/<abi>/` 下需要有：

- `libLiteRt.so`
- `libLiteRtCompilerPlugin_Aml.so`
- `libLiteRtDispatch_Aml.so`
- `libaml_compiler_core.so`
- `libcompiler_adla.so`（或 `libadla_compiler.so`）

## 编译

```bash
cd Aml_LiteRT_Out/Aml_CompiledModel_Case
./build_aml_case.sh
```

按平台跳过：

```bash
SKIP_ANDROID_ARM64=1 SKIP_ANDROID_ARM=1 SKIP_YOCTO=1 ./build_aml_case.sh   # 只编 Linux
SKIP_ANDROID_ARM64=1 SKIP_ANDROID_ARM=1 SKIP_LINUX=1 ./build_aml_case.sh   # 只编 Yocto
SKIP_YOCTO=1 SKIP_LINUX=1 ./build_aml_case.sh                               # 只编 Android
```

可选覆盖：`NDK`、`CMAKE`、`YOCTO_ENV`、`AARCH64_TOOLCHAIN_ROOT`、`TOOLCHAIN_FILE`。

Linux / Yocto 使用 `-fPIE -pie`。非 PIE 可执行文件会把 `std::cout` / iostream 虚表 COPY 进主程序，可能截胡 `libcompiler_adla.so`，在 `Compile()` 内段错误。

## 部署与运行

把某个 `out/<platform>/` 整目录拷到板端。可执行文件和全部 `.so` 放在同一目录，在该目录下执行：

```bash
cd /data/native_case   # 或任意部署目录
export LD_LIBRARY_PATH="$(pwd):${LD_LIBRARY_PATH}"
chmod +x aml_compiled_model_case
mkdir -p cache /data/vendor/nn

./aml_compiled_model_case <model.tflite> [input_file]
```

示例：

```bash
# 随机输入
./aml_compiled_model_case /data/models/mobilenet_v2_quant.tflite

# 二进制输入（文件大小需覆盖所有模型输入）
./aml_compiled_model_case ./model.tflite ./input.bin
```

注意：

- 编译产物写到 `/data/vendor/nn/<model_name>.adla`（及 fingerprint），目录需可写。
- plugin / dispatch / 编译缓存从 `.` / `./cache` 加载。
- Android 上若 plugin 依赖 `libcompiler_adla.so`，也请放在同目录；不要为找 `libnnsdk.so` 把 `/vendor/lib64` 加进 `LD_LIBRARY_PATH`（可能导致进程起不来）。

## 退出码

| 码 | 含义 |
|---|---|
| `0` | 推理完成 |
| `1` | 参数错误 / I/O / LiteRT API 失败 |
| `139` / 编译阶段段错误 | 多为非 PIE 的 case 截胡了 `compiler_adla`；用当前 `CMakeLists.txt` 重新编译 |

## 相关

- 完整 LiteRT + AML online 用例：`compiled_litert/linux-aarch64/benchmark_online_aml`
- LiteRT 全平台编译：仓库根目录 `build_all_aml.sh`
