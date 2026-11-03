# aml_compiled_model_case

[中文](README_zh.md)

Minimal LiteRT JIT sample for AML NPU. It loads a `.tflite`, compiles through
the AML compiler plugin, then runs once via the AML dispatch library.

Unlike `benchmark_online_aml` inside the LiteRT tree, this case only links
`libLiteRt.so` at build time. The compiler plugin and dispatch libraries are
`dlopen`'d from the current directory at runtime.

## What it does

1. Creates a LiteRT environment with:
   - dispatch library dir: `.`
   - compiler plugin library dir: `.`
   - compiler cache dir: `./cache`
2. Enables NPU and sets AML options:
   - model path: `/data/vendor/nn/`
   - model name: basename of the `.tflite` without extension
3. Builds a `CompiledModel` (JIT compile via `libLiteRtCompilerPlugin_Aml.so`)
4. Fills inputs from an optional binary file, or with random bytes
5. Runs inference once and prints each output buffer size

## Layout

```
Aml_CompiledModel_Case/
  aml_case.cc              # main sample
  case_runtime_shims.cc    # LrtAmlOptions / runtime shims (no Abseil)
  include/                 # LiteRT headers used by the case
  CMakeLists.txt           # Linux / Yocto build
  Android.mk               # Android ndk-build
  Application.mk
  build_aml_case.sh        # build + stage all platforms
  out/<platform>/          # staged binary + required .so files
```

Platform outputs:

| Directory | Source prebuilts |
|---|---|
| `out/android-arm64/` | `../compiled_litert/arm64` |
| `out/android-arm/` | `../compiled_litert/arm` |
| `out/linux-aarch64/` | `../compiled_litert/linux-aarch64` |
| `out/yocto-aarch64/` | `../compiled_litert/yocto-aarch64` |

## Prerequisites

Build LiteRT AML artifacts first, for example:

```bash
cd /path/to/LiteRT
./build_all_aml.sh
# or only one platform, e.g.:
# SKIP_ANDROID_LITERT=1 SKIP_YOCTO_LITERT=1 ./build_all_aml.sh
```

Required files under `Aml_LiteRT_Out/compiled_litert/<abi>/`:

- `libLiteRt.so`
- `libLiteRtCompilerPlugin_Aml.so`
- `libLiteRtDispatch_Aml.so`
- `libaml_compiler_core.so`
- `libcompiler_adla.so` (or `libadla_compiler.so`)

## Build

```bash
cd Aml_LiteRT_Out/Aml_CompiledModel_Case
./build_aml_case.sh
```

Skip platforms with:

```bash
SKIP_ANDROID_ARM64=1 SKIP_ANDROID_ARM=1 SKIP_YOCTO=1 ./build_aml_case.sh   # Linux only
SKIP_ANDROID_ARM64=1 SKIP_ANDROID_ARM=1 SKIP_LINUX=1 ./build_aml_case.sh   # Yocto only
SKIP_YOCTO=1 SKIP_LINUX=1 ./build_aml_case.sh                               # Android only
```

Optional overrides: `NDK`, `CMAKE`, `YOCTO_ENV`, `AARCH64_TOOLCHAIN_ROOT`,
`TOOLCHAIN_FILE`.

Linux / Yocto builds use `-fPIE -pie`. A non-PIE executable COPY-relocates
`std::cout` / iostream vtables into the main binary; those copies can interpose
`libcompiler_adla.so` and crash inside `Compile()`.

## Deploy and run

Copy one `out/<platform>/` folder to the device. Keep the binary and all `.so`
files in the same directory, then run from that directory.

```bash
cd /data/native_case   # or any deploy dir
export LD_LIBRARY_PATH="$(pwd):${LD_LIBRARY_PATH}"
chmod +x aml_compiled_model_case
mkdir -p cache /data/vendor/nn

./aml_compiled_model_case <model.tflite> [input_file]
```

Examples:

```bash
# Random inputs
./aml_compiled_model_case /data/models/mobilenet_v2_quant.tflite

# Binary input blob (must be large enough for every model input)
./aml_compiled_model_case ./model.tflite ./input.bin
```

Notes:

- Compiler output is written under `/data/vendor/nn/` as
  `<model_name>.adla` (and fingerprint). Ensure that directory is writable.
- Plugin / dispatch / compiler cache are loaded from `.` / `./cache`.
- On Android, also keep `libcompiler_adla.so` next to the binary if the plugin
  needs it; do not prepend `/vendor/lib64` to `LD_LIBRARY_PATH` just to find
  `libnnsdk.so` (that can break process startup).

## Exit status

| Code | Meaning |
|---|---|
| `0` | Inference finished |
| `1` | Bad args / I/O / LiteRT API error |
| `139` / segfault during compile | Often a non-PIE case binary interposing `compiler_adla`; rebuild with the current `CMakeLists.txt` |

## Related

- Full LiteRT + AML online benchmark: `compiled_litert/linux-aarch64/benchmark_online_aml`
- LiteRT all-platform build: repo-root `build_all_aml.sh`
