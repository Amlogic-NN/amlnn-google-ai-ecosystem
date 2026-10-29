# LiteRT

This directory holds on-device Android inference samples based on Google LiteRT.

## YoloX_Litert_Demo

Real-time Android object detection demo. The camera preview is letterboxed, then run through a YOLOX model (`yolox_s_w8a16_f16_dispatch.tflite`) with LiteRT `CompiledModel`. Detection boxes are drawn back onto the preview. Frames are dropped while inference is busy.

Dependencies are split into three layers:

- **API**: Maven `com.google.ai.edge.litert:litert:2.1.0`
- **Native**: Bazel-built `libLiteRt.so`, placed in `jniLibs/arm64-v8a/`
- **AML NPU**: `libLiteRtDispatch_Aml.so` in the same directory; NPU is used when `USE_NPU=true`

Open `YoloX_Litert_Demo` in Android Studio and run it on an `arm64-v8a` device. See that directory's README for dependency and packaging details. For Android, copy the libraries under `libs/arm64` into `YoloX_Litert_Demo/app/src/main/jniLibs/arm64-v8a/`.

## libs

Prebuilt LiteRT and AML NPU shared libraries, grouped by target platform:

| Directory | Platform |
|-----------|----------|
| `arm` / `arm64` / `x86` | Android |
| `linux-aarch64` | Linux aarch64 |
| `yocto-aarch64` | Yocto aarch64 |

Libraries in each directory:

- `libLiteRt.so`: LiteRT native runtime
- `libLiteRtDispatch_Aml.so`: AML NPU dispatch, required for offline inference
- `libLiteRtCompilerPlugin_Aml.so`: AML compiler plugin, required for online JIT
- `libaml_compiler_core.so`: AML compiler core, version **1.02**
- `libcompiler_adla.so`: ADLA compiler, version **v3.5.3** (`linux-aarch64` and `yocto-aarch64` only)
