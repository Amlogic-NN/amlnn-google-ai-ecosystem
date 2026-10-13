# LiteRT

本目录存放基于 Google LiteRT 的 Android 端侧推理示例。

## YoloX_Litert_Demo

Android 实时目标检测 Demo。摄像头预览画面经 letterbox 预处理后，用 LiteRT `CompiledModel` 跑 YOLOX 模型（`yolox_s_w8a16_f16_dispatch.tflite`），再把检测框画回预览上。推理忙时会丢帧。

依赖拆成三层：

- **API**：Maven `com.google.ai.edge.litert:litert:2.1.0`
- **Native**：Bazel 编出的 `libLiteRt.so`，放到 `jniLibs/arm64-v8a/`
- **AML NPU**：`libLiteRtDispatch_Aml.so` 等同目录，`USE_NPU=true` 时走 NPU

用 Android Studio 打开 `YoloX_Litert_Demo`，在 `arm64-v8a` 设备上运行。更多依赖与打包说明见该目录下的 README。
