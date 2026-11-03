# Interpreter Demo (Android)

摄像头 YOLOX，推理走 TFLite `Interpreter` + `AmlDelegate`。

源模型固定为 `app/src/main/assets/yolox_s_w8a16_f16.tflite`。不传 `cache_path`，delegate 走在线 JIT：

- `.adla` 不存在：`BuildGraph` → `AmlContextCompile`，写到应用 cache 目录 `cache/yolox_s_w8a16_f16.adla`（`Context.getCacheDir()`，经 `AmlDelegate.Options.model_path` 传给 delegate）
- `.adla` 已存在：跳过编译，直接加载

要重新 JIT，删掉设备上该 `.adla` 后重启应用。

## 依赖

| 层 | 来源 | 放哪里 |
|----|------|--------|
| Interpreter API + `libtensorflowlite_jni.so` | Maven `org.tensorflow:tensorflow-lite:2.16.1` | `app/build.gradle.kts` |
| `AmlDelegate` + `libtensorflowlite_aml_jni.so` + compiler so | `compiled_interpreter/arm64/LiteRT_AML.aar` | `app/libs/LiteRT_AML.aar` |
| `libnnsdk.so` / `libadla.so` / `libteec.so` | 板端运行时，delegate `dlopen` | `app/src/main/jniLibs/arm64-v8a/` |

构建产物里的 `libAML_Delegate.so` 是 C++ external delegate，这条 Java 路径不加载它。

## 打开工程

1. Android Studio 打开本目录
2. Sync Gradle，在 `arm64-v8a` 设备上 Run

首次加载会做 JIT，界面停在 “Loading model (AML JIT)…” 直到编译结束。
