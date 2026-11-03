# CompiledModel Demo (Android)

最简 LiteRT `CompiledModel`：assets 模型 → 随机输入 → APK 内推理。

## 依赖拆分（当前方案）

| 层 | 来源 | 放哪里 |
|----|------|--------|
| **API** | Maven `com.google.ai.edge.litert:litert:2.1.0` | `app/build.gradle.kts` |
| **Native** | 你 Bazel 编的 `libLiteRt.so` | `app/src/main/jniLibs/arm64-v8a/` |
| **AML** | `libLiteRtDispatch_Aml.so` 等 | 同上 `jniLibs/arm64-v8a/` |

Bazel `//litert/kotlin:litert` 产出的 AAR 经常 **只有 so、classes.jar 为空**，不能单独当 API 依赖。从该 AAR 解出 so 即可：

```bash
bazel build --config=android_arm64 --android_ndk_min_sdk_version=26 \
  //litert/kotlin:litert

# 解压 bazel-bin/litert/kotlin/litert.aar
# 拷贝 jni/arm64-v8a/libLiteRt.so -> app/src/main/jniLibs/arm64-v8a/
```

AML（示例）：

```text
app/src/main/jniLibs/arm64-v8a/
  libLiteRt.so
  libLiteRtDispatch_Aml.so          # offline 必需
  libLiteRtCompilerPlugin_Aml.so    # 仅 online JIT 需要
  libc++_shared.so
  libadla.so / libnnsdk.so / ...    # 板端依赖，按你环境补齐
```

`USE_NPU=true` 时走 `BuiltinNpuAcceleratorProvider` → `nativeLibraryDir` 动态加载。

## 打开工程

1. Android Studio Open `CompiledModel_Demo`
2. Sync Gradle → Run（`arm64-v8a` 设备）

## 模型

放到 `app/src/main/assets/`，并在 `app/build.gradle.kts` 配置：

```kotlin
buildConfigField("boolean", "USE_NPU", "true")
buildConfigField("String", "MODEL_ASSET", "\"resnet_float_dispatch.tflite\"")
```

## packaging 说明

`useLegacyPackaging = true`：so 解压到可访问的 `nativeLibraryDir`，供 dispatch 动态 `dlopen`。  
`pickFirsts`：本地 `jniLibs` 与 Maven AAR 内嵌 so 冲突时优先合并策略不炸包。
