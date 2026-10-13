plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.example.compiledmodeldemo"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.example.compiledmodeldemo"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"

        // NPU offline: needs libLiteRtDispatch_Aml.so (+ deps) in jniLibs.
        buildConfigField("boolean", "USE_NPU", "true")
        buildConfigField("String", "MODEL_ASSET", "\"yolox_s_w8a16_f16_dispatch.tflite\"")
        buildConfigField("String", "INPUT_IMAGE_ASSET", "\"zidane.jpg\"")

        ndk {
            // Match your Bazel/AML .so ABI.
            abiFilters += listOf("arm64-v8a")
        }
    }

    buildFeatures {
        buildConfig = true
        viewBinding = true
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    packaging {
        jniLibs {
            // Keep .so extractable so BuiltinNpuAcceleratorProvider can dlopen
            // from applicationInfo.nativeLibraryDir.
            useLegacyPackaging = true
            // Prefer app/src/main/jniLibs over Maven AAR embedded natives.
            pickFirsts += listOf(
                "lib/arm64-v8a/libLiteRt.so",
                "lib/arm64-v8a/libc++_shared.so",
                "lib/arm64-v8a/libLiteRtDispatch_Aml.so",
                "lib/arm64-v8a/libLiteRtCompilerPlugin_Aml.so",
            )
        }
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }

    androidResources {
        noCompress += "tflite"
    }
}

dependencies {
    // ---------------------------------------------------------------------------
    // Stack:
    //   1) API     : Maven litert (Kotlin CompiledModel / Environment classes)
    //   2) Native  : your Bazel libLiteRt.so in app/src/main/jniLibs/arm64-v8a/
    //   3) AML     : libLiteRtDispatch_Aml.so (+ plugin/deps) in the same jniLibs
    //
    // Do NOT use Bazel //litert/kotlin:litert.aar alone — its classes.jar is often
    // empty. Extract only libLiteRt.so from that AAR into jniLibs.
    // ---------------------------------------------------------------------------
    implementation("com.google.ai.edge.litert:litert:2.1.0")

    // Align with litert:2.1.0 (Kotlin metadata 2.2.0).
    implementation("org.jetbrains.kotlin:kotlin-stdlib:2.2.0")

    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
    implementation("androidx.lifecycle:lifecycle-common:2.8.4")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.4")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-guava:1.8.1")
    implementation("com.google.guava:guava:33.2.1-android")

    val cameraX = "1.3.4"
    implementation("androidx.camera:camera-core:$cameraX")
    implementation("androidx.camera:camera-camera2:$cameraX")
    implementation("androidx.camera:camera-lifecycle:$cameraX")
    implementation("androidx.camera:camera-view:$cameraX")
}
