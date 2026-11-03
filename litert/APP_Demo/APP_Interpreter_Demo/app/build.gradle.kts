import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

base {
    archivesName.set("YOLOX_Interprerter")
}

android {
    namespace = "com.example.interpreterdemo"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.example.yoloxinterpreter"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"

        buildConfigField("String", "MODEL_ASSET", "\"yolox_s_w8a16_f16.tflite\"")
        buildConfigField("String", "MODEL_NAME", "\"yolox_s_w8a16_f16\"")

        ndk {
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

    packaging {
        jniLibs {
            // Extract .so so AmlDelegate can dlopen libnnsdk.so from nativeLibraryDir.
            useLegacyPackaging = true
            pickFirsts += listOf(
                "lib/arm64-v8a/libc++.so",
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

kotlin {
    compilerOptions {
        jvmTarget.set(JvmTarget.JVM_17)
    }
}

dependencies {
    // Interpreter Java API + libtensorflowlite_jni.so.
    // AmlDelegate / libtensorflowlite_aml_jni.so / compiler libs come from the local AAR.
    implementation("org.tensorflow:tensorflow-lite:2.16.1")
    implementation(files("libs/LiteRT_AML.aar"))

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
