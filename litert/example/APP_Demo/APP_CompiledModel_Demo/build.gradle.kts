plugins {
    id("com.android.application") version "8.7.3" apply false
    // litert-api:2.3.0 ships Kotlin metadata 2.4.0; need Kotlin >= 2.3 to read it.
    id("org.jetbrains.kotlin.android") version "2.3.0" apply false
}
