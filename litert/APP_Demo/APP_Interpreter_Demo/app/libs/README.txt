LiteRT_AML.aar
  Java: org.tensorflow.lite.AmlDelegate
  jni/arm64-v8a:
    libtensorflowlite_aml_jni.so
    libaml_compiler_core.so
    libcompiler_adla.so

libAML_Delegate.so (next to the AAR in the build output) is the C++ external
delegate. This APK does not load it. The Java Interpreter path uses the .so
files inside LiteRT_AML.aar.

NPU runtime loaded by dlopen (not inside the AAR) lives in
app/src/main/jniLibs/arm64-v8a/:
  libnnsdk.so
  libadla.so
  libteec.so
