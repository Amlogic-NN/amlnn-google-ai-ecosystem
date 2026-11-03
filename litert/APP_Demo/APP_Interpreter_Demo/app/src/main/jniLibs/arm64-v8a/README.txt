Board runtime libs dlopened by the AML delegate. Not part of LiteRT_AML.aar.

libnnsdk.so        AmlDelegateKernel dlopen
libadla.so         libnnsdk.so dlopen
libteec.so         libadla.so dlopen
libc++.so          required by libteec.so (not std::__ndk1)
# Do not add libc++_shared.so. libcompiler_adla.so already contains
# std::__ndk1, and a second copy crashes Compiler::GetVersionBuildInfo.

Compiler libs (libaml_compiler_core.so, libcompiler_adla.so) and
libtensorflowlite_aml_jni.so come from app/libs/LiteRT_AML.aar.
libtensorflowlite_jni.so comes from Maven org.tensorflow:tensorflow-lite.
