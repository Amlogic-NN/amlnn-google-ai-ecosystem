# arm64-v8a native libs (preferred over Maven AAR .so)

Required for this demo stack:

1. libLiteRt.so                  <- from your Bazel litert.aar (jni/ only)
2. libLiteRtDispatch_Aml.so      <- AML offline/online dispatch
3. libLiteRtCompilerPlugin_Aml.so<- optional; online JIT only
4. libc++_shared.so / libadla.so / libnnsdk.so / ...  <- board deps as needed

API classes come from Maven litert:2.1.0, not from these .so files.
