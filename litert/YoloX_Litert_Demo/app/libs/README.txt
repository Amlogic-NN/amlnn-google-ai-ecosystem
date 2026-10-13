# Do not use incomplete Bazel litert.aar as the Gradle API dependency.
# (classes.jar is often empty → Unresolved reference: ai)
#
# Use Maven for API:
#   implementation("com.google.ai.edge.litert:litert:2.1.0")
#
# Put Bazel-built libLiteRt.so under:
#   app/src/main/jniLibs/arm64-v8a/
