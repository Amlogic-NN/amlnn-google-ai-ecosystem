LOCAL_PATH := $(call my-dir)

# Paths are relative to this Android.mk. compiled_litert sits next to this folder.
ifeq ($(TARGET_ARCH_ABI),arm64-v8a)
  AML_LITERT_PREBUILT := ../compiled_litert/arm64
else ifeq ($(TARGET_ARCH_ABI),armeabi-v7a)
  AML_LITERT_PREBUILT := ../compiled_litert/arm
else ifeq ($(TARGET_ARCH_ABI),x86)
  AML_LITERT_PREBUILT := ../compiled_litert/x86
else ifeq ($(TARGET_ARCH_ABI),x86_64)
  AML_LITERT_PREBUILT := ../compiled_litert/x86
else
  AML_LITERT_PREBUILT := ../compiled_litert/arm64
endif

include $(CLEAR_VARS)
LOCAL_MODULE := LiteRt
LOCAL_SRC_FILES := $(AML_LITERT_PREBUILT)/libLiteRt.so
LOCAL_EXPORT_C_INCLUDES := $(LOCAL_PATH)/include
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := aml_compiled_model_case
LOCAL_SRC_FILES := aml_case.cc case_runtime_shims.cc
LOCAL_C_INCLUDES := $(LOCAL_PATH)/include
LOCAL_SHARED_LIBRARIES := LiteRt
LOCAL_LDLIBS := -ldl -llog
LOCAL_CPPFLAGS := -std=c++20 -DNDEBUG -DLITERT_NO_ABSL -fexceptions -frtti
include $(BUILD_EXECUTABLE)
