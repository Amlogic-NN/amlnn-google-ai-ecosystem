cmake_minimum_required(VERSION 3.20)

project(AmlCompiledModelCase LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Sibling of this folder: Aml_LiteRT_Out/compiled_litert/<abi>/libLiteRt.so
# Override with -DLITERT_LIB_DIR=... for linux-aarch64, arm64, or arm.
set(LITERT_LIB_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../compiled_litert/yocto-aarch64"
    CACHE PATH "Directory containing libLiteRt.so")

set(LITERT_SHARED_LIB "${LITERT_LIB_DIR}/libLiteRt.so")
if(NOT EXISTS "${LITERT_SHARED_LIB}")
  message(FATAL_ERROR
    "libLiteRt.so not found: ${LITERT_SHARED_LIB}\n"
    "  Pass -DLITERT_LIB_DIR=/path/to/compiled_litert/<abi>")
endif()

add_library(LiteRt SHARED IMPORTED)
set_target_properties(LiteRt PROPERTIES
  IMPORTED_LOCATION "${LITERT_SHARED_LIB}"
  IMPORTED_NO_SONAME TRUE
)

add_executable(aml_compiled_model_case aml_case.cc case_runtime_shims.cc)
# Non-PIC executables COPY-relocate std::cout and iostream vtables. Those
# copies interpose libcompiler_adla.so and crash inside Compile().
set_target_properties(aml_compiled_model_case PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)
target_include_directories(aml_compiled_model_case PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_compile_definitions(aml_compiled_model_case PRIVATE
  NDEBUG
  LITERT_NO_ABSL
)
target_link_libraries(aml_compiled_model_case PRIVATE LiteRt)

if(UNIX)
  target_link_libraries(aml_compiled_model_case PRIVATE dl)
endif()
if(ANDROID)
  target_link_libraries(aml_compiled_model_case PRIVATE log)
endif()
if(UNIX AND NOT APPLE)
  target_compile_options(aml_compiled_model_case PRIVATE -fPIE)
  target_link_options(aml_compiled_model_case PRIVATE
    -pie
    "LINKER:-rpath,$ORIGIN"
    "LINKER:--disable-new-dtags"
  )
endif()
