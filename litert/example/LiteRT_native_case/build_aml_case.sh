#!/usr/bin/env bash
# Build aml_compiled_model_case and install one folder per platform:
#
#   out/android-arm64/     # ndk-build, libs from ../compiled_litert/arm64
#   out/android-arm/       # ndk-build, libs from ../compiled_litert/arm
#   out/yocto-aarch64/     # CMake + OE SDK, libs from ../compiled_litert/yocto-aarch64
#   out/linux-aarch64/     # CMake + gcc-arm, libs from ../compiled_litert/linux-aarch64
#
# Each folder contains the binary and the .so files from that platform's
# compiled_litert directory. The case loads the compiler plugin and dispatch
# from the current directory at runtime.
#
# Optional:
#   SKIP_ANDROID_ARM64=1 / SKIP_ANDROID_ARM=1 / SKIP_YOCTO=1 / SKIP_LINUX=1
#   NDK / CMAKE / YOCTO_ENV / AARCH64_TOOLCHAIN_ROOT / TOOLCHAIN_FILE
#
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi
set -euo pipefail

CASE_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "${CASE_DIR}/../.." && pwd)"
LITERT_PREBUILT="${CASE_DIR}/../compiled_litert"
OUT_DIR="${CASE_DIR}/out"
BUILD_DIR="${CASE_DIR}/build"

NDK="${NDK:-/aidata/junyi/100_work_dir/android-ndk-r28c}"
CMAKE="${CMAKE:-/aidata/junyi/100_work_dir/cmake-3.20.6-linux-x86_64/bin/cmake}"
YOCTO_ENV="${YOCTO_ENV:-/aidata/junyi/100_work_dir/kirkstone/64/environment-setup-armv8a-poky-linux}"
AARCH64_TOOLCHAIN_ROOT="${AARCH64_TOOLCHAIN_ROOT:-/aidata/junyi/100_work_dir/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu}"
TOOLCHAIN_FILE="${TOOLCHAIN_FILE:-${REPO}/tflite/tools/cmake/aarch64-linux-gcc10-gnu.cmake}"

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

stage_case() {
  local platform="$1"
  local lib_dir="$2"
  local bin="$3"
  local dest="${OUT_DIR}/${platform}"
  local so
  local sos=()

  [[ -f "${bin}" ]] || die "built case not found: ${bin}"
  [[ -d "${lib_dir}" ]] || die "prebuilt dir not found: ${lib_dir}"

  rm -rf "${dest}"
  mkdir -p "${dest}"
  cp -f "${bin}" "${dest}/aml_compiled_model_case"
  chmod a+x "${dest}/aml_compiled_model_case" || true

  shopt -s nullglob
  sos=("${lib_dir}"/*.so)
  shopt -u nullglob
  [[ ${#sos[@]} -gt 0 ]] || die "no .so under ${lib_dir}"
  for so in "${sos[@]}"; do
    cp -f "${so}" "${dest}/"
  done
  echo "[OK] ${dest}"
  ls -l "${dest}"
}

build_android() {
  local abi="$1"
  local platform="$2"
  local lib_name="$3"
  local lib_dir="${LITERT_PREBUILT}/${lib_name}"
  local ndk_build="${NDK}/ndk-build"

  [[ -x "${ndk_build}" ]] || die "ndk-build not found: ${ndk_build}"
  [[ -f "${lib_dir}/libLiteRt.so" ]] || die "missing ${lib_dir}/libLiteRt.so"

  echo "[INFO] ndk-build ${abi}"
  "${ndk_build}" -j"$(nproc 2>/dev/null || echo 8)" \
    NDK_PROJECT_PATH="${CASE_DIR}" \
    APP_BUILD_SCRIPT="${CASE_DIR}/Android.mk" \
    NDK_APPLICATION_MK="${CASE_DIR}/Application.mk" \
    NDK_OUT="${BUILD_DIR}/ndk/obj" \
    NDK_LIBS_OUT="${BUILD_DIR}/ndk/libs" \
    APP_ABI="${abi}"

  stage_case "${platform}" "${lib_dir}" \
    "${BUILD_DIR}/ndk/libs/${abi}/aml_compiled_model_case"
}

build_cmake() {
  local platform="$1"
  local lib_dir="$2"
  local build_dir="${BUILD_DIR}/${platform}"
  shift 2

  [[ -x "${CMAKE}" ]] || die "cmake not found: ${CMAKE}"
  [[ -f "${lib_dir}/libLiteRt.so" ]] || die "missing ${lib_dir}/libLiteRt.so"

  echo "[INFO] cmake ${platform}"
  "${CMAKE}" -S "${CASE_DIR}" -B "${build_dir}" \
    -DLITERT_LIB_DIR="${lib_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    "$@"
  "${CMAKE}" --build "${build_dir}" -j"$(nproc 2>/dev/null || echo 8)"

  stage_case "${platform}" "${lib_dir}" \
    "${build_dir}/aml_compiled_model_case"
}

build_yocto() {
  local lib_dir="${LITERT_PREBUILT}/yocto-aarch64"
  [[ -f "${YOCTO_ENV}" ]] || die "Yocto environment-setup not found: ${YOCTO_ENV}"
  [[ -f "${lib_dir}/libLiteRt.so" ]] || die "missing ${lib_dir}/libLiteRt.so"
  (
    set +u
    # shellcheck disable=SC1090
    source "${YOCTO_ENV}"
    set -u
    [[ -n "${OE_CMAKE_TOOLCHAIN_FILE:-}" && -f "${OE_CMAKE_TOOLCHAIN_FILE}" ]] ||
      die "OE_CMAKE_TOOLCHAIN_FILE not set after sourcing ${YOCTO_ENV}"
    build_cmake yocto-aarch64 "${lib_dir}" \
      -DCMAKE_TOOLCHAIN_FILE="${OE_CMAKE_TOOLCHAIN_FILE}"
  )
}

build_linux() {
  local lib_dir="${LITERT_PREBUILT}/linux-aarch64"
  [[ -f "${TOOLCHAIN_FILE}" ]] || die "TOOLCHAIN_FILE not found: ${TOOLCHAIN_FILE}"
  export AARCH64_TOOLCHAIN_ROOT
  build_cmake linux-aarch64 "${lib_dir}" \
    -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}" \
    -DAARCH64_TOOLCHAIN_ROOT="${AARCH64_TOOLCHAIN_ROOT}"
}

FAILED=()
PASSED=()

run_platform() {
  local skip_var="$1"
  local name="$2"
  shift 2
  if [[ "${!skip_var:-0}" == "1" ]]; then
    echo "[SKIP] ${name} (${skip_var}=1)"
    return 0
  fi
  if "$@"; then
    PASSED+=("${name}")
  else
    FAILED+=("${name}")
    echo "[WARN] ${name} failed"
  fi
}

echo "===== aml_compiled_model_case ====="
echo "CASE_DIR=${CASE_DIR}"
echo "OUT_DIR=${OUT_DIR}"
echo "==================================="

run_platform SKIP_ANDROID_ARM64 android-arm64 \
  build_android arm64-v8a android-arm64 arm64

run_platform SKIP_ANDROID_ARM android-arm \
  build_android armeabi-v7a android-arm arm

run_platform SKIP_YOCTO yocto-aarch64 build_yocto

run_platform SKIP_LINUX linux-aarch64 build_linux

echo
echo "===== Summary ====="
if [[ ${#PASSED[@]} -gt 0 ]]; then
  echo "OK:   ${PASSED[*]}"
fi
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "FAIL: ${FAILED[*]}"
  exit 1
fi
echo "Output: ${OUT_DIR}"
