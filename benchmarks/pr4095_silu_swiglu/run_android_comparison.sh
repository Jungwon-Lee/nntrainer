#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(git rev-parse --show-toplevel)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

BEFORE_REF=${BEFORE_REF:-2c9764aa^}
AFTER_REF=${AFTER_REF:-2c9764aa}
THREADS=${THREADS:-"1 2 4 6 8 10"}
WIDTH=${WIDTH:-11008}
PREFILL_HEIGHT=${PREFILL_HEIGHT:-128}
PREFILL_ITERATIONS=${PREFILL_ITERATIONS:-50}
DECODE_ITERATIONS=${DECODE_ITERATIONS:-500}
SAMPLES=${SAMPLES:-15}
ARM_ARCH=${ARM_ARCH:-armv8.2-a}
ADB_SERIAL=${ADB_SERIAL:-}
DEVICE_DIR=${DEVICE_DIR:-/data/local/tmp/nntr-pr4095-benchmark}
RESULT_DIR=${RESULT_DIR:-"${ROOT_DIR}/benchmark-results/pr4095-silu-swiglu-android"}

if [[ $(uname -s) != Linux ]]; then
  echo "The NNTrainer Android packaging script requires a Linux host." >&2
  exit 1
fi

if [[ -z ${ANDROID_NDK:-} ]]; then
  echo "ANDROID_NDK is not set." >&2
  echo "Example: export ANDROID_NDK=/opt/android-ndk-r26d" >&2
  exit 1
fi
export PATH="${ANDROID_NDK}:${PATH}"

for command in adb git meson ninja ndk-build python3 tar; do
  if ! command -v "${command}" >/dev/null 2>&1; then
    echo "Missing required command: ${command}" >&2
    exit 1
  fi
done

ADB=(adb)
if [[ -n ${ADB_SERIAL} ]]; then
  ADB+=( -s "${ADB_SERIAL}" )
fi

DEVICE_ABI=$("${ADB[@]}" shell getprop ro.product.cpu.abi | tr -d '\r')
if [[ ${DEVICE_ABI} != arm64-v8a ]]; then
  echo "An arm64-v8a device is required; connected device reports ${DEVICE_ABI}." >&2
  exit 1
fi

for submodule in iniparser OpenBLAS ruy; do
  if [[ ! -d "${ROOT_DIR}/subprojects/${submodule}" ]] ||
     [[ -z $(find "${ROOT_DIR}/subprojects/${submodule}" -mindepth 1 -maxdepth 1 -print -quit) ]]; then
    echo "subprojects/${submodule} is not initialized." >&2
    echo "Run: git submodule update --init --depth 1" >&2
    exit 1
  fi
done

WORK_DIR=$(mktemp -d "${TMPDIR:-/tmp}/nntrainer-pr4095-android.XXXXXX")
cleanup() {
  if [[ -n ${WORK_DIR:-} ]] && [[ -d ${WORK_DIR} ]]; then
    rm -rf -- "${WORK_DIR}"
  fi
}
trap cleanup EXIT

snapshot_source() {
  local ref=$1
  local destination=$2

  mkdir -p "${destination}"
  git -C "${ROOT_DIR}" archive "${ref}" | tar -x -C "${destination}"
  mkdir -p "${destination}/subprojects"
  cp -a "${ROOT_DIR}/subprojects/." "${destination}/subprojects/"
}

build_variant() {
  local variant=$1
  local source_dir="${WORK_DIR}/${variant}"
  local jni_dir="${WORK_DIR}/${variant}-jni"
  local package_dir="${source_dir}/builddir/android_build_result"

  echo "Building nntrainer for Android: ${variant}"
  "${source_dir}/tools/package_android.sh" "${source_dir}" \
    "--arm-arch=${ARM_ARCH}"

  mkdir -p "${jni_dir}"
  cp "${SCRIPT_DIR}/Android.mk" "${SCRIPT_DIR}/Application.mk" \
    "${SCRIPT_DIR}/layer_benchmark.cpp" "${jni_dir}/"
  cp "${source_dir}/Applications/CausalLM/layers/swiglu.cpp" \
    "${source_dir}/Applications/CausalLM/layers/swiglu.h" "${jni_dir}/"

  echo "Building benchmark executable: ${variant}"
  ndk-build \
    -C "${jni_dir}" \
    NDK_PROJECT_PATH="${jni_dir}" \
    NDK_OUT="${jni_dir}/obj" \
    NDK_LIBS_OUT="${jni_dir}/libs" \
    APP_BUILD_SCRIPT="${jni_dir}/Android.mk" \
    NDK_APPLICATION_MK="${jni_dir}/Application.mk" \
    NNTRAINER_PACKAGE="${package_dir}" \
    layer_benchmark \
    -j"$(getconf _NPROCESSORS_ONLN)"

  local install_dir="${DEVICE_DIR}/${variant}"
  "${ADB[@]}" shell "mkdir -p '${install_dir}'"
  "${ADB[@]}" push "${jni_dir}/obj/local/arm64-v8a/layer_benchmark" \
    "${install_dir}/" >/dev/null
  "${ADB[@]}" push "${package_dir}/lib/arm64-v8a/libnntrainer.so" \
    "${package_dir}/lib/arm64-v8a/libc++_shared.so" \
    "${install_dir}/" >/dev/null
  "${ADB[@]}" shell "chmod 755 '${install_dir}/layer_benchmark'"
}

run_one() {
  local workload=$1
  local variant=$2
  local thread_count=$3
  local height=$4
  local iterations=$5
  local raw_file=$6
  local meta_file=$7
  local install_dir="${DEVICE_DIR}/${variant}"
  local output

  output=$("${ADB[@]}" shell \
    "cd '${install_dir}' && LD_LIBRARY_PATH='${install_dir}' ./layer_benchmark '${thread_count}' '${height}' '${WIDTH}' '${iterations}' '${SAMPLES}' 2>&1")
  output=${output//$'\r'/}
  printf '%s\n' "${output}" | grep '^effective_threads=' >>"${meta_file}"
  while IFS= read -r row; do
    [[ ${row} =~ ^(SiLU|SwiGLU), ]] || continue
    printf '%s,%s,%s,%s\n' \
      "${workload}" "${variant}" "${thread_count}" "${row}" >>"${raw_file}"
  done <<<"${output}"
}

run_workload() {
  local workload=$1
  local height=$2
  local iterations=$3
  local raw_file=$4
  local meta_file=$5
  local thread_count variant

  for thread_count in ${THREADS}; do
    for variant in before after; do
      run_one "${workload}" "${variant}" "${thread_count}" "${height}" \
        "${iterations}" "${raw_file}" "${meta_file}"
    done
  done
}

mkdir -p "${RESULT_DIR}"
RAW_FILE="${RESULT_DIR}/raw.csv"
META_FILE="${RESULT_DIR}/run_meta.log"

echo "Preparing source snapshots: before=${BEFORE_REF}, after=${AFTER_REF}"
snapshot_source "${BEFORE_REF}" "${WORK_DIR}/before"
snapshot_source "${AFTER_REF}" "${WORK_DIR}/after"
build_variant before
build_variant after

printf 'workload,version,threads,layer,median_us,mean_us,min_us,max_us,checksum\n' \
  >"${RAW_FILE}"
{
  echo "device_serial=$("${ADB[@]}" get-serialno)"
  echo "device_model=$("${ADB[@]}" shell getprop ro.product.model | tr -d '\r')"
  echo "device_abi=${DEVICE_ABI}"
  echo "android_release=$("${ADB[@]}" shell getprop ro.build.version.release | tr -d '\r')"
  echo "before_ref=$(git -C "${ROOT_DIR}" rev-parse "${BEFORE_REF}")"
  echo "after_ref=$(git -C "${ROOT_DIR}" rev-parse "${AFTER_REF}")"
  echo "arm_arch=${ARM_ARCH}"
} >"${META_FILE}"

echo "Running Android decode benchmark"
run_workload "decode_h1" 1 "${DECODE_ITERATIONS}" "${RAW_FILE}" "${META_FILE}"
echo "Running Android prefill benchmark"
run_workload "prefill_h${PREFILL_HEIGHT}" "${PREFILL_HEIGHT}" \
  "${PREFILL_ITERATIONS}" "${RAW_FILE}" "${META_FILE}"

python3 "${SCRIPT_DIR}/summarize_results.py" "${RAW_FILE}" "${RESULT_DIR}"

echo "Results:"
echo "  ${RAW_FILE}"
echo "  ${RESULT_DIR}/summary.csv"
echo "  ${RESULT_DIR}/summary.md"
echo "  ${META_FILE}"
