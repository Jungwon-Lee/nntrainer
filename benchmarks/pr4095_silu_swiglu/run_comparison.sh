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
RESULT_DIR=${RESULT_DIR:-"${ROOT_DIR}/benchmark-results/pr4095-silu-swiglu"}
CXX=${CXX:-c++}

if [[ $(uname -s) != Linux ]]; then
  echo "This benchmark requires Linux because ThreadManager is not implemented for macOS." >&2
  exit 1
fi

for command in git meson ninja tar python3 "${CXX}"; do
  if ! command -v "${command}" >/dev/null 2>&1; then
    echo "Missing required command: ${command}" >&2
    exit 1
  fi
done

for submodule in iniparser googletest; do
  if [[ ! -d "${ROOT_DIR}/subprojects/${submodule}" ]] ||
     [[ -z $(find "${ROOT_DIR}/subprojects/${submodule}" -mindepth 1 -maxdepth 1 -print -quit) ]]; then
    echo "subprojects/${submodule} is not initialized." >&2
    echo "Run: git submodule update --init --depth 1" >&2
    exit 1
  fi
done

WORK_DIR=$(mktemp -d "${TMPDIR:-/tmp}/nntrainer-pr4095-benchmark.XXXXXX")
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
  for submodule in iniparser googletest; do
    mkdir -p "${destination}/subprojects/${submodule}"
    cp -a "${ROOT_DIR}/subprojects/${submodule}/." \
      "${destination}/subprojects/${submodule}/"
  done
}

build_variant() {
  local variant=$1
  local source_dir="${WORK_DIR}/${variant}"
  local build_dir="${source_dir}/build-benchmark"
  local binary="${WORK_DIR}/layer_benchmark_${variant}"

  meson setup "${build_dir}" "${source_dir}" \
    -Dbuildtype=release \
    -Denable-app=false \
    -Denable-test=false \
    -Denable-transformer=false \
    -Denable-tflite-backbone=false \
    -Denable-tflite-interpreter=false
  # Meson 0.61 can emit this source-relative include path for CMake subprojects.
  mkdir -p "${source_dir}/subprojects/iniparser/__CMake_build"
  ninja -C "${build_dir}" nntrainer/libnntrainer.so

  "${CXX}" -O3 -std=c++17 -march=native \
    -DMIN_CPP_VERSION=201703L \
    -DMMAP_READ=1 \
    -DUSE_MMAP=1 \
    -DML_API_COMMON=0 \
    -DUSE_BLAS=1 \
    -DNNTR_NUM_THREADS=4 \
    -D__LOGGING__=1 \
    -I"${source_dir}/nntrainer" \
    -I"${source_dir}/api" \
    -I"${source_dir}/api/ccapi/include" \
    -I"${source_dir}/nntrainer/compiler" \
    -I"${source_dir}/nntrainer/schema" \
    -I"${source_dir}/nntrainer/dataset" \
    -I"${source_dir}/nntrainer/layers" \
    -I"${source_dir}/nntrainer/layers/loss" \
    -I"${source_dir}/nntrainer/models" \
    -I"${source_dir}/nntrainer/optimizers" \
    -I"${source_dir}/nntrainer/tensor" \
    -I"${source_dir}/nntrainer/tensor/cpu_backend" \
    -I"${source_dir}/nntrainer/tensor/cpu_backend/fallback" \
    -I"${source_dir}/nntrainer/tensor/cpu_backend/arm" \
    -I"${source_dir}/nntrainer/tensor/cpu_backend/x86" \
    -I"${source_dir}/nntrainer/utils" \
    -I"${source_dir}/nntrainer/graph" \
    -I"${source_dir}/Applications/CausalLM/layers" \
    "${SCRIPT_DIR}/layer_benchmark.cpp" \
    "${source_dir}/Applications/CausalLM/layers/swiglu.cpp" \
    -L"${build_dir}/nntrainer" \
    -lnntrainer -lopenblas -lpthread -ldl -lm \
    -Wl,-rpath,"${build_dir}/nntrainer" \
    -o "${binary}"
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
      "${WORK_DIR}/layer_benchmark_${variant}" \
        "${thread_count}" "${height}" "${WIDTH}" "${iterations}" "${SAMPLES}" \
        2>>"${meta_file}" |
        while IFS= read -r row; do
          printf '%s,%s,%s,%s\n' \
            "${workload}" "${variant}" "${thread_count}" "${row}" >>"${raw_file}"
        done
    done
  done
}

mkdir -p "${RESULT_DIR}"
RAW_FILE="${RESULT_DIR}/raw.csv"
META_FILE="${RESULT_DIR}/run_meta.log"

echo "Preparing source snapshots: before=${BEFORE_REF}, after=${AFTER_REF}"
snapshot_source "${BEFORE_REF}" "${WORK_DIR}/before"
snapshot_source "${AFTER_REF}" "${WORK_DIR}/after"

echo "Building before"
build_variant before
echo "Building after"
build_variant after

printf 'workload,version,threads,layer,median_us,mean_us,min_us,max_us,checksum\n' \
  >"${RAW_FILE}"
: >"${META_FILE}"
export OPENBLAS_NUM_THREADS=1

echo "Running decode benchmark"
run_workload "decode_h1" 1 "${DECODE_ITERATIONS}" "${RAW_FILE}" "${META_FILE}"
echo "Running prefill benchmark"
run_workload "prefill_h${PREFILL_HEIGHT}" "${PREFILL_HEIGHT}" \
  "${PREFILL_ITERATIONS}" "${RAW_FILE}" "${META_FILE}"

python3 "${SCRIPT_DIR}/summarize_results.py" "${RAW_FILE}" "${RESULT_DIR}"

echo "Results:"
echo "  ${RAW_FILE}"
echo "  ${RESULT_DIR}/summary.csv"
echo "  ${RESULT_DIR}/summary.md"
