# PR #4094 GELU and PR #4095 SiLU / SwiGLU layer benchmarks

This benchmark builds and compares the parent of PR #4095 with the PR commit.
It calls the actual NNTrainer layers directly:

- SiLU: `nntrainer::ActivationLayer::forwarding()`
- SwiGLU: `causallm::SwiGLULayer::incremental_forwarding()`

The PR #4094 runners compare `933679c5^` with `933679c5` and call the actual
FP32 GELU layer through `nntrainer::ActivationLayer::forwarding()`. FP16 GELU
is not compared because PR #4094 introduces its FP16 implementation, so the
parent revision is not a valid performance baseline for that data type.

The default workloads are FP32 decode `(1, 1, 1, 11008)` and prefill
`(1, 1, 128, 11008)`, using 1, 2, 4, 6, 8, and 10 compute threads.

## Linux

### Prerequisites

The script is intended for Linux. On Ubuntu 22.04 or later:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake flatbuffers-compiler git \
  libgmock-dev libgtest-dev libopenblas-dev \
  meson ninja-build pkg-config python3
git submodule update --init --depth 1
```

### Run

From the repository root:

```bash
./benchmarks/pr4095_silu_swiglu/run_comparison.sh
```

For PR #4094 GELU:

```bash
./benchmarks/pr4095_silu_swiglu/run_gelu_comparison.sh
```

GELU results are written to `benchmark-results/pr4094-gelu/`.

Results are written to `benchmark-results/pr4095-silu-swiglu/`:

- `raw.csv`: median, mean, min, and max latency emitted by each process
- `summary.csv`: paired before/after latency and calculated speedup
- `summary.md`: review-ready Markdown tables
- `run_meta.log`: effective thread count and tensor shape for each run

### Configuration

Environment variables can override the defaults:

```bash
THREADS="1 2 4 8" \
WIDTH=4096 \
PREFILL_HEIGHT=256 \
PREFILL_ITERATIONS=100 \
DECODE_ITERATIONS=1000 \
SAMPLES=20 \
RESULT_DIR=/tmp/pr4095-results \
./benchmarks/pr4095_silu_swiglu/run_comparison.sh
```

The compared refs can also be changed:

```bash
BEFORE_REF=2c9764aa^ AFTER_REF=2c9764aa \
  ./benchmarks/pr4095_silu_swiglu/run_comparison.sh
```

The script uses 10 warmup calls per layer and reports the median latency per
call. `OPENBLAS_NUM_THREADS` is forced to 1 so that only NNTrainer's
`ThreadManager` thread count changes.

## Android

The Android runner builds both revisions for `arm64-v8a`, creates a standalone
NDK benchmark executable for each revision, pushes the artifacts with ADB, and
runs both revisions on the same device. It requires Android API 29 or later.

### Prerequisites

- Linux host with the normal NNTrainer Android build dependencies
- Android NDK and `ANDROID_NDK` set to its absolute path
- `adb` in `PATH`
- One unlocked `arm64-v8a` device visible in `adb devices`
- Initialized submodules

```bash
export ANDROID_NDK=/opt/android-ndk-r26d
export PATH="$ANDROID_NDK:$PATH"
git submodule update --init --depth 1
adb devices
```

If more than one device is connected, select one with `ADB_SERIAL`.

### Run

From the repository root:

```bash
./benchmarks/pr4095_silu_swiglu/run_android_comparison.sh
```

For PR #4094 GELU:

```bash
./benchmarks/pr4095_silu_swiglu/run_gelu_android_comparison.sh
```

Android GELU results are written to
`benchmark-results/pr4094-gelu-android/`.

Android results are written to
`benchmark-results/pr4095-silu-swiglu-android/`. The files have the same format
as the Linux results. `run_meta.log` additionally records the device model,
Android version, ABI, exact Git revisions, and effective thread count.

### Configuration

The workload variables accepted by the Linux runner are also accepted by the
Android runner. Android-only variables are `ARM_ARCH`, `ADB_SERIAL`,
`DEVICE_DIR`, and `COOLDOWN_SECONDS`. The runner waits 10 seconds after every
device-side benchmark process by default. Set `COOLDOWN_SECONDS=0` to disable
the wait; decimal values such as `2.5` are accepted.

```bash
ADB_SERIAL=R3CT... \
THREADS="1 2 4 8" \
ARM_ARCH=armv8.2-a \
PREFILL_ITERATIONS=100 \
DECODE_ITERATIONS=1000 \
SAMPLES=20 \
COOLDOWN_SECONDS=15 \
RESULT_DIR=/tmp/pr4095-android-results \
./benchmarks/pr4095_silu_swiglu/run_android_comparison.sh
```

For stable Android measurements, close foreground applications, keep the
device temperature and power mode consistent, and repeat the full run at least
three times. The runner alternates `before` and `after` for every thread count
and applies the configured cooldown after each execution to reduce
time-dependent and thermal bias. It does not change device thermal or CPU
governor settings.

## Result interpretation

`summary.md` reports `speedup = before median / after median`. Values greater
than `1.0x` indicate that PR #4095 is faster; values below `1.0x` indicate a
regression. Compare results only within the same host or Android device and
workload shape.
