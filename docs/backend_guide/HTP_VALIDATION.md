---
title: Validating the Hexagon HTP Backend
...

# Validating the Hexagon HTP Backend

Validate native HTP at several levels: serialized weights, host arithmetic
and ownership, ARM/DSP integration, whole-model output, and application
lifecycle. A passing host check covers its fixture; it does not establish
DSP hardware correctness or model quality.

Use [HTP Build](HTP_BUILD.md) for the runtime and package setup,
[HTP Model Preparation](HTP_MODEL_PREPARATION.md) for conversion checks,
and [HTP Inference](HTP_INFERENCE.md) for ordinary execution.

## Select checks for the change

| Change | Relevant checks | Additional evidence needed |
|---|---|---|
| Weight converter or layout | Synthetic serialization, record read-back, size/hash checks | Matching production pack load and output regression |
| HVX/HMX arithmetic | Host reference checks; optional native emulation | Device kernels against independent references |
| Expert cache, prefetch or worker ownership | Routing, read-pool, slot reuse and cancellation fixtures | Repeated device generation and teardown |
| FastRPC or shared graph structures | Regenerate both sides, build ARM and DSP | Matching binaries on the device; malformed/bounds cases |
| Attention/KV state | KV manager, attention, scale and snapshot checks | Full/windowed-cache and continuation comparisons |
| Public API or Android app | Handle and instrumentation lifecycle tests | Repeated prompts, cancellation, reload and failed-load recovery |
| Performance optimization | Correctness checks above | Matched workloads and repeated measurements from [HTP Performance](HTP_PERFORMANCE.md) |

## Host checks

Run from the repository root:

```bash
bash test/htp/host/run_host_checks.sh
```

The script builds fixtures in a temporary directory and stops on failure.
It needs host C/C++ compilers and Python 3. Portions invoke `gcc`/`g++`
directly even when `CC`/`CXX` are set. The ordinary scalar checks need no
phone or Hexagon SDK. The suite can take substantial time; preserve the
complete log and its exit status.

The suite covers expert routing, slot ownership, direct reads, DMA-buffer
reuse, worker pools, graph/token protocols, TL2 packing and conversion,
affine arithmetic, normalization, attention and convolution fixtures.
The scalar stand-ins model operations and ownership; they do not reproduce
all device timing, arithmetic or memory behavior.

Optional paths are reported separately:

- **Qualcomm native emulation:** Selected checks run when
  `$DEFAULT_HEXAGON_TOOLS_ROOT/Tools/libnative/lib/libnative.a` is available.
- **AArch64/NEON emulation:** The abs-max check uses
  `aarch64-linux-gnu-g++` and `qemu-aarch64` when installed.

A skip message is not a pass for that path. Likewise, a sanitizer that
cannot start establishes no sanitizer result.

### Conversion fixtures

The SR15 synthetic fixture exercises a small mixed convolution/attention
model, weight layouts, rotation records and activation-sidecar encoding:

```bash
python3 test/htp/host/lfm2_conversion_check.py
```

Use Python 3.11+, NumPy and a host C compiler. Its small synthetic rotation
geometry checks serialization; it is not a supported production model
configuration for the HTP runtime.

The original Gemma4 FP32 fused exporter has an independent tiny-model
check:

```bash
python3 test/unittest/models/causallm_reference/generators/check_gemma4_htp_converter.py
```

That fixture documents PyTorch 2.14.1, Transformers 5.18.0, NumPy and
safetensors as dependencies. It creates a local random checkpoint and
checks the pinned serializer, logits and greedy tokens. It does not test
the entire production TL2/affine conversion path. For those packs, retain
the converter's read-back report and perform the
[pack integrity check](HTP_MODEL_PREPARATION.md#verify-the-pack-before-deployment).

### Meson model and KV tests

In a host build configured with transformer support, the relevant targets
include `unittest_causallm_models`, `unittest_kv_cache_manager`,
`unittest_hexkl_kv_q`, `unittest_hexkl_attn_q_plan`,
`unittest_hvx_softmax_q` and `unittest_cpu_kv_q_attention`.
List the tests available in that build before selecting them:

```bash
meson test -C build --list
NNTR_ENGINE=cpu meson test -C build unittest_causallm_models unittest_kv_cache_manager
```

`NNTR_ENGINE=cpu` selects the CPU host path when an OpenCL-enabled build
would otherwise choose GPU. The tiny model fixtures cover graph contracts,
reference outputs, malformed inputs and attention dispatch; they do not
measure device throughput. See [Getting Started](../getting-started.md)
for the host build prerequisites.

## Device and whole-model checks

Build the ARM runtime and DSP skeleton from the same checkout and IDL,
using explicit SDK paths and `HEX_ARCH=v81`. Record their hashes with
the result. Package verification checks dependencies and DSP image
preservation, while a device run checks whether that combination executes.

Maintained source fixtures include:

| Fixture | Scope |
|---|---|
| [TL2 device check](../../tools/tl2_device_check.cpp) | Native TL2 kernel behavior on the DSP |
| [Affine device check](../../tools/wh_affine_device_check.cpp) | Affine INT4 arithmetic against reference values |
| [Gemma4 prefill check](../../tools/gemma4_prefill_device_check.cpp) | Model-specific prefill integration |
| [Public API validation](../../test/htp/quick_dot_ai_validation.cpp) | Failed-load recovery, CPU coexistence and LFM2 SR15 acceptance/repeatability |
| [Gemma4 API lifecycle fixture](../../Applications/CausalLM/api-app/htp_lifecycle_test.h) | Load, repeated generation, cancellation and release through `quick_dot_ai_test` |

These standalone tools have their own build and input requirements; they
are not all emitted by `build_android.sh`. Inspect each fixture's usage
and build dependencies before running it. In particular, historical device
scripts can assume older NDK paths and engine output directories; use the
current [build guide](HTP_BUILD.md) as the runtime packaging reference.

For every whole-model regression:

1. Freeze the pack, tokenizer, template, rendered prompt, generation
   settings and runtime environment. Use greedy generation for exact-ID
   comparisons.
2. Capture baseline and candidate output IDs with `NNTR_GENERATED_IDS`.
   `NNTR_HTP_TOKEN_TRACE` additionally writes JSON input/output token IDs.
   Use writable, distinct output files for each run.
3. Check process exit status and logs as well as output equality. An
   interrupted process without a preserved exit status is incomplete evidence.
4. Repeat generation in one loaded API handle, then after cancellation,
   unload/reload and failed-load recovery. Check that prompt state resets.
5. Keep diagnostic/profiling runs separate from ordinary output and
   throughput comparisons.

For example, on the host after pulling two generated-ID files:

```bash
cmp baseline-output-ids.txt candidate-output-ids.txt
sha256sum baseline-output-ids.txt candidate-output-ids.txt
```

Exact IDs establish parity for that fixture and decoding policy. They do
not establish corpus perplexity, equivalence to an external ONNX model,
or task quality across a dataset.

## Android application lifecycle

Build/install the HTP app as in HTP Build. Then build and install the
instrumentation APK from the repository root:

```bash
(
  cd Applications/CausalLM/Android
  ./gradlew "-PnntrainerNdkPath=$ANDROID_NDK" :SampleTestAPP:assembleDebugAndroidTest
)
adb install -r Applications/CausalLM/Android/SampleTestAPP/build/outputs/apk/androidTest/debug/SampleTestAPP-debug-androidTest.apk
```

Set `ANDROID_SERIAL` for the target device. Grant model-file access and
acknowledge first-launch dialogs before instrumentation. Supply a prepared
model base directory and a device-side `messages.json` containing an
OpenAI messages array. Choose a greedy prompt and output budget long enough
to exercise cancellation, such as 96 generated tokens.

```bash
adb shell am instrument -w -r \
  -e class 'com.example.sampletestapp.HtpAppLifecycleTest#loadGenerateStopRepeatReloadAndReportInvalidPath' \
  -e htpModelId lfm2-24ba2b-htp \
  -e htpModelBase /sdcard/Download/aistudio-mobile/models \
  -e htpMessages /sdcard/Download/lfm2/messages.json \
  com.example.sampletestapp.test/androidx.test.runner.AndroidJUnitRunner
```

Use `gemma4-26b-a4b-htp` and its matching messages for Gemma4. The test
checks repeatability, stop, unload during generation, explicit reload and
recovery from an invalid path. It writes reports and traces beneath the
app's `files/htp-app-validation` directory. Read the instrumentation result
for failures and skipped tests; process completion alone is insufficient.

The LFM2 normal-launch path has a separate opt-in test:

```bash
adb shell am instrument -w -r \
  -e class 'com.example.sampletestapp.HtpAppLifecycleTest#lfm2ChatOpensFromDefaultPathAfterActivityRestart' \
  -e htpDefaultPathCheck true \
  com.example.sampletestapp.test/androidx.test.runner.AndroidJUnitRunner
```

It requires the model at the normal default base path
`/sdcard/Download/aistudio-mobile/models/`, compares Chat and OpenAI output,
and repeats after Activity restart. Reports go under
`files/htp-default-path-validation`. The separate Gemma4 chat-format test
and its inputs are described in the
[app validation record](../gemma4-htp-quickdotai.md#validation).

## Recorded coverage and remaining limits

The October 19 common-runtime record reports Android ARM64 and v81 builds,
host arithmetic/ownership checks, LFM2 repeated generation and app lifecycle
coverage, and paired Gemma4 output regression. The
[LFM2 validation record](../lfm2-htp-user-guide.md#validation-and-limits)
and [Gemma4 record](../gemma4-htp-tl2-guide.md#current-validation-coverage)
describe the conditions. These are recorded results, not tests newly run
by publishing this guide.

The later windowed-KV fix adds `check_amax_floor` in
[`attn_m1_host_check.c`](../../test/htp/host/attn_m1_host_check.c) and
`compact_folds_the_dropped_rows_amax` in
[`unittest_kv_cache_manager.cpp`](../../test/unittest/layers/unittest_kv_cache_manager.cpp).
The device record reports matching Gemma4 IDs for context sizes 2304,
4096 and 30272 for its 2048-token fixture. Loaded-prefix prompt overflow
is now rejected before cache loading and prefill. See the
[windowed-cache investigation](../gemma4-htp-kv-cache-s25u.md) for that
specific validation; it does not prove general snapshot/full-prefill parity.

Keep these limits visible when reporting results:

- Host emulation and successful compilation do not replace device tests.
- Prior TSan execution was blocked by a host runtime mapping error;
  optional NEON checks were unavailable in the recorded environment.
- The rotated Gemma4 record includes an unexplained intermittent expert
  rebind failure that did not recur in eleven repeats.
- Current app validation uses Android 16 with 4 KB pages; it does not
  certify 16 KB-page devices.
- Full platform CI, sustained stability and corpus-level quality require
  their own evidence.

Preserve a result bundle with revisions, binary/pack hashes, configuration,
commands, environment, logs, exit status, output IDs and all skips. Some
historical raw logs remain outside the repository; summaries in the
model-specific records should not be presented as a downloadable raw dataset.
