---
title: Running Inference with the Hexagon HTP Backend
...

# Running Inference with the Hexagon HTP Backend

Run native HTP text generation through the `nntr_causallm` CLI, the
QuickDotAI C API, or the QuickDotAI Android AAR. These interfaces use the
same CausalLM model implementations and converted weights. They differ
in model selection, prompt handling and process-level defaults.

Before running inference:

- Build and deploy the matching ARM libraries and v81 DSP skeleton using
  [HTP Build](HTP_BUILD.md).
- Prepare a complete supported pack using
  [HTP Model Preparation](HTP_MODEL_PREPARATION.md).
- Use a validated device configuration from the
  [HTP overview](../hexagon-htp.md#supported-configurations).
- Choose a prompt and generation budget that fit the pack's context
  settings. Keep its weight formats, engines and rotation metadata intact.

## Choose an interface

| Interface | Model selection | Prompt input | Typical use |
|---|---|---|---|
| `nntr_causallm` | Pack directory | Configuration's `chat_input` or `sample_input`, or an optional command-line prompt | A standalone run, debugging or a controlled measurement |
| QuickDotAI C API | Catalog ID, backend, quantization and model base directory | Message array or JSON request through a loaded handle | Native application integration |
| QuickDotAI Android AAR | `LoadModelRequest` with a catalog ID | Kotlin messages and a `StreamSink` | Android application integration |
| SampleTestAPP | Model family and runtime/backend selectors | OpenAI or Chat tab | Manual application checks |

Use the native HTP build for all four paths. `NPU` is the API backend
selection; it does not by itself distinguish native HTP from the separate
QNN implementation. The model ID and build variant must also match.

## Model directories

The CLI takes the directory containing `nntr_config.json` directly.
QuickDotAI takes its parent model base directory and resolves a catalog
subdirectory:

| Model | Catalog ID | Directory below the model base path |
|---|---|---|
| Gemma4 26B A4B | `gemma4-26b-a4b-htp` | `gemma4-26b-a4b-htp-w4a32/` |
| LFM2 24B A2B | `lfm2-24ba2b-htp` | `lfm2-24ba2b-htp-w4a32/` |

Both API entries use `NPU` and `W4A32`. The latter is the API selection for
these mixed-format packs, not a request to convert every weight to four
bits. LFM2 requires the SR15 `sr15_v2` pack. Gemma4 accepts the supported
original QS2/QS4 configuration or native TL2/affine configuration described
in the model preparation guide.

For example, arrange a prepared LFM2 pack under the expected directory
name and copy it from the host:

```bash
adb shell mkdir -p /sdcard/Download/aistudio-mobile/models
adb push /path/to/lfm2-24ba2b-htp-w4a32 \
  /sdcard/Download/aistudio-mobile/models/
```

Use the equivalent Gemma4 directory for that model. The runtime build and
APK do not contain model weights. A path readable by `adb shell` also needs
appropriate file access when opened by an Android application.

## Run the native CLI

After deployment with `build_android.sh --htp --install`, run from the host:

```bash
adb shell /data/local/tmp/Quick.AI/run_causallm.sh \
  /sdcard/Download/aistudio-mobile/models/lfm2-24ba2b-htp-w4a32
```

The wrapper sets the ARM and DSP library paths, selects the build's compute
thread count and forwards arguments to `nntr_causallm`. Replace only the
pack path to run Gemma4.

For a manually deployed runtime, enter an Android shell and set the paths
before invoking the executable:

```sh
runtime_dir=/data/local/tmp/Quick.AI
model_dir=/sdcard/Download/aistudio-mobile/models/lfm2-24ba2b-htp-w4a32
export LD_LIBRARY_PATH="$runtime_dir"
export ADSP_LIBRARY_PATH="$runtime_dir"
export NNTR_NUM_THREADS=8
"$runtime_dir/nntr_causallm" "$model_dir"
```

Each invocation loads the model, runs one prompt, reports results and exits.
This is different from keeping an API model handle loaded across requests.

### Prompt selection and chat templates

The CLI syntax is `nntr_causallm <model_path> [input_prompt]`. It selects
input in this order:

1. If supplied, `input_prompt` is used directly. The CLI does not apply
   the pack's chat template to this argument.
2. Otherwise, `chat_input` in `nntr_config.json` is rendered with the
   available chat template.
3. If `chat_input` is absent, the CLI uses `sample_input`. If a template
   is unavailable for `chat_input`, it warns and falls back to `sample_input`.

For a custom chat prompt, merge this field into the existing
`nntr_config.json`, preserving the rest of the pack configuration:

```json
{
  "chat_input": {
    "messages": [
      {"role": "user", "content": "Explain how a community garden works."}
    ],
    "add_generation_prompt": true
  }
}
```

Run the CLI without its optional prompt argument to use this field. The
template provides model-specific turn markers and the assistant prefix.
For a raw prompt supplied through `sample_input` or the command line,
provide the formatting expected by the checkpoint. Existing
`system_prompt.head_prompt` and `tail_prompt` settings still participate in
the raw-input path; the CLI clears them after successfully rendering
`chat_input`.

### Context and generation settings

| Setting | Role |
|---|---|
| `init_seq_len` | Prefill activation capacity; it also limits an unchunked prompt |
| `max_seq_len` | Total context capacity, including generated tokens |
| `num_to_generate` | Configured decode budget; the first output token selected by prefill is accounted for separately in runtime measurements |
| `generation_config.json` → `do_sample` | Sampling selection; use `false` for greedy repeatability checks |
| EOS and `bad_word_ids` | Determine whether generation can stop at the checkpoint's end-of-turn tokens |

The runtime reserves context space for generation and any loaded KV prefix.
If a prefix is present and the prompt does not fit, it rejects the request
before cache loading and prefill, reporting the prefix size and available
prompt space. Without a prefix, overlong prompts can lose tail tokens and
produce a truncation warning. That can remove the
question or assistant prefix, so check the token budget after applying the
template. LFM2 can process longer prompts in chunks while keeping the
prefill buffers at `init_seq_len`; the total context limit still applies.

Keep normal EOS handling for interactive use. Fixed-length benchmark packs
may deliberately suppress EOS and should not be used unchanged as chat
presets. For TL2 Gemma4, preserve the generated dtype and rotation keys when
adapting chat settings from an older preset.

## Integrate the C API

Include
[`quick_dot_ai_api.h`](../../Applications/CausalLM/api/quick_dot_ai_api.h)
and link `libquick_dot_ai_api.so`, with its runtime dependencies available
to the platform loader. Use `loadModelHandleByName()` with a catalog string
ID. Pass an absolute `native_lib_dir` containing the matching
`libnntr_hvx_skel.so`, and a model base directory from the table above.

This example loads either supported HTP catalog model, streams one message
and releases the handle. Call it on a worker thread:

```c
#include <stdio.h>
#include "quick_dot_ai_api.h"

static int print_delta(const char *delta, void *user_data) {
  (void)user_data;
  fputs(delta, stdout);
  fflush(stdout);
  return 0;
}

ErrorCode generate_htp(const char *model_id, const char *model_base_path,
                       const char *native_lib_dir) {
  CausalLmHandle handle = NULL;
  ErrorCode status = loadModelHandleByName(
    CAUSAL_LM_BACKEND_NPU, model_id, CAUSAL_LM_QUANTIZATION_W4A32,
    native_lib_dir, model_base_path, &handle);
  if (status != CAUSAL_LM_ERROR_NONE) {
    fprintf(stderr, "HTP load failed: %s\n", getLastModelLoadError());
    return status;
  }

  const CausalLMChatMessage message = {
    "user", "Explain how a community garden works."
  };
  status = runModelHandleWithMessagesStreaming(
    handle, &message, 1, true, print_delta, NULL);
  ErrorCode cleanup_status = destroyModelHandle(handle);
  return status != CAUSAL_LM_ERROR_NONE ? status : cleanup_status;
}
```

The message API applies the pack's chat template. Supply plain message
content rather than text that already includes model turn markers.
`runModelHandleWithJsonStreaming()` accepts a JSON request containing a
`messages` array through the same local API; this does not start an HTTP
server.

Streaming calls block until the run finishes, is cancelled, or fails.
Callbacks execute during the call. A callback's `delta` pointer is valid
only during that callback; copy text that must be retained. Keep callbacks
short and dispatch UI work to the UI thread.

### Runtime ownership and cancellation

- **One active HTP model per process:** Unload or destroy the current HTP
  handle before loading another one, including when switching model families.
- **Repeated requests:** Keep the loaded handle to reuse its weights. Each
  run starts an independent prompt and resets attention state; LFM2 also
  resets its convolution state. Provide the desired message history in
  each request if the application needs conversational context.
- **Stop:** Return nonzero from the C callback or call
  `cancelModelHandle()` from another thread. Cancellation takes effect at
  a token boundary; it is not an immediate interruption of a running DSP op.
- **Release:** Cancel first when an early stop is needed, wait for the run
  to return, then destroy the handle. Unload and destruction wait for the
  active run's mutex. Do not destroy a handle inside its streaming callback.
- **Unload versus destroy:** `unloadModelHandle()` releases model resources
  but keeps the handle object. Subsequent inference requires a newly loaded
  handle; destroy the old object when finished with it.

The API fixes the canonical DSP library directory for the process. An
existing `ADSP_LIBRARY_PATH` must exactly match that directory, and a later
load cannot switch it to another location. For standalone API consumers,
leave this variable unset before the first load or set it to that exact
directory. A colon-separated search list accepted by a raw CLI launch can
be rejected by this API check. Changing the runtime library location
requires a new process.

On a load failure, read `getLastModelLoadError()` on the thread that
performed the load. Copy the diagnostic before passing it to another
thread or attempting another load.

## Integrate the Android AAR

Use `NativeQuickDotAI` and the current `LoadModelRequest.modelId` string
field. For example, with an application `context`:

```kotlin
val engine = NativeQuickDotAI(context.applicationContext)
val request = LoadModelRequest(
    modelId = ModelIds.LFM2_24BA2B_HTP,
    backend = BackendType.NPU,
    quantization = QuantizationType.W4A32,
    modelBasePath = "/sdcard/Download/aistudio-mobile/models",
    nativeLibDir = context.applicationInfo.nativeLibraryDir
)
```

Use `ModelIds.GEMMA4_26B_A4B_HTP` for Gemma4. The types above are in
`com.example.quickdotai`. `modelBasePath` is the parent of the model pack;
`nativeLibDir` points to the extracted runtime libraries. The
`htpBackendConfigPath` field belongs to QNN configuration and is not how
this native HTP path selects its runtime.

On a worker thread, call `engine.load(request)` and check its
`BackendResult`. After a successful load, call
`engine.runModelHandleWithMessagesStreaming(messages, sink)`, using
`QuickAiChatMessage` entries containing `PromptPart.Text`. Handle
`StreamSink.onDelta`, `onDone` and `onError`, and check the returned result.
Use `engine.cancel()` to request a stop, and close the engine after the
active call has returned.

Calling `load()` on an already loaded `NativeQuickDotAI` instance returns
success without replacing its model. To change the request, close the old
engine and create/load a new instance. This also releases its native
handle. Keep the package extraction and DSP no-stripping settings from
[HTP Build](HTP_BUILD.md#packaging-in-a-consuming-app).

## Run SampleTestAPP

1. Build and install the HTP APK with `build_android.sh --htp --app --install`
   as described in HTP Build. Stage the model pack separately.
2. Grant the sample app access to the chosen model directory. For the
   shared Download directory, enable its all-files access through Android
   settings or the development command below.
3. Select family `gemma4-26b-a4b` or `lfm2`, runtime `NATIVE`, backend `NPU`
   and quantization `W4A32`.
4. Set **MODEL BASE PATH** to the parent directory, such as
   `/sdcard/Download/aistudio-mobile/models`.
5. Use **Load model** and **Run (streaming)** in the OpenAI tab, or **Open**
   and Run in the Chat tab. Enter plain message content for template-based
   requests.
6. Use **Stop**, **Unload** or **Reload** as needed. Reload replaces the
   handle, and an invalid model path should produce a load diagnostic.

```bash
adb shell appops set com.example.sampletestapp MANAGE_EXTERNAL_STORAGE allow
```

The sample's normal default model base is
`/sdcard/Download/aistudio-mobile/models/`. A model path supplied to an
instrumentation test is not saved for later normal launches.
The HTP Chat flow also starts independent prompts; opening a chat session
does not enable persistent HTP KV history between calls.

## CLI defaults and performance settings

For `lfm2_moe` with `sr15_v2` and `htp_fused`, the standalone CLI installs
the SR15 performance preset before model creation. It sets only environment
variables that are absent. Explicit values, including zero, take precedence.
The log reports when the defaults are active.

The preset selects eight compute threads, direct model-file I/O, no load
prefetch and startup preload 128, together with the selected prefill/decode
optimizations. It grows the expert cache by one slot per layer after the
first decode step. See the
[preset definition](../../Applications/CausalLM/lfm2_runtime_defaults.h)
for the complete list.

For example, in the Android shell using the manual runtime setup above:

```sh
# Keep the initial expert-cache size while retaining the other defaults.
NNTR_MOE_CACHE_GROW=0 "$runtime_dir/nntr_causallm" "$model_dir"

# Skip installation of the automatic SR15 preset.
NNTR_LFM2_DEFAULT_PRESET=0 "$runtime_dir/nntr_causallm" "$model_dir"
```

Disabling the preset does not clear variables already exported in the
shell. Review overrides when reproducing a result. The CLI does not change
the pack's prompt, rotation, context or generation budget.

QuickDotAI and the AAR do not install this CLI preset. Some backend options
are read once per process, so set any intentional overrides before loading
the first model rather than changing them between requests. Identical
weights alone do not establish identical CLI and app performance.

Use `getPerformanceMetricsHandle()` or `engine.metrics()` after a completed
run to inspect inference metrics. Keep model initialization, prefill,
decode and complete request latency distinct. For comparisons, record the
pack, rendered prompt, token budget, settings and device conditions with
the result. [HTP Performance](HTP_PERFORMANCE.md) describes existing
measurements, their scope and a reproducible capture procedure.

## Snapshots and common failures

Native CLI/model snapshot support is separate from public API support.
The QuickDotAI HTP loaders reject enabled `use_kvcache` or `save_kvcache` options.
LFM2 native snapshots require a matching `<snapshot>.conv` companion for
convolution state. Gemma4 snapshot or chunked-prefill output can differ
from a single prefill because quantized KV calibration depends on the
processed rows. The current windowed-cache path preserves the maxima of
dropped rows for decode scale seeding; that fixes a specific window/full
cache mismatch and does not establish parity for every snapshot workflow.
See [HTP Validation](HTP_VALIDATION.md) and the model-specific
[Gemma4 validation limits](../gemma4-htp-tl2-guide.md#current-validation-coverage)
and [LFM2 state handling](../lfm2-htp-user-guide.md#build-and-run).

| Symptom | Check |
|---|---|
| Model files cannot be found | Pass the pack directory to the CLI and its parent base directory to QuickDotAI. Check the catalog subdirectory name and app file access. |
| HTP initialization fails | Check the matching v81 DSP skeleton, ARM dependencies and `native_lib_dir`. Read the load diagnostic on the loading thread. |
| Conflicting runtime option or fixed-directory error | Remove stale `ADSP_LIBRARY_PATH` before first API load, or use the exact canonical directory. Restart the process to change that directory. |
| A second HTP model cannot load | Finish the active request and unload/destroy the existing HTP model first. |
| Unexpected text or truncated instructions | Check the template, tokenizer, rotation metadata, EOS settings and prompt truncation warning. Avoid adding turn markers twice. |
| New pack or settings are ignored by the AAR | Replace the loaded engine; `load()` alone does not reload an already loaded instance. |
| CLI and app throughput differ | Compare the full workload and environment, including the CLI-only SR15 preset. |
