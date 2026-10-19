---
title: Hexagon HTP Support
...

# Hexagon HTP Support

NNTrainer provides a native Hexagon HTP backend for text generation on
Qualcomm Snapdragon devices. It runs supported CausalLM operations on the
DSP while the ARM host manages model loading, tokenization, weight delivery
and application control. The current integrations cover Gemma4 26B A4B and
LFM2 24B A2B, including their mixture-of-experts (MoE) layers.

This page describes the implementation in
[PR #4428](https://github.com/nntrainer/nntrainer/pull/4428) at
[`89aa8c1f8`](https://github.com/nntrainer/nntrainer/commit/89aa8c1f8b263c30e426980eaf2e17b210d734eb),
as of October 19, 2026. Support below refers to inference with the listed
model packs. Enabling the backend does not automatically make an arbitrary
model or training workload executable on the DSP.

## Native HTP and QNN

Hexagon is Qualcomm's DSP architecture. This backend uses the Hexagon Tensor
Processor (HTP), including HVX vector operations and HMX matrix operations.
NNTrainer controls its own DSP kernels and execution graph through FastRPC
and DSP queues, using the Hexagon SDK and HexKL libraries.

NNTrainer also has a separate QNN integration. Both paths can target
Qualcomm accelerators, but they use different runtimes and model artifacts.

| Path | Build selection | Runtime and model preparation |
|---|---|---|
| Native HTP | `build_android.sh --htp` | Hexagon SDK/toolchain, HexKL and NNTrainer HTP model packs |
| QNN | `build_android.sh --qnn` | Qualcomm QNN runtime and the artifacts required by the selected QNN model |

The Android build script rejects combining `--htp` and `--qnn`. The
QuickDotAI `NPU` backend label alone does not distinguish them; select the
appropriate model catalog entry and build variant. For the QNN path, see
[QNN build instructions](backend_guide/QNN_BUILD.md).

## Supported configurations

### Devices and build environment

The standard native HTP Android build targets **ARM64 and Hexagon v81**.
Device validation includes Galaxy S26 Ultra variants SM-S948N and SM-S948U.
The documented app setup uses Android 16 on a device with 4 KB pages.

The tested tool combination includes Android NDK r30, Hexagon SDK 6.4.0.1,
Hexagon Tools 19.0.04 and HexKL 1.0 beta 2 with its `6.4.0.2` library
directory. The SDK version and the HexKL library directory are separate
settings. Qualcomm SDK binaries and production model weights are supplied
separately from this repository.

There is also a [v79 execution record](gemma4-htp-kv-cache-s25u.md) for the
Galaxy S25 Ultra using a separately rebuilt DSP skeleton. The standard
`build_android.sh --htp` path still requires `HEX_ARCH=v81`; the v79 record
does not establish equivalent app or packaging support.

### Models and weight formats

A model pack contains weights, tokenizer files, configuration and any
required conversion or rotation metadata. The configuration must describe
the actual binary layout. Renaming a dtype or format does not convert the
weights.

| Model pack | Expert weights | Other principal weights | Native CLI | QuickDotAI API |
|---|---|---|---|---|
| Gemma4, original fused pack | `QS2CX_WH` | `QS4CX` projections; `Q4_0` embedding/head | Supported | Accepted |
| Gemma4, native TL2/affine pack | `TL2CX_WH`, with affine expert overrides for mixed packs | `QS4CX_WH_AFFINE` projections/head; `Q4_0` embedding | Supported | Accepted for the configured HTP contract |
| LFM2, `htp_fused_v1` | `TL2CX_WH` | Affine INT4 projections/head; FP32 embedding/router | Supported | Accepted |
| LFM2 SR15, `sr15_v2` | `TL2CX_WH` | Affine INT4 projections/head/router; `Q4_0` embedding; packed short-convolution weights | Supported | Not accepted by the current API loader |

Both models require their explicit `htp_fused` execution profile and run at
batch size one. Their geometry, attention-cache settings and rotation
contracts differ. Use the conversion path for the chosen checkpoint:
[Gemma4 formats and conversion](gemma4-htp-tl2-guide.md#conversion) or
[LFM2 formats and conversion](lfm2-htp-user-guide.md#convert-the-weights).
The Gemma4 rotated converter expects a checkpoint with rotation and
quantization already applied.

### CLI and application interfaces

The native CLI and QuickDotAI call the same CausalLM model implementations.
QuickDotAI adds model catalog selection, load checks and handle lifecycle
management. Its accepted configurations can therefore be narrower than the
underlying model's capabilities.

| Capability | Native CLI | QuickDotAI API |
|---|---|---|
| Text generation | Model pack passed to `nntr_causallm` | Catalog model loaded with `NPU` and `W4A32` |
| LFM2 SR15 loading | Supported | Loader currently requires `htp_fused_v1` |
| KV snapshot save/load | Available through model configuration | Snapshot options are rejected |
| Model ownership | One model per CLI invocation | One active HTP model handle per process |

The API catalog identifiers are `gemma4-26b-a4b-htp` and
`lfm2-24ba2b-htp`. `W4A32` is the API selection for these mixed-format packs;
it does not mean that every stored weight uses four bits. Repeated API
generation requests start independent prompts while retaining loaded
weights. Destroy or unload the active HTP model before loading another one.
See the [API implementation](../Applications/CausalLM/api/quick_dot_ai_api.cpp)
and [runtime ownership](../Applications/CausalLM/api/htp_model_runtime.h).

## Current performance

The following results were recorded on an **SM-S948U / Hexagon v81 on
October 19, 2026**. Each row reports medians from three runs of its own
workload. Input lengths and packs differ, so the rows are not a controlled
comparison between models.

| Model pack and execution settings | Input tokens | Measured decode steps / total output IDs | Prefill tok/s | Decode tok/s |
|---|---:|---:|---:|---:|
| Gemma4 native TL2/affine; C16, preload 128, LFM2 preset inactive | 1,024 | 64 / 65 | 632.10 | 12.99 |
| LFM2 SR15; automatic CLI preset, initial C16, preload 128 | 2,048 | 127 / 128 | 1,157.72 | 17.39 |

`C16` means an initial capacity of 16 resident experts per MoE layer.
Startup preload is a separate setting. The SR15 preset adds one expert
slot per layer after the first decode step and is applied by the CLI only.
It does not change the model pack's prompt or generation budget; the
published workload sets those explicitly.

These rates use inference timers and exclude model initialization. They do
not measure complete time to first token or sustained throughput. The
recorded output IDs matched their nntrainer references; that is distinct
from model-quality evaluation against the original checkpoint. See the
[workload, preset and validation results](lfm2-htp-user-guide.md#validated-2048-token-workload)
for the exact conditions, ranges and pack identifiers. Memory results from
other experiments should not be combined with these rows as one measured
speed-and-memory result.

## Where to go next

| Task | Starting point |
|---|---|
| Build the native runtime or Android app | [SDK setup and Android build](gemma4-htp-quickdotai.md#android-build) |
| Prepare a Gemma4 model pack | [Weight formats and conversion](gemma4-htp-tl2-guide.md) |
| Prepare an LFM2 model pack | [Supported packs and conversion](lfm2-htp-user-guide.md#supported-model-packs) |
| Run from the command line | [Gemma4 CLI](gemma4-htp-user-guide.md#7-run-from-the-command-line), [LFM2 CLI](lfm2-htp-user-guide.md#build-and-run) |
| Integrate an application | [QuickDotAI C API](../Applications/CausalLM/api/README.md), [Android AAR](../Applications/CausalLM/Android/QuickDotAI/README.md) |
| Check correctness and lifecycle behavior | [Host and device test coverage](gemma4-htp-code-guide.md#8-tests-and-what-each-one-proves), [API and app validation](gemma4-htp-quickdotai.md#validation) |
| Understand or extend the backend | [Execution architecture and code map](gemma4-htp-code-guide.md) |

## Operational limits

- The current Android package is not validated for 16 KB-page devices.
  HexKL library alignment and DSP library packaging are described in the
  [packaging limits](gemma4-htp-quickdotai.md#known-limits).
- LFM2 snapshots include convolution state as well as attention KV state.
  Gemma4 snapshot or chunked-prefill output can differ from a single
  prefill because quantized KV calibration depends on the processed rows.
  See [LFM2 state handling](lfm2-htp-user-guide.md#build-and-run) and
  [Gemma4 validation limits](gemma4-htp-tl2-guide.md#current-validation-coverage).
- CPU reference tests and ordinary Android, Windows, Tizen or Yocto builds
  do not establish DSP device support. Native HTP validation requires the
  matching ARM libraries and DSP skeleton on compatible hardware.
