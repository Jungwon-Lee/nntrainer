---
title: Preparing Models for the Hexagon HTP Backend
...

# Preparing Models for the Hexagon HTP Backend

The native HTP runtime loads an NNTrainer model pack: a weight binary,
model and execution configuration, tokenizer files, and any rotation
metadata required by the checkpoint. These files describe one model and
must stay together through conversion, deployment and loading.

This guide covers the preparation paths for the supported Gemma4 26B A4B
and LFM2 24B A2B packs. Building the runtime is a separate step; see
[HTP Build](HTP_BUILD.md). See the [support overview](../hexagon-htp.md)
for device support and measured performance.

## Choose a conversion path

Choose by the checkpoint's actual values and metadata, not just its model
name or advertised bit width.

| Available input | Preparation path | Result |
|---|---|---|
| Pre-rotated, fake-quantized Gemma4 BF16 checkpoint with `rotation_meta.json` | [Convert the rotated checkpoint](#convert-a-rotated-gemma4-checkpoint) | Native TL2 experts, optional affine expert layers, affine projections/head, and rotation signs |
| Verified Gemma4 QS2-expert/Q4_0 pack with a complete conversion report, plus its corresponding BF16 checkpoint | [Convert an existing intermediate pack](#convert-an-existing-gemma4-intermediate-pack) | Native TL2 experts and affine projections/head |
| LFM2 SR15 checkpoint, rotation matrices, activation encodings and matching tokenizer | [Convert the SR15 checkpoint](#convert-an-lfm2-sr15-checkpoint) | The `sr15_v2` pack required by both CLI and QuickDotAI |
| Existing Gemma4 QS2-expert/QS4CX-projection HTP pack | Keep its matching configuration and tokenizer | The original supported Gemma4 layout; this is not the QS2/Q4_0 input expected by the TL2 wrapper |

The rotated conversion paths expect quantization and rotation to have
already been applied to the source checkpoint. They package and verify
those values for NNTrainer; they do not train, calibrate or rotate an
arbitrary full-precision checkpoint.

The older
[`gemma4/htp_weight_converter.py`](../../Applications/CausalLM/res/gemma4/htp_weight_converter.py)
exports FP32 weights in the fused record order. Its `--output_name` sets a
filename, not a quantization format. Naming that output `qs2cx` does not
produce the original quantized HTP pack or the verified intermediate
required by the TL2 wrapper.

LFM2 HTP accepts only `sr15_v2`. The former unrotated LFM2
`htp_fused_v1` pack cannot be upgraded by editing its format label.

## Host requirements

- **Python:** 3.11 or newer for the conversion paths below; shared helpers
  use `hashlib.file_digest`.
- **Python packages:** NumPy. These converters read safetensors records
  directly and do not require PyTorch or the Python safetensors package.
- **Native tools:** A host C compiler for the TL2 packing helper, plus a
  C++17 compiler for Gemma4. The rotated Gemma4 entry point invokes
  `gcc-13` and `g++` by name; LFM2 invokes `cc`.
- **Input files:** Local checkpoints with the configuration, tensor names,
  quantization grids and rotation metadata expected by the chosen converter.
- **Storage:** Space for the source and output packs, plus intermediate
  files. Rotated Gemma4 checks for three times its stage-zero pack size
  plus 1 GiB of free space. This is a converter check, not a complete
  source-and-output storage budget.
- **Output location:** A new directory. The converters below refuse an
  existing output directory. For rotated Gemma4, create the parent first
  and leave both the output directory and its `<output>.qs2` sibling absent.

Run commands below from the repository root. Conversion runs on the host;
it does not require a connected phone or the Hexagon SDK.

## What belongs in a model pack

| File | Purpose |
|---|---|
| Binary named by `nntr_config.json` → `model_file_name` | Serialized model weights in the declared record order and formats |
| `config.json` | Model architecture, layer geometry, attention settings and expert counts |
| `nntr_config.json` | NNTrainer execution profile, weight dtypes, engines, file paths and initial inference settings |
| `generation_config.json` | Generation settings and special-token behavior |
| `tokenizer.json`, `tokenizer_config.json` | The matching tokenizer and its metadata |
| `chat_template.jinja` | The checkpoint's chat formatting, when supplied as a separate file |
| `conversion_report.json` | Binary size/hash, tensor layout and conversion checks; also required at runtime for LFM2 rotation records |
| `rotation_signs.json` | Required by the rotated Gemma4 configuration generated below |
| `activation_scales.bin` | LFM2 SR15 conversion sidecar; retained with the pack but unused by the current dynamic-quantization runtime |

Gemma4 converters copy auxiliary files when present. Verify that the source
supplies the tokenizer and generation files needed for the intended CLI or
app flow. The LFM2 converter requires its listed configuration, template
and tokenizer inputs and copies them into the output.

## Convert a rotated Gemma4 checkpoint

Use
[`gemma4-26b-a4b/weight_converter.py`](../../Applications/CausalLM/res/gemma4-26b-a4b/weight_converter.py)
for the matching pre-rotated, fake-quantized BF16 checkpoint.

Required source contents include:

- `config.json`, with `architectures: ["Gemma4ForCausalLM"]`.
- `model.safetensors.index.json` and the referenced BF16 shards.
- `rotation_meta.json`, describing Hadamard mode, both rotations, the
  checkpoint dimensions and the sign-generation seed.
- The matching tokenizer, generation configuration and chat template.

```bash
python3 Applications/CausalLM/res/gemma4-26b-a4b/weight_converter.py \
  --source /path/to/rotated-gemma4-checkpoint \
  --output /path/to/models/gemma4-rotated-htp
```

The converter creates a QS2 intermediate, builds the native TL2 helper,
converts ternary experts to TL2, recovers affine INT4 grids for projections
and the head, and writes the rotation signs. It also fills missing global
attention geometry from the checkpoint's projection shapes. The resulting
binary is `nntr_gemma4_26b_a4b_tl2_rotated.bin`.

By default, `--affine-layers 28,29` stores those zero-based expert layers
as `QS4CX_WH_AFFINE`. Other layers use `TL2CX_WH` when their values are
exactly ternary. The converter also selects affine storage when its layer
inspection detects non-ternary values, and verifies records during writing.
The generated `moe_layer_dtype_per_layer` records the overrides.

`--affine-layers ''` removes the explicit layer list; it does not quantize
non-ternary weights into ternary weights. The stored container affects
expert-cache memory even when the numerical values happen to be ternary.
Keep the converter's dtype assignments with its binary.

The default helper build directory is `/tmp/gemma4_convert_build`.
Use `--build-dir /path/to/new-helper-build` after changing the helper's
source, because the converter reuses an existing helper executable.
`--keep-stage0` retains the `<output>.qs2` intermediate for inspection;
otherwise it is removed after successful conversion.

## Convert an existing Gemma4 intermediate pack

Use
[`convert_gemma4_tl2.py`](../../Applications/CausalLM/res/gemma4/convert_gemma4_tl2.py)
when a verified intermediate already exists. Its input contract is specific:

- `moe_layer_dtype=QS2CX_WH`, with exact ternary expert values.
- `fc_layer_dtype`, `embedding_dtype` and `lmhead_dtype` set to `Q4_0`.
- A complete `conversion_report.json` describing the binary, contiguous
  tensor offsets and sizes, verified expert count and SHA-256.
- The corresponding BF16-dequantized checkpoint with recoverable affine
  INT4 projection/head grids, its safetensors index and referenced shards.

Build the host helper and run the wrapper:

```bash
mkdir -p /tmp/nntr-tl2-tools
cc -O3 -c nntrainer/tensor/htp_backend/hvx/hvx_gemv_u8tl2.c \
  -I nntrainer/tensor/htp_backend/hvx \
  -o /tmp/nntr-tl2-tools/hvx_gemv_u8tl2.o
c++ -O3 -std=c++17 -pthread tools/convert_tl2_model.cpp \
  /tmp/nntr-tl2-tools/hvx_gemv_u8tl2.o \
  -I nntrainer/tensor -I nntrainer/tensor/htp_backend \
  -I nntrainer/tensor/htp_backend/hvx \
  -o /tmp/nntr-tl2-tools/convert_tl2_model
python3 Applications/CausalLM/res/gemma4/convert_gemma4_tl2.py \
  /path/to/verified-qs2-q40-pack \
  --output /path/to/models/gemma4-tl2-affine \
  --converter /tmp/nntr-tl2-tools/convert_tl2_model \
  --hf-source /path/to/matching-bf16-checkpoint
```

The wrapper verifies the source size/hash and tensor plan before conversion.
It preserves the embedding and FP32 records, converts expert codes to
native TL2, and recovers affine projection/head records from `--hf-source`.
The output binary is `nntr_gemma4_ternary_tl2_qs4cx_wh_affine.bin`.

This wrapper does not generate rotation signs or repair missing model
geometry. Use the complete rotated-checkpoint entry point above when those
steps are needed. Do not substitute the original QS2/QS4CX HTP pack for
this wrapper's QS2/Q4_0 intermediate.

## Convert an LFM2 SR15 checkpoint

Use [`convert_lfm2_sr15.py`](../../tools/convert_lfm2_sr15.py) with:

- **Checkpoint directory:** `model.safetensors`, `config.json`,
  `generation_config.json` and `chat_template.jinja`. The current reader
  expects a single safetensors file containing BF16 or F32 records.
- **Rotation matrices in that directory:** `R_in_f32.npy` and
  `R_down_f32.npy`, containing the checkpoint's normalized FP32 transforms.
- **Encoding directory:** `lfm2_24b_w4a8kv8_generation.encodings` and
  `lfm2_24b_w4a8kv8_prefill.encodings`.
- **Tokenizer directory:** `tokenizer.json` and `tokenizer_config.json`
  matching the checkpoint.

```bash
python3 tools/convert_lfm2_sr15.py /path/to/sr15-checkpoint \
  --encodings /path/to/encodings \
  --output /path/to/models/lfm2-sr15-htp \
  --tokenizer-source /path/to/tokenizer \
  --rotation both
```

The converter builds its host TL2 helper, writes and reads back the weight
records, appends compact rotation records to
`LFM2-24B-A2B-Ternary-INT4.bin`, and records their offsets in
`conversion_report.json`. Keep that report beside the binary: the runtime
uses it to locate and validate the rotation signs and basis.

The output uses ternary experts, affine INT4 projections/head/router,
Q4_0 embeddings, and packed `DW3_INT4` short-convolution weights. The
runtime decodes the convolution weights to FP32 at load time.

The converter's initial settings are `init_seq_len=256`,
`max_seq_len=2048` and `num_to_generate=32`. These are smoke-run settings,
not the workload used for published throughput results.

Although the converter exposes other `--rotation` values, the rotated
SR15 checkpoint requires `both` and `moe_hadamard_scope=all` for correct
inference. Its activation-encoding sidecar does not enable static
quantization: keep `activation_quantization=dynamic`.

## Preserve the model contract

The formats below describe different parts of each pack. `TL2CX_WH`
stores ternary expert codes with channel scales. `QS4CX_WH_AFFINE` stores
INT4 codes with scales and offsets; dropping the offsets changes the
weights. `Q4_0` is a separate block-quantized layout used for embeddings.

| Configuration | Gemma4 native TL2/affine | LFM2 SR15 |
|---|---|---|
| Execution profile | `gemma4_execution_profile=htp_fused` | `lfm2_execution_profile=htp_fused` |
| Weight format | `gemma4_weight_format=htp_fused_v1` | `lfm2_weight_format=sr15_v2` |
| Expert dtype | `TL2CX_WH`, with generated per-layer overrides for mixed packs | `TL2CX_WH` |
| Projections and head | `QS4CX_WH_AFFINE` | `QS4CX_WH_AFFINE` |
| Embedding | `Q4_0` | `Q4_0` |
| Router | FP32 | `QS4CX_WH_AFFINE` |
| Attention KV dtype | `q8` | `q8_row` |
| Rotation metadata | `rotation_signs.json` for the rotated pack | Binary rotation records described by `conversion_report.json` |

Preserve model geometry in `config.json`, including Gemma4's local/global
attention dimensions and LFM2's convolution/attention layer layout.
Preserve the generated engine, dtype and rotation settings in
`nntr_config.json`; both paths use batch size one.

Gemma4's rotated pack uses H44-based transforms and sign vectors of lengths
2816 and 704. LFM2 SR15 uses a 2048-wide input transform and an oriented
H12 × H128 down transform. Their metadata is not interchangeable. Keep
rotation off for an unrotated Gemma4 pack. Clear stale
`NNTR_MOE_HADAMARD` and `NNTR_MOE_HADAMARD_SCOPE` environment overrides
when switching packs, because they take precedence over configuration.

Prompt, generation budget and context settings can be adjusted within
the model's supported limits. Keep its tokenizer, special-token IDs and
chat template consistent. In particular, do not replace a generated TL2
`nntr_config.json` wholesale with the older Gemma4 `htp_chat` preset: that
preset declares different weight dtypes. Apply the desired chat settings
while retaining the converted pack's weight and execution contract.

## Verify the pack before deployment

For any of the three conversion paths above, check that the configured
binary matches its report. This reads the whole binary to compute SHA-256:

```bash
python3 - /path/to/converted-pack <<'PY'
import hashlib
import json
from pathlib import Path
import sys

pack = Path(sys.argv[1])
config = json.loads((pack / "nntr_config.json").read_text())
report = json.loads((pack / "conversion_report.json").read_text())
name = config["model_file_name"]
assert name == report["model_file"], "Configuration/report filename mismatch"
binary = pack / name
assert binary.stat().st_size == report["bytes"], "Binary length mismatch"
with binary.open("rb") as stream:
    digest = hashlib.file_digest(stream, "sha256").hexdigest()
assert digest == report["sha256"], "Binary SHA-256 mismatch"
for name in ("config.json", "generation_config.json", "tokenizer.json",
             "tokenizer_config.json"):
    assert (pack / name).is_file(), f"Missing {name}"
if config.get("moe_hadamard_signs"):
    assert (pack / config["moe_hadamard_signs"]).is_file(), "Missing rotation signs"
print("Pack files and binary hash verified:", pack)
PY
```

Review the conversion report as well:

- **Ternary experts:** The converter checks serialized codes and their
  scales against the source representation.
- **Affine weights:** Exact BF16 round-trip recovery and FP32-grid error
  are separate metrics. Values that round back to the source BF16 need
  not equal those rounded values in FP32 arithmetic.
- **Embeddings:** Q4_0 requantization can introduce error. LFM2 reports
  embedding error separately; the Gemma4 wrapper preserves the Q4_0
  embedding already in its intermediate.
- **Rotation:** Signs, basis, normalization and orientation must match
  the checkpoint, in addition to the record dimensions.

File checks and converter read-back checks do not establish model quality
or device correctness. Validate generation with the matching ARM runtime
and DSP skeleton before treating a new pack as validated. Retain source
identifiers, converter revision, conversion report and final inference
configuration so later results can be tied to the same pack.

## Arrange the pack for CLI or QuickDotAI

The native CLI accepts the pack directory directly. QuickDotAI resolves a
catalog model under a model base directory:

| Model ID | Directory below the model base path | API selection |
|---|---|---|
| `gemma4-26b-a4b-htp` | `gemma4-26b-a4b-htp-w4a32/` | `NPU` / `W4A32` |
| `lfm2-24ba2b-htp` | `lfm2-24ba2b-htp-w4a32/` | `NPU` / `W4A32` |

Copy the complete pack into the appropriate directory. `W4A32` is the API
selection for these mixed-format packs; it does not describe every tensor's
stored bit width or perform conversion. The CLI and API use the same model
implementations, while the public loader imposes additional catalog and
configuration checks. Keep CLI snapshot options out of API configurations.

For deployment of runtime binaries, follow [HTP Build](HTP_BUILD.md#deploy-to-a-device).
For model provisioning and loading, continue with
[HTP Inference](HTP_INFERENCE.md). Use
[HTP Validation](HTP_VALIDATION.md) to check a new pack and
[HTP Performance](HTP_PERFORMANCE.md) to interpret or reproduce measurements.
