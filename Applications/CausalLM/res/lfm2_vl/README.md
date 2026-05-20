# LFM2-VL Validation Workflow

This directory contains helper scripts for implementing LFM2.5-VL support
incrementally. The rule for this port is to compare each stage against the
original Hugging Face model before moving to the next stage.

## Stages

1. Processor output
   - `input_ids`
   - `attention_mask`
   - `pixel_values`
   - `spatial_shapes`
   - `pixel_attention_mask`
2. SigLIP2 NaFlex vision tower
   - `vision_last_hidden_state`
3. Multimodal projector
   - `image_features`
4. Image-token merge
   - `inputs_embeds`
   - `inputs_embeds_after_image_merge`
5. Language prefill
   - `prefill_logits`

## Reference Dump

Use a local model directory when possible. Passing a Hugging Face model id also
works if the environment can access the network.

```bash
/Users/jwon/miniconda3/bin/conda run -n nntrainer python \
  Applications/CausalLM/res/lfm2_vl/dump_lfm2_vl_reference.py \
  --model LiquidAI/LFM2.5-VL-450M \
  --output /tmp/lfm2_vl_reference.npz
```

For repeatable processor checks without an external image file, omit
`--image`; the script generates a deterministic RGB test image.

## Compare

Once nntrainer emits matching `.npz` files for a stage, compare them with:

```bash
/Users/jwon/miniconda3/bin/conda run -n nntrainer python \
  Applications/CausalLM/res/lfm2_vl/compare_lfm2_vl_npz.py \
  /tmp/lfm2_vl_reference.npz /tmp/lfm2_vl_nntrainer.npz \
  --keys pixel_values spatial_shapes pixel_attention_mask
```

## Stage 1 Candidate

`dump_lfm2_vl_processor_candidate.py` is the local processor implementation used
before the C++ inference path is wired up. It intentionally avoids
`Lfm2VlProcessor` and `Lfm2VlImageProcessor`, while still using the model
tokenizer to match token ids.

```bash
/Users/jwon/miniconda3/bin/conda run -n nntrainer python \
  Applications/CausalLM/res/lfm2_vl/dump_lfm2_vl_processor_candidate.py \
  --model LiquidAI/LFM2.5-VL-450M \
  --output /tmp/lfm2_vl_processor_candidate.npz

/Users/jwon/miniconda3/bin/conda run -n nntrainer python \
  Applications/CausalLM/res/lfm2_vl/compare_lfm2_vl_npz.py \
  /tmp/lfm2_vl_reference.npz /tmp/lfm2_vl_processor_candidate.npz \
  --keys input_ids attention_mask pixel_values spatial_shapes pixel_attention_mask \
  --rtol 1e-6 --atol 1e-6
```

## Stage 2 Vision Tower

`Siglip2NaFlexVision` can run only the SigLIP2 NaFlex tower, or the tower plus
the LFM2-VL multimodal projector when `include_projector=true` is set in
`nntr_config.json`.

```bash
/Users/jwon/miniconda3/bin/conda run -n nntrainer python \
  Applications/CausalLM/res/lfm2_vl/convert_lfm2_vl_vision_weights.py \
  --model /path/to/LFM2.5-VL-450M \
  --output /tmp/lfm2_vl_siglip2_vision_model/nntr_siglip2_vision_fp32.bin

build-x86/Applications/CausalLM/nntr_causallm \
  /tmp/lfm2_vl_siglip2_vision_model

/Users/jwon/miniconda3/bin/conda run -n nntrainer python \
  Applications/CausalLM/res/lfm2_vl/pack_siglip2_vision_output.py \
  --input-dir /tmp/lfm2_vl_siglip2_inputs \
  --output /tmp/lfm2_vl_siglip2_vision_nntr.npz
```

Current validation on the deterministic test image:
`vision_last_hidden_state` has `max_abs=5.7029724e-4` against the HF FP32
reference. The largest differences come from accumulated CPU math differences
through 12 transformer layers.

## Stage 3 Projector

Use `--include-projector` to append `multi_modal_projector` weights after the
vision tower weights. The runner emits `image_features.f32`.

```bash
/Users/jwon/miniconda3/bin/conda run -n nntrainer python \
  Applications/CausalLM/res/lfm2_vl/convert_lfm2_vl_vision_weights.py \
  --model /path/to/LFM2.5-VL-450M \
  --output /tmp/lfm2_vl_siglip2_vision_model/nntr_siglip2_vision_fp32.bin \
  --include-projector

/Users/jwon/miniconda3/bin/conda run -n nntrainer python \
  Applications/CausalLM/res/lfm2_vl/pack_lfm2_vl_projector_output.py \
  --input-dir /tmp/lfm2_vl_siglip2_inputs \
  --output /tmp/lfm2_vl_projector_nntr.npz
```

`analyze_lfm2_vl_projector_error.py` separates projector error from incoming
vision-tower error. Current validation shows the C++ projector itself at
`max_abs=5.7983398e-4` versus HF projector on the same nntrainer vision input;
the end-to-end `image_features` difference is `max_abs=9.0179443e-3` because
the vision output difference is amplified by projector weights.

## Stage 4 Image Merge

`Lfm2VlEmbeddingMerge` validates token embedding lookup and image-token
replacement independently. It expects `input_ids.f32` and padded
`image_features.f32` in the input directory.

```bash
/Users/jwon/miniconda3/bin/conda run -n nntrainer python \
  Applications/CausalLM/res/lfm2_vl/convert_lfm2_vl_embedding_merge_weights.py \
  --model /path/to/LFM2.5-VL-450M \
  --output /tmp/lfm2_vl_embedding_merge_model/nntr_lfm2_vl_embedding_merge_fp32.bin

build-x86/Applications/CausalLM/nntr_causallm \
  /tmp/lfm2_vl_embedding_merge_model

/Users/jwon/miniconda3/bin/conda run -n nntrainer python \
  Applications/CausalLM/res/lfm2_vl/pack_lfm2_vl_embedding_merge_output.py \
  --input-dir /tmp/lfm2_vl_siglip2_inputs \
  --output /tmp/lfm2_vl_embedding_merge_nntr.npz \
  --sequence-length 115
```

Current validation with HF reference `image_features` is exact:
`inputs_embeds_after_image_merge` has `max_abs=0`.

## Stage 5 Language Prefill

`Lfm2VlPrefill` validates the LFM2 language blocks from already-merged
multimodal embeddings. It reads `inputs_embeds_after_image_merge.f32` from the
input directory and emits `prefill_logits.f32`.

```bash
/Users/jwon/miniconda3/bin/conda run -n nntrainer python \
  Applications/CausalLM/res/lfm2_vl/convert_lfm2_vl_prefill_weights.py \
  --model /path/to/LFM2.5-VL-450M \
  --output /tmp/lfm2_vl_prefill_model/nntr_lfm2_vl_prefill_fp32.bin

build-x86/Applications/CausalLM/nntr_causallm \
  /tmp/lfm2_vl_prefill_model

/Users/jwon/miniconda3/bin/conda run -n nntrainer python \
  Applications/CausalLM/res/lfm2_vl/pack_lfm2_vl_prefill_output.py \
  --input-dir /tmp/lfm2_vl_siglip2_inputs \
  --output /tmp/lfm2_vl_prefill_nntr.npz
```

For layer-by-layer debugging, set `output_mode` to `hidden`, `operator`, or
`norm` and set `num_validate_layers` in `nntr_config.json`. HF language hidden
states can be dumped with `dump_lfm2_vl_language_reference.py`.

Current validation with exact HF `inputs_embeds_after_image_merge`:

- `hidden_01`: `max_abs=1.5258789e-5`
- `hidden_02`: `max_abs=1.5258789e-5`
- `hidden_03`: `max_abs=1.6903877e-4`
- `prefill_logits`: `max_abs=4.887104e-3`, passing `rtol=5e-3, atol=5e-3`

Current staged end-to-end validation
(`Siglip2NaFlexVision --include-projector` -> `Lfm2VlEmbeddingMerge` ->
`Lfm2VlPrefill`) has `prefill_logits max_abs=5.0024986e-3`, passing
`rtol=1e-2, atol=1e-2`.

## End-to-End Wrapper

`Lfm2VlForConditionalGeneration` runs the three validated stages in sequence
inside one app invocation. Use `prepare_lfm2_vl_e2e_model.py` to create a model
directory with the wrapper config and all three nntrainer weight files.

```bash
/Users/jwon/miniconda3/bin/conda run -n nntrainer python \
  Applications/CausalLM/res/lfm2_vl/prepare_lfm2_vl_e2e_model.py \
  --model /path/to/LFM2.5-VL-450M \
  --output-dir /tmp/lfm2_vl_e2e_model \
  --sample-input /tmp/lfm2_vl_siglip2_inputs \
  --sequence-length 115

build-x86/Applications/CausalLM/nntr_causallm \
  /tmp/lfm2_vl_e2e_model
```

Current wrapper validation matches the staged run:
`prefill_logits max_abs=5.0024986e-3`, with identical top-5 token ids against
the HF FP32 reference.

## Greedy Text Generation

Set `num_to_generate` to a positive value and make `max_seq_len` at least
`init_seq_len + num_to_generate`. `Lfm2VlPrefill` uses
`decode_embedding_file` to look up generated-token embeddings during decode;
`prepare_lfm2_vl_e2e_model.py --num-to-generate N` fills this field
automatically.

```bash
/Users/jwon/miniconda3/bin/conda run -n nntrainer python \
  Applications/CausalLM/res/lfm2_vl/decode_lfm2_vl_generated_ids.py \
  --model /path/to/LFM2.5-VL-450M \
  --input-dir /tmp/lfm2_vl_siglip2_inputs
```

Current 16-token greedy generation test:

- generated ids match HF exactly:
  `[542, 35381, 521, 30700, 19450, 4810, 916, 46986, 48264, 803, 3460, 521, 16942, 521, 9897, 521]`
- decoded text:
  `A colorful, diamond-shaped pattern with alternating stripes of red, orange, yellow,`
