---
title: Hexagon HTP Performance
...

# Hexagon HTP Performance

The latest recorded three-run CLI medians are **632.10 prefill / 12.99
decode tok/s for Gemma4** and **1157.72 prefill / 17.39 decode tok/s for
LFM2 SR15** on Galaxy S26 Ultra SM-S948U. These are different workloads,
recorded on October 19, 2026, with the settings below.

This page consolidates existing measurements. Later source changes,
including the windowed-KV and prefix-budget fixes, do not establish a new
throughput result. For execution instructions, see
[HTP Inference](HTP_INFERENCE.md); for correctness checks, see
[HTP Validation](HTP_VALIDATION.md).

## Recorded workloads

| Condition | Gemma4 | LFM2 |
|---|---|---|
| Model pack | Unrotated native TL2/affine, 26B A4B | SR15 `sr15_v2`, 24B A2B |
| Device / DSP | SM-S948U / Hexagon v81 | SM-S948U / Hexagon v81 |
| Interface | Native CLI | Native CLI |
| Input tokens | 1,024 | 2,048 |
| Maximum context | 2,048 | 4,096 |
| Timed decode steps / total output IDs | 64 / 65 | 127 / 128 |
| Initial expert-cache capacity | 16 per MoE layer (C16) | 16 per MoE layer (C16) |
| Startup preload | 128 | 128 |
| Cache growth | Disabled | One additional slot per layer after the first decode step |
| Runtime settings | Eight threads, direct file I/O, no load prefetch | Automatic SR15 CLI preset |
| Prefill median (range), tok/s | 632.099 (622.871–641.202) | 1157.72 (1151.21–1159.03) |
| Decode median (range), tok/s | 12.9923 (12.9215–13.0240) | 17.3877 (17.2813–17.4259) |

The first output token is selected by prefill; it is included in total
output IDs but not in the listed timed decode-step count. Both workloads
used greedy generation, and all runs reproduced their respective recorded
NNTrainer output IDs. This is runtime regression evidence, not a quality
comparison with the original checkpoints.

The records required five seconds without another inference process before
launch, HAL AP temperature at most 34 °C and skin temperature at most
36.5 °C. Layer and HTP profiling were disabled. These conditions and short
generation lengths do not establish sustained throughput under thermal
load. The original records retain the workload hashes and configuration:
[Gemma4 measurement](../gemma4-htp-user-guide.md) and
[LFM2 workload and automatic-preset runs](../lfm2-htp-user-guide.md#validated-2048-token-workload).

## Matched comparisons

The LFM2 optimization comparison used the same 2,048-token workload on
alternating baseline/candidate runs, with three runs per runtime:

| LFM2 runtime | Prefill median (range), tok/s | Decode median (range), tok/s |
|---|---:|---:|
| Prefill optimization only | 1147.98 (1146.70–1153.80) | 16.303 (16.2967–16.3176) |
| Combined prefill and decode | 1153.15 (1145.41–1157.72) | 17.402 (17.3996–17.4187) |

The matched decode improvement was 6.74%. A separate final-CLI series
verified automatic preset selection and produced the 17.3877 median in
the first table. Do not pool those two series into one six-run statistic.

The paired Gemma4 regression comparison recorded baseline medians of
621.736 prefill / 12.9110 decode tok/s and candidate medians of
632.099 / 12.9923. All six runs matched the same 65 output IDs. This supports
no observed regression for that fixture; it is not a guarantee for every
Gemma4 pack. The LFM2 preset did not activate for Gemma4.

Earlier 1,024-token LFM2 runs, rotated Gemma4 runs, and the 2,048-token
Gemma4 decode optimization experiment used other conditions. Their
numbers remain in the model-specific records and are not baselines for
the table above.

## Understand the metrics

| Metric | Interpretation |
|---|---|
| Prefill tok/s | Input tokens divided by measured prefill duration |
| Decode tok/s | Timed decode tokens divided by measured generation duration |
| Model initialization | Loading, preparation and registration costs; excluded from the throughput rates above |
| Time to first token | Time from a defined request boundary to the first delivered token; requires its own measurement |
| End-to-end latency | The complete operation at the chosen boundary, such as process launch or application request |
| Memory | A separate measurement requiring an explicit accounting method and sampling interval |

Use the API's `getPerformanceMetricsHandle()` or the AAR's `metrics()` for
the last run's fields. Do not call an inference-only rate a complete
first-token latency. Also keep the CLI's process-level elapsed time
separate from application request latency with an already loaded handle.
The public metrics fields are defined in
[`quick_dot_ai_api.h`](../../Applications/CausalLM/api/quick_dot_ai_api.h).

## Memory observations

The following observations come from separate Gemma4 memory experiments,
not the latest throughput series:

| Experiment | Observation |
|---|---|
| C16 staging-related changes | Active-decode sampled median 3173.25 → 3124.42 MiB, a 48.83 MiB reduction |
| Sampled external peak in that experiment | 3752.35 → 3490.64 MiB; sparse sampling does not establish a complete peak |
| Compact TL2 slots at fixed C24 | 128 MiB less allocated DMA; logical payload saving 160.31 MiB |
| Compact C24 versus the old C16 | Active-decode memory was 158.97 MiB higher despite compact storage |

The active-decode statistic used samples from 6–9 seconds into a run. Its
external total was ARM smaps RSS plus unique process-owned DMA-buffer FD
sizes, minus DMA-mapped RSS overlap. DSP heap was logged separately and
was not merged into that total. These are not complete ARM+DMA+DSP budgets.
See the [memory experiment definitions](../gemma4-htp-tl2-guide.md#memory-changes-and-their-measurement-scope).

Cache capacity, speculative slots, arena allocation granularity, model
format and context size all affect memory. In particular, a larger expert
cache can consume more memory even after compact TL2 storage reduces each
slot. LFM2's preset changes capacity during the run; it is not a fixed-C16
memory experiment. Do not combine savings from overlapping lifetimes or
different experiments into a claimed total reduction.

## Reproduce or measure a workload

1. Record the source revision, ARM/DSP binary hashes, SDK/toolchain versions
   and device model. Rebuild both sides after interface changes.
2. Preserve the model binary hash, tokenizer/template, `nntr_config.json`
   and generation configuration. Use a copy when changing a benchmark prompt.
3. Record the rendered input token count, context, decode budget, sampling,
   EOS policy, cache size, preload and all environment overrides.
4. Establish a repeatable launch condition, including temperatures,
   background inference activity and any warm-up procedure.
5. Capture each run's exit status, logs and output IDs. Check correctness
   before including its timing in a comparison.
6. Run repeated measurements and report the median and range. Alternate
   baseline/candidate order when comparing implementations.

To reproduce the recorded LFM2 workload, use its
[exact prompt and configuration recipe](../lfm2-htp-user-guide.md#validated-2048-token-workload).
The converter's 256-token/32-decode smoke preset is a different workload.
QuickDotAI does not install the CLI's SR15 performance preset.

For a single measurement, the following commands run **inside an Android
shell** after the runtime and prepared pack have been deployed. Start
without stale performance overrides; apply the intended preset explicitly
or allow the CLI to select SR15 defaults.

```sh
runtime_dir=/data/local/tmp/Quick.AI
model_dir=/sdcard/Download/aistudio-mobile/models/lfm2-24ba2b-htp-w4a32
result_dir=/data/local/tmp/htp-results/run-1
mkdir -p "$result_dir"
export LD_LIBRARY_PATH="$runtime_dir"
export ADSP_LIBRARY_PATH="$runtime_dir"
export NNTR_GENERATED_IDS="$result_dir/output-ids.txt"
"$runtime_dir/nntr_causallm" "$model_dir" > "$result_dir/run.log" 2>&1
run_status=$?
echo "$run_status" > "$result_dir/exit-status.txt"
```

Choose a new result directory for each run. Pull the directory to the host
and compare its IDs with the matching reference. The capture command does
not impose the thermal conditions or verify that the configuration matches
a published workload.

Use profiling and layer dumps for diagnosis in separate runs. They can
affect scheduling, timing and sometimes output IDs; one SR15 diagnostic
run with `NNTR_HTP_PROFILE=2` was excluded for changing baseline IDs.
Published CLI results do not establish app throughput, corpus perplexity,
long-duration stability or a complete device memory budget.
