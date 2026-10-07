# nntr_trace

Timeline profiling for nntrainer runs that mix CPU, Hexagon HTP (HMX / HVX /
DMA) and QNN layers.

**Start with [`GUIDE.md`](GUIDE.md)** (Korean): branches, a five-minute demo
without a device, capturing on a phone, reading the numbers, sharing a
report, troubleshooting. The design and the phased plan are in
[`docs/backend_guide/HTP_TRACE_PROFILER.md`](../../docs/backend_guide/HTP_TRACE_PROFILER.md);
the viewer's own roadmap is [`VIEWER_PLAN.md`](VIEWER_PLAN.md). This directory
holds the viewer, the converters for the logs HTP builds already print,
samples, and tests; the runtime recorder that writes a per-call trace is
`nntrainer/tensor/htp_backend/htp_trace.{h,cpp}` (see below).

| file | what it does |
|---|---|
| `viewer.html` | Standalone timeline viewer in the chrome://tracing idiom: no build, no server, no external scripts. Open it in a browser and press `Load` to pick a `trace.json`. |
| `bundle.py` | Bakes one or more traces into the viewer so a single HTML file carries the data (mail, chat, an artifact link). |
| `make_sample_trace.py` | Emits a synthetic `trace.json` in the target schema, durations scaled from the numbers measured in the HTP branch docs. `--pipelined` projects the doc-35 T2 overlap; `--with-warnings` adds a CPU fallback, dropped records and clock violations; `--tokens N` sets the decode length. |
| `stage_us_to_trace.py` | Converts `FC_STAGE` / `ATTN_STAGE` marker logs from `unittest_hvx_fc` / `unittest_hvx_attn` into the schema, stage totals laid out back to back. |
| `summarize.py` | The viewer's metrics from the command line (`--range`, `--fail-if` gates), held to the viewer's numbers by a test. |
| `qnn_optrace_to_nntr.py` | Rebases a QAIRT HTP optrace (`*_chromeTrace_opTrace.json`) onto the schema as pid 3 so it opens beside ours. Written against a fixture: its docstring lists the field assumptions to verify on a real file. |
| `test/run.sh` | Generates the samples, bundles them, runs `test/check.js` (headless Chromium) and the python unit tests. |

```bash
python3 tools/nntr_trace/make_sample_trace.py            # trace_asbuilt.json
python3 tools/nntr_trace/make_sample_trace.py --pipelined
python3 tools/nntr_trace/stage_us_to_trace.py /tmp/hvx_fc_device_run.log -o fc.json
python3 tools/nntr_trace/bundle.py -o report.html --trace "run 1=trace_asbuilt.json"
# then open tools/nntr_trace/viewer.html (Load) or report.html,
# or drop the json on https://ui.perfetto.dev
bash tools/nntr_trace/test/run.sh                         # before committing
```

## Capturing from a device run

Two ways, both ending in the same viewer.

**A. No rebuild.** Any HTP build already prints a per-shape summary at exit;
that becomes one representative call per shape (averages, no timeline):

```bash
adb shell "cd /data/local/tmp/nntrainer/causallm && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
  NNTR_NUM_THREADS=8 NNTR_HTP_PROFILE=2 ./nntrainer_causallm ./models/<model>" 2>&1 | tee run.log
python3 tools/nntr_trace/htp_profile_to_trace.py run.log -o profile.json
python3 tools/nntr_trace/bundle.py -o report.html --trace "run=profile.json"
```

**B. Per-call timeline.** Needs a build with the `HtpTrace` recorder
(`nntrainer/tensor/htp_backend/htp_trace.{h,cpp}`); build it with
`Applications/CausalLM/build_android.sh --htp` (not `--cache`).
`NNTR_TRACE` names the output and implies profile level 2 so the timed
FastRPC entries run:

```bash
adb shell "cd /data/local/tmp/nntrainer/causallm && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
  NNTR_NUM_THREADS=8 NNTR_TRACE=/data/local/tmp/trace.json ./nntrainer_causallm ./models/<model>"
adb pull /data/local/tmp/trace.json
python3 tools/nntr_trace/bundle.py -o run.html --trace "run=trace.json"
```

Every FastRPC call gets a host wait span, its seam halves and a DSP entry
holding that call's stage totals on their lanes, with prefill and each
decode token marked as phases. Stage totals are laid out sequentially
inside a call, which the metadata states; the one measured overlap is the
MoE layer kernel's SwiGLU worker time, drawn under its HMX span. General
lane overlap needs DSP timestamps; use the bounded capture below for MoE or conv.

### Measured parallel MoE / conv capture

Build **both** the DSP skel (`test/htp/build.sh`) and ARM client/library
(`generate_stub.sh`, then your Android build) from the same IDL revision.
`build.sh` supports the flat SDK addon layout and the versioned beta layout.
For the older two-argument HexKL initializer, set `NNTR_HEXKL_HW_INIT_2ARG=1`.

Capture one matching production call, without stage probes or repetition:

```bash
# Add to the model run's environment (an absolute writable device path):
export NNTR_DSP_LANE_TRACE=/data/local/tmp/model-m512.bin
export NNTR_DSP_LANE_KIND=moe # default moe; conv selects the fused conv block
export NNTR_DSP_LANE_M=512   # default 1; exact M match
export NNTR_DSP_LANE_SKIP=1  # default 0; skip matching calls, e.g. a warmup
# Run the model here, then pull/convert on the host:

adb pull /data/local/tmp/model-m512.bin
python3 tools/nntr_trace/lane_trace_to_trace.py model-m512.bin \
  -o model-m512.json --metrics model-m512-metrics.json
python3 tools/nntr_trace/bundle.py -o lanes.html --trace model=model-m512.json
```

The selected call is excluded from the aggregate `NNTR_HTP_PROFILE` summary,
since its stage probes are disabled. Extraction and file I/O happen after
the call timer stops; end-to-end inference time still includes extraction.
Capture prefill and decode in separate runs. Selection is one-shot per process
and serialized by the existing invoke mutex. Unset the variable for normal use.
An old skel without the appended trace methods produces an explicit RPC error.

For conv, enable `"conv_block_engine": "htp"` in `nntr_config.json` and set
`NNTR_DSP_LANE_KIND=conv`. The current model path offloads prefill (`M>1`)
with Q4_0 projections; decode remains on the CPU. `M` must match the actual
prefill row count. At `M=512`, `SKIP=1` skips the load-time conv warmup;
other kinds do not consume the skip count. Rebuild both the library and
DSP skel for the new selection and conv event names.

Conv tracks distinguish HMX a/c, b and out_proj batches, a/c dequant +
multiply, b/out dequant, and the fused conv/gate/requant/pack stage. Stage
units own four rows by default, with four-row-aligned AH pack stores.
Short units let workers return sooner to foreground dequant when a stage
callback overlaps its publication; the worker pool cannot interrupt a
callback already running. Device A/B timing is needed to verify the tradeoff
between foreground response time and the extra unit scheduling overhead.
`HEXKL_CONV_STAGE_UNIT_ROWS=4`, `8` or `16` when building the skel allows
device comparisons of scheduling overhead and foreground response time.
Each unit completes its own gate before scanning and packing those rows;
there is no block-wide gate barrier or single-worker requant job.
The out_proj waits for all units of its input block before reading it.

The isolated conv check builds and saves output **and state** for M=1, 3,
63, 65, 150 and 512, with four distinct projection weights:

```bash
ANDROID_NDK=/path/to/ndk HEXAGON_SDK_ROOT=/path/to/sdk \
  bash test/htp/build_conv_block_bench.sh
# Deploy conv_block_bench and a matching skel in an isolated directory:
ADSP_LIBRARY_PATH=. ./conv_block_bench baseline 40
ADSP_LIBRARY_PATH=. ./conv_block_bench optimized 40 lane
```

Use the baseline skel for the first command and the optimized skel for the
second. Compare each corresponding `*-m<M>.f32` file byte for byte; these
contain the output followed by the two-row state. The `lane` run also
saves `*-m<M>.bin` captures for `lane_trace_to_trace.py`. Timing samples
exclude capture and extraction and have stage probes enabled on both
sides. M=1 tests the kernel's empty-history edge, not stateful decode.
The current FP32 HVX kernels build for v79; targeting v75 with this SDK
fails instruction selection for the existing IEEE vector add/multiply
intrinsics and requires a separate arithmetic implementation.

All timestamps share the DSP qtimer; they are **not** aligned with the host
clock. HVX tracks use physical worker IDs, not callback slice/unit indices.
Lane slices display the operation (FC matmul gate/up/down or conv projection,
dequant, quant/requant, pack, and fused callbacks). Hover or click for the
original event and job/block details; fused operations retain one measured
window. `Color: op` assigns colors by operation. For labeled SVG exports:

```bash
python3 tools/nntr_trace/render_lane_timeline.py model-m512.json -o timeline.svg
python3 tools/nntr_trace/render_lane_timeline.py model-m512.json \
  --start-us 0 --end-us 1000 -o timeline-detail.svg
```

HMX spans include the synchronous batch API and accumulator read. DMA spans
are issue-to-observed-completion **bounds**, with the last observed incomplete
timestamp retained; they do not prove hardware DMA activity. The converter
rejects malformed clocks and exits nonzero for dropped records/unfinished DMA.

Use `exposed_wait_us`, job publish-to-start bounds, worker finish tails, and
`longest_hmx_no_api_windows` to find dependencies delaying the caller.
Background waits subtract time the caller spends helping HVX work.
`callback_threads.no_callback_us` means no instrumented callback, **not**
hardware idle or scheduler sleep. Publish-to-start also includes publication,
wake costs and (for background jobs) dependency gating, not just scheduling.
These measurements identify exposed stalls, not a full dependency-DAG critical
path or PMU utilization. Generic stage-total idle/compression metrics are
intentionally replaced in the viewer's Wall time pane and CLI summary.

For a model-independent output/overhead check, build the isolated client:

```bash
ANDROID_NDK=/path/to/ndk HEXAGON_SDK_ROOT=/path/to/sdk \
  bash test/htp/build_lane_trace_bench.sh
# Deploy the bench and matching skel to a separate device directory, then:
ADSP_LIBRARY_PATH=. ./lane_trace_bench lane 40
ADSP_LIBRARY_PATH=. ./lane_trace_bench timed-lane 40 timed
```

This uses synthetic distinct weights, K=2048/I=1792, top-4/32 experts,
M=1 and 512. It alternates ON/OFF order and verifies identical finite outputs.
Default results time production RPCs; `timed` additionally compares DSP totals
with the existing stage probes enabled on **both** sides. Neither is a model
throughput claim. The trace buffer is bounded (4096 records per physical
thread, eight threads); inspect capture completeness before drawing conclusions.
Captures longer than the 32-bit qtimer-relative range (~224 seconds) are
marked incomplete. Use a single process/session during capture, as the
underlying DMA ring and existing stage probes are also DSP-global.

## Viewer

- **Timeline.** One process per pid (host, HTP, later QNN), one row per
  thread, nested slices, flow arrows from the FastRPC seam to the DSP entry.
  Every slice label is `op: name`; `Color:` switches between op, name and
  engine. Under each process header a **units busy** strip shows how many
  lanes (HMX, HVX-i, DMA; or CPU threads) are busy at each instant, which is
  parallel compression unrolled over time. Counters (VTCM) draw their
  `metadata.budgets` line and hi-water mark.
- **Navigation.** W/S zoom, A/D pan, `0` or double-click fit, drag to pan,
  wheel to zoom at the cursor. The overview strip under the toolbar shows
  DSP and host busy density with the viewport; drag it, click to center,
  wheel to zoom. Click a process header to collapse it, double-click a
  thread label to fold that thread to one row. Find: Enter / Shift+Enter
  walk the matches in time order (`k / n` beside the box), `f` fits the
  view to the current match, Esc clears. The view, selection, tab, range,
  color mode and filter live in the URL hash, so a zoomed-in finding can be
  shared as a link to the same bundled page.
- **Compare.** `Compare:` loads a second trace (an embedded one or a file)
  as B under A; `View:` shows A, B or both. The Compare tab puts the
  wall-time buckets, parallel compression, idle and transport share side
  by side with deltas, joins the kernels on `engine + op: name` (calls,
  total, mean, cy/elem, Δ), and the layers on name + phase. A converted QNN
  optrace loads the same way (pid 3); the Kernels tab's `QNN ref` column
  shows ref_16's cycles/element for each kernel class.
- **Range.** Drag in the ruler (or shift+drag) to select a time range, `m`
  to make the selected slice's span the range, `Esc` to clear; `Range:` also
  offers each prefill / decode phase. Every analysis tab is computed over
  the range.
- **Warnings banner.** CPU fallbacks (`args.fallback`, drawn hatched),
  dropped records (`metadata.dropped`), clock-sync violations and a disabled
  HTP backend are listed under the toolbar.
- **Analysis tabs.** Selection (title, track, duration, cy/elem, MACs, TOPS
  and share of `metadata.hmx_peak_tops` when known, args), Wall time
  (buckets: CPU, FastRPC seam, HMX only, HMX ∥ HVX, HVX only, DMA only, DSP
  idle, load, other; parallel compression; idle inside DSP calls; CPU
  fallbacks; copy the metrics JSON), Engines (busy per track, and the HVX
  pool tail per fork-join kind against the `units ≥ threads × 4` rule),
  Kernels (`op: name` rows with cycles/element, TOPS and outlier flags;
  click to filter), Layers (per-layer stack and numbers; click to zoom),
  Transport (FastRPC seam time against payload bytes, least-squares
  fixed + per-MB fit, outliers beyond 2σ; click to select the call), Tokens
  (per-token bars stacked by CPU / FastRPC / DSP busy / other, calls per
  token, TTFT and median token time; click to zoom). Tables have Copy and
  Download CSV buttons.

## summarize.py

The same metrics from the command line, for the device gate:

```bash
python3 tools/nntr_trace/summarize.py trace.json -o metrics.json
python3 tools/nntr_trace/summarize.py trace.json --range phase:1
python3 tools/nntr_trace/summarize.py trace.json --fail-if 'compression<1.5' --fail-if 'idle_ratio>0.05'
```

`test/test_summarize.py` holds it to the viewer's numbers (1e-6 relative)
on the sample, so the two cannot drift. Schema `nntr_trace.metrics.v1`:
`range`, `wall_us`, `buckets`, `compression`, `idle_ratio`, `engines`,
`kernels`, `layers`, `tokens`, `token_summary`, `transport`, `pool_tail`,
`warnings`, `hmx_peak_tops`.

## Format

Chrome Trace Event Format (`X` spans, `M` names, `s`/`f` flows, `C`
counters). DSP thread ids follow the QNN optrace convention (DMA 256, HVX
512.., HMX 768) so a vendor optrace and ours line up side by side. Compute
spans carry `args.engine`, `args.elems` and (when the tracer has them)
`args.cycles`; `metadata.dsp_clock_mhz` is the fallback for cycles/element.
The metrics the viewer derives are exposed as `window.__nntr.metricsJSON()`
(schema `nntr_trace.metrics.v1`).

Stdlib only, on purpose: these run on whatever machine the device is
plugged into.
