#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Convert one bounded DSP qtimer capture and report exposed dependency waits.

Do NOT use stage-total viewer idle/compression metrics for this capture.
HMX spans are synchronous batch-call windows (including accumulator read).
HVX spans are physical-thread callback windows, not hardware busy counters.
DMA spans bound descriptor outstanding lifetime, not DMA engine activity.
No ARM/DSP clock alignment is inferred. All timestamps are DSP-entry-relative.
"""
import argparse
import json
from pathlib import Path
import struct

NAMES = {
    1: "HMX gate_up + acc_read", 2: "HMX down + acc_read",
    3: "HVX callback", 4: "HVX background unit", 5: "foreground wait",
    6: "background wait (includes caller help)", 7: "DMA outstanding upper bound",
    8: "DMA wait", 9: "DMA descriptor push", 10: "job publish",
    11: "quantize / requantize", 12: "activation pack",
    13: "gate_up dequant + SwiGLU", 14: "down dequant + requant/scatter",
    15: "tail gate_up", 16: "tail requant", 17: "tail down",
    18: "HMX conv a/c + acc_read", 19: "HMX conv b + acc_read",
    20: "HMX conv out_proj + acc_read", 21: "conv a/c dequant + multiply",
    22: "conv b/out dequant", 23: "conv gate + requant + pack",
}
HVX = {3, 4, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23}
CONV_HMX = {18, 19, 20}
HMX = {1, 2} | CONV_HMX

# Labels describe the instrumented window; fused callbacks are not split into
# invented sub-spans. Keep NAMES stable for metrics and existing consumers.
OPERATIONS = {
    1: "FC matmul gate/up + acc_read", 2: "FC matmul down + acc_read",
    3: "callback (operation unknown)", 4: "background callback (operation unknown)",
    5: "wait foreground", 6: "wait background / caller help",
    7: "DMA transfer (outstanding bound)", 8: "wait DMA",
    9: "DMA enqueue", 10: "publish job",
    11: "quant / requant", 12: "quant + activation pack",
    13: "dequant + SwiGLU", 14: "dequant + requant / scatter",
    15: "tail gate/up", 16: "tail requant", 17: "tail down",
    18: "FC matmul a/c + acc_read", 19: "FC matmul b + acc_read",
    20: "FC matmul out_proj + acc_read", 21: "dequant + multiply",
    22: "dequant b/out", 23: "conv + gate + requant + pack",
}


def union(intervals):
    out = []
    for a, b in sorted(intervals):
        if b <= a:
            continue
        if out and a <= out[-1][1]:
            out[-1][1] = max(b, out[-1][1])
        else:
            out.append([a, b])
    return out


def length(intervals):
    return sum(b - a for a, b in union(intervals))


def intersect(left, right):
    a, b = union(left), union(right)
    i = j = 0
    out = []
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            out.append([lo, hi])
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def subtract(left, right):
    out = []
    for a, b in union(left):
        for c, d in union(right):
            if d <= a:
                continue
            if c >= b:
                break
            if c > a:
                out.append([a, c])
            a = max(a, d)
        if a < b:
            out.append([a, b])
    return out


def convert(raw):
    if len(raw) < 32 or len(raw) % 4:
        raise ValueError("invalid capture byte length")
    words = struct.unpack("<%dI" % (len(raw) // 4), raw)
    magic, version, hz, count, dropped, duration, stride, slots = words[:8]
    if (magic, version, hz, stride) != (0x4e4c5431, 1, 19200000, 8):
        raise ValueError("unsupported capture header")
    if len(words) != 8 + count * stride or not slots or slots > 8:
        raise ValueError("inconsistent capture length/thread capacity")
    us = 1e6 / hz
    entry = duration * us
    recs = []
    for offset in range(8, len(words), 8):
        k, w, t, d, a, b, c, e = words[offset:offset + 8]
        if k not in NAMES or w >= slots or t + d > duration:
            raise ValueError("invalid record kind/thread/clock bound")
        recs.append(dict(kind=k, writer=w, start=t * us, end=(t + d) * us,
                         a=a, b=b, c=c, d=e))
    recs.sort(key=lambda r: (r["start"], r["end"]))
    last_end = {}
    for r in recs:
        if r["kind"] in HVX:
            if r["start"] < last_end.get(r["writer"], 0):
                raise ValueError("overlapping callbacks on one physical writer")
            last_end[r["writer"]] = r["end"]
    ev = []
    threads = {0: "DSP entry", 1: "HMX batch API windows",
               2: "DMA outstanding bounds (NOT hardware busy)",
               3: "Caller dependency waits / publication"}
    for r in recs:
        k = r["kind"]
        args = {}
        if k in HVX:
            tid, cat = 10 + r["writer"], "dsp.hvx"
            threads[tid] = "HVX physical thread %d" % r["writer"]
            args = dict(job_id=r["a"], unit=r["b"], units=r["c"],
                        background=bool(r["d"]), physical_thread=r["writer"])
        elif k in HMX:
            tid, cat = 1, "dsp.hmx"
            if k in CONV_HMX:
                args = dict(block=r["a"], batch=r["b"], column_tile=r["c"],
                            tiles=r["d"])
            else:
                args = dict(expert=r["a"], block=r["b"], batch=r["c"], tiles=r["d"])
            args["fidelity"] = "API window including accumulator read"
        elif k == 7:
            tid, cat = 2, "dsp.dma.bound"
            args = dict(bytes=r["a"], descriptor_id=r["b"], slot=r["d"],
                        completion_lower_us=r["c"] * us,
                        completion_upper_us=r["end"],
                        fidelity="issue-to-observation bound; not hardware execution")
            if not r["start"] <= args["completion_lower_us"] <= r["end"]:
                raise ValueError("invalid DMA completion bound")
        else:
            tid, cat = 3, "dsp.sync"
            args = dict(target_id=r["a"], target_kind=r["b"],
                        target=NAMES.get(r["b"], "unknown"), detail=r["c"])
            if k == 8:
                args = dict(descriptor_id=r["a"], slot=r["b"])
        args["operation"] = OPERATIONS[k]
        args["display_label"] = OPERATIONS[k]
        ev.append(dict(ph="X", pid=2, tid=tid, cat=cat, name=NAMES[k],
                       ts=r["start"], dur=r["end"] - r["start"], args=args))
    operation = "conv" if any(r["kind"] in CONV_HMX for r in recs) else "moe"
    entry_name = "Conv block DSP entry" if operation == "conv" else "MoE DSP entry"
    ev.insert(0, dict(ph="X", pid=2, tid=0, cat="dsp.entry", name=entry_name,
                      ts=0, dur=entry, args=dict(capture="one call")))
    ev += [dict(ph="M", pid=2, tid=tid, name="thread_name", args=dict(name=name))
           for tid, name in threads.items()]
    ev.append(dict(ph="M", pid=2, tid=0, name="process_name",
                   args=dict(name="DSP qtimer timeline")))
    hmx = union((r["start"], r["end"]) for r in recs if r["kind"] in HMX)
    hvx = union((r["start"], r["end"]) for r in recs if r["kind"] in HVX)
    caller = union((r["start"], r["end"]) for r in recs
                   if r["kind"] in HVX and r["writer"] == 0)
    gaps = subtract([[0, entry]], hmx)
    waits = {}
    for r in recs:
        if r["kind"] not in (5, 6, 8):
            continue
        name = "DMA" if r["kind"] == 8 else NAMES.get(r["b"], "unknown")
        span = [[r["start"], r["end"]]]
        if r["kind"] == 6:
            span = subtract(span, caller)
        waits.setdefault(name, []).extend(span)
    by_writer = {}
    for tid in sorted({r["writer"] for r in recs if r["kind"] in HVX}):
        busy = union((r["start"], r["end"]) for r in recs
                     if r["kind"] in HVX and r["writer"] == tid)
        no_cb = subtract([[0, entry]], busy)
        by_writer[str(tid)] = dict(callback_us=length(busy),
                                  no_callback_us=length(no_cb),
                                  longest_no_callback_us=max((b-a for a, b in no_cb), default=0))
    published = {r["a"]: r for r in recs if r["kind"] == 10}
    jobs = {}
    for r in recs:
        if r["kind"] in HVX:
            jobs.setdefault(r["a"], []).append(r)
    job_details = []
    for job, callbacks in jobs.items():
        publication = published.get(job)
        starts = [r["start"] for r in callbacks]
        ends = sorted(r["end"] for r in callbacks)
        job_details.append(dict(
            job_id=job, operation=NAMES[callbacks[0]["kind"]],
            background=bool(callbacks[0]["d"]), callbacks=len(callbacks),
            callback_span_us=max(ends)-min(starts),
            summed_callback_us=sum(r["end"]-r["start"] for r in callbacks),
            publish_to_first_start_lower_us=max(0, min(starts)-publication["end"]) if publication else None,
            publish_to_first_start_upper_us=max(0, min(starts)-publication["start"]) if publication else None,
            publish_to_latest_start_upper_us=max(0, max(starts)-publication["start"]) if publication else None,
            callbacks_running_at_publish=[dict(
                operation=NAMES[r["kind"]], physical_thread=r["writer"],
                job_id=r["a"], remaining_window_us=r["end"]-publication["start"])
                for r in recs if publication and r["kind"] in HVX and
                r["start"] <= publication["start"] < r["end"]],
            last_callback_finish_tail_us=ends[-1]-ends[-2] if len(ends) > 1 else 0))
    grouped_jobs = {}
    for j in job_details:
        g = grouped_jobs.setdefault(j["operation"], dict(jobs=0, summed_callback_us=0,
                                                       max_finish_tail_us=0, max_publish_to_first_start_upper_us=0))
        g["jobs"] += 1
        g["summed_callback_us"] += j["summed_callback_us"]
        g["max_finish_tail_us"] = max(g["max_finish_tail_us"], j["last_callback_finish_tail_us"])
        g["max_publish_to_first_start_upper_us"] = max(
            g["max_publish_to_first_start_upper_us"], j["publish_to_first_start_upper_us"] or 0)
    metrics = dict(
        schema="nntr_lane_trace.metrics.v1", operation=operation, entry_us=entry, records=count,
        dropped_records_or_unfinished_dma=dropped, complete=dropped == 0,
        hmx_api_window_us=length(hmx), hmx_no_api_window_us=length(gaps),
        hvx_any_callback_us=length(hvx),
        hmx_hvx_overlap_us=length(intersect(hmx, hvx)),
        callback_threads=by_writer,
        jobs_by_operation=grouped_jobs,
        longest_publish_to_first_start_jobs=sorted(
            (j for j in job_details if j["publish_to_first_start_upper_us"] is not None),
            key=lambda j: -j["publish_to_first_start_upper_us"])[:10],
        exposed_wait_us={k: length(v) for k, v in waits.items()},
        exposed_wait_with_no_hmx_api_us={k: length(intersect(v, gaps)) for k, v in waits.items()},
        longest_hmx_no_api_windows=sorted(
            [dict(start_us=a, duration_us=b-a,
                  exposed_wait_us={k: length(intersect(v, [[a, b]])) for k, v in waits.items()})
             for a, b in gaps], key=lambda x: -x["duration_us"])[:10],
        dma=dict(descriptors=sum(r["kind"] == 7 for r in recs),
                 bytes=sum(r["a"] for r in recs if r["kind"] == 7),
                 outstanding_upper_union_us=length((r["start"], r["end"]) for r in recs if r["kind"] == 7),
                 max_completion_uncertainty_us=max(
                     (r["end"]-r["c"]*us for r in recs if r["kind"] == 7), default=0)),
        caveats=["No-callback time is not proven hardware idle or scheduler sleep.",
                 "Publish-to-start bounds include wake/publication costs; background jobs also include dependency gating.",
                 "HMX API spans include acc_read; DMA outstanding spans are bounds, not utilization.",
                 "Exposed waits are dependency stalls, not a complete DAG critical path.",
                 "Use only complete captures for bottleneck conclusions."])
    trace = dict(traceEvents=ev, metadata=dict(
        clock_domain="DSP qtimer 19.2MHz, entry-relative", fidelity="measured callback/API windows",
        dropped_records=dropped, lane_metrics=metrics))
    return trace, metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("capture", type=Path)
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--metrics", type=Path)
    args = ap.parse_args()
    try:
        trace, metrics = convert(args.capture.read_bytes())
    except ValueError as exc:
        ap.error(str(exc))
    args.output.write_text(json.dumps(trace, indent=2) + "\n")
    if args.metrics:
        args.metrics.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    return 0 if metrics["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
