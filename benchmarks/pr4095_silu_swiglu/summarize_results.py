#!/usr/bin/env python3
"""Summarize PR #4095 layer benchmark CSV results."""

from __future__ import annotations

import csv
import pathlib
import sys


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: summarize_results.py RAW_CSV RESULT_DIR", file=sys.stderr)
        return 2

    raw_path = pathlib.Path(sys.argv[1])
    result_dir = pathlib.Path(sys.argv[2])

    with raw_path.open(newline="", encoding="utf-8") as raw_file:
        rows = list(csv.DictReader(raw_file))

    indexed = {
        (row["workload"], row["layer"], int(row["threads"]), row["version"]): row
        for row in rows
    }

    summary_rows: list[dict[str, str | int | float]] = []
    workloads = sorted({row["workload"] for row in rows})
    layers = ("SiLU", "SwiGLU")
    thread_counts = sorted({int(row["threads"]) for row in rows})

    for workload in workloads:
        for layer in layers:
            for threads in thread_counts:
                before = indexed[(workload, layer, threads, "before")]
                after = indexed[(workload, layer, threads, "after")]
                before_us = float(before["median_us"])
                after_us = float(after["median_us"])
                summary_rows.append(
                    {
                        "workload": workload,
                        "layer": layer,
                        "threads": threads,
                        "before_us": before_us,
                        "after_us": after_us,
                        "speedup": before_us / after_us,
                        "improvement_pct": (1.0 - after_us / before_us) * 100.0,
                    }
                )

    result_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = result_dir / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    summary_md = result_dir / "summary.md"
    with summary_md.open("w", encoding="utf-8") as output_file:
        output_file.write("# PR #4095 SiLU / SwiGLU layer benchmark\n\n")
        output_file.write(
            "Median latency; speedup = before / after; improvement = "
            "1 - after / before.\n\n"
        )
        for workload in workloads:
            for layer in layers:
                output_file.write(f"## {workload} — {layer}\n\n")
                output_file.write(
                    "| Threads | Before (us) | After (us) | Speedup | Improvement |\n"
                )
                output_file.write("|---:|---:|---:|---:|---:|\n")
                for row in summary_rows:
                    if row["workload"] != workload or row["layer"] != layer:
                        continue
                    output_file.write(
                        f"| {row['threads']} | {row['before_us']:.3f} | "
                        f"{row['after_us']:.3f} | {row['speedup']:.2f}x | "
                        f"{row['improvement_pct']:.1f}% |\n"
                    )
                output_file.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
