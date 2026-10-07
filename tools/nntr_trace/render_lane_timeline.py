#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Render DSP lane traces as labeled SVG panels (stdlib only).

Example: render_lane_timeline.py moe.json conv.json -o timeline.svg
Use --start-us / --end-us to inspect short callbacks at a shared time scale.
Blank regions mean no recorded window, not proven hardware idle.
"""
import argparse
import html
import json
from pathlib import Path

def render(paths, start_us=0, end_us=None):
    panels = []
    labels = set()
    for path in paths:
        doc = json.loads(path.read_text())
        if 'lane_metrics' not in doc.get('metadata', {}):
            raise ValueError(f'{path}: requires a measured DSP lane capture')
        events = [e for e in doc['traceEvents'] if e['ph'] == 'X' and e['tid'] != 0]
        names = {e['tid']: e['args']['name'] for e in doc['traceEvents']
                 if e['ph'] == 'M' and e['name'] == 'thread_name'}
        labels.update(e['args'].get('display_label', e['name']) for e in events)
        panels.append((path.stem, doc['metadata']['lane_metrics'], events, names))
    legend = sorted(labels)
    colors = {label: f'hsl({i * 137.508 % 360:.3f},65%,78%)'
              for i, label in enumerate(legend)}
    codes = {label: str(i + 1) for i, label in enumerate(legend)}
    top = 104
    legend_top = top + sum(90 + len({e['tid'] for e in p[2]}) * 25 for p in panels)
    height = legend_top + 60 + ((len(legend) + 2) // 3) * 30
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1800" height="{height}">',
           f'<rect width="1800" height="{height}" fill="#f8fafc"/>',
           '<g font-family="sans-serif" fill="#0f172a">',
           '<text x="24" y="30" font-size="22" font-weight="bold">DSP operation timeline</text>',
           '<text x="24" y="53" font-size="13">DMA: outstanding bounds. HMX: batch API + acc_read. HVX: callback windows. Blank: no recorded window.</text>',
           '<text x="24" y="73" font-size="13">Colors and numbers identify operations in the legend below. Short spans omit text. Hover for full details.</text>']
    for name, metrics, events, names in panels:
        stop = min(end_us, metrics['entry_us']) if end_us is not None else metrics['entry_us']
        if stop <= start_us:
            raise ValueError(f'{name}: requested range is outside the capture')
        ids = sorted({e['tid'] for e in events})
        left, width = 350, 1420
        svg.append(f'<text x="24" y="{top}" font-size="17" font-weight="bold">{html.escape(name)} | entry {metrics["entry_us"]/1000:.3f} ms | HMX/HVX overlap {metrics["hmx_hvx_overlap_us"]/1000:.3f} ms | complete: {metrics["complete"]}</text>')
        for tick in range(8):
            x = left + width * tick / 7
            t = start_us + (stop-start_us) * tick / 7
            svg.append(f'<path d="M{x} {top+26} V{top+35+len(ids)*25}" stroke="#cbd5e1"/><text x="{x}" y="{top+21}" font-size="11" text-anchor="middle">{t:.1f} us</text>')
        for row, tid in enumerate(ids):
            y = top + 35 + row * 25
            svg.append(f'<text x="24" y="{y+14}" font-size="11">{html.escape(names[tid])}</text><rect x="{left}" y="{y}" width="{width}" height="20" fill="#e2e8f0"/>')
            # Nested DMA descriptors can overlap; retain individual hover titles.
            for e in events:
                if e['tid'] != tid or e['ts'] >= stop or e['ts']+e['dur'] <= start_us:
                    continue
                label = e['args'].get('display_label', e['name'])
                x = left + (max(start_us, e['ts'])-start_us)/(stop-start_us)*width
                w = (min(stop, e['ts']+e['dur'])-max(start_us, e['ts']))/(stop-start_us)*width
                title = html.escape(f'{label}: start {e["ts"]:.2f} us; duration {e["dur"]:.2f} us; {json.dumps(e["args"])}')
                svg.append(f'<rect x="{x:.3f}" y="{y}" width="{max(.35,w):.3f}" height="20" fill="{colors[label]}" stroke="#475569" stroke-width=".3"><title>{title}</title></rect>')
                # Show complete names only when they fit; otherwise use the
                # legend number, or color alone for very short windows.
                full = codes[label] + ': ' + label
                text = full if w >= len(full)*6.5+8 else codes[label]
                if w >= len(text)*6.5+8:
                    svg.append(f'<text x="{x+4:.3f}" y="{y+14}" font-size="11">{html.escape(text)}</text>')
        top += 90 + len(ids)*25
    svg.append(f'<text x="24" y="{legend_top}" font-size="18" font-weight="bold">Operation legend</text>')
    for i, label in enumerate(legend):
        x = 24 + (i % 3) * 590
        y = legend_top + 20 + (i // 3) * 30
        svg += [f'<rect x="{x}" y="{y}" width="28" height="20" fill="{colors[label]}" stroke="#475569" stroke-width=".3"/>',
                f'<text x="{x+14}" y="{y+14}" font-size="11" text-anchor="middle">{codes[label]}</text>',
                f'<text x="{x+38}" y="{y+14}" font-size="13">{html.escape(label)}</text>']
    return '\n'.join(svg + ['</g></svg>'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('traces', type=Path, nargs='+')
    ap.add_argument('-o', '--output', type=Path, required=True)
    ap.add_argument('--start-us', type=float, default=0)
    ap.add_argument('--end-us', type=float)
    args = ap.parse_args()
    args.output.write_text(render(args.traces, args.start_us, args.end_us))


if __name__ == '__main__':
    main()
