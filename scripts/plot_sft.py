"""Plot SFT metrics from a run's train.log into a self-contained HTML file (no dependencies).

    python scripts/plot_sft.py outputs/sft_runs/lr1e-4-r16 [--steps-per-epoch 246]

Reads the `{"type": "metrics", ...}` lines the cookbook prints and writes <run>/metrics.html:
train loss (raw + moving average) with eval-loss points, and throughput. Rerun while training.
"""

import argparse
import json
import math
from pathlib import Path

W, H, M = 760, 260, 48  # panel size and margin


def read(log: Path):
    train, evals = [], []
    for line in log.read_text().splitlines():
        if not line.startswith('{"type": "metrics"'):
            continue
        r = json.loads(line)
        (evals if "eval_loss" in r else train).append(r)
    return train, evals


def moving_average(xs, k=10):
    return [sum(xs[max(0, i - k + 1): i + 1]) / len(xs[max(0, i - k + 1): i + 1]) for i in range(len(xs))]


def panel(title, series, xmax, log_y=False, vlines=(), ylabel=""):
    """series: list of (label, color, [(x, y)], kind) with kind 'line' | 'dots'."""
    ys = [y for _, _, pts, _ in series for _, y in pts if y > 0]
    if not ys:
        return f"<p>{title}: no data yet</p>"
    f = (lambda v: math.log10(v)) if log_y else (lambda v: v)
    lo, hi = f(min(ys)), f(max(ys))
    pad = (hi - lo) * 0.08 or 1
    lo, hi = lo - pad, hi + pad
    sx = lambda x: M + (W - 2 * M) * x / max(xmax, 1)
    sy = lambda y: H - M + (2 * M - H) * (f(y) - lo) / (hi - lo)
    out = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="{title}">']
    for i in range(5):  # y grid + labels
        v = lo + (hi - lo) * i / 4
        yy = H - M + (2 * M - H) * i / 4
        label = f"{10 ** v:.3g}" if log_y else f"{v:.3g}"
        out.append(f'<line x1="{M}" x2="{W - M}" y1="{yy:.1f}" y2="{yy:.1f}" class="grid"/>'
                   f'<text x="{M - 6}" y="{yy + 4:.1f}" text-anchor="end" class="tick">{label}</text>')
    for x in vlines:
        out.append(f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{M / 2}" y2="{H - M}" class="epoch"/>'
                   f'<text x="{sx(x) + 3:.1f}" y="{M / 2 + 8}" class="tick">epoch end</text>')
    out.append(f'<text x="{W / 2}" y="{H - 8}" text-anchor="middle" class="tick">step</text>')
    out.append(f'<text x="12" y="{M / 2 - 8}" class="tick">{ylabel}</text>')
    for step in range(0, int(xmax) + 1, max(1, int(xmax) // 6 or 1)):
        out.append(f'<text x="{sx(step):.1f}" y="{H - M + 16}" text-anchor="middle" class="tick">{step}</text>')
    for label, color, pts, kind in series:
        pts = [(x, y) for x, y in pts if y > 0]
        if kind == "line":
            d = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in pts)
            width = 1 if "raw" in label else 2
            opacity = 0.35 if "raw" in label else 1
            out.append(f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="{width}" opacity="{opacity}"/>')
        else:
            for x, y in pts:
                out.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="5" fill="{color}"><title>{label} step {x}: {y:.4f}</title></circle>')
    legend = " ".join(f'<span><i style="background:{c}"></i>{l}</span>' for l, c, _, _ in series)
    return f"<h2>{title}</h2><div class='legend'>{legend}</div>" + "".join(out) + "</svg>"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("--steps-per-epoch", type=int, default=0)
    args = p.parse_args()
    train, evals = read(args.run_dir / "train.log")
    steps = [r["step"] for r in train]
    loss = [r["ce_loss"] for r in train]
    xmax = max(steps + [e["step"] for e in evals] + [args.steps_per_epoch * 3 if args.steps_per_epoch else 0, 1])
    vl = [args.steps_per_epoch * k for k in (1, 2) if args.steps_per_epoch and args.steps_per_epoch * k < xmax]
    last = train[-1] if train else {}
    p1 = panel("Loss (log scale)", [
        ("train raw", "#4c78a8", list(zip(steps, loss)), "line"),
        ("train (10-step avg)", "#1f4e79", list(zip(steps, moving_average(loss))), "line"),
        ("eval (dev)", "#e45756", [(e["step"], e["eval_loss"]) for e in evals], "dots"),
    ], xmax, log_y=True, vlines=vl, ylabel="cross-entropy")
    p2 = panel("Throughput", [("tokens/sec", "#54a24b", [(r["step"], r.get("tokens_per_sec", 0)) for r in train], "line")],
               xmax, ylabel="tokens/s")
    ev = "".join(f"<li>step {e['step']}: eval loss {e['eval_loss']:.4f} (ppl {e['eval_ppl']:.3f})</li>" for e in evals)
    html = f"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SFT metrics</title><style>
body{{font:14px system-ui,sans-serif;max-width:800px;margin:24px auto;padding:0 16px;color:#222;background:#fff}}
h1{{font-size:18px}}h2{{font-size:15px;margin:22px 0 4px}}.grid{{stroke:#ddd}}.epoch{{stroke:#999;stroke-dasharray:4}}
.tick{{font-size:11px;fill:#666}}.legend span{{margin-right:14px;font-size:12px}}.legend i{{display:inline-block;width:10px;height:10px;margin-right:4px}}
@media(prefers-color-scheme:dark){{body{{background:#161616;color:#ddd}}.grid{{stroke:#333}}.tick{{fill:#999}}}}
</style><h1>{args.run_dir.name}: step {last.get('step', '-')} of {xmax}</h1>
<p>latest train loss {last.get('ce_loss', float('nan')):.4f} · {last.get('tokens_per_sec', 0):,.0f} tokens/s</p>
{p1}<ul>{ev}</ul>{p2}"""
    out = args.run_dir / "metrics.html"
    out.write_text(html)
    print(f"wrote {out} ({len(train)} train points, {len(evals)} eval points)")


if __name__ == "__main__":
    main()
