"""Turn bench/results/*.json into SVG charts in bench/results/charts/.

Hand-written SVG (no plotting library). Each chart carries light and dark
colours and switches with prefers-color-scheme, so it reads on GitHub in
either theme. Palette checked with the dataviz validator (CVD and contrast).

    python tools/reference/charts.py
"""

import glob
import json
import os
from html import escape

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "bench", "results")
OUT = os.path.join(ROOT, "charts")

ENGINES = [("glassbox", "Glassbox"), ("llamacpp", "llama.cpp"), ("transformers", "transformers")]
STYLE = """<style>
  .bg { fill: #fbfcfb } .ink { fill: #121413 } .ink2 { fill: #444a46 } .grid { stroke: #dfe3df }
  .glassbox { fill: #c8692e } .llamacpp { fill: #2f72a8 } .transformers { fill: #6f8a2a } .mark { fill: #c92a22 }
  .base { fill: #444a46 } .base-line { stroke: #444a46 }
  .glassbox-line { stroke: #c8692e } .llamacpp-line { stroke: #2f72a8 } .transformers-line { stroke: #6f8a2a } .mark-line { stroke: #c92a22 }
  text { font-family: Archivo, system-ui, sans-serif; font-variant-numeric: tabular-nums }
  @media (prefers-color-scheme: dark) {
    .bg { fill: #1c1814 } .ink { fill: #efe7da } .ink2 { fill: #b3a899 } .grid { stroke: #3a332c }
    .llamacpp { fill: #3f86c4 } .transformers { fill: #7f9a35 } .mark { fill: #ff6b3d }
    .llamacpp-line { stroke: #3f86c4 } .transformers-line { stroke: #7f9a35 } .mark-line { stroke: #ff6b3d }
  }
</style>"""


def load_single():
    """{model: {(engine, dtype): (pp512, tg128)}}"""
    rows = {}
    for f in glob.glob(os.path.join(ROOT, "single", "*.json")):
        engine, rest = os.path.basename(f)[:-5].split("-", 1)
        model, dtype = rest.rsplit("-", 1)
        d = json.load(open(f))
        if isinstance(d, list):  # llama-bench output
            pp = next(r["avg_ts"] for r in d if r["n_prompt"] > 0)
            tg = next(r["avg_ts"] for r in d if r["n_gen"] > 0)
        else:
            pp, tg = d["pp512_tok_s"], d["tg128_tok_s"]
        rows.setdefault(model, {})[(engine, dtype)] = (pp, tg)
    return rows


def nice_max(v):
    for step in [1, 2, 2.5, 5]:
        for mag in range(0, 6):
            m = step * 10**mag
            if m >= v:
                return m
    return v


def text(x, y, s, cls="ink", size=13, anchor="start", weight=400):
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}">{escape(s)}</text>'


def legend(x, y, engines):
    out, cx = [], x
    for key, name in engines:
        out.append(f'<rect x="{cx}" y="{y - 10}" width="12" height="12" rx="2" class="{key}"/>')
        out.append(text(cx + 18, y, name, size=13))
        cx += 30 + 7.5 * len(name)
    return out


def single_chart(rows, metric, title, subtitle, fname):
    models = [m for m in ["smollm2-135m-standin", "smollm2-360m-standin", "smollm2-1.7b-standin",
                          "smollm2-135m", "smollm2-360m", "smollm2-1.7b"] if m in rows]
    dtypes = ["f32", "q8_0", "q4_0"]
    pw, ph, top, left = 300, 230, 96, 48
    W = left + len(models) * (pw + 24)
    H = top + ph + 64
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-labelledby="t">',
           STYLE, f'<title id="t">{escape(title)}</title>', f'<rect width="{W}" height="{H}" class="bg"/>',
           text(24, 32, title, size=18, weight=700), text(24, 54, subtitle, "ink2", 13)]
    out += legend(24, 80, ENGINES)
    for mi, model in enumerate(models):
        x0 = left + mi * (pw + 24)
        vals = [v[metric] for v in rows[model].values()]
        ymax = nice_max(max(vals) * 1.1)
        y = lambda v: top + ph - v / ymax * ph
        for k in range(5):
            gv = ymax * k / 4
            out.append(f'<line x1="{x0}" x2="{x0 + pw}" y1="{y(gv):.1f}" y2="{y(gv):.1f}" class="grid" stroke-width="1"/>')
            out.append(text(x0 - 6, y(gv) + 4, f"{gv:g}", "ink2", 11, "end"))
        gw = pw / len(dtypes)
        for di, dt in enumerate(dtypes):
            present = [(k, n) for k, n in ENGINES if (k, dt) in rows[model]]
            bw = min(28, (gw - 20) / max(1, len(present)) - 2)
            gx = x0 + di * gw + (gw - len(present) * (bw + 2)) / 2
            for ei, (key, name) in enumerate(present):
                v = rows[model][(key, dt)][metric]
                bx = gx + ei * (bw + 2)
                h = v / ymax * ph
                out.append(f'<path d="M{bx:.1f},{top + ph:.1f} V{top + ph - h + 3:.1f} q0,-3 3,-3 H{bx + bw - 3:.1f} q3,0 3,3 V{top + ph:.1f} Z" class="{key}"><title>{escape(name)} {dt.upper()}: {v:.1f} tok/s</title></path>')
                out.append(text(bx + bw / 2, top + ph - h - 5, f"{v:.0f}", "ink", 11, "middle"))
            out.append(text(x0 + di * gw + gw / 2, top + ph + 20, dt.upper(), "ink2", 12, "middle"))
        label = model.replace("smollm2-", "SmolLM2-").replace("-standin", " (stand-in)").replace("m", "M").replace("b", "B")
        out.append(text(x0 + pw / 2, top + ph + 44, label, "ink", 13, "middle", 700))
    out.append("</svg>")
    open(os.path.join(OUT, fname), "w").write("\n".join(out))


def progression_chart():
    f = os.path.join(ROOT, "progression.json")
    if not os.path.exists(f):
        return
    d = json.load(open(f))
    steps = d["steps"]
    W, rowh, left, top = 760, 44, 200, 80
    H = top + rowh * len(steps) + 40
    vmax = nice_max(max(s["tg32_tok_s"] for s in steps) * 1.1)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-labelledby="t">',
           STYLE, '<title id="t">How Glassbox got fast: decode speed at each step</title>',
           f'<rect width="{W}" height="{H}" class="bg"/>',
           text(24, 32, "How it got fast: decode tokens per second", size=18, weight=700),
           text(24, 54, f"{os.path.basename(d['model'])}, {d['meta']['cpu']}, commit {d['meta']['commit']}", "ink2", 13)]
    for i, s in enumerate(steps):
        yy = top + i * rowh
        w = s["tg32_tok_s"] / vmax * (W - left - 80)
        out.append(text(left - 12, yy + 22, s["step"], "ink", 13, "end"))
        out.append(f'<rect x="{left}" y="{yy + 6}" width="{w:.1f}" height="24" rx="3" class="glassbox"><title>{escape(s["step"])}: {s["tg32_tok_s"]:.1f} tok/s</title></rect>')
        out.append(text(left + w + 8, yy + 23, f'{s["tg32_tok_s"]:.1f}', "ink", 13, "start", 700))
    out.append("</svg>")
    open(os.path.join(OUT, "progression.svg"), "w").write("\n".join(out))


def line_chart(fname, title, subtitle, series, xlabel, fmt="{:.0f}"):
    """series: [(css class, legend name, [(x, y), ...])]"""
    W, H, L, T, B = 760, 380, 64, 96, 52
    xs = sorted({x for _, _, pts in series for x, _ in pts})
    ymax = nice_max(max(y for _, _, pts in series for _, y in pts) * 1.1)
    px = lambda x: L + (xs.index(x) / max(1, len(xs) - 1)) * (W - L - 40)
    py = lambda y: T + (H - T - B) * (1 - y / ymax)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-labelledby="t">',
           STYLE, f'<title id="t">{escape(title)}</title>', f'<rect width="{W}" height="{H}" class="bg"/>',
           text(24, 32, title, size=18, weight=700), text(24, 54, subtitle, "ink2", 13)]
    out += legend(24, 80, [(k, n) for k, n, _ in series])
    for k in range(5):
        v = ymax * k / 4
        out.append(f'<line x1="{L}" x2="{W - 24}" y1="{py(v):.1f}" y2="{py(v):.1f}" class="grid" stroke-width="1"/>')
        out.append(text(L - 8, py(v) + 4, fmt.format(v), "ink2", 11, "end"))
    for x in xs:
        out.append(text(px(x), H - B + 20, f"{x:g}", "ink2", 12, "middle"))
    out.append(text(W - 24, H - 10, xlabel, "ink2", 12, "end"))
    for cls, name, pts in series:
        d = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in pts)
        out.append(f'<polyline points="{d}" fill="none" stroke-width="2.5" class="{cls}-line"/>')
        for x, y in pts:
            out.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="5" class="{cls}"><title>{escape(name)}, {x:g}: {fmt.format(y)}</title></circle>')
    out.append("</svg>")
    open(os.path.join(OUT, fname), "w").write("\n".join(out))


def serving_charts():
    f = os.path.join(ROOT, "serving.json")
    if not os.path.exists(f):
        return
    d = json.load(open(f))
    mixed = [r for r in d["runs"] if r["workload"] == "mixed"]
    pts = lambda mode, key: [(r["result"]["rate_rps"], r["result"][key]) for r in mixed if r["mode"] == mode]
    modes = [("glassbox", "continuous batching"), ("transformers", "one at a time")]
    line_chart("serving-ttft.svg", "Time to first token, median", "Seconds, lower is better. 48 chat-sized requests, Poisson arrivals, 135M shapes Q8_0.",
               [(c, m, pts(m, "ttft_p50_s")) for c, m in modes], "requests per second", "{:.1f}")
    line_chart("serving-throughput.svg", "Throughput under load", "Generated tokens per second, higher is better. Same workload.",
               [(c, m, pts(m, "throughput_tok_s")) for c, m in modes], "requests per second")


def spec_chart():
    runs = []
    for name in ["speculative.json", "speculative-trained.json"]:
        f = os.path.join(ROOT, name)
        if os.path.exists(f):
            runs += json.load(open(f))["runs"]
    if not runs:
        return
    d = {"runs": runs}
    pairs = list(dict.fromkeys(r["pair"] for r in d["runs"]))
    classes = ["glassbox", "llamacpp", "transformers", "mark", "base", "glassbox"]
    line_chart("speculative.svg", "Speculative decoding: speedup over the target alone", "Greedy, identical output. 1.0 means no gain; below 1 the drafting costs more than it saves.",
               [(classes[i % len(classes)], p, [(r["k"], r["speedup"]) for r in d["runs"] if r["pair"] == p]) for i, p in enumerate(pairs)],
               "tokens drafted per round", "{:.2f}")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    rows = load_single()
    if rows:
        cpu = "4 vCPU Xeon (Emerald Rapids), 4 threads"
        single_chart(rows, 1, "Decode speed, one stream (tg128)", f"Tokens per second, higher is better. {cpu}. Same GGUF weights for Glassbox and llama.cpp.", "single-decode.svg")
        single_chart(rows, 0, "Prompt processing, one stream (pp512)", f"Tokens per second, higher is better. {cpu}.", "single-prefill.svg")
    progression_chart()
    serving_charts()
    spec_chart()
    for name in sorted(os.listdir(OUT)):
        print(os.path.join("bench/results/charts", name))
