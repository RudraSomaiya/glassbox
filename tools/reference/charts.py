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
  .glassbox { fill: #c8692e } .llamacpp { fill: #2f72a8 } .transformers { fill: #6f8a2a }
  text { font-family: Archivo, system-ui, sans-serif; font-variant-numeric: tabular-nums }
  @media (prefers-color-scheme: dark) {
    .bg { fill: #1c1814 } .ink { fill: #efe7da } .ink2 { fill: #b3a899 } .grid { stroke: #3a332c }
    .llamacpp { fill: #3f86c4 } .transformers { fill: #7f9a35 }
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


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    rows = load_single()
    if rows:
        cpu = "4 vCPU Xeon (Emerald Rapids), 4 threads"
        single_chart(rows, 1, "Decode speed, one stream (tg128)", f"Tokens per second, higher is better. {cpu}. Same GGUF weights for Glassbox and llama.cpp.", "single-decode.svg")
        single_chart(rows, 0, "Prompt processing, one stream (pp512)", f"Tokens per second, higher is better. {cpu}.", "single-prefill.svg")
    progression_chart()
    for name in sorted(os.listdir(OUT)):
        print(os.path.join("bench/results/charts", name))
