import './benchmarks.css'

type Json = Record<string, any>
const files = import.meta.glob('../../../bench/results/**/*.json', { eager: true, import: 'default' }) as Record<string, Json>
const byName = (suffix: string) => Object.entries(files).find(([k]) => k.endsWith(suffix))?.[1]

const ENGINES = [
  { key: 'glassbox', name: 'Glassbox' },
  { key: 'llamacpp', name: 'llama.cpp' },
  { key: 'transformers', name: 'transformers' },
] as const
const DTYPES = ['f32', 'q8_0', 'q4_0']
const MODELS = ['smollm2-135m-standin', 'smollm2-360m-standin', 'smollm2-1.7b-standin']
const label = (m: string) => m.replace('smollm2-', 'SmolLM2-').replace('-standin', '').toUpperCase().replace('SMOLLM2', 'SmolLM2')

function single() {
  const rows: Record<string, Record<string, [number, number]>> = {}
  for (const [path, d] of Object.entries(files)) {
    const m = path.match(/single\/(\w+)-(.+)-(f32|q8_0|q4_0)\.json$/)
    if (!m) continue
    const [, engine, model, dtype] = m
    const v: [number, number] = Array.isArray(d)
      ? [d.find((r: Json) => r.n_prompt > 0).avg_ts, d.find((r: Json) => r.n_gen > 0).avg_ts]
      : [d.pp512_tok_s, d.tg128_tok_s]
    ;(rows[model] ??= {})[`${engine}/${dtype}`] = v
  }
  return rows
}

function Legend({ items }: { items: { key: string; name: string }[] }) {
  return (
    <ul className="legend">
      {items.map((i) => (
        <li key={i.key}>
          <i className={`sw sw--${i.key}`} /> {i.name}
        </li>
      ))}
    </ul>
  )
}

/** Small multiples, one panel per model, bars per engine within each dtype. */
function SingleChart({ metric, title }: { metric: 0 | 1; title: string }) {
  const rows = single()
  const W = 300
  const H = 200
  return (
    <figure className="chart">
      <figcaption>
        <h3>{title}</h3>
        <Legend items={[...ENGINES]} />
      </figcaption>
      <div className="multiples">
        {MODELS.filter((m) => rows[m]).map((model) => {
          const vals = Object.values(rows[model]).map((v) => v[metric])
          const max = Math.max(...vals) * 1.15
          return (
            <svg key={model} viewBox={`0 0 ${W} ${H + 44}`} role="img" aria-label={`${title}, ${label(model)}`}>
              {[0, 0.5, 1].map((f) => (
                <line key={f} x1={0} x2={W} y1={H - f * H} y2={H - f * H} className="grid" />
              ))}
              {DTYPES.map((dt, di) => {
                const present = ENGINES.filter((e) => rows[model][`${e.key}/${dt}`])
                const gw = W / DTYPES.length
                const bw = 26
                const gx = di * gw + (gw - present.length * (bw + 3)) / 2
                return (
                  <g key={dt}>
                    {present.map((e, ei) => {
                      const v = rows[model][`${e.key}/${dt}`][metric]
                      const h = (v / max) * H
                      const x = gx + ei * (bw + 3)
                      return (
                        <g key={e.key}>
                          <rect x={x} y={H - h} width={bw} height={h} rx={3} className={`bar bar--${e.key}`}>
                            <title>{`${e.name} ${dt.toUpperCase()}: ${v.toFixed(1)} tokens/s`}</title>
                          </rect>
                          <text x={x + bw / 2} y={H - h - 5} className="val" textAnchor="middle">
                            {v >= 100 ? Math.round(v) : v.toFixed(1)}
                          </text>
                        </g>
                      )
                    })}
                    <text x={di * gw + gw / 2} y={H + 18} className="tick" textAnchor="middle">
                      {dt.toUpperCase()}
                    </text>
                  </g>
                )
              })}
              <text x={W / 2} y={H + 40} className="panel" textAnchor="middle">
                {label(model)}
              </text>
            </svg>
          )
        })}
      </div>
      <details>
        <summary>The numbers</summary>
        <table>
          <thead>
            <tr>
              <th>Model</th>
              <th>Type</th>
              {ENGINES.map((e) => (
                <th key={e.key}>{e.name}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {MODELS.filter((m) => rows[m]).flatMap((m) =>
              DTYPES.map((dt) => (
                <tr key={m + dt}>
                  <td>{label(m)}</td>
                  <td>{dt.toUpperCase()}</td>
                  {ENGINES.map((e) => (
                    <td key={e.key}>{rows[m][`${e.key}/${dt}`]?.[metric].toFixed(1) ?? '–'}</td>
                  ))}
                </tr>
              )),
            )}
          </tbody>
        </table>
      </details>
    </figure>
  )
}

function Progression() {
  const d = byName('progression.json')
  if (!d) return null
  const max = Math.max(...d.steps.map((s: Json) => s.tg32_tok_s))
  return (
    <figure className="chart">
      <figcaption>
        <h3>How it got fast</h3>
        <p>Decode tokens per second on the 135M shapes, one change at a time.</p>
      </figcaption>
      <ol className="hbars">
        {d.steps.map((s: Json) => (
          <li key={s.step}>
            <span className="hbars__label">{s.step}</span>
            <span className="hbars__track">
              <span className="hbars__bar" style={{ '--w': s.tg32_tok_s / max } as React.CSSProperties} />
            </span>
            <span className="hbars__val">{s.tg32_tok_s.toFixed(1)}</span>
          </li>
        ))}
      </ol>
    </figure>
  )
}

/** Lines over request rate, one per mode. */
function LineChart({ title, note, series, y, unit }: { title: string; note: string; series: { key: string; name: string; pts: [number, number][] }[]; y: string; unit: string }) {
  const W = 560
  const H = 220
  const L = 48
  const xs = series.flatMap((s) => s.pts.map((p) => p[0]))
  const ys = series.flatMap((s) => s.pts.map((p) => p[1]))
  const xmax = Math.max(...xs)
  const ymax = Math.max(...ys) * 1.1
  const px = (v: number) => L + (v / xmax) * (W - L - 12)
  const py = (v: number) => H - (v / ymax) * (H - 12)
  return (
    <figure className="chart">
      <figcaption>
        <h3>{title}</h3>
        <p>{note}</p>
        <Legend items={series} />
      </figcaption>
      <svg viewBox={`0 0 ${W} ${H + 36}`} role="img" aria-label={title}>
        {[0, 0.25, 0.5, 0.75, 1].map((f) => (
          <g key={f}>
            <line x1={L} x2={W} y1={py(ymax * f)} y2={py(ymax * f)} className="grid" />
            <text x={L - 6} y={py(ymax * f) + 4} className="tick" textAnchor="end">
              {(ymax * f).toFixed(ymax < 10 ? 1 : 0)}
            </text>
          </g>
        ))}
        {[...new Set(xs)].map((x) => (
          <text key={x} x={px(x)} y={H + 18} className="tick" textAnchor="middle">
            {x}
          </text>
        ))}
        <text x={W} y={H + 34} className="tick" textAnchor="end">
          {y}
        </text>
        {series.map((s) => (
          <g key={s.key}>
            <polyline points={s.pts.map((p) => `${px(p[0])},${py(p[1])}`).join(' ')} className={`line line--${s.key}`} />
            {s.pts.map((p) => (
              <circle key={p[0]} cx={px(p[0])} cy={py(p[1])} r={4.5} className={`dot dot--${s.key}`}>
                <title>{`${s.name}, ${p[0]}: ${p[1].toFixed(2)} ${unit}`}</title>
              </circle>
            ))}
          </g>
        ))}
      </svg>
    </figure>
  )
}

function Serving() {
  const d = byName('serving.json')
  if (!d) return null
  const mixed = (mode: string, field: string) =>
    d.runs.filter((r: Json) => r.workload === 'mixed' && r.mode === mode).map((r: Json) => [r.result.rate_rps, r.result[field]] as [number, number])
  const modes = [
    { key: 'glassbox', name: 'Continuous batching', mode: 'continuous batching' },
    { key: 'base', name: 'One request at a time', mode: 'one at a time' },
  ]
  const shared = d.runs.filter((r: Json) => r.workload === 'shared system prompt')
  return (
    <>
      <LineChart
        title="Throughput under load"
        note="Generated tokens per second as requests arrive faster (Poisson arrivals, 48 chat-sized requests)."
        y="requests per second"
        unit="tokens/s"
        series={modes.map((m) => ({ key: m.key, name: m.name, pts: mixed(m.mode, 'throughput_tok_s') }))}
      />
      <LineChart
        title="Time to first token, 99th percentile"
        note="How long the unluckiest requests wait before their first token, in seconds."
        y="requests per second"
        unit="s"
        series={modes.map((m) => ({ key: m.key, name: m.name, pts: mixed(m.mode, 'ttft_p99_s') }))}
      />
      <figure className="chart">
        <figcaption>
          <h3>Prefix caching</h3>
          <p>Every request starts with the same 384-token system prompt. With the cache, those blocks are computed once.</p>
        </figcaption>
        <table>
          <thead>
            <tr>
              <th>Rate</th>
              <th>Mode</th>
              <th>Tokens/s</th>
              <th>TTFT p50</th>
              <th>TTFT p99</th>
              <th>Prompt tokens reused</th>
            </tr>
          </thead>
          <tbody>
            {shared.map((r: Json, i: number) => (
              <tr key={i}>
                <td>{r.result.rate_rps} req/s</td>
                <td>{r.mode}</td>
                <td>{r.result.throughput_tok_s.toFixed(0)}</td>
                <td>{r.result.ttft_p50_s.toFixed(2)} s</td>
                <td>{r.result.ttft_p99_s.toFixed(2)} s</td>
                <td>
                  {r.result.prefix_hit_tokens.toLocaleString()} of {r.result.prompt_tokens.toLocaleString()}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </figure>
    </>
  )
}

function Spec() {
  const random = byName('/speculative.json')
  const trained = byName('/speculative-trained.json')
  if (!random && !trained) return null
  const d = { runs: [...(random?.runs ?? []), ...(trained?.runs ?? [])] }
  const pairs = [...new Set(d.runs.map((r: Json) => r.pair))] as string[]
  const keys = ['glassbox', 'llamacpp', 'transformers', 'mark', 'base', 'glassbox']
  return (
    <>
      <LineChart
        title="Speculative decoding: speedup by draft length"
        note="Decode speedup over the target alone (greedy, identical output). Below 1, drafting costs more than it saves: on stand-in weights the drafters rarely agree with the targets, and the trained stand-in is too small for a drafter to be much cheaper."
        y="tokens drafted per round"
        unit="×"
        series={pairs.map((p, i) => ({ key: keys[i % keys.length], name: p, pts: d.runs.filter((r: Json) => r.pair === p).map((r: Json) => [r.k, r.speedup]) }))}
      />
      <details className="chart">
        <summary>Acceptance rates</summary>
        <table>
          <thead>
            <tr>
              <th>Pair</th>
              <th>k</th>
              <th>Accepted</th>
              <th>Speedup</th>
            </tr>
          </thead>
          <tbody>
            {d.runs.map((r: Json, i: number) => (
              <tr key={i}>
                <td>{r.pair}</td>
                <td>{r.k}</td>
                <td>{(r.acceptance * 100).toFixed(0)}%</td>
                <td>{r.speedup.toFixed(2)}×</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </>
  )
}

function Quality() {
  const q = byName('quality-135m-standin.json')
  const l = byName('quality-135m-standin-llamacpp.json')
  if (!q) return null
  return (
    <figure className="chart">
      <figcaption>
        <h3>Quantization against llama.cpp</h3>
        <p>
          Perplexity on {q.scored_tokens.toLocaleString()} scored tokens, the same GGUF files in both engines. With stand-in
          random weights the value sits near the vocabulary size; what matters is that the engines agree.
        </p>
      </figcaption>
      <table>
        <thead>
          <tr>
            <th>Type</th>
            <th>Glassbox</th>
            <th>llama.cpp</th>
            <th>Top-1 agreement with f32</th>
          </tr>
        </thead>
        <tbody>
          {q.results.map((r: Json) => (
            <tr key={r.dtype}>
              <td>{r.dtype.toUpperCase()}</td>
              <td>{r.ppl.toFixed(2)}</td>
              <td>{l?.[r.dtype]?.toFixed(2) ?? '–'}</td>
              <td>{(r.top1_agreement_vs_f32 * 100).toFixed(1)}%</td>
            </tr>
          ))}
        </tbody>
      </table>
    </figure>
  )
}

export function Benchmarks() {
  const meta = Object.values(files).find((f) => f.meta?.cpu)?.meta
  const browser = byName('browser.json')
  return (
    <div className="page bench">
      <header className="bench__head">
        <h1>Benchmarks</h1>
        <p className="bench__lede">
          Every number here comes from a script in the repository and a JSON file in <code>bench/results</code>. One
          machine: {meta?.cpu ?? 'a 4-vCPU cloud VM'}, 4 threads, no GPU. llama.cpp built from source (commit{' '}
          <code>0c1e570</code>) reads the same GGUF bytes.
        </p>
        <p className="bench__warn">
          The model weights are stand-ins with SmolLM2’s exact shapes and random values, because Hugging Face was blocked
          from the build machine. Speed depends only on the shapes, so speed numbers stand; anything that depends on what
          the model knows (perplexity, speculative acceptance) will change with real weights.
        </p>
      </header>
      <section aria-labelledby="b-single">
        <h2 id="b-single">One stream</h2>
        <SingleChart metric={1} title="Writing: decode tokens per second (tg128)" />
        <SingleChart metric={0} title="Reading: prompt tokens per second (pp512)" />
        <p className="bench__read">
          Decode, where memory bandwidth rules, runs at 50 to 80 percent of llama.cpp. Prompt processing is three to seven
          times slower: llama.cpp uses AVX-512 VNNI and weight layouts repacked for its matrix kernels, Glassbox uses AVX2
          dot products.
        </p>
        <Progression />
      </section>
      <section aria-labelledby="b-serve">
        <h2 id="b-serve">Serving</h2>
        <Serving />
      </section>
      <section aria-labelledby="b-spec">
        <h2 id="b-spec">Speculative decoding</h2>
        <Spec />
      </section>
      <section aria-labelledby="b-quality">
        <h2 id="b-quality">Quality</h2>
        <Quality />
      </section>
      {browser && (
        <section aria-labelledby="b-browser">
          <h2 id="b-browser">In the browser</h2>
          <p className="bench__big">
            <strong>{browser.tg64_tok_s.toFixed(1)}</strong> tokens per second
          </p>
          <p>
            SmolLM2-135M shapes at Q8_0, single-threaded WebAssembly with SIMD, {browser.note}.
          </p>
        </section>
      )}
    </div>
  )
}
