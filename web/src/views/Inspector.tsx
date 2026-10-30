import { useLayoutEffect, useMemo, useRef, useState } from 'react'
import type { Inspect } from '../engine/protocol'
import { density, describe, FLIP, Glyphs, pct } from './FilmStrip'
import './inspector.css'

export type Tok = { id: number; text: string }

/**
 * Logit lens: for the chosen token, what each layer would have predicted if
 * the residual stream had stopped there. The first layer whose guess matches
 * the final choice is where the model "made up its mind".
 */
export function Lens({ data, chosen, vocabText }: { data: Inspect; chosen: number; vocabText: (id: number) => string }) {
  const decided = useMemo(() => {
    // The earliest layer from which every later layer agrees with the final choice.
    let d = data.lens.length
    for (let l = data.lens.length - 1; l >= 0; l--) {
      if (data.lens[l].top[0]?.[0] === chosen) d = l
      else break
    }
    return d
  }, [data, chosen])
  return (
    <figure className="lens" aria-labelledby="lens-cap">
      <figcaption id="lens-cap">
        <strong>Layer by layer.</strong>{' '}
        {decided < data.lens.length
          ? `The final answer is on top from layer ${decided + 1} of ${data.lens.length} onward.`
          : `No layer before the last settles on the final answer.`}
      </figcaption>
      <ol className="lens__rows">
        {data.lens.map((l, i) => {
          const [id, p] = l.top[0] ?? [0, 0]
          const hit = id === chosen
          return (
            <li key={i} className={'lens__row' + (hit ? ' lens__row--hit' : '') + (i === decided ? ' lens__row--decided' : '')}>
              <span className="lens__layer">{i + 1}</span>
              <span className={'lens__tok' + (density(p) > FLIP ? ' lens__tok--dense' : '')} style={{ '--d': density(p) } as React.CSSProperties}>
                <Glyphs text={vocabText(id)} />
              </span>
              <span className="lens__p">{pct(p)}</span>
              <span className="lens__rest" aria-label="runners-up">
                {l.top.slice(1, 3).map(([rid, rp]) => (
                  <span key={rid} className="lens__alt">
                    <span className="lens__chip">
                      <Glyphs text={vocabText(rid)} />
                    </span>
                    {pct(rp)}
                  </span>
                ))}
              </span>
            </li>
          )
        })}
      </ol>
    </figure>
  )
}

/**
 * Attention of one token over everything before it: small multiples of every
 * head in a layer, and, for the selected head, the weights as heat on the text
 * with arcs from the query to the tokens it leaned on most.
 */
export function Attention({ data, tokens, query }: { data: Inspect; tokens: Tok[]; query: number }) {
  const layers = data.attention.length
  const heads = data.attention[0]?.length ?? 0
  const [layer, setLayer] = useState(Math.min(layers - 1, Math.floor(layers / 2)))
  const [head, setHead] = useState(0)
  const w = data.attention[layer]?.[head] ?? []
  // Show the most recent stretch of context so arcs stay legible.
  const from = Math.max(0, query - 47)
  const shown = tokens.slice(from, query + 1)
  const max = Math.max(1e-9, ...w.slice(from))
  const top = w
    .map((v, i) => [v, i] as const)
    .slice(from, query + 1)
    .sort((a, b) => b[0] - a[0])
    .slice(0, 5)
  const textRef = useRef<HTMLDivElement>(null)
  const [geo, setGeo] = useState<{ x: number; y: number }[]>([])
  const [size, setSize] = useState({ w: 0, h: 0 })

  // Token positions, measured after layout, for the arcs.
  useLayoutEffect(() => {
    const el = textRef.current
    if (!el) return
    const measure = () => {
      const box = el.getBoundingClientRect()
      setSize({ w: box.width, h: box.height })
      setGeo([...el.querySelectorAll<HTMLElement>('.attn__tok')].map((t) => {
        const r = t.getBoundingClientRect()
        return { x: r.left - box.left + r.width / 2, y: r.top - box.top }
      }))
    }
    measure()
    const ro = new ResizeObserver(measure)
    ro.observe(el)
    return () => ro.disconnect()
  }, [layer, head, query, tokens.length])

  const q = geo[query - from]
  return (
    <figure className="attn" aria-labelledby="attn-cap">
      <figcaption id="attn-cap">
        <strong>Where it looked.</strong> The query is {describe(tokens[query]?.text ?? '')} at position {query}, layer{' '}
        {layer + 1}, head {head + 1}. Darker tiles got more of its attention; the arcs go to the five it leaned on most.
      </figcaption>
      <label className="attn__layer">
        <span>Layer {layer + 1}</span>
        <input type="range" min={0} max={layers - 1} value={layer} onChange={(e) => setLayer(Number(e.target.value))} />
      </label>
      <div className="attn__heads" role="group" aria-label="Attention head">
        {Array.from({ length: heads }, (_, h) => {
          const row = data.attention[layer][h]
          const m = Math.max(1e-9, ...row.slice(from))
          return (
            <button
              key={h}
              type="button"
              aria-pressed={h === head}
              className={'attn__head' + (h === head ? ' attn__head--on' : '')}
              onClick={() => setHead(h)}
            >
              <span className="attn__thumb" aria-hidden>
                {row.slice(from).map((v, i) => (
                  <i key={i} style={{ '--a': v / m } as React.CSSProperties} />
                ))}
              </span>
              <span>Head {h + 1}</span>
            </button>
          )
        })}
      </div>
      <div className="attn__text" ref={textRef}>
        {q && (
          <svg className="attn__arcs" width={size.w} height={size.h} aria-hidden>
            {top.map(([v, i]) => {
              const k = geo[i - from]
              if (!k || i === query) return null
              const lift = 18 + Math.min(60, Math.abs(q.x - k.x) / 6) + Math.abs(q.y - k.y) / 3
              const cy = Math.min(q.y, k.y) - lift
              return (
                <path
                  key={i}
                  d={`M${q.x},${q.y + 2} Q${(q.x + k.x) / 2},${cy} ${k.x},${k.y + 2}`}
                  style={{ '--a': v / max } as React.CSSProperties}
                />
              )
            })}
          </svg>
        )}
        {shown.map((t, n) => {
          const i = from + n
          const rank = top.findIndex(([, j]) => j === i)
          return (
            <span
              key={i}
              className={'attn__tok' + (i === query ? ' attn__tok--q' : '')}
              style={{ '--a': (w[i] ?? 0) / max } as React.CSSProperties}
            >
              <Glyphs text={t.text} />
              {rank >= 0 && rank < 3 && <small className="attn__pct">{pct(w[i])}</small>}
            </span>
          )
        })}
      </div>
      <ol className="sr-only">
        {top.map(([v, i]) => (
          <li key={i}>
            {describe(tokens[i]?.text ?? '')} at position {i}: {pct(v)}
          </li>
        ))}
      </ol>
    </figure>
  )
}
