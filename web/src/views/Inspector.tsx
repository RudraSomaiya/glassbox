import { useMemo, useState } from 'react'
import type { Inspect } from '../engine/protocol'
import { Glyphs, pct } from './FilmStrip'
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
              <span className="lens__tok" style={{ '--p': p } as React.CSSProperties}>
                <Glyphs text={vocabText(id)} />
              </span>
              <span className="lens__p">{pct(p)}</span>
              <span className="lens__rest" aria-label="runners-up">
                {l.top.slice(1, 3).map(([rid, rp]) => (
                  <span key={rid}>
                    <Glyphs text={vocabText(rid)} /> {pct(rp)}
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
 * head in a layer, and arcs over the text for the selected head.
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

  return (
    <figure className="attn" aria-labelledby="attn-cap">
      <figcaption id="attn-cap">
        <strong>Where it looked.</strong> Layer {layer + 1}, head {head + 1}: the darker the tile, the more this token’s
        query matched that earlier token’s key.
      </figcaption>
      <label className="attn__layer">
        <span>Layer {layer + 1}</span>
        <input type="range" min={0} max={layers - 1} value={layer} onChange={(e) => setLayer(Number(e.target.value))} />
      </label>
      <div className="attn__heads" role="radiogroup" aria-label="Attention head">
        {Array.from({ length: heads }, (_, h) => {
          const row = data.attention[layer][h]
          const m = Math.max(1e-9, ...row.slice(from))
          return (
            <button
              key={h}
              type="button"
              role="radio"
              aria-checked={h === head}
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
      <div className="attn__text">
        {shown.map((t, i) => {
          const a = (w[from + i] ?? 0) / max
          return (
            <span
              key={from + i}
              className={'attn__tok' + (from + i === query ? ' attn__tok--q' : '')}
              style={{ '--a': a } as React.CSSProperties}
              title={`${pct(w[from + i] ?? 0)} of the attention`}
            >
              <Glyphs text={t.text} />
            </span>
          )
        })}
      </div>
    </figure>
  )
}
