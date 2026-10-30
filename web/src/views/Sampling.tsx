import type { Alt, Sampler } from '../engine/protocol'
import { Glyphs, pct } from './FilmStrip'
import './sampling.css'

/** The sampler's filters, in the engine's order: temperature, top-k, min-p, top-p. */
export function reshape(dist: Alt[], s: Sampler): number[] {
  const l = dist.map((a) => a.logit ?? Math.log(a.p))
  if (s.temperature <= 0) return l.map((_, i) => (i === 0 ? 1 : 0))
  const max = Math.max(...l)
  let p = l.map((v) => Math.exp((v - max) / s.temperature))
  const keep = p.map((_, i) => s.top_k === 0 || i < s.top_k)
  p = p.map((v, i) => (keep[i] ? v : 0))
  let sum = p.reduce((a, b) => a + b, 0)
  p = p.map((v) => v / sum)
  if (s.min_p > 0) p = p.map((v) => (v >= p[0] * s.min_p ? v : 0))
  if (s.top_p < 1) {
    let acc = 0
    let cut = false
    p = p.map((v) => {
      if (cut) return 0
      acc += v
      if (acc >= s.top_p) cut = true
      return v
    })
  }
  sum = p.reduce((a, b) => a + b, 0)
  return p.map((v) => v / sum)
}

const Slider = (props: { label: string; value: number; min: number; max: number; step: number; show: string; onChange: (v: number) => void }) => (
  <label className="dial">
    <span>
      {props.label} <output>{props.show}</output>
    </span>
    <input
      type="range"
      min={props.min}
      max={props.max}
      step={props.step}
      value={props.value}
      onChange={(e) => props.onChange(Number(e.target.value))}
    />
  </label>
)

/**
 * The next-token distribution after the last token, reshaped live by the
 * sampling dials: ghost bars are the raw model (temperature 1), solid bars
 * what the sampler will actually draw from.
 */
export function Sampling({ dist, sampler, onChange }: { dist: Alt[]; sampler: Sampler; onChange: (s: Sampler) => void }) {
  const raw = reshape(dist, { temperature: 1, top_k: 0, top_p: 1, min_p: 0, seed: 0 })
  const now = reshape(dist, sampler)
  const shown = 16
  const max = Math.max(...raw.slice(0, shown), ...now.slice(0, shown))
  const alive = now.filter((p) => p > 0).length
  return (
    <figure className="samp" aria-labelledby="samp-cap">
      <figcaption id="samp-cap">
        <strong>Choosing the next token.</strong> {alive === 1 ? 'Only one candidate survives' : `${alive} candidates survive`}{' '}
        the dials; the sampler draws from the solid bars. Ghost bars are the model’s raw opinion.
      </figcaption>
      <div className="samp__dials">
        <Slider label="Temperature" value={sampler.temperature} min={0} max={2} step={0.05} show={sampler.temperature === 0 ? '0 (greedy)' : sampler.temperature.toFixed(2)} onChange={(v) => onChange({ ...sampler, temperature: v })} />
        <Slider label="Top-k" value={sampler.top_k} min={0} max={40} step={1} show={sampler.top_k === 0 ? 'off' : String(sampler.top_k)} onChange={(v) => onChange({ ...sampler, top_k: v })} />
        <Slider label="Top-p" value={sampler.top_p} min={0.05} max={1} step={0.05} show={sampler.top_p >= 1 ? 'off' : sampler.top_p.toFixed(2)} onChange={(v) => onChange({ ...sampler, top_p: v })} />
        <Slider label="Min-p" value={sampler.min_p} min={0} max={0.5} step={0.01} show={sampler.min_p === 0 ? 'off' : sampler.min_p.toFixed(2)} onChange={(v) => onChange({ ...sampler, min_p: v })} />
      </div>
      <ol className="samp__bars">
        {dist.slice(0, shown).map((a, i) => (
          <li key={a.id} className={now[i] === 0 ? 'samp__row samp__row--cut' : 'samp__row'}>
            <span className="samp__tok">
              <Glyphs text={a.text} />
            </span>
            <span className="samp__track" aria-hidden>
              <span className="samp__ghost" style={{ '--w': raw[i] / max } as React.CSSProperties} />
              <span className="samp__solid" style={{ '--w': now[i] / max } as React.CSSProperties} />
            </span>
            <span className="samp__p">{now[i] === 0 ? 'cut' : pct(now[i])}</span>
          </li>
        ))}
      </ol>
      <p className="samp__note">Top {dist.length} of the vocabulary shown; the long tail beyond it holds the rest of the raw probability.</p>
    </figure>
  )
}
