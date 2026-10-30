import { useEffect, useMemo, useState } from 'react'
import './replay.css'

export type SchedTrace = {
  model: string
  standin?: boolean
  requests: { id: number; arrival_ms: number; prompt_tokens: number; tokens: { t_ms: number }[] }[]
  steps: {
    step: number
    t_ms: number
    ms: number
    batch: [number, 'prefill' | 'decode', number][]
    preempted: number[]
    finished: number[]
    waiting: number
    kv_used: number
  }[]
}

/**
 * Requests as lanes on a timeline. Each forward pass is a column: wide amber
 * blocks are prompt chunks being read (prefill), thin ticks are single decode
 * tokens, and the band on top is how many tokens that pass carried.
 */
export function Scheduler({ trace }: { trace: SchedTrace }) {
  const ids = useMemo(() => trace.requests.map((r) => r.id).sort((a, b) => a - b), [trace])
  const steps = trace.steps.filter((s) => s.batch.length)
  const t0 = steps[0]?.t_ms ?? 0
  const end = Math.max(...steps.map((s) => s.t_ms + s.ms)) - t0
  const [play, setPlay] = useState(() => (matchMedia('(prefers-reduced-motion: reduce)').matches ? 1 : 0))
  useEffect(() => {
    if (play >= 1) return
    const id = requestAnimationFrame(() => setPlay((p) => Math.min(1, p + 0.004)))
    return () => cancelAnimationFrame(id)
  }, [play])

  const W = 960
  const lane = 18
  const band = 48
  const top = band + 16
  const H = top + ids.length * lane + 24
  const x = (t: number) => ((t - t0) / end) * W
  const maxBatch = Math.max(...steps.map((s) => s.batch.reduce((a, b) => a + b[2], 0)))
  const cutoff = t0 + play * end
  const shown = steps.filter((s) => s.t_ms <= cutoff)
  const last = shown[shown.length - 1]
  const preempts = steps.reduce((a, s) => a + s.preempted.length, 0)

  return (
    <figure className="replay" aria-labelledby="sched-cap">
      <figcaption id="sched-cap">
        <strong>Continuous batching.</strong> {trace.requests.length} requests arriving at different times share each forward
        pass. New prompts are read in chunks between other requests’ decode steps, so nobody waits for a long prompt to
        finish.{preempts > 0 && ` ${preempts} sequences were preempted when memory ran out and recomputed later.`}
      </figcaption>
      <div className="replay__controls">
        <button type="button" className="btn btn--quiet" onClick={() => setPlay(play >= 1 ? 0 : 1)}>
          {play >= 1 ? 'Replay' : 'Skip to end'}
        </button>
        <dl className="replay__stats">
          <div>
            <dt>Time</dt>
            <dd>{Math.round(play * end)} ms</dd>
          </div>
          <div>
            <dt>Running together</dt>
            <dd>{last ? last.batch.length : 0}</dd>
          </div>
          <div>
            <dt>Waiting</dt>
            <dd>{last ? last.waiting : 0}</dd>
          </div>
        </dl>
      </div>
      <div className="sched__scroll">
        <svg viewBox={`0 0 ${W} ${H}`} className="sched" role="img" aria-label="Timeline of forward passes per request">
          {shown.map((s) => {
            const tokens = s.batch.reduce((a, b) => a + b[2], 0)
            const h = (tokens / maxBatch) * band
            return <rect key={s.step} x={x(s.t_ms)} y={band - h} width={Math.max(1, x(s.t_ms + s.ms) - x(s.t_ms) - 0.5)} height={h} className="sched__load" />
          })}
          {ids.map((id, i) => {
            const r = trace.requests.find((q) => q.id === id)!
            const y = top + i * lane
            return (
              <g key={id}>
                <rect x={0} y={y + 2} width={W} height={lane - 4} className="sched__lane" />
                <rect x={x(r.arrival_ms)} y={y + 2} width={2} height={lane - 4} className="sched__arrive" />
              </g>
            )
          })}
          {shown.map((s) =>
            s.batch.map(([seq, kind]) => {
              const i = ids.indexOf(seq)
              if (i < 0) return null
              const y = top + i * lane
              const x0 = x(s.t_ms)
              const w = Math.max(1.2, x(s.t_ms + s.ms) - x0 - 0.5)
              return (
                <rect
                  key={`${s.step}-${seq}`}
                  x={x0}
                  y={kind === 'prefill' ? y + 3 : y + 6}
                  width={w}
                  height={kind === 'prefill' ? lane - 6 : lane - 12}
                  className={kind === 'prefill' ? 'sched__prefill' : 'sched__decode'}
                />
              )
            }),
          )}
          {shown.map((s) =>
            s.preempted.map((seq) => {
              const i = ids.indexOf(seq)
              const cx = x(s.t_ms)
              const cy = top + i * lane + lane / 2
              return <path key={`p${s.step}-${seq}`} d={`M${cx - 5},${cy - 5}L${cx + 5},${cy + 5}M${cx + 5},${cy - 5}L${cx - 5},${cy + 5}`} className="sched__preempt" />
            }),
          )}
          <line x1={x(cutoff)} x2={x(cutoff)} y1={0} y2={H - 20} className="sched__head" />
          <text x={0} y={H - 4} className="sched__axis">
            0 ms
          </text>
          <text x={W} y={H - 4} className="sched__axis" textAnchor="end">
            {Math.round(end)} ms
          </text>
        </svg>
      </div>
      <ul className="replay__legend">
        <li>
          <i className="key key--prefill" /> reading the prompt (prefill chunk)
        </li>
        <li>
          <i className="key key--decode" /> writing one token (decode)
        </li>
        <li>
          <i className="key key--arrive" /> request arrives
        </li>
        <li>
          <i className="key key--load" /> tokens in each forward pass
        </li>
      </ul>
    </figure>
  )
}
