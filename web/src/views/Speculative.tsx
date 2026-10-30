import { useEffect, useState } from 'react'
import { Glyphs } from './FilmStrip'
import './replay.css'

export type SpecTrace = {
  model: string
  standin: boolean
  draft: string
  prompt: string
  k: number
  first: string
  rounds: { drafted: string[]; accepted: number; next: string }[]
  acceptance: number
  target_passes: number
  tokens: number
  alone_s: number
  spec_s: number
}

type Frame = { text: string; kind: 'target' | 'accepted' | 'rejected' | 'draft' }

/**
 * Replays speculative decoding round by round: the drafter's guesses land as
 * latent (ghost) frames, then the target's single verification pass develops
 * the ones it agrees with and crosses out the rest, adding its own token.
 */
export function Speculative({ trace }: { trace: SpecTrace }) {
  const n = trace.rounds.length
  const still = matchMedia('(prefers-reduced-motion: reduce)').matches
  // With reduced motion, start from the finished run; Step and Play still work.
  const [round, setRound] = useState(still ? n : 0)
  const [phase, setPhase] = useState<'draft' | 'verify'>('verify')
  const [playing, setPlaying] = useState(!still)

  useEffect(() => {
    if (!playing) return
    const id = setTimeout(
      () => {
        if (phase === 'draft') setPhase('verify')
        else if (round < n) {
          setRound(round + 1)
          setPhase('draft')
        } else {
          setRound(0)
          setPhase('verify')
        }
      },
      phase === 'draft' ? 700 : round >= n ? 2500 : 500,
    )
    return () => clearTimeout(id)
  }, [playing, phase, round, n])

  const frames: Frame[] = [{ text: trace.first, kind: 'target' }]
  let drafted = 0
  let accepted = 0
  for (let r = 0; r < round; r++) {
    const R = trace.rounds[r]
    const last = r === round - 1
    if (last && phase === 'draft') {
      R.drafted.forEach((t) => frames.push({ text: t, kind: 'draft' }))
      break
    }
    R.drafted.forEach((t, i) => frames.push({ text: t, kind: i < R.accepted ? 'accepted' : 'rejected' }))
    frames.push({ text: R.next, kind: 'target' })
    drafted += R.drafted.length
    accepted += R.accepted
  }
  const passes = Math.min(round, n) + 1
  const produced = frames.filter((f) => f.kind === 'accepted' || f.kind === 'target').length

  return (
    <figure className="replay" aria-labelledby="spec-cap">
      <figcaption id="spec-cap">
        <strong>Speculative decoding.</strong> The drafter ({trace.draft}) guesses {trace.k} tokens at a time; the target (
        {trace.model}) checks all of them in one pass. Output is identical to the target alone.
      </figcaption>
      <div className="replay__controls">
        <button type="button" className="btn btn--quiet" onClick={() => setPlaying(!playing)}>
          {playing ? 'Pause' : 'Play'}
        </button>
        <button
          type="button"
          className="btn btn--quiet"
          onClick={() => {
            setPlaying(false)
            setRound(Math.min(n, round + (phase === 'verify' ? 1 : 0)))
            setPhase(phase === 'verify' ? 'draft' : 'verify')
          }}
        >
          Step
        </button>
        <dl className="replay__stats">
          <div>
            <dt>Accepted</dt>
            <dd>{drafted ? `${Math.round((accepted / drafted) * 100)}%` : '–'}</dd>
          </div>
          <div>
            <dt>Tokens per target pass</dt>
            <dd>{(produced / passes).toFixed(2)}</dd>
          </div>
          <div>
            <dt>Whole run</dt>
            <dd>{(trace.alone_s / trace.spec_s).toFixed(2)}× the target alone</dd>
          </div>
        </dl>
      </div>
      <div className="spec__strip" aria-live="off">
        {frames.map((f, i) => (
          <span key={i} className={`spec__frame spec__frame--${f.kind}`}>
            <Glyphs text={f.text} />
          </span>
        ))}
      </div>
      <ul className="replay__legend">
        <li>
          <span className="spec__frame spec__frame--draft">draft</span> guessed, not yet checked
        </li>
        <li>
          <span className="spec__frame spec__frame--accepted">kept</span> the target agreed
        </li>
        <li>
          <span className="spec__frame spec__frame--rejected">crossed</span> the target disagreed
        </li>
        <li>
          <span className="spec__frame spec__frame--target">target</span> written by the target itself
        </li>
      </ul>
    </figure>
  )
}
