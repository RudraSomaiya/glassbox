import { useEffect, useState } from 'react'
import { engine, greedy, MODEL_BYTES, useEngine } from '../engine/client'
import type { Token } from '../engine/protocol'
import { FilmStrip } from '../views/FilmStrip'
import { LoadButton } from '../views/LoadButton'
import hero from '../traces/hero.json'
import './landing.css'

const PROMPT = hero.prompt

/** Replays a recorded run at the speed it was recorded, then holds and loops. */
function useReplay(tokens: Token[], perSecond = 14) {
  const [n, setN] = useState(0)
  useEffect(() => {
    if (matchMedia('(prefers-reduced-motion: reduce)').matches) {
      setN(tokens.length)
      return
    }
    const id = setInterval(() => setN((k) => (k >= tokens.length + perSecond * 4 ? 0 : k + 1)), 1000 / perSecond)
    return () => clearInterval(id)
  }, [tokens, perSecond])
  return tokens.slice(0, Math.min(n, tokens.length))
}

export function Landing() {
  const s = useEngine()
  const replay = useReplay(hero.tokens as Token[])
  const live = s.tokens.length > 0
  const tokens = live ? s.tokens : replay

  useEffect(() => {
    // Once the model is loaded, the hero runs it for real.
    if (s.status === 'ready' && !live) engine.start(PROMPT, greedy, 96)
  }, [s.status, live])

  return (
    <div className="page landing">
      <section className="hero" aria-labelledby="hero-title">
        <h1 id="hero-title">Watch a language model write</h1>
        <p className="hero__lede">
          Glassbox is an inference engine I wrote from scratch in Rust. Here it runs a small model in your browser.
          Every frame on the strip is one token, and the clearer the frame, the surer the model was.
        </p>
        <div className="hero__run">
          <p className="hero__prompt">
            <span className="sr-only">Prompt: </span>
            {PROMPT}
          </p>
          <FilmStrip
            tokens={tokens}
            label={live ? 'The model’s answer, live' : 'A recorded answer, replaying'}
            live={s.status === 'running'}
            onFork={live ? (i, alt) => engine.fork(i, alt, greedy, 80) : undefined}
          />
          <div className="hero__actions">
            <LoadButton />
            <a className="btn btn--quiet" href="#/play">
              Open the playground
            </a>
            <p className="hero__note">
              {live
                ? `Live: SmolLM2-135M, ${s.tokPerSec.toFixed(0)} tokens a second on your machine. Hover a frame for the alternatives; click one to fork.`
                : hero.standin
                  ? 'Replaying a run recorded with stand-in weights, so the words are noise. Run the real model to see real text.'
                  : `Replaying a recorded run. The live model is a ${Math.round(MODEL_BYTES / 1e6)} MB download, cached after the first time.`}
            </p>
          </div>
        </div>
      </section>
    </div>
  )
}
