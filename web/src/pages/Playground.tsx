import { useState } from 'react'
import { engine, useEngine } from '../engine/client'
import type { Sampler } from '../engine/protocol'
import { FilmStrip } from '../views/FilmStrip'
import { LoadButton } from '../views/LoadButton'
import './playground.css'

export function Playground() {
  const s = useEngine()
  const [prompt, setPrompt] = useState('Write a short poem about light passing through glass.')
  const [sampler, setSampler] = useState<Sampler>({ temperature: 0.7, top_k: 40, top_p: 0.95, min_p: 0, seed: 1 })
  const ready = s.status === 'ready' || s.status === 'running'

  return (
    <div className="page play">
      <h1 className="play__title">Playground</h1>
      <form
        className="play__form"
        onSubmit={(e) => {
          e.preventDefault()
          if (s.status === 'ready') engine.start(prompt, sampler, 200)
        }}
      >
        <label className="field">
          <span>Your message</span>
          <textarea value={prompt} rows={2} onChange={(e) => setPrompt(e.target.value)} />
        </label>
        <label className="field field--range">
          <span>
            Temperature <output>{sampler.temperature.toFixed(2)}</output>
          </span>
          <input
            type="range"
            min={0}
            max={1.5}
            step={0.05}
            value={sampler.temperature}
            onChange={(e) => setSampler({ ...sampler, temperature: Number(e.target.value) })}
          />
        </label>
        <div className="play__actions">
          {ready ? (
            s.status === 'running' ? (
              <button type="button" className="btn" onClick={() => engine.stop()}>
                Stop
              </button>
            ) : (
              <button type="submit" className="btn">
                Write
              </button>
            )
          ) : (
            <LoadButton />
          )}
        </div>
      </form>

      <section className="play__stream" aria-labelledby="stream-title">
        <h2 id="stream-title" className="sr-only">
          Token stream
        </h2>
        {s.tokens.length ? (
          <FilmStrip
            tokens={s.tokens}
            label="Generated tokens"
            live={s.status === 'running'}
            onFork={(i, alt) => engine.fork(i, alt, sampler, 160)}
          />
        ) : (
          <p className="play__empty">
            {ready
              ? 'Write a message and press Write. Each token lands as a frame on the strip.'
              : 'Load the model to start. It runs entirely on your machine.'}
          </p>
        )}
        <p className="play__stats">
          {s.prefill && `Read ${s.prefill.tokens} prompt tokens in ${Math.round(s.prefill.ms)} ms. `}
          {s.tokPerSec > 0 && `Writing ${s.tokPerSec.toFixed(1)} tokens a second. `}
          {s.branches.length > 0 && `${s.branches.length} fork${s.branches.length > 1 ? 's' : ''} so far.`}
        </p>
      </section>
    </div>
  )
}
