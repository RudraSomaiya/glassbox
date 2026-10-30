import { useEffect, useState } from 'react'
import { CONTEXT, engine, useEngine } from '../engine/client'
import type { Sampler } from '../engine/protocol'
import { FilmStrip } from '../views/FilmStrip'
import { Attention, Lens } from '../views/Inspector'
import { LoadButton } from '../views/LoadButton'
import { Memory } from '../views/Memory'
import { Sampling } from '../views/Sampling'
import { Scheduler, type SchedTrace } from '../views/Scheduler'
import { Speculative, type SpecTrace } from '../views/Speculative'
import scheduler from '../traces/scheduler.json'
import speculative from '../traces/speculative.json'
import lookup from '../traces/lookup.json'
import hero from '../traces/hero.json'
import type { Token } from '../engine/protocol'
import './playground.css'

const VIEWS = ['Layers', 'Attention', 'Memory', 'Sampling', 'Batching', 'Speculative'] as const
type View = (typeof VIEWS)[number]

export function Playground() {
  const s = useEngine()
  const [prompt, setPrompt] = useState('Write a short poem about light passing through glass.')
  const [sampler, setSampler] = useState<Sampler>({ temperature: 0.7, top_k: 40, top_p: 0.95, min_p: 0, seed: 1 })
  const [view, setView] = useState<View>('Layers')
  const [selected, setSelected] = useState<number | null>(null)
  const ready = s.status === 'ready' || s.status === 'running'
  const all = [...s.prompt, ...s.tokens]
  const textOf = (id: number) => all.find((t) => t.id === id)?.text ?? s.tokens.flatMap((t) => t.alts).find((a) => a.id === id)?.text ?? `#${id}`

  // Inspect the step that produced the selected token: the forward pass at the token before it.
  useEffect(() => {
    if (selected !== null && selected > 0 && s.status === 'ready') engine.inspect(selected - 1)
  }, [selected, s.status])
  const chosen = selected === null ? null : (s.tokens.find((t) => t.index === selected) ?? null)
  const inspected = s.inspect && selected !== null && s.inspect.pos === selected - 1 ? s.inspect.data : null

  return (
    <div className="page play">
      <h1 className="play__title">Playground</h1>
      <form
        className="play__form"
        onSubmit={(e) => {
          e.preventDefault()
          setSelected(null)
          if (s.status === 'ready') engine.start(prompt, sampler, 200)
        }}
      >
        <label className="field">
          <span>Your message</span>
          <textarea value={prompt} rows={2} onChange={(e) => setPrompt(e.target.value)} />
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
            selected={selected}
            onInspect={(i) => setSelected(i)}
            onFork={(i, alt) => {
              setSelected(null)
              engine.fork(i, alt, sampler, 160)
            }}
          />
        ) : (
          <>
            <p className="play__empty">
              {ready
                ? 'Write a message and press Write. Each token lands as a frame on the strip.'
                : 'Load the model to start. It runs entirely on your machine; nothing is sent anywhere. Until then, here is a recorded reply; Batching and Speculative work without the model.'}
            </p>
            {!ready && (
              <div className="play__sample">
                <p className="play__sample-label">Recorded: “{hero.prompt}”</p>
                <FilmStrip tokens={hero.tokens as Token[]} label="A recorded reply" />
              </div>
            )}
          </>
        )}
        <p className="play__stats">
          {s.prefill && `Read ${s.prefill.tokens} prompt tokens in ${Math.round(s.prefill.ms)} ms. `}
          {s.tokPerSec > 0 && `Writing ${s.tokPerSec.toFixed(1)} tokens a second. `}
          {s.branches.length > 0 && `${s.branches.length} fork${s.branches.length > 1 ? 's' : ''} so far. `}
          {s.tokens.length > 0 && selected === null && 'Click a frame to look inside the step that wrote it.'}
        </p>
      </section>

      <section className="play__bench" aria-labelledby="views-title">
          <h2 id="views-title" className="play__h2">
            Inside the model
          </h2>
          <div
            className="tabs"
            role="tablist"
            aria-label="Views"
            onKeyDown={(e) => {
              const step = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0
              if (!step) return
              const next = VIEWS[(VIEWS.indexOf(view) + step + VIEWS.length) % VIEWS.length]
              setView(next)
              document.getElementById(`tab-${next}`)?.focus()
            }}
          >
            {VIEWS.map((v) => (
              <button
                key={v}
                id={`tab-${v}`}
                type="button"
                role="tab"
                aria-selected={v === view}
                aria-controls="view-panel"
                tabIndex={v === view ? 0 : -1}
                className="tab"
                onClick={() => setView(v)}
              >
                {v}
              </button>
            ))}
          </div>
          <div className="play__panel" id="view-panel" role="tabpanel" aria-labelledby={`tab-${view}`}>
            {(view === 'Layers' || view === 'Attention') &&
              (inspected && chosen ? (
                view === 'Layers' ? (
                  <Lens data={inspected} chosen={chosen.id} vocabText={textOf} />
                ) : (
                  <Attention data={inspected} tokens={all} query={selected! - 1} />
                )
              ) : (
                <p className="play__empty">
                  {!s.info
                    ? 'Load the model and write something, then pick a frame to see how the model arrived at it.'
                    : selected === null
                    ? 'Pick a frame on the strip to see how the model arrived at it.'
                    : s.status === 'running'
                      ? 'Stop or wait for the writing to finish, then the model can be traced.'
                      : 'Tracing that step…'}
                </p>
              ))}
            {view === 'Batching' && <Scheduler trace={scheduler as unknown as SchedTrace} />}
            {view === 'Speculative' && (
              <div className="play__spec">
                <Speculative trace={speculative as unknown as SpecTrace} />
                <Speculative trace={lookup as unknown as SpecTrace} />
              </div>
            )}
            {(view === 'Batching' || view === 'Speculative') && scheduler.standin && (
              <p className="play__empty">
                Recorded from the native server with a small stand-in model trained locally, because the real weights could
                not be downloaded when this was recorded.
              </p>
            )}
            {view === 'Memory' &&
              (s.info && s.cache ? (
                <Memory cache={s.cache} bytesPerToken={s.info.kv_bytes_per_token} context={CONTEXT} />
              ) : (
                <p className="play__empty">Load the model and write something; the cache fills as it writes.</p>
              ))}
            {view === 'Sampling' &&
              (s.dist ? (
                <Sampling dist={s.dist} sampler={sampler} onChange={setSampler} />
              ) : (
                <p className="play__empty">Write something first; the dials act on what the model would say next.</p>
              ))}
          </div>
        </section>
    </div>
  )
}
