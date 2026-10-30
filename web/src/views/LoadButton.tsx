import { engine, MODEL_BYTES, useEngine } from '../engine/client'

const mb = (b: number) => `${Math.round(b / 1e6)} MB`

/** Starts the model download; shows progress while it runs. */
export function LoadButton() {
  const s = useEngine()
  if (s.status === 'idle' || s.status === 'error')
    return (
      <>
        <button type="button" className="btn" onClick={() => engine.load()}>
          Run it in your browser ({mb(MODEL_BYTES)})
        </button>
        {s.error && (
          <p className="load-error" role="alert">
            The model didn’t load: {s.error} Check your connection and try again.
          </p>
        )}
      </>
    )
  if (s.status === 'loading') {
    const p = s.progress
    const frac = p && p.total ? p.loaded / p.total : 0
    const label =
      p?.label === 'weights'
        ? `${p.cached ? 'Reading cached weights' : 'Downloading weights'} ${mb(p.loaded)} of ${mb(p.total)}`
        : p?.label === 'quantizing'
          ? 'Quantizing to 8 bits'
          : 'Starting the engine'
    return (
      <div className="load" role="status">
        <span className="load__bar" style={{ '--f': frac } as React.CSSProperties} aria-hidden />
        <span>{label}</span>
      </div>
    )
  }
  if (s.status === 'running')
    return (
      <button type="button" className="btn" onClick={() => engine.stop()}>
        Stop
      </button>
    )
  return null
}
