import { useEffect, useRef, useState } from 'react'
import type { Alt, Token } from '../engine/protocol'
import './filmstrip.css'

/** Visible form of a token: whitespace drawn, not swallowed. */
export function Glyphs({ text }: { text: string }) {
  if (text === '') return <span className="glyph-empty" aria-hidden />
  // A token that is only spaces would be invisible: draw the space.
  if (/^ +$/.test(text))
    return (
      <svg className="glyph-sp" viewBox="0 0 14 8" aria-hidden>
        <path d="M1 2v5h12V2" fill="none" stroke="currentColor" strokeWidth="1.6" />
      </svg>
    )
  const parts = text.split(/(\n)/)
  return (
    <>
      {parts.map((p, i) =>
        p === '\n' ? (
          <svg key={i} className="glyph-nl" viewBox="0 0 12 12" aria-hidden>
            <path d="M10 2v5H3M5.5 4.5 3 7l2.5 2.5" fill="none" stroke="currentColor" strokeWidth="1.6" />
          </svg>
        ) : (
          <span key={i}>{p.replace(/\t/g, '    ')}</span>
        ),
      )}
    </>
  )
}

/**
 * Optical density of a frame that lets through fraction p of the light,
 * D = log10(1/p), read off an 11-step wedge (steps of D = 0.3) and scaled so
 * D = 3 (p = 0.001) is fully dense. Quantizing, as a real step wedge does,
 * keeps every tone clear of the point where text has to change colour.
 */
export function density(p: number): number {
  const d = Math.min(1, Math.max(0, -Math.log10(Math.max(p, 1e-6)) / 3))
  return Math.round(d * 10) / 10
}

/**
 * Density from which frame text turns light. Between the wedge steps either
 * side of it, black and white text both keep at least 4.5:1 on the
 * light-table film; the darkroom film is dark enough for light text throughout.
 */
export const FLIP = 0.15

export function pct(p: number): string {
  if (p >= 0.995) return '>99%'
  if (p < 0.001) return '<0.1%'
  return p < 0.1 ? `${(p * 100).toFixed(1)}%` : `${Math.round(p * 100)}%`
}

function spoken(text: string): string {
  return text.replace(/\n/g, ' newline ').trim() || 'space'
}

type Props = {
  tokens: Token[]
  /** batch index of the first token, so frame numbers match the engine's positions */
  offset?: number
  onFork?: (index: number, alt: Alt) => void
  onInspect?: (index: number) => void
  selected?: number | null
  label: string
  live?: boolean
}

/**
 * Generated text as a strip of film. Each token is a frame; its density is
 * the probability the model gave it (clear = sure, dense = a long shot).
 */
export function FilmStrip({ tokens, onFork, onInspect, selected, label, live }: Props) {
  const [open, setOpen] = useState<number | null>(null)
  const [pinned, setPinned] = useState(false)
  const stripRef = useRef<HTMLDivElement>(null)
  const tok = open === null ? null : (tokens.find((t) => t.index === open) ?? null)
  const [anchor, setAnchor] = useState<{ x: number; y: number; flip: boolean } | null>(null)

  // Place the loupe under the frame it magnifies, flipped left near the right edge.
  useEffect(() => {
    const wrap = stripRef.current
    const el = wrap?.querySelector<HTMLElement>(`[data-index="${open}"]`)
    if (!wrap || !el) return setAnchor(null)
    const w = wrap.getBoundingClientRect()
    const r = el.getBoundingClientRect()
    const x = r.left - w.left
    setAnchor({ x, y: r.bottom - w.top + 14, flip: x > w.width - 360 })
  }, [open, tokens.length])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setOpen(null)
        setPinned(false)
      }
    }
    addEventListener('keydown', onKey)
    return () => removeEventListener('keydown', onKey)
  }, [])

  return (
    <div className="strip-wrap" ref={stripRef}>
      <div className="strip" role="list" aria-label={label} aria-live={live ? 'polite' : undefined}>
        {tokens.map((t) => (
          <div role="listitem" key={t.index} className="strip__cell">
            <button
              type="button"
              className={'frame' + (density(t.p) > FLIP ? ' frame--dense' : '') + (selected === t.index ? ' frame--selected' : '')}
              style={{ '--d': density(t.p) } as React.CSSProperties}
              data-index={t.index}
              data-n={t.index % 5 === 0 ? t.index : undefined}
              aria-label={`${spoken(t.text)}, ${pct(t.p)}`}
              aria-expanded={open === t.index}
              onMouseEnter={() => !pinned && setOpen(t.index)}
              onMouseLeave={() => !pinned && setOpen(null)}
              onFocus={() => setOpen(t.index)}
              onClick={() => {
                setOpen(t.index)
                setPinned(true)
                onInspect?.(t.index)
              }}
            >
              <Glyphs text={t.text} />
            </button>
          </div>
        ))}
        {live && <div className="strip__gate" aria-hidden />}
      </div>
      {tok && (
        <Loupe
          token={tok}
          anchor={anchor}
          onFork={
            onFork &&
            ((alt) => {
              setOpen(null)
              setPinned(false)
              onFork(tok.index, alt)
            })
          }
          onClose={() => {
            setOpen(null)
            setPinned(false)
          }}
          pinned={pinned}
        />
      )}
    </div>
  )
}

function Loupe({
  token,
  anchor,
  onFork,
  onClose,
  pinned,
}: {
  token: Token
  anchor: { x: number; y: number; flip: boolean } | null
  onFork?: (alt: Alt) => void
  onClose: () => void
  pinned: boolean
}) {
  return (
    <aside
      className={'loupe' + (anchor?.flip ? ' loupe--flip' : '')}
      style={anchor ? ({ '--x': `${anchor.x}px`, '--y': `${anchor.y}px` } as React.CSSProperties) : undefined}
      aria-label={`What else could have come at position ${token.index}`}
      onMouseLeave={() => !pinned && onClose()}
    >
      <div className="loupe__lens" aria-hidden style={{ '--len': Math.max(3, [...token.text].length) } as React.CSSProperties}>
        <span>
          <Glyphs text={token.text} />
        </span>
      </div>
      <div className="loupe__body">
        <p className="loupe__head">
          Position {token.index}: chose this at <strong>{pct(token.p)}</strong>
          {token.entropy > 0 && <>, uncertainty {token.entropy.toFixed(2)} nats</>}
        </p>
        <ol className="wedge">
          {token.alts.map((a) => {
            const chosen = a.id === token.id
            return (
              <li key={a.id}>
                <button
                  type="button"
                  className={'wedge__row' + (chosen ? ' wedge__row--chosen' : '')}
                  disabled={!onFork || chosen}
                  onClick={() => onFork?.(a)}
                  title={chosen ? 'The token the model chose' : 'Rewrite the story from here with this token'}
                >
                  <span className="wedge__bar" style={{ '--p': a.p } as React.CSSProperties} aria-hidden />
                  <span className="wedge__text">
                    <Glyphs text={a.text} />
                  </span>
                  <span className="wedge__p">{pct(a.p)}</span>
                </button>
              </li>
            )
          })}
        </ol>
        {onFork && <p className="loupe__hint">Pick another token to fork the story from here.</p>}
      </div>
      <button type="button" className="loupe__close" onClick={onClose} aria-label="Close the loupe">
        <svg viewBox="0 0 12 12" width="14" height="14" aria-hidden>
          <path d="M2 2l8 8M10 2l-8 8" stroke="currentColor" strokeWidth="1.8" />
        </svg>
      </button>
    </aside>
  )
}
