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

/** A token in running prose: quoted, or named when it is only whitespace. */
export function describe(text: string): string {
  if (text === '\n') return 'a line break'
  if (/^\n+$/.test(text)) return 'line breaks'
  if (/^ +$/.test(text)) return 'a space'
  return `“${text.trim() || text}”`
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
  /** Lay out every token but show only the first `visible`: a replay that
   * reserves its final height instead of pushing the page down as it grows. */
  visible?: number
}

/**
 * Generated text as a strip of film. Each token is a frame; its density is
 * the probability the model gave it (clear = sure, dense = a long shot).
 *
 * Keyboard: the strip is one tab stop; arrow keys move between frames, Enter
 * opens the loupe and moves into its alternatives, Escape comes back.
 */
export function FilmStrip({ tokens, onFork, onInspect, selected, label, live, visible }: Props) {
  const [open, setOpen] = useState<number | null>(null)
  const [pinned, setPinned] = useState(false)
  const [focus, setFocus] = useState<number | null>(null)
  const stripRef = useRef<HTMLDivElement>(null)
  const loupeRef = useRef<HTMLElement>(null)
  const tok = open === null ? null : (tokens.find((t) => t.index === open) ?? null)
  const [anchor, setAnchor] = useState<{ x: number; y: number; flip: boolean } | null>(null)
  const current = focus ?? tokens[0]?.index ?? null

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

  // Keep the newest frame in view when the strip scrolls inside itself (phones).
  useEffect(() => {
    const strip = stripRef.current?.firstElementChild
    if (live && strip) strip.scrollTop = strip.scrollHeight
  }, [live, tokens.length])

  // A preview on hover or keyboard focus; on a phone the loupe is a sheet that
  // would land under the finger mid-tap, so there it opens on a tap only.
  const preview = (i: number | null) => {
    if (!pinned && matchMedia('(min-width: 641px)').matches) setOpen(i)
  }

  const frame = (i: number) => stripRef.current?.querySelector<HTMLElement>(`[data-index="${i}"]`)
  const close = (refocus: boolean) => {
    const was = open
    setOpen(null)
    setPinned(false)
    if (refocus && was !== null) frame(was)?.focus()
  }

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && open !== null) close(stripRef.current?.contains(document.activeElement) ?? false)
    }
    // A pinned loupe goes away when you work elsewhere on the page, such as switching views.
    const onDown = (e: PointerEvent) => {
      if (open !== null && !stripRef.current?.contains(e.target as Node)) close(false)
    }
    addEventListener('keydown', onKey)
    addEventListener('pointerdown', onDown)
    return () => {
      removeEventListener('keydown', onKey)
      removeEventListener('pointerdown', onDown)
    }
  })

  const onKeyDown = (e: React.KeyboardEvent) => {
    const at = tokens.findIndex((t) => t.index === current)
    const step: Record<string, number> = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }
    let next = -1
    if (e.key in step) next = Math.min(tokens.length - 1, Math.max(0, at + step[e.key]))
    else if (e.key === 'Home') next = 0
    else if (e.key === 'End') next = tokens.length - 1
    else if ((e.key === 'Enter' || e.key === ' ') && current !== null) {
      e.preventDefault()
      setOpen(current)
      setPinned(true)
      onInspect?.(current)
      // Into the alternatives, once the loupe has rendered.
      requestAnimationFrame(() => loupeRef.current?.querySelector<HTMLElement>('button:not(:disabled)')?.focus())
      return
    }
    if (next < 0) return
    e.preventDefault()
    const i = tokens[next].index
    setFocus(i)
    frame(i)?.focus()
  }

  return (
    <div className="strip-wrap" ref={stripRef}>
      <div className="strip" role="list" aria-label={label} onKeyDown={onKeyDown}>
        {tokens.map((t, n) => (
          <div role="listitem" key={t.index} className={'strip__cell' + (visible !== undefined && n >= visible ? ' strip__cell--latent' : '')}>
            <button
              type="button"
              tabIndex={t.index === current ? 0 : -1}
              className={'frame' + (density(t.p) > FLIP ? ' frame--dense' : '') + (selected === t.index ? ' frame--selected' : '')}
              style={{ '--d': density(t.p) } as React.CSSProperties}
              data-index={t.index}
              data-n={t.index % 5 === 0 ? t.index : undefined}
              aria-label={`${spoken(t.text)}, ${pct(t.p)}`}
              aria-expanded={open === t.index}
              onMouseEnter={() => preview(t.index)}
              onMouseLeave={() => preview(null)}
              onFocus={(e) => {
                setFocus(t.index)
                if (e.currentTarget.matches(':focus-visible')) preview(t.index)
              }}
              onClick={() => {
                setFocus(t.index)
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
      <p className="sr-only" role="status">
        {live ? (tokens.length % 10 === 0 ? `${tokens.length} tokens written` : '') : tokens.length ? `${tokens.length} tokens. Arrow keys move between them; Enter shows the alternatives.` : ''}
      </p>
      {tok && (
        <Loupe
          ref={loupeRef}
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
          onClose={() => close(true)}
          pinned={pinned}
        />
      )}
    </div>
  )
}

function Loupe({
  ref,
  token,
  anchor,
  onFork,
  onClose,
  pinned,
}: {
  ref: React.Ref<HTMLElement>
  token: Token
  anchor: { x: number; y: number; flip: boolean } | null
  onFork?: (alt: Alt) => void
  onClose: () => void
  pinned: boolean
}) {
  // The chosen token belongs on the wedge even when it wasn't in the top ten.
  const rows = token.alts.some((a) => a.id === token.id)
    ? token.alts
    : [...token.alts, { id: token.id, p: token.p, text: token.text }]
  const top = Math.max(...rows.map((a) => a.p))
  const options = Math.exp(token.entropy)
  return (
    <aside
      ref={ref}
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
          {token.entropy > 0 && (
            <>
              , choosing between about {options < 10 ? options.toFixed(1) : Math.round(options).toLocaleString()} equally likely
              options
            </>
          )}
        </p>
        <ol className="wedge">
          {rows.map((a) => {
            const chosen = a.id === token.id
            const d = density(a.p)
            return (
              <li key={a.id}>
                <button
                  type="button"
                  className={'wedge__row' + (chosen ? ' wedge__row--chosen' : '')}
                  disabled={!onFork || chosen}
                  onClick={() => onFork?.(a)}
                  aria-label={`${spoken(a.text)}, ${pct(a.p)}${chosen ? ', the token it chose' : ', fork the story with this'}`}
                >
                  <span
                    className={'wedge__swatch' + (d > FLIP ? ' wedge__swatch--dense' : '')}
                    style={{ '--d': d } as React.CSSProperties}
                    aria-hidden
                  >
                    <Glyphs text={a.text} />
                  </span>
                  <span className="wedge__track" aria-hidden>
                    <span className="wedge__bar" style={{ '--w': a.p / top } as React.CSSProperties} />
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
