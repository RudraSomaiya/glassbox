import { useEffect, useState } from 'react'
import { Landing } from './pages/Landing'
import { Playground } from './pages/Playground'

const ROUTES: Record<string, { title: string; page: () => React.ReactNode }> = {
  '': { title: 'Glassbox', page: Landing },
  play: { title: 'Playground', page: Playground },
}

function route(): string {
  const r = location.hash.replace(/^#\/?/, '').split(/[?#]/)[0]
  return r in ROUTES ? r : ''
}

export const GITHUB = 'https://github.com/RudraSomaiya/glassbox'

function ThemeToggle() {
  const [theme, setTheme] = useState<string | null>(() => {
    try {
      return localStorage.getItem('theme')
    } catch {
      return null
    }
  })
  useEffect(() => {
    if (theme) document.documentElement.dataset.theme = theme
    else delete document.documentElement.dataset.theme
  }, [theme])
  const dark = theme ? theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches
  const next = dark ? 'light' : 'dark'
  return (
    <button
      type="button"
      className="theme"
      aria-label={`Switch to the ${next === 'dark' ? 'darkroom (dark)' : 'light table (light)'} theme`}
      onClick={() => {
        setTheme(next)
        try {
          localStorage.setItem('theme', next)
        } catch {
          /* private mode: the choice lasts for this visit */
        }
      }}
    >
      <svg width="22" height="22" viewBox="0 0 22 22" aria-hidden>
        <circle cx="11" cy="11" r="9" fill="none" stroke="currentColor" strokeWidth="2" />
        <path d="M11 2a9 9 0 0 1 0 18z" fill="currentColor" />
      </svg>
    </button>
  )
}

export function App() {
  const [r, setR] = useState(route)
  useEffect(() => {
    const on = () => {
      setR(route())
      scrollTo(0, 0)
    }
    addEventListener('hashchange', on)
    return () => removeEventListener('hashchange', on)
  }, [])
  const { title, page: Page } = ROUTES[r]
  useEffect(() => {
    document.title = r ? `${title} · Glassbox` : 'Glassbox: watch a language model write'
  }, [r, title])
  return (
    <>
      <header className="bar">
        <a className="bar__mark" href="#/">
          Glassbox
        </a>
        <nav className="bar__links" aria-label="Main">
          <a href="#/play" aria-current={r === 'play' ? 'page' : undefined}>
            Playground
          </a>
          <a href={GITHUB}>Source</a>
        </nav>
        <ThemeToggle />
      </header>
      <main id="main">
        <Page />
      </main>
    </>
  )
}
