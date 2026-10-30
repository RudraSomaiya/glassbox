// Screenshots of every view at 1440px and 390px in both themes, for review and
// for docs/screenshots. Serves the app with a local model (default: the
// stand-in) so the live states can be captured too.
//
//   node tests/shots.mjs [outDir] [model]
import { createRequire } from 'node:module'
import { spawn } from 'node:child_process'
import { mkdirSync } from 'node:fs'

const { chromium } = createRequire(import.meta.url)('playwright')
const out = process.argv[2] ?? '../docs/screenshots'
const model = process.argv[3] ?? 'standin-tiny'
const only = process.env.ONLY?.split(',')
mkdirSync(out, { recursive: true })

const port = 5198
const server = spawn('node_modules/.bin/vite', ['--port', String(port), '--strictPort'], {
  stdio: 'ignore',
  env: { ...process.env, VITE_MODEL_BASE: `/models/${model}/` },
})
for (let i = 0; ; i++) {
  if (await fetch(`http://localhost:${port}/`).then((r) => r.ok, () => false)) break
  if (i > 100) throw new Error('vite did not start')
  await new Promise((r) => setTimeout(r, 200))
}
const base = `http://localhost:${port}/`

const widths = [
  [1440, 900],
  [390, 844],
]
const themes = ['light', 'dark']

async function shoot(page, name) {
  await page.waitForTimeout(400)
  await page.screenshot({ path: `${out}/${name}.png`, fullPage: true })
  console.log('saved', name)
}

const browser = await chromium.launch()
try {
  for (const [w, h] of widths) {
    for (const theme of themes) {
      const ctx = await browser.newContext({ viewport: { width: w, height: h }, colorScheme: theme, reducedMotion: 'reduce' })
      const page = await ctx.newPage()
      page.on('pageerror', (e) => console.log('[pageerror]', e.message))
      const tag = `${w}-${theme}`
      for (const route of ['', 'explain', 'bench']) {
        const name = `${route || 'landing'}-${tag}`
        if (only && !only.includes(route || 'landing')) continue
        await page.goto(base + '#/' + route)
        await page.waitForLoadState('load')
        await page.waitForTimeout(800)
        await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight))
        await page.waitForTimeout(300)
        await page.evaluate(() => window.scrollTo(0, 0))
        await shoot(page, name)
      }
      if (!only || only.includes('play')) {
        await page.goto(base + '#/play')
        await shoot(page, `play-empty-${tag}`)
        await page.getByRole('button', { name: /Run it in your browser/ }).click()
        await page.getByRole('button', { name: 'Write' }).waitFor({ timeout: 120000 })
        await page.getByRole('button', { name: 'Write' }).click()
        await page.getByRole('button', { name: 'Write' }).waitFor({ timeout: 120000 })
        const frames = page.locator('.frame')
        await frames.nth(6).click()
        await page.waitForSelector('.lens', { timeout: 30000 })
        await shoot(page, `play-layers-${tag}`)
        await page.getByRole('tab', { name: 'Attention' }).click()
        await shoot(page, `play-attention-${tag}`)
        await page.keyboard.press('Escape')
        await page.getByRole('tab', { name: 'Memory' }).click()
        await shoot(page, `play-memory-${tag}`)
        await page.getByRole('tab', { name: 'Sampling' }).click()
        await shoot(page, `play-sampling-${tag}`)
        await page.getByRole('tab', { name: 'Batching' }).click()
        await shoot(page, `play-batching-${tag}`)
        await page.getByRole('tab', { name: 'Speculative' }).click()
        await shoot(page, `play-speculative-${tag}`)
      }
      await ctx.close()
    }
  }
} finally {
  await browser.close()
  server.kill()
}
