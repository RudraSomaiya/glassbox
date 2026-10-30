// Headless Chromium runs the WebAssembly engine on a model under models/ and
// checks it reproduces the native greedy continuation from golden.json for
// three prompts, then measures Q8_0 decode speed for bench/results/browser.json.
//
//   npm run test:wasm [-- standin-135M]
// Needs: scripts/build_wasm.sh, and web/public/models -> ../../models.
import { createRequire } from 'node:module'
const { chromium } = createRequire(import.meta.url)('playwright')
import { spawn } from 'node:child_process'
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs'
import assert from 'node:assert/strict'

const model = process.argv[2] ?? 'standin-135M'
const golden = JSON.parse(readFileSync(new URL(`../../models/${model}/golden.json`, import.meta.url)))
const cases = golden.cases.slice(0, 3).map((c) => ({ ids: c.ids, want: c.greedy.slice(0, 32) }))

const port = 5199
const server = spawn('npx', ['vite', '--port', String(port), '--strictPort'], { stdio: 'ignore' })
for (let i = 0; ; i++) {
  if (await fetch(`http://localhost:${port}/`).then((r) => r.ok, () => false)) break
  if (i > 100) throw new Error('vite did not start')
  await new Promise((r) => setTimeout(r, 200))
}

const browser = await chromium.launch()
try {
  const page = await browser.newPage()
  page.on('console', (m) => console.log('[page]', m.text()))
  await page.goto(`http://localhost:${port}/`)
  const result = await page.evaluate(
    async ({ model, cases }) => {
      const m = await import('/src/wasm/glassbox.js')
      await m.default()
      const get = (f) => fetch(`/models/${model}/${f}`)
      const [config, tok, weights] = await Promise.all([
        get('config.json').then((r) => r.text()),
        get('tokenizer.json').then((r) => r.text()),
        get('model.safetensors').then((r) => r.arrayBuffer()),
      ])
      const bytes = new Uint8Array(weights)
      const greedy = '{"temperature":0}'
      const out = []
      const f32 = new m.Engine(config, tok, bytes, 'f32', 256)
      for (const c of cases) {
        f32.reset()
        f32.feed(new Uint32Array(c.ids), 1)
        const got = []
        for (let i = 0; i < c.want.length; i++) {
          const t = f32.sample(greedy)
          got.push(t)
          f32.feed(new Uint32Array([t]), 1)
        }
        out.push(got)
      }
      f32.free()
      // Speed: Q8_0, the configuration the site runs.
      const q8 = new m.Engine(config, tok, bytes, 'q8_0', 512)
      q8.feed(new Uint32Array(cases[0].ids), 1)
      let t = q8.sample(greedy)
      const t0 = performance.now()
      for (let i = 0; i < 64; i++) {
        q8.feed(new Uint32Array([t]), 1)
        t = q8.sample(greedy)
      }
      const tokPerSec = 64 / ((performance.now() - t0) / 1000)
      return { out, tokPerSec, simd: JSON.parse(q8.info()).simd, ua: navigator.userAgent }
    },
    { model, cases },
  )
  cases.forEach((c, i) => assert.deepEqual(result.out[i], c.want, `prompt ${i}: browser greedy differs from native`))
  console.log(`${cases.length} prompts: browser greedy output identical to native (32 tokens each)`)
  console.log(`Q8_0 decode in headless Chromium: ${result.tokPerSec.toFixed(1)} tok/s`)
  mkdirSync(new URL('../../bench/results/', import.meta.url), { recursive: true })
  writeFileSync(
    new URL('../../bench/results/browser.json', import.meta.url),
    JSON.stringify(
      {
        engine: 'glassbox-wasm',
        model,
        dtype: 'Q8_0',
        threads: 1,
        simd128: result.simd,
        tg64_tok_s: result.tokPerSec,
        note: 'headless Chromium on a 4-vCPU cloud VM; real laptops will differ',
        browser: result.ua,
        date: new Date().toISOString(),
      },
      null,
      2,
    ) + '\n',
  )
} finally {
  await browser.close()
  server.kill()
}
