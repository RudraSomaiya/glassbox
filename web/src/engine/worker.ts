// The engine's Web Worker: downloads weights once (Cache API), owns the
// WebAssembly engine, and streams tokens back to the page.
import init, { Engine } from '../wasm/glassbox.js'
import type { Alt, FromWorker, ToWorker, Token } from './protocol'

let engine: Engine | null = null
let running = 0 // generation id; bumping it stops the current loop
let eos = 2
const decoder = new TextDecoder()
const post = (m: FromWorker) => postMessage(m)

const CACHE = 'glassbox-weights-v1'

async function fetchCached(url: string, label: string, expect = 0): Promise<Uint8Array> {
  const cache = await caches.open(CACHE).catch(() => null)
  const hit = await cache?.match(url)
  if (hit) {
    const buf = new Uint8Array(await hit.arrayBuffer())
    post({ type: 'progress', label, loaded: buf.length, total: buf.length, cached: true })
    return buf
  }
  const res = await fetch(url)
  if (!res.ok || !res.body) throw new Error(`${label}: HTTP ${res.status} from ${url}`)
  const total = Number(res.headers.get('content-length')) || expect
  const reader = res.body.getReader()
  const parts: Uint8Array[] = []
  let loaded = 0
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    parts.push(value)
    loaded += value.length
    post({ type: 'progress', label, loaded, total, cached: false })
  }
  const buf = new Uint8Array(loaded)
  let off = 0
  for (const p of parts) {
    buf.set(p, off)
    off += p.length
  }
  await cache?.put(url, new Response(buf)).catch(() => {})
  return buf
}

function text(id: number): string {
  return decoder.decode(engine!.token_bytes(id))
}

type Summary = { top: [number, number][]; entropy: number }

function alts(s: Summary): Alt[] {
  return s.top.map(([id, p]) => ({ id, p, text: text(id) }))
}

// Streaming UTF-8: a token can end mid-character, so text is attributed to
// the token that completes it.
let stream = new TextDecoder()

async function generate(gen: number, first: Summary, max: number, sampler: object, startIndex: number) {
  let summary = first
  for (let i = 0; i < max && gen === running; i++) {
    const id = engine!.sample(JSON.stringify(sampler))
    const chosen = summary.top.find(([t]) => t === id)
    const tok: Token = {
      index: startIndex + i,
      id,
      text: stream.decode(engine!.token_bytes(id), { stream: true }),
      p: chosen ? chosen[1] : probOf(id),
      entropy: summary.entropy,
      alts: alts(summary),
      special: engine!.is_special(id),
    }
    if (id === eos) {
      post({ type: 'done', reason: 'eos' })
      return
    }
    if (engine!.len() + 1 >= ctx) {
      post({ type: 'token', token: tok })
      post({ type: 'done', reason: 'length' })
      return
    }
    const t0 = performance.now()
    summary = JSON.parse(engine!.feed(new Uint32Array([id]), 10))
    post({ type: 'token', token: tok, ms: performance.now() - t0 })
    // Yield so a 'stop' or 'fork' message can arrive between tokens.
    await new Promise((r) => setTimeout(r, 0))
  }
  if (gen === running) post({ type: 'done', reason: 'max' })
}

function probOf(id: number): number {
  const l = engine!.logits()
  let max = -Infinity
  for (const v of l) max = Math.max(max, v)
  let sum = 0
  for (const v of l) sum += Math.exp(v - max)
  return Math.exp(l[id] - max) / sum
}

let ctx = 1024

onmessage = async (e: MessageEvent<ToWorker>) => {
  const m = e.data
  try {
    if (m.type === 'load') {
      await init()
      const [config, tokenizer] = await Promise.all([
        fetchCached(m.base + 'config.json', 'config'),
        fetchCached(m.base + 'tokenizer.json', 'tokenizer'),
      ])
      const weights = await fetchCached(m.base + 'model.safetensors', 'weights', m.expectBytes)
      post({ type: 'progress', label: 'quantizing', loaded: 0, total: 0, cached: false })
      ctx = m.context
      engine = new Engine(decoder.decode(config), decoder.decode(tokenizer), weights, m.quant, ctx)
      const info = JSON.parse(engine.info())
      eos = info.eos
      post({ type: 'ready', info })
    } else if (m.type === 'start') {
      const gen = ++running
      engine!.reset()
      stream = new TextDecoder()
      const prompt = m.chat ? engine!.chat(m.prompt) : engine!.encode(m.prompt)
      const promptTokens = Array.from(prompt).map((id) => ({ id, text: text(id), special: engine!.is_special(id) }))
      post({ type: 'prompt', tokens: promptTokens })
      const t0 = performance.now()
      const first: Summary = JSON.parse(engine!.feed(prompt, 10))
      post({ type: 'prefill', ms: performance.now() - t0, tokens: prompt.length })
      await generate(gen, first, m.max, m.sampler, prompt.length)
    } else if (m.type === 'fork') {
      const gen = ++running
      // Keep the first `keep` tokens, then force the chosen alternative.
      engine!.truncate(m.keep, 10)
      stream = new TextDecoder()
      const s: Summary = JSON.parse(engine!.feed(new Uint32Array([m.token]), 10))
      post({ type: 'forked', keep: m.keep, token: { index: m.keep, id: m.token, text: text(m.token), p: m.p, entropy: 0, alts: m.alts, special: false } })
      await generate(gen, s, m.max, m.sampler, m.keep + 1)
    } else if (m.type === 'stop') {
      running++
      post({ type: 'done', reason: 'stopped' })
    } else if (m.type === 'inspect') {
      post({ type: 'inspect', pos: m.pos, data: JSON.parse(engine!.inspect(m.pos, 5)) })
    } else if (m.type === 'dist') {
      // The raw next-token logits, top k, for the sampling playground.
      const l = engine!.logits()
      const idx = Array.from(l.keys()).sort((a, b) => l[b] - l[a]).slice(0, m.k)
      post({ type: 'dist', top: idx.map((id) => ({ id, p: 0, logit: l[id], text: text(id) })) })
    } else if (m.type === 'cache') {
      post({ type: 'cache', data: JSON.parse(engine!.cache()) })
    }
  } catch (err) {
    post({ type: 'error', message: err instanceof Error ? err.message : String(err) })
  }
}
