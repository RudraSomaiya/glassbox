// Bundled replay traces must be well formed: the views trust them blindly.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readdirSync, readFileSync } from 'node:fs'

const dir = new URL('../../src/traces/', import.meta.url)
const load = (f) => JSON.parse(readFileSync(new URL(f, dir), 'utf8'))

for (const f of readdirSync(dir).filter((f) => f.endsWith('.json'))) {
  test(`trace ${f} names its model and whether it is a stand-in`, () => {
    const t = load(f)
    assert.equal(typeof t.model, 'string')
    assert.equal(typeof t.standin, 'boolean')
  })
}

test('hero token stream: probabilities in range, alternatives sorted', () => {
  for (const tok of load('hero.json').tokens) {
    assert.ok(tok.p >= 0 && tok.p <= 1, `p out of range at ${tok.index}`)
    const ps = tok.alts.map((a) => a.p)
    assert.deepEqual(ps, [...ps].sort((a, b) => b - a), `alts not sorted at ${tok.index}`)
  }
})

test('anatomy: one lens entry and one attention map per layer', () => {
  const a = load('anatomy.json')
  assert.equal(a.layers.length, a.config.layers)
  assert.equal(a.attention.length, a.config.layers)
  assert.equal(a.attention[0].length, a.config.heads)
  assert.equal(a.attention[0][0].length, a.tokens.length)
})

for (const f of ['speculative.json', 'lookup.json']) {
  test(`${f}: accepted drafts never exceed drafted`, () => {
    const s = load(f)
    for (const r of s.rounds) assert.ok(r.accepted <= r.drafted.length)
  })
}

test('scheduler: every batch entry refers to a recorded request', () => {
  const s = load('scheduler.json')
  const ids = new Set(s.requests.map((r) => r.id))
  for (const step of s.steps) for (const [seq] of step.batch) assert.ok(ids.has(seq), `unknown sequence ${seq}`)
})
