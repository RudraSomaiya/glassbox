// Bundled replay traces must be well-formed: the views trust them blindly.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readdirSync, readFileSync } from 'node:fs'

const dir = new URL('../../src/traces/', import.meta.url)

for (const f of readdirSync(dir).filter((f) => f.endsWith('.json'))) {
  test(`trace ${f} is well formed`, () => {
    const t = JSON.parse(readFileSync(new URL(f, dir), 'utf8'))
    assert.equal(typeof t.model, 'string')
    assert.equal(typeof t.standin, 'boolean')
    for (const tok of t.tokens ?? []) {
      assert.ok(tok.p >= 0 && tok.p <= 1, `p out of range at ${tok.index}`)
      const ps = tok.alts.map((a) => a.p)
      assert.deepEqual(ps, [...ps].sort((a, b) => b - a), `alts not sorted at ${tok.index}`)
    }
  })
}
