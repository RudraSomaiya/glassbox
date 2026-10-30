import { useEffect, useRef, useState } from 'react'
import type { Alt, Inspect, Sampler } from '../engine/protocol'
import anatomy from '../traces/anatomy.json'
import scheduler from '../traces/scheduler.json'
import speculative from '../traces/lookup.json'
import { Glyphs, pct } from '../views/FilmStrip'
import { Attention, Lens } from '../views/Inspector'
import { Memory } from '../views/Memory'
import { Sampling } from '../views/Sampling'
import { Scheduler, type SchedTrace } from '../views/Scheduler'
import { Speculative, type SpecTrace } from '../views/Speculative'
import './explainer.css'

type Anatomy = typeof anatomy

/** Sections fade up as they reach the reader, once. */
function useReveal() {
  const ref = useRef<HTMLElement>(null)
  const [seen, setSeen] = useState(() => matchMedia('(prefers-reduced-motion: reduce)').matches)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const io = new IntersectionObserver(([e]) => e.isIntersecting && setSeen(true), { rootMargin: '0px 0px -15% 0px' })
    io.observe(el)
    return () => io.disconnect()
  }, [])
  return [ref, seen] as const
}

function Step({ id, title, children, figure }: { id: string; title: string; children: React.ReactNode; figure: React.ReactNode }) {
  const [ref, seen] = useReveal()
  return (
    <section id={id} ref={ref} className={'step' + (seen ? ' step--in' : '')} aria-labelledby={`${id}-h`}>
      <div className="step__text">
        <h2 id={`${id}-h`}>{title}</h2>
        {children}
      </div>
      <div className="step__figure">{figure}</div>
    </section>
  )
}

/** An embedding as a strip of densities, one column per dimension. */
function Barcode({ values }: { values: number[] }) {
  const max = Math.max(...values.map(Math.abs))
  return (
    <div className="barcode" role="img" aria-label={`${values.length} numbers between ${(-max).toFixed(2)} and ${max.toFixed(2)}`}>
      {values.map((v, i) => (
        <i key={i} className={v < 0 ? 'neg' : ''} style={{ '--a': Math.abs(v) / max } as React.CSSProperties} />
      ))}
    </div>
  )
}

function Norms({ layers }: { layers: Anatomy['layers'] }) {
  const max = Math.max(...layers.map((l) => l.norm))
  return (
    <div className="norms" role="img" aria-label="Size of the hidden state after each layer, growing as layers add to it">
      {layers.map((l, i) => (
        <i key={i} style={{ '--h': l.norm / max } as React.CSSProperties} title={`layer ${i + 1}: ${l.norm.toFixed(1)}`} />
      ))}
    </div>
  )
}

export function Explainer() {
  const a = anatomy as Anatomy
  const tokens = a.tokens
  const last = tokens[tokens.length - 1]
  const winner = a.logits[0]
  // The anatomy records the lens with token texts; adapt it to the playground views' shape.
  const inspect: Inspect = {
    lens: a.layers.map((l) => ({ top: l.top.map((t) => [t.id, t.p] as [number, number]), entropy: 0 })),
    attention: a.attention as number[][][],
  }
  const texts = new Map<number, string>()
  a.layers.forEach((l) => l.top.forEach((t) => texts.set(t.id, t.text)))
  a.logits.forEach((t) => texts.set(t.id, t.text))
  const dist: Alt[] = a.logits.map((t) => ({ id: t.id, p: t.p, text: t.text, logit: t.logit }))
  const [sampler, setSampler] = useState<Sampler>({ temperature: 1, top_k: 0, top_p: 1, min_p: 0, seed: 0 })
  const kvTokens = tokens.length
  const block = 16
  const cache = { blocks: Array.from({ length: Math.ceil(kvTokens / block) }, (_, i) => i), len: kvTokens, num_blocks: 64, used: Math.ceil(kvTokens / block), block }
  const maxLogit = Math.max(...a.logits.map((l) => l.logit))
  const minLogit = Math.min(...a.logits.map((l) => l.logit))

  return (
    <article className="page explainer">
      <header className="ex__head">
        <h1>What happens when a language model writes one word</h1>
        <p className="ex__lede">
          One forward pass, from the text you type to the token that comes out, using numbers recorded from the Glassbox
          engine running {a.model}
          {a.standin ? ', a small stand-in model trained locally, because the real SmolLM2 weights were out of reach when this was recorded' : ''}.
          The prompt is <q>{a.prompt}</q>
        </p>
      </header>

      <Step
        id="tokens"
        title="Text becomes tokens"
        figure={
          <div className="ex__tokens">
            {tokens.map((t, i) => (
              <span key={i} className={'ex__tok' + (t.special ? ' ex__tok--special' : '')}>
                <Glyphs text={t.text} />
                <small>{t.id}</small>
              </span>
            ))}
          </div>
        }
      >
        <p>
          The model never sees letters. A tokenizer cuts the text into pieces from a fixed vocabulary of{' '}
          {a.config.vocab.toLocaleString()} and replaces each with its number. Common words are one token; rare ones split.
          The chat template wraps your message in special markers so the model knows who is speaking.
        </p>
        <p>
          This prompt is {tokens.length} tokens. Byte-level BPE means any text can be encoded, emoji and Hindi included,
          because the fallback is raw bytes.
        </p>
      </Step>

      <Step id="embed" title="Each token becomes a list of numbers" figure={<Barcode values={a.embedding} />}>
        <p>
          The token’s number picks one row of the embedding table: {a.config.hidden} numbers that place it in a space where
          related tokens sit near each other. Shown here is the row for <q>{last.text.trim() || last.text}</q>, darker
          for larger values, warm for positive and cool for negative.
        </p>
        <p>From here on, every token is this vector, and the model only adds to it.</p>
      </Step>

      <Step
        id="layers"
        title={`${a.config.layers} layers refine a guess`}
        figure={
          <>
            <Norms layers={a.layers} />
            <Lens data={inspect} chosen={winner.id} vocabText={(id) => texts.get(id) ?? `#${id}`} />
          </>
        }
      >
        <p>
          Each layer reads the stream of vectors, mixes in information from other tokens (attention), transforms it (the
          MLP) and adds the result back. The bars above show the vector growing as layers write into it.
        </p>
        <p>
          The list is a <em>logit lens</em>: if we stopped after layer n and asked for a prediction, this is what we would
          get. Early layers guess generic tokens; the answer firms up as the layers stack.
        </p>
      </Step>

      <Step id="attention" title="Attention decides what to look at" figure={<Attention data={inspect} tokens={tokens} query={tokens.length - 1} />}>
        <p>
          Inside each layer, every token asks a question (its query) and every earlier token offers an answer (its key).
          Where they match, the earlier token’s value flows in. Each layer has {a.config.heads} heads asking different
          questions; only {a.config.kv_heads} sets of keys and values are stored, shared between heads (grouped-query
          attention), which keeps the cache small.
        </p>
        <p>Pick a layer and a head: some look at the previous token, some at the start, some at the word that matters.</p>
      </Step>

      <Step
        id="logits"
        title="The last vector scores every token"
        figure={
          <ol className="ex__logits">
            {a.logits.slice(0, 12).map((l) => (
              <li key={l.id}>
                <span className="ex__lt">
                  <Glyphs text={l.text} />
                </span>
                <span className="ex__lb" style={{ '--w': (l.logit - minLogit + 0.5) / (maxLogit - minLogit + 0.5) } as React.CSSProperties} />
                <span className="ex__lv">{l.logit.toFixed(2)}</span>
                <span className="ex__lp">{pct(l.p)}</span>
              </li>
            ))}
          </ol>
        }
      >
        <p>
          After the final layer, the last token’s vector is compared with every row of the vocabulary. The result is one
          score (a logit) per token, {a.config.vocab.toLocaleString()} of them. Softmax turns scores into probabilities: here
          the top choice is <q>{winner.text}</q> at {pct(winner.p)}.
        </p>
      </Step>

      <Step id="sampling" title="Sampling picks one" figure={<Sampling dist={dist} sampler={sampler} onChange={setSampler} />}>
        <p>
          Greedy decoding always takes the top token. Sampling draws from the distribution instead, after reshaping it:
          temperature sharpens or flattens it, top-k and top-p cut the tail, min-p drops anything far below the leader.
          Move the dials and watch what survives.
        </p>
      </Step>

      <Step id="cache" title="Keys and values are kept for next time" figure={<Memory cache={cache} bytesPerToken={a.kv_bytes_per_token} context={1024} />}>
        <p>
          To write the next word, the model needs every earlier token’s keys and values again. Recomputing them would
          make each word cost as much as the whole prompt, so they are cached: {Math.round(a.kv_bytes_per_token / 1024)} kB
          per token for this model.
        </p>
        <p>
          Glassbox stores the cache in blocks of {block} tokens. A sequence only holds the blocks it has filled, and two
          requests that start with the same text share the same blocks.
        </p>
      </Step>

      <Step id="batching" title="Many requests share one pass" figure={<Scheduler trace={scheduler as unknown as SchedTrace} />}>
        <p>
          Writing one token for one person reads every weight in the model from memory. Writing one token for ten people
          reads them once. A server batches requests into the same forward pass, adding new ones as soon as there is room
          rather than waiting for a batch to finish: continuous batching.
        </p>
      </Step>

      <Step id="speculative" title="Guess ahead, check in bulk" figure={<Speculative trace={speculative as unknown as SpecTrace} />}>
        <p>
          A forward pass over five tokens costs little more than over one, because the weights are read once either way. So
          a cheap drafter guesses several tokens and the real model checks them together, keeping every guess it would
          have made itself. The output is exactly what the real model alone would write; only the time changes.
        </p>
      </Step>

      <footer className="ex__end">
        <p>
          That is one word. The playground runs all of this live in your browser, and the source is a few thousand lines of
          Rust.
        </p>
        <a className="btn" href="#/play">
          Open the playground
        </a>
      </footer>
    </article>
  )
}
