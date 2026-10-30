import { useEffect, useState } from 'react'
import { engine, greedy, MODEL_BYTES, useEngine } from '../engine/client'
import type { Token } from '../engine/protocol'
import { density, FilmStrip } from '../views/FilmStrip'
import { LoadButton } from '../views/LoadButton'
import hero from '../traces/hero.json'
import { GITHUB } from '../App'
import './landing.css'

const PROMPT = hero.prompt

/** Replays a recorded run at the speed it was recorded, then holds and loops. */
function useReplay(tokens: Token[], perSecond = 14) {
  const [n, setN] = useState(0)
  useEffect(() => {
    if (matchMedia('(prefers-reduced-motion: reduce)').matches) {
      setN(tokens.length)
      return
    }
    const id = setInterval(() => setN((k) => (k >= tokens.length + perSecond * 4 ? 0 : k + 1)), 1000 / perSecond)
    return () => clearInterval(id)
  }, [tokens, perSecond])
  return tokens.slice(0, Math.min(n, tokens.length))
}

export function Landing() {
  const s = useEngine()
  const replay = useReplay(hero.tokens as Token[])
  const live = s.tokens.length > 0
  const tokens = live ? s.tokens : replay

  useEffect(() => {
    // Once the model is loaded, the hero runs it for real.
    if (s.status === 'ready' && !live) engine.start(PROMPT, greedy, 96)
  }, [s.status, live])

  return (
    <div className="page landing">
      <section className="hero" aria-labelledby="hero-title">
        <h1 id="hero-title">Watch a language model write</h1>
        <p className="hero__lede">
          Glassbox is an inference engine I wrote from scratch in Rust. Here it runs a small model in your browser.
          Every frame on the strip is one token, and the clearer the frame, the surer the model was.
        </p>
        <div className="hero__run">
          <p className="hero__prompt">
            <span className="sr-only">Prompt: </span>
            {PROMPT}
          </p>
          <FilmStrip
            tokens={tokens}
            label={live ? 'The model’s answer, live' : 'A recorded answer, replaying'}
            live={s.status === 'running'}
            onFork={live ? (i, alt) => engine.fork(i, alt, greedy, 80) : undefined}
          />
          <div className="hero__actions">
            <LoadButton />
            <a className="btn btn--quiet" href="#/play">
              Open the playground
            </a>
            <p className="hero__note">
              {live
                ? `Live: SmolLM2-135M, ${s.tokPerSec.toFixed(0)} tokens a second on your machine. Hover a frame for the alternatives; click one to fork.`
                : hero.standin
                  ? 'Replaying a run from a small stand-in model trained locally on Python’s documentation, recorded while the real weights were out of reach. Run it to see SmolLM2 itself.'
                  : `Replaying a recorded run. The live model is a ${Math.round(MODEL_BYTES / 1e6)} MB download, cached after the first time.`}
            </p>
          </div>
        </div>
      </section>

      <section className="wedge-sec" aria-labelledby="read-title">
        <h2 id="read-title">Reading the strip</h2>
        <p>
          Each frame is exposed like film: the probability the model gave the token is how much light gets through. Clear
          frames are tokens it was sure of; dense ones were long shots it took anyway, which sampling with a temperature
          above zero lets through now and then. As on film, density is logarithmic, so 10% and 1% look clearly different.
        </p>
        <ol className="stepwedge" aria-label="Density scale from certain to unlikely">
          {[1, 0.7, 0.5, 0.3, 0.15, 0.08, 0.04, 0.02, 0.01, 0.002].map((p) => (
            <li key={p} className={density(p) > 0.3 ? 'dense' : ''} style={{ '--d': density(p) } as React.CSSProperties}>
              <span>{p >= 0.01 ? `${Math.round(p * 100)}%` : `${(p * 100).toFixed(1)}%`}</span>
            </li>
          ))}
        </ol>
      </section>

      <section className="sheet" aria-labelledby="sheet-title">
        <h2 id="sheet-title">Technical data</h2>
        <dl className="sheet__grid">
          <div>
            <dt>Engine</dt>
            <dd>Rust, no ML framework. Tokenizer, kernels, model, sampler, scheduler written for this project.</dd>
          </div>
          <div>
            <dt>Kernels</dt>
            <dd>AVX2 and FMA on x86, SIMD128 in WebAssembly, each checked against a scalar twin.</dd>
          </div>
          <div>
            <dt>Weights</dt>
            <dd>safetensors, or GGUF Q8_0 and Q4_0, bit-identical to llama.cpp’s quantizer.</dd>
          </div>
          <div>
            <dt>Memory</dt>
            <dd>Paged KV cache in 16-token blocks, shared between requests with the same prefix.</dd>
          </div>
          <div>
            <dt>Serving</dt>
            <dd>Continuous batching with chunked prefill and preemption, behind an OpenAI-compatible API.</dd>
          </div>
          <div>
            <dt>Decoding</dt>
            <dd>Speculative decoding (draft model or prompt lookup) and JSON-schema constrained output.</dd>
          </div>
          <div>
            <dt>Checked against</dt>
            <dd>Hugging Face transformers token for token; llama.cpp perplexity to five significant figures.</dd>
          </div>
          <div>
            <dt>This page</dt>
            <dd>SmolLM2-135M quantized to 8 bits in your browser, one thread, no server.</dd>
          </div>
        </dl>
      </section>

      <nav className="next" aria-labelledby="next-title">
        <h2 id="next-title">Keep looking</h2>
        <a href="#/play">
          <strong>Playground</strong>
          <span>Every view at once: the strip, layer by layer, attention, memory, the sampling dials.</span>
        </a>
        <a href="#/explain">
          <strong>How it works</strong>
          <span>One word, followed from text to token through every stage, with real numbers.</span>
        </a>
        <a href="#/bench">
          <strong>Benchmarks</strong>
          <span>Measured against llama.cpp and transformers on the same machine, including where it is slower.</span>
        </a>
        <a href={GITHUB}>
          <strong>Source</strong>
          <span>The Rust engine, the server, this site, and the scripts behind every number.</span>
        </a>
      </nav>

      <footer className="foot">
        <p className="foot__mark" aria-hidden>
          Glassbox
        </p>
        <p>Built by Rudra Somaiya. MIT licence.</p>
      </footer>
    </div>
  )
}
