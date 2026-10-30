// One engine worker per page, shared by every view. State lives here and
// components subscribe with useEngine().
import { useSyncExternalStore } from 'react'
import type { Alt, CacheState, EngineInfo, FromWorker, Inspect, Sampler, Token, ToWorker } from './protocol'

export const MODEL_BASE: string =
  import.meta.env.VITE_MODEL_BASE ?? 'https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main/'
/** bf16 safetensors of SmolLM2-135M-Instruct, shown before the download starts */
export const MODEL_BYTES = 269_060_552
export const CONTEXT = 1024

export type Status = 'idle' | 'loading' | 'ready' | 'running' | 'error'

export type PromptToken = { id: number; text: string; special: boolean }

export type EngineState = {
  status: Status
  progress: { label: string; loaded: number; total: number; cached: boolean } | null
  info: EngineInfo | null
  prompt: PromptToken[]
  tokens: Token[]
  /** branches abandoned by forking: the tokens that were replaced, keyed by fork position */
  branches: { at: number; tokens: Token[] }[]
  tokPerSec: number
  prefill: { ms: number; tokens: number } | null
  error: string | null
  inspect: { pos: number; data: Inspect } | null
  cache: CacheState | null
}

let state: EngineState = {
  status: 'idle',
  progress: null,
  info: null,
  prompt: [],
  tokens: [],
  branches: [],
  tokPerSec: 0,
  prefill: null,
  error: null,
  inspect: null,
  cache: null,
}
const listeners = new Set<() => void>()
let worker: Worker | null = null
let stepTimes: number[] = []

function set(patch: Partial<EngineState>) {
  state = { ...state, ...patch }
  listeners.forEach((l) => l())
}

function send(m: ToWorker) {
  worker?.postMessage(m)
}

function onMessage(e: MessageEvent<FromWorker>) {
  const m = e.data
  switch (m.type) {
    case 'progress':
      set({ progress: m })
      break
    case 'ready':
      set({ status: 'ready', info: m.info, progress: null })
      break
    case 'prompt':
      set({ prompt: m.tokens, tokens: [], branches: [], inspect: null })
      break
    case 'prefill':
      set({ prefill: m })
      break
    case 'token':
      if (m.ms) stepTimes = [...stepTimes.slice(-15), m.ms]
      set({
        tokens: [...state.tokens, m.token],
        tokPerSec: stepTimes.length ? 1000 / (stepTimes.reduce((a, b) => a + b, 0) / stepTimes.length) : 0,
      })
      send({ type: 'cache' })
      break
    case 'forked': {
      const cut = m.keep - state.prompt.length
      set({
        branches: [...state.branches, { at: m.keep, tokens: state.tokens.slice(cut) }],
        tokens: [...state.tokens.slice(0, cut), m.token],
        inspect: null,
      })
      break
    }
    case 'done':
      set({ status: 'ready' })
      break
    case 'inspect':
      set({ inspect: { pos: m.pos, data: m.data } })
      break
    case 'cache':
      set({ cache: m.data })
      break
    case 'error':
      set({ status: 'error', error: m.message })
      break
  }
}

export const engine = {
  load() {
    if (worker) return
    worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' })
    worker.onmessage = onMessage
    worker.onerror = (e) => set({ status: 'error', error: e.message || 'The engine worker failed to start.' })
    set({ status: 'loading', error: null })
    send({ type: 'load', base: MODEL_BASE, quant: 'q8_0', context: CONTEXT, expectBytes: MODEL_BYTES })
  },
  start(prompt: string, sampler: Sampler, max = 160, chat = true) {
    stepTimes = []
    set({ status: 'running', tokens: [], branches: [], inspect: null })
    send({ type: 'start', prompt, chat, max, sampler })
  },
  fork(index: number, alt: Alt, sampler: Sampler, max = 120) {
    set({ status: 'running' })
    send({ type: 'fork', keep: index, token: alt.id, p: alt.p, alts: [], max, sampler })
  },
  stop() {
    send({ type: 'stop' })
  },
  inspect(pos: number) {
    send({ type: 'inspect', pos })
  },
  get: () => state,
}

export function useEngine(): EngineState {
  return useSyncExternalStore(
    (l) => {
      listeners.add(l)
      return () => listeners.delete(l)
    },
    () => state,
  )
}

export const greedy: Sampler = { temperature: 0, top_k: 0, top_p: 1, min_p: 0, seed: 0 }
