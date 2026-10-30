// Messages between the page and the engine worker.

export type Alt = { id: number; p: number; text: string; logit?: number }

export type Token = {
  index: number
  id: number
  text: string
  /** probability the model gave this token when it was chosen */
  p: number
  /** entropy (nats) of the distribution it was drawn from */
  entropy: number
  /** top alternatives at this position, most likely first */
  alts: Alt[]
  special: boolean
}

export type Sampler = { temperature: number; top_k: number; top_p: number; min_p: number; seed: number }

export type EngineInfo = {
  layers: number
  heads: number
  kv_heads: number
  hidden: number
  vocab: number
  context: number
  block: number
  weight_bytes: number
  kv_bytes_per_token: number
  simd: boolean
  eos: number
}

export type Inspect = {
  /** per layer: logit-lens distribution summary */
  lens: { top: [number, number][]; entropy: number }[]
  /** [layer][head][position] */
  attention: number[][][]
}

export type CacheState = { blocks: number[]; len: number; num_blocks: number; used: number; block: number }

export type ToWorker =
  | { type: 'load'; base: string; quant: 'q8_0' | 'f32'; context: number; expectBytes: number }
  | { type: 'start'; prompt: string; chat: boolean; max: number; sampler: Sampler }
  | { type: 'fork'; keep: number; token: number; p: number; alts: Alt[]; max: number; sampler: Sampler }
  | { type: 'stop' }
  | { type: 'inspect'; pos: number }
  | { type: 'cache' }
  | { type: 'dist'; k: number }

export type FromWorker =
  | { type: 'progress'; label: string; loaded: number; total: number; cached: boolean }
  | { type: 'ready'; info: EngineInfo }
  | { type: 'prompt'; tokens: { id: number; text: string; special: boolean }[] }
  | { type: 'prefill'; ms: number; tokens: number }
  | { type: 'token'; token: Token; ms?: number }
  | { type: 'forked'; keep: number; token: Token }
  | { type: 'done'; reason: 'eos' | 'length' | 'max' | 'stopped' }
  | { type: 'inspect'; pos: number; data: Inspect }
  | { type: 'cache'; data: CacheState }
  | { type: 'dist'; top: Alt[] }
  | { type: 'error'; message: string }
