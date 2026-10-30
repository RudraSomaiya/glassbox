import type { CacheState } from '../engine/protocol'
import './memory.css'

const mb = (b: number) => (b >= 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${Math.round(b / 1e3)} kB`)

/**
 * The paged KV cache as a reel of 16-slot blocks. A sequence holds only the
 * blocks it has filled, where a contiguous cache reserves the whole context
 * up front.
 */
export function Memory({ cache, bytesPerToken, context }: { cache: CacheState; bytesPerToken: number; context: number }) {
  const owned = new Map(cache.blocks.map((b, i) => [b, i]))
  const used = cache.blocks.length * cache.block * bytesPerToken
  const naive = context * bytesPerToken
  return (
    <figure className="mem" aria-labelledby="mem-cap">
      <figcaption id="mem-cap">
        <strong>Memory.</strong> {cache.len} tokens of keys and values sit in {cache.blocks.length} blocks of{' '}
        {cache.block}: {mb(used)}. A contiguous cache sized for the full {context}-token context would hold {mb(naive)}{' '}
        from the first token.
      </figcaption>
      <div className="mem__grid" role="img" aria-label={`${cache.blocks.length} of ${cache.num_blocks} blocks in use`}>
        {Array.from({ length: cache.num_blocks }, (_, b) => {
          const i = owned.get(b)
          const fill = i === undefined ? 0 : Math.min(1, (cache.len - i * cache.block) / cache.block)
          return (
            <span key={b} className={'mem__block' + (i !== undefined ? ' mem__block--on' : '')}>
              <span className="mem__fill" style={{ '--f': fill } as React.CSSProperties} />
            </span>
          )
        })}
      </div>
      <div className="mem__bars" aria-hidden>
        <span className="mem__bar mem__bar--paged" style={{ '--f': used / naive } as React.CSSProperties}>
          paged {mb(used)}
        </span>
        <span className="mem__bar mem__bar--naive">contiguous {mb(naive)}</span>
      </div>
    </figure>
  )
}
