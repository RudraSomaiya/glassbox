# Glassbox design

## Concept: the light table and the darkroom

Inference shown as photography. Each token the model writes is a **frame** exposed on a strip of film; the frame's **density** is the probability the model gave it (clear means sure, dense means a long shot). The visitor inspects frames with a **loupe** on a **light table**, sees the **step wedge** of what else the model could have written, and **forks** the strip with a grease-pencil mark. Layers are exposures stacked through the glass; the KV cache is a reel of fixed-size frames.

Light theme is the light table: luminous cool white, black ink, the C-41 negative's orange base as the one committed field, grease-pencil red for marks. Dark theme is the darkroom: warm black, paper-white ink, safelight red-orange. The theme follows the system setting and can be switched in the top bar.

## Palette

| Role | Light table | Darkroom | Use |
|---|---|---|---|
| `--ground` | `#eef1ee` | `#13100d` | page ground |
| `--surface` | `#fbfcfb` | `#1c1814` | raised things that float (loupe, fields) |
| `--sunk` | `#e2e6e2` | `#0c0a08` | wells, inactive bars |
| `--ink` | `#121413` | `#efe7da` | all running text |
| `--ink-2` | `#444a46` | `#b3a899` | secondary notes only |
| `--film` | `#c8692e` | `#8a4419` | the film base: strips, bars, lens |
| `--film-ink` | `#1d0d03` | `#fbeadb` | text on clear frames |
| `--film-dense` | `rgb(29 10 2)` | `rgb(8 3 0)` | D-max, mixed into frames by `1 - p` |
| `--mark` | `#c92a22` | `#ff6b3d` | selection, focus, forks, the action on hover |

Probability is never shown by hue. It is density: `color-mix(film, film-dense, (1 - p) * 88%)`, with the text flipping to light below p = 0.4.

## Type

One family: **Archivo** (variable, self-hosted), using its width axis as the second voice.

| Step | Size | Setting |
|---|---|---|
| display (`--step-3`) | clamp(2.6rem, 6rem) | weight 900, width 125, tracking -0.035em, leading 0.92 |
| heading (`--step-2`) | clamp(1.75rem, 2.5rem) | weight 800, width 125 |
| lede (`--step-1`) | 1.375rem | weight 400-700, width 90-100 |
| body (`--step-0`) | 1.0625rem | weight 400, width 100, leading 1.5 |
| note (`--step--1`) | 0.875rem | secondary ink |

Edge print (frame numbers in the sprocket band) uses width 70 at 12px, numerals only. Tabular figures everywhere. No monospace outside code.

## Space and shape

- One outer gutter, `clamp(16px, 3vw, 40px)`, both sides at every width.
- Strip rows are a fixed `--row` (72px desktop, 64px phone) with a `--band` of sprockets above and below, so the sprocket tile lines up with every wrapped line.
- Square corners on buttons and fields; 3px on frames (film frames have slightly rounded corners); the lens is the only circle.
- Surfaces separate by a step in tone. Only things that float (the loupe) cast a shadow.

## Motion

Personality: precision. One curve, `cubic-bezier(0.2, 0, 0, 1)`, three durations: 120ms (feedback), 240ms (panels), 480ms (the signature).

- **Signature: develop.** A new frame appears as a latent image, blurred and pale, and sharpens to its true density in 480ms.
- The gate (the red cursor at the end of a running strip) blinks in two steps, like a projector gate.
- Step-wedge bars grow from the left when the loupe opens.
- `prefers-reduced-motion`: everything lands immediately; the hero replay shows the whole strip at once.

## Never

- Frosted-glass cards, backdrop blur, purple or blue-violet gradients, neon on black.
- A generic dark dashboard of rounded cards with soft grey shadows.
- Hue as the probability scale, or rainbow heatmaps.
- Eyebrow labels, tracked uppercase labels, index numbers on sections, emoji icons.
- Stand-in or illustrative data presented as the model's real output.
