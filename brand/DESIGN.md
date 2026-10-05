# runa brandbook

Only what is specific to runa. Everything else is the [Pyrlyn base layer](https://github.com/pyrlyn/brand/blob/v0.4.0/base/DESIGN.md) from `@pyrlyn/brand` (pinned in `package.json`).
Tokens: [`tokens.json`](tokens.json) (extends `@pyrlyn/brand/base/tokens.json`), built to
`dist/tokens.css` by `node build.mjs`. Source: `pyrlyn/runa` at `947399f`, `docs/brand/DESIGN.md` and
`docs/brand/tokens.css`.

## Idea

Local fire, cloud when you need it: a geometric rune stave with an ember glowing at its base. Warm charcoal
structure, ember coral for heat and compute. Nerd + AI + glass + flat, the same family as the other Pyrlyn products.

## Colour

- Light is the default theme (warm ash `#F8F3F0`); dark is first-class (warm charcoal `#120E0C`).
- Accent: ember `#E85A3C` on light (hover `#C94830`), `#F07050` on dark (hover `#FF9A6A`). As text on light use
  `accent-fg` `#B6412C`; the ember fill carries charcoal text (`on-accent` `#1C1412`), not white.
- Ember highlight (`--runa-ember`): `#FF9A6A` on light, `#FFB088` on dark; the CLI highlight (`--runa-code-fg`) is
  `#FFB088` in both themes on a dark code ground (`--runa-code-bg`).
- Mark tile is always warm charcoal `#1C1412`, so the ember stave reads on both canvases.
- Status: fills success `#3D7A4A` / `#6BBF7A`, warn `#D4893A` / `#E8A05C`, danger `#C9302C` / `#F07068`
  (light / dark), plus `--runa-info` `#5A6B8A` / `#8A9BB8`. Use the `*-fg` variant for text; the light ones are
  darkened to pass AA (`#3A7547`, `#945C20`, `#C52F2B`).
- Ember only for agent and compute cues (soft glow `--runa-accent-glow`, gradient hairline, status chips).
- No purple or violet "AI" gradients, no neon pink, no cool teal or cyan.
- Role mapping from the `docs/brand/` names: `bg-elevated`→`surface`, `surface-1`→`surface-2`,
  `surface-2`→`surface-3`, `accent-soft`→`accent-muted`, `shadow`→`shadow-e1`. Kept as `--runa-*`:
  `accent-hover`, `ember`, `info`, `surface-pressed` (`docs/brand/` `surface-3`), `border-hairline`, `code-bg`,
  `code-fg`, `glass-fill`, `glass-border`, `accent-glow`.

## Type

IBM Plex Mono for CLI, labels, chips and docs code (the base font); `docs/brand/` adds IBM Plex Sans for marketing
copy, which the base does not ship. Scale 12 / 14 / 16 / 20 / 28 / 40, weights 400 / 500 / 600.

## Logo

| File (`logo/`) | Use |
|---|---|
| `runa-mark.svg` | Mark 64×64: charcoal tile (12px radius), ember stave with angled arms, glowing ember at the base |
| `runa-wordmark.svg` | Mark + `runa` in mono 600, charcoal text (live text: needs IBM Plex Mono or JetBrains Mono; light grounds only) |
| `runa-favicon.svg` | Simplified 32×32 mark |
| `png/runa-favicon-{32x32,64x64}.png`, `png/runa-apple-touch-icon-180.png`, `png/runa-icon-logo-{512,1024}.png` | Rendered from the SVGs |

Keep the rune geometric and sharp at 16px; one geometric metaphor only, no ornate historical rune fonts.
No wordmark PNG: the wordmark is live text, so a raster depends on the installed font.
