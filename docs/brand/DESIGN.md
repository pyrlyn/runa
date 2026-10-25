# runa — Design System

## Overview

**runa** is a local+cloud AI CLI (GGUF / llama.cpp + APIs). Metaphor: rune / local fire / compute — a geometric stave with an ember at the base. Warm charcoal surfaces and ember coral accents; deliberately not purple AI gradient.

Identity: ember coral + warm charcoal. Distinct hue family from copper bindsmith, forest cox, and indigo stator.


**Shared visual lock (Listepo landing v1):** **nerd + ai + glass + flat** — same family as ketch brand v1. IBM Plex Mono eyebrows/labels/chips, hairline borders, mono CLI cards; subtle agent/compute cues (soft accent glow, gradient hairline, status chips) using **ember only** (no purple AI gradients); translucent glass panels with `backdrop-filter` plus opaque `@media (prefers-reduced-transparency: reduce)` fallbacks; flat CTAs, 4/8pt spacing, surface ladder, radii 8–12.

## Colors

### Light

| Token | Hex | Use |
|-------|-----|-----|
| bg | `#F8F3F0` | Page background (warm ash) |
| bg-elevated | `#FFFFFF` | Cards |
| surface-1 | `#F0E8E3` | Nested |
| surface-2 | `#E6D9D1` | Hover |
| surface-3 | `#D8C8BE` | Pressed |
| border | `#C4B0A4` | Default border |
| border-hairline | `#E0D4CC` | Divider |
| fg | `#1C1412` | Primary text |
| fg-muted | `#5A4A44` | Secondary |
| fg-subtle | `#8A7870` | Tertiary |
| accent | `#E85A3C` | Primary CTA / ember |
| accent-hover | `#C94830` | Hover |
| accent-muted | `#F07050` | Soft accent |
| accent-soft | `#FCE8E0` | Accent wash |
| ember | `#FF9A6A` | Highlight / local-fire |
| code-bg | `#1C1412` | Code / CLI |
| code-fg | `#FFB088` | CLI highlight |

### Dark

| Token | Hex | Use |
|-------|-----|-----|
| bg | `#120E0C` | Page background |
| bg-elevated | `#1C1412` | Cards |
| surface-1 | `#261E1A` | Nested |
| surface-2 | `#322824` | Hover |
| surface-3 | `#403430` | Pressed |
| border | `#4E403A` | Default border |
| border-hairline | `#2A221E` | Divider |
| fg | `#F0E8E3` | Primary text |
| fg-muted | `#B0A098` | Secondary |
| fg-subtle | `#786860` | Tertiary |
| accent | `#F07050` | Primary CTA |
| accent-hover | `#FF9A6A` | Hover |
| accent-muted | `#E85A3C` | Soft accent |
| accent-soft | `#2A1814` | Accent wash |
| ember | `#FFB088` | Highlight |
| code-bg | `#0C0908` | CLI |
| code-fg | `#FFB088` | Prompt / tokens |

## Typography

- **Mono (CLI, labels, docs code):** IBM Plex Mono (fallback JetBrains Mono)
- **Sans (marketing):** IBM Plex Sans
- Scale: 12 / 14 / 16 / 20 / 28 / 40
- Weights: 400 / 500 / 600

## Layout

- CLI-first density; marketing max width 840px
- 8px grid; hairline borders; warm surface ladder
- Logo tile: 64×64, 12px radius

## Components

- **CLI prompt:** charcoal elevated, ember caret, mono
- **Model chips:** GGUF / API tags — accent-soft bg, mono 12px
- **Buttons:** solid ember; ghost hairline
- **Local vs cloud badge:** ember for local fire; muted charcoal for remote API
- **Progress / tokens:** ember soft wash bars

## Mini landing wire

1. Hero: rune mark + `runa` + “Local fire. Cloud when you need it.”
2. Dual path: GGUF local | API remote
3. CLI snippet: `runa ask "…"`
4. Footer: Listepo + docs

## Do / Don't

**Do**
- Ember for heat/compute; warm charcoal for structure
- Keep the rune geometric and sharp at 16px
- Mono-forward for CLI credibility

**Don't**
- No purple/violet AI gradients, no neon pink
- Don’t use cool teal or cyan
- Avoid ornate historical rune fonts — one geometric metaphor only
