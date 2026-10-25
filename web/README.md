# web — live usage page for `runa web`

Vite, React, and TanStack Query. The page is served by the `runa` binary
at `GET /` when you run `runa web`. Live numbers arrive on `GET /ws`.
The first paint loads `GET /api/snapshot`.

The journal never leaves this machine. There is no telemetry.

Colours are the runa brand (warm ash, charcoal, ember). The stack matches
the token CLI admin page: a Vite build embedded in the binary, a websocket for
the live snapshot.

## Build

From this directory, with Node (pinned in the repo `mise.toml`):

```sh
npm install
npm run build
```

`npm run build` writes three stable files:

- `dist/index.html`
- `dist/stats.js`
- `dist/stats.css`

`crates/runa` embeds those files with `include_str!`. Rebuild the UI, then
rebuild `runa`, after a frontend change. Commit the regenerated `dist/`
files with the source so CI can compile without Node.

`npm run dev` serves the page for layout work and proxies `/api`, `/health`,
and `/ws` to `http://127.0.0.1:3939` (`runa web`).

The snapshot fields are documented in [`docs/stats.md`](../docs/stats.md).
