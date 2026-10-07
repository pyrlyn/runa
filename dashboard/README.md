# dashboard — local UI for `runa serve`

React, TanStack Query, and Tailwind. The page is served by the `runa` binary
at `GET /dashboard` on the same host and port as the OpenAI-compatible API.
Live numbers arrive on `GET /dashboard/ws`. The first paint loads
`GET /dashboard/snapshot`.

Nothing in this UI calls a host outside the machine. There is no telemetry.

## Build

From this directory, with Node (pinned in the repo `mise.toml`):

```sh
npm install
npm run build
```

`npm run build` writes three stable files:

- `dist/index.html`
- `dist/dashboard.js`
- `dist/dashboard.css`

`crates/runa` embeds those files with `include_str!`. Rebuild the UI, then
rebuild `runa`, after a frontend change. Commit the regenerated `dist/` files
with the source so CI can compile without Node.

`npm run dev` serves the page for layout work and proxies `/dashboard/snapshot`
and `/dashboard/ws` to `http://127.0.0.1:8080` (`runa serve --port 8080`).

The snapshot fields are documented in [`docs/dashboard.md`](../docs/dashboard.md).
