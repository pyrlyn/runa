# Roadmap

Approved work that is not yet in the active plan.

All four post-1.0 items entered the plan as P9.1–P9.4 on 2026-09-15;
all four are done (see `done.md`).

## P18. Model sources and loading (`runa-models` crate)

Approved by the creator. Research: `research.md` §15. P18.1 is done (`done.md`). Each
item stays under the 500-line task limit and ships with tests.

- **P18.3. Network download and cache.** HTTPS only (plain HTTP for loopback tests),
  host allowlist, `.part` file with `Range` resume, fsync and atomic rename.
  Content-addressed blobs (`sha256-<hex>`) plus a ref record (ETag, size,
  Last-Modified); revalidate with `If-None-Match`; an offline mode that never touches
  the network. Moves the `.verified` sidecar logic out of `pull.rs`.
- **P18.4. One interface for every source.** `ModelRef { Local, Hf, Url, Wasm }` and
  `ModelStore::resolve` returning `ResolvedModel { path, format, digest, size, source,
  fresh }`; `runa pull`, `find_local` and the `runa-fit` refs move onto it with no
  CLI change.
- **P18.5. Wasm source.** Behind a `wasm` feature: validate the module and load it
  through the extism host with a memory cap, a deadline and no file or network
  access.
- **P18.6. Errors and logging.** A typed `LoadError` (unreachable URL, corrupt file,
  no disk space checked before and during the download, format mismatch, too large,
  unauthorized, host not allowed) and one `tracing` span per load: source, URL without
  query, bytes, digest, cache hit or miss, duration. No stderr output from the
  library.
- **P18.7. Sign-in.** `runa login | logout | whoami <provider>`; tokens in the OS
  keychain, bound to their host; `Authorization` dropped on a cross-host redirect.
  Token first (`HF_TOKEN` or paste); the Hugging Face device code flow once the
  creator registers a public OAuth app and supplies its client id.
- **P18.8. Examples and docs.** `examples/host.rs`, `examples/local.rs`,
  `examples/hf.rs`; a docs page with `docs/ru/` and `docs/uk/` translations;
  `docs/memory.md` entries.
- **P18.9. OCI sources.** Ollama registry, Docker Hub `ai/`, GHCR through
  `oci-client`, blobs verified by digest.
- **P18.10. ModelScope and Hugging Face mirrors.** ModelScope downloads with its SDK
  token; a configurable Hub endpoint for mirrors.
