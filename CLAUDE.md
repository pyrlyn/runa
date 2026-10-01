# AGENTS.md — agent coordination protocol (symlinked as AGENTS.md → CLAUDE.md, single source of truth)

This file is normative for every coding agent working in this repo (human or AI).
Source of truth for active work: the table at the top of `plan.md`.
Claim registry (code + CI): `docs/tasks.md`. Public-method docs: `docs/memory.md`.
Finished tasks live in `done.md`. Approved-but-not-started work lives in `roadmap.md`.
Documentation index: `docs/README.md`. Contribution checks: `CONTRIBUTING.md`.

`CLAUDE.md` is the real file; `AGENTS.md` is a symlink to it. Edit `CLAUDE.md`
(editing either path writes the same inode) — never fork the protocol into two
files, and keep §1–§7 numbered as they are: `scripts/lint-tasks.py` matches the
§2 claim-first marker, and `moon.yml` cites §6.

## 1. Read before touching anything

1. Read `plan.md` (header table, then Reference) and `README.md`.
2. Read `docs/tasks.md` — it lists only **active** tasks (`free` or `in progress`).
   Finished work is not listed there.
3. Touching docs or a public surface? Read `docs/README.md` (the index) and the
   affected page first; touching a crate? Read its `Cargo.toml` description in
   §8 and the matching ADR in `docs/adr/`.

## 2. Claim protocol (mandatory)

- Take a task only if the `plan.md` table status is `todo` (and `docs/tasks.md` is `free` or has no row).
- To take it: set `plan.md` to `in progress` and write your **provider and model** in Agent; add or update the `docs/tasks.md` row to `in progress` plus agent name and `started_at` (UTC, RFC 3339) so the registry lint stays valid.
- An `in progress` row with another agent — do not take it.
- When **stopped**: `plan.md` back to `todo` with empty Agent; `docs/tasks.md` back to `free`.
- When **done**: move the whole task (id, title, description) to `done.md`; remove it from the `plan.md` table and cards, from `todo.md`, and from `docs/tasks.md`.
- Never hold two tasks at once unless the human explicitly allows it.
- No tree edits without a claim row in `docs/tasks.md`: every tree edit must be covered by your `in progress` claim.

## 3. Ask-before-steal

- An `in progress` task belongs to its owner. To work on it, **ask the
  owner first** (or the human if the owner is unreachable) and proceed
  **only on explicit approval**. No approval = pick a `free` task.
- Stale claims (owner gone, `started_at` older than 7 days): still ask the
  human first — never silently steal. CI surfaces old claims.

## 4. Ask flow (copy/paste)

> Task `P2.6` is `in progress` (owner: `<agent>`, since `<started_at>`).
> I want to take it over because `<reason>`. Approve? If not, I will take
> `<free-task-id>` instead.

## 5. Memory discipline for agents

- Respect the adaptive-memory design (D17): idle → shrink to floor, heavy →
  bounded grow. Do not add caches, pools, or background jobs without a
  release path wired to `MemoryManager::on_idle`.
- Every new public method gets docs in `docs/memory.md` (P7.5, P6.4 rule:
  docs lint fails otherwise).

## 6. Machine checks

- Registry lint: every `in progress` row has agent + RFC 3339 `started_at`;
  no task held twice. Runs in CI (P7.6).
- `cargo test -p runa-memory` covers claim/release/double-claim (P7.4).

## 7. Kernel language (D23)

- Do **not** rewrite ggml, whisper.cpp, or llama.cpp in Zig or Rust.
- New code in `runa-kernels`: prefer **Zig** (C ABI `export fn`, `@Vector` SIMD,
  slices instead of `malloc`/`qsort`) over new C for softmax, sampling
  (top-k/top-p/min-p), and image/audio front-ends.
- Keep **C or `.S`** only where Zig is worse or missing: wrapping existing C
  headers, SME2/AMX assembly, or a measured C path that already wins the D1 gate.
- Same merge gate as D1. Pin Zig with mise (`mise.toml`, D21); do not install
  a floating Zig outside mise.

## 8. Workspace layout (what lives where)

| Path | Responsibility |
|------|----------------|
| `crates/runa` | The CLI: `main.rs` (the clap surface), `config.rs`, `tui.rs`, `serve.rs`, `daemon.rs`, `daemon_proto.rs`, `mcp.rs`, `pull.rs`, `bench.rs`, `fit.rs`, `pool.rs`, `cloud.rs` |
| `crates/runa-core` | `Backend` trait, `Request`/`Event`, `ThinkConfig`, `Mode`, errors. No engine dependency |
| `crates/runa-engine` | `llama-cpp-2` wrapper: load, placement, sampling loop, mtmd, state save; vendored patches behind features (`rpc`, `hexagon`/`openvino` stubs) |
| `crates/runa-fit` | GGUF header (local file / HTTP range), hardware probe, estimator, placement planner, calibration DB. Must not depend on the engine (D5) |
| `crates/runa-memory` | `MemoryManager` (D17) + `TaskRegistry` (D18) |
| `crates/runa-media` | Audio/video decode, resample, frame sampling, ASR bridge (`whisper-rs`) |
| `crates/runa-cloud` | OpenAI + Anthropic adapters, price table (`docs/prices.toml`) |
| `crates/runa-kernels` | Own kernels behind the D1 gate; Zig preferred (D23) |
| `tests/fixtures` | Fixtures — read its `README.md` before adding a file |
| `scripts/` | `lint-tasks.py`, fixture guards/cleanup, `perf-regress.py`, `release.sh` (the release version rule, P13.3), dist and smoke helpers |
| `docs/` | Documentation, indexed by `docs/README.md`; ADRs in `docs/adr/` |

Rust owns orchestration, C owns the compute core (D1); do not rewrite ggml,
llama.cpp or whisper.cpp (§7). Shared dependencies come from the workspace root
`Cargo.toml` (`[workspace.dependencies]`) or the workspace-root `rust.md`
sibling conventions; extract duplicated helpers instead of copying them.

## 9. Local checks (CI parity)

Run these before requesting a merge or closing a task. They mirror
`.github/workflows/ci.yml` (P0.2), so a green local run is evidence, not a guess.

| Check | Command |
|-------|---------|
| Bootstrap tools | `mise install` (`rust-toolchain.toml` stays the rustup source of truth) |
| Format | `cargo fmt --all -- --check` |
| Lint | `cargo clippy --workspace -- -D warnings` |
| Build | `cargo build --workspace` |
| Tests | `cargo test --workspace --lib`, then `moon run :test` for the full set |
| Registry lint | `moon run root:lint-tasks` (or `python3 scripts/lint-tasks.py docs/tasks.md`) |
| Memory crate | `cargo test -p runa-memory` (claim/release/double-claim, P7.4) |
| Fixture budget | `python3 scripts/check-fixture-size.py` (≤ 3 GiB per file) |
| Perf gate self-test | `python3 scripts/perf-regress.py --self-test` |
| Feature builds | `cargo check -p runa-engine --features hexagon,openvino` and `cargo test -p runa-engine -p runa --features runa/rpc --lib --test rpc --test doctor --test trycmd` |
| Full pass + cleanup | `moon run root:test-with-cleanup` (drops downloaded weights, compacts `target/` with dunnage) |

Job time budgets stay under 5 minutes (P11.8/P11.9); when a step grows, report
it instead of loosening the gate.

## 10. Documentation rules

- Prose is English (repository rule, `CONTRIBUTING.md`).
- Document only what the tree ships. Never invent a flag, key, or public method
  to fill a table — if the docs promise it, the code must have it.
- `docs/README.md` is the index: a new doc is added there **and** to the Docs
  table in `README.md` in the same change.
- One topic per file, with the `# <file> — <purpose>` H1 that every file in
  `docs/` already uses.
- Public API: every new public method gets an entry in `docs/memory.md` (§5,
  P7.5); the docs update is part of the task, not a follow-up.
- Decisions are recorded as ADRs in `docs/adr/` (`d01`–`d18`, `d23`; D19–D22
  live in `plan.md` §1). ADRs are append-only: supersede, never rewrite.
- Task bookkeeping is documentation too: `plan.md`, `docs/tasks.md`, `done.md`,
  `roadmap.md` and `todo.md` move together when a task changes state (§2).
- Never commit downloaded weights or generated payloads; the fixture inventory
  and the 3 GiB budget live in `tests/fixtures/README.md`.

## 11. Commit and PR conventions

- English messages with a conventional prefix: `feat:`, `fix:`, `docs:`, `ci:`,
  `test:`, `chore:`, `style:`, optional scope (`ci(trial):`). Name the task when
  one exists (`feat: P9.3 real RPC backend behind rpc feature`). No trailers.
- One concern per commit; scope `git add` to the files the task card lists.
  Never commit another agent's uncommitted work.
- `main` is the reference. Work that touches CI, or that risks a red `main`,
  goes through a topic branch plus a PR (the P11 flow); a scoped push is fine
  while CI is green. The `ci.yml` `revert-on-failure` job reverts a red push to
  `main`, so verify locally first (§9).
- Never force-push a shared branch. `git pull --rebase` before pushing; on a
  conflict stop and report instead of guessing.
- Doc-only commits still use the same prefixes (`docs: …`) and mention the task
  id in the subject when they close one.

## 12. Definition of done

A task is done when all of these hold:

1. The machine check in its `plan.md` card passes, and the evidence (command
   plus output, run id, or file path) is written into the closeout text.
2. `docs/` reflects the change — new public methods in `docs/memory.md`, new
   pages in `docs/README.md` + `README.md`, new behavior in the relevant guide.
3. `moon run :test` and `moon run root:lint-tasks` are green with the claim
   still `in progress`.
4. Bookkeeping then moves in one commit while the claim is still held: the
   whole task (id, title, description) is appended to `done.md`, its row and
   card are removed from `plan.md`, and its row is deleted from
   `docs/tasks.md`. An empty registry table is the normal end state.
5. No claim is left behind on a task you are not actively working: stop and
   done both clear it (§2).

## Host agents

Save tokens. If anything is unclear, ask the creator first. Write a short execution plan into that task's card in `plan.md`, then claim and work. Default cap: **5** parallel agents per project unless the creator says otherwise. Never use max effort or fast mode without permission. Cheapest model for scripts, commands, repo scans, web, file moves, tests. On Cursor: **grok 4.6** (no fast) for planning, refactoring, bug hunts; **composer 2.5** (no fast) for file moves, tests, commands, scans, web. Before writing code, decide whether a ready library or framework should be used. A new dependency is allowed only if it is current (not abandoned) and the creator approved it. Packages already in `toolchain.md` may be reused without asking again. Prefer the latest versions of tools and packages, but bump already-installed ones only with the creator’s permission. Rust: reuse crates already used by sibling projects in this workspace (workspace-root `rust.md`). If this repo lacks one it should use, add a `plan.md` task — do not add the dependency silently. Extract duplicated helpers into `packages/` and depend via local `{ path = "..." }`. No version bumps without permission.

If a directory above this repository contains an `AGENTS.md` or `CLAUDE.md`, follow it too. If it conflicts with this file, ask the creator.

**Config files.** A config file this project owns has a schema generated from its types (Rust: `schemars`), committed and checked by a drift test, and one module owns all config loading, validation and editing. A config file another program owns (an agent host's or an editor's) gets no schema from us: check only our own entry in it and leave the rest byte-for-byte, comments included.
