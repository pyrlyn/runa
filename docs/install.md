# Installing runa

runa is one executable. A release carries it for macOS on Apple Silicon and for Linux (glibc) on
x86-64 and arm64. Until the first release is out, build it from the source: see the
[README](../README.md#build).

## The install script

```sh
curl -fsSL https://github.com/pyrlyn/runa/releases/latest/download/install.sh | sh
```

The script picks the archive for this machine, downloads it together with the `SHA256SUMS` of the
release, checks the archive against it, puts `runa` into `~/.local/bin` and runs
`runa --version`. An archive whose SHA-256 differs is not installed. A second run replaces the
executable; a `runa` that is running keeps working until it is restarted.

| Variable | What it changes |
| --- | --- |
| `RUNA_VERSION` | The release to install (`v1.2.3` or `1.2.3`) instead of the latest one |
| `RUNA_INSTALL_DIR` | The directory `runa` goes to instead of `~/.local/bin` |
| `RUNA_DOWNLOAD_BASE` | Where the archives and `SHA256SUMS` are (a mirror, or `file:///path`) instead of the GitHub release |

```sh
curl -fsSL https://github.com/pyrlyn/runa/releases/latest/download/install.sh |
  RUNA_VERSION=1.2.3 RUNA_INSTALL_DIR=/usr/local/bin sh
```

The script needs `curl`, `tar`, `awk` and `sha256sum` or `shasum`. It says so when the directory
is not in `PATH`. It refuses an Intel Mac and a Linux with musl (Alpine): there is no runa for
them.

To remove runa, delete the executable and its data directory, `~/.runa`, which holds the
downloaded models.

## Homebrew

Every release carries its formula, `runa.rb`. Homebrew installs a formula from a tap only, so
the file goes into a local one:

```sh
brew tap-new --no-git runa/release
curl -fsSL -o "$(brew --repository runa/release)/Formula/runa.rb" \
  https://github.com/pyrlyn/runa/releases/latest/download/runa.rb
brew install runa/release/runa
```

The formula installs the same archives, on macOS (Apple Silicon) and on Linux, and the shell
completions for bash, zsh and fish. To upgrade, download the new `runa.rb` over the old one and
run `brew upgrade runa`.

## By hand

The assets of a release:

| Asset | What it is |
| --- | --- |
| `runa-<tag>-aarch64-apple-darwin.tar.gz` | macOS, Apple Silicon; signed and notarized when the release had the signing identity |
| `runa-<tag>-x86_64-unknown-linux-gnu.tar.gz` | Linux, x86-64, glibc |
| `runa-<tag>-aarch64-unknown-linux-gnu.tar.gz` | Linux, arm64, glibc |
| `SHA256SUMS` | The SHA-256 of every archive |
| `install.sh` | The install script |
| `runa.rb` | The Homebrew formula |

An archive holds one directory, `runa-<tag>-<target>`, with `runa` and `LICENSE`.

```sh
tag=v1.2.3
target=x86_64-unknown-linux-gnu
base="https://github.com/pyrlyn/runa/releases/download/$tag"
curl -fsSLO "$base/runa-$tag-$target.tar.gz" -O "$base/SHA256SUMS"
sha256sum --check --ignore-missing SHA256SUMS   # macOS: shasum -a 256 --check --ignore-missing
tar -xzf "runa-$tag-$target.tar.gz"
"runa-$tag-$target/runa" --version
```

`runa --version` prints the version of the release, which is its tag without the `v`, and the
commit it was built from.

## What Linux must have

The Linux executable is linked dynamically against glibc and uses a few libraries and files of
the system. A desktop or server installation has them; a minimal container image may not.

| Needed | Where it is missing | Package |
| --- | --- | --- |
| `libcrypt.so.1` | Fedora | `libxcrypt-compat` |
| `libz.so.1` | very small images | `zlib1g` (Debian, Ubuntu), `zlib` (Fedora) |
| glibc's gconv modules | openSUSE minimal images | `glibc-locale-base` |
| The CA certificates, for downloads over HTTPS | Debian and Ubuntu minimal images | `ca-certificates` |

Without `libcrypt.so.1` or `libz.so.1` runa does not start, and the install script names these
packages when that happens. Without the CA certificates runa starts, and `runa get` cannot reach
the hub.

The Linux builds compute on the CPU. The macOS build uses Metal.

## How a release is made

A release is made by the `Bump and release` workflow (`.github/workflows/bump.yml`) and by
nothing else; nobody pushes a `v*` tag by hand.

```sh
scripts/release.sh minor --dry-run   # what the next version would be; changes nothing
scripts/release.sh minor             # starts the workflow (gh workflow run bump.yml)
```

The version is the `version` of `gradle.properties` without its `-dev` suffix; when that version
is released already, the level (`patch`, `minor` or `major`) raises it.

1. `bump.yml` (the shared workflow of pyrlyn/ci) runs `scripts/release.sh <level> --local`, which
   writes the version into `gradle.properties` in one `release: vX.Y.Z` commit, and opens a pull
   request with it.
2. When the required checks of that pull request pass, the workflow merges it, tags the merged
   commit, creates a draft release and starts `release.yml` on the tag.
3. `release.yml` builds each target with `scripts/build-release.sh`, signs and notarizes the
   macOS executable, compares `runa --version` with the tag, packs the archives and uploads them
   with `SHA256SUMS` to the draft.
4. Its `assets` job installs from those archives with `scripts/install.sh`, writes `runa.rb` with
   `scripts/formula.sh`, attaches both and publishes the release.
5. Its `verify` jobs install the published release on macOS, on Linux x86-64 and on Linux arm64
   with the install script, and on macOS with the formula.

Signing uses the organisation secrets `MACOS_CERTIFICATE` and `MACOS_CERTIFICATE_PWD`, and
notarization `APPSTORE_CONNECT_KEY`, `APPSTORE_CONNECT_KEY_ID` and `APPSTORE_CONNECT_ISSUER_ID`.
Without them the macOS executable keeps the ad-hoc signature of the linker, and the workflow says
so in a notice.

`app/integration/install.sh <runa executable>` tests the install script and the formula against
archives packed the same way, with no network; CI runs it on every pull request.
