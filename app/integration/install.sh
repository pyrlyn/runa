#!/usr/bin/env bash
# Integration test of the release packaging (PLAN.md, phase 10): an archive made the way
# release.yml makes it is installed by scripts/install.sh, and scripts/formula.sh names it.
#
#   app/integration/install.sh <runa executable>
set -euo pipefail

runa="$1"
root="$(cd "$(dirname "$0")/../.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

fail() {
  echo "FAILED: $1" >&2
  exit 1
}

sha256() {
  if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi
}

version="$("$runa" --version | cut -d' ' -f2)"
tag="v$version"
dist="$work/dist"
mkdir -p "$dist"

# The same executable under every target's name: the installer takes the one of this machine.
echo "archives as release.yml packs them"
for target in aarch64-apple-darwin x86_64-unknown-linux-gnu aarch64-unknown-linux-gnu; do
  name="runa-$tag-$target"
  mkdir -p "$work/stage/$name"
  cp "$runa" "$work/stage/$name/runa"
  cp "$root/LICENSE" "$work/stage/$name/"
  (cd "$work/stage" && tar -czf "$dist/$name.tar.gz" "$name")
  (cd "$dist" && sha256 "$name.tar.gz" >"$name.tar.gz.sha256")
done
(cd "$dist" && cat ./*.sha256 >SHA256SUMS)

install() {
  RUNA_VERSION="$1" RUNA_DOWNLOAD_BASE="file://$dist" RUNA_INSTALL_DIR="$2" sh "$root/scripts/install.sh"
}

echo "install.sh installs the release that is named"
install "$tag" "$work/bin" >"$work/out"
grep -q "^Installed runa $version " "$work/out" || fail "install.sh did not say what it installed: $(cat "$work/out")"
[ "$("$work/bin/runa" --version)" = "$("$runa" --version)" ] || fail "the installed runa is not the packed one"
grep -q "is not in PATH" "$work/out" || fail "install.sh did not say that the directory is not in PATH"

echo "a version without the v, and a second install over the first"
PATH="$work/bin:$PATH" install "$version" "$work/bin" >"$work/out"
! grep -q "is not in PATH" "$work/out" || fail "install.sh warned about a directory that is in PATH"
[ -z "$(find "$work/bin" -name '.runa.new.*')" ] || fail "install.sh left its temporary file"

echo "an archive that is not what the release says is refused"
for archive in "$dist"/*.tar.gz; do echo tampered >>"$archive"; done
if install "$tag" "$work/other" >"$work/out" 2>&1; then fail "install.sh installed a tampered archive"; fi
grep -q "nothing was installed" "$work/out" || fail "install.sh did not say why it stopped: $(cat "$work/out")"
[ ! -e "$work/other/runa" ] || fail "install.sh left a runa from a tampered archive"

echo "an archive that the checksums of the release do not name is refused"
cp "$dist/SHA256SUMS" "$work/sums"
: >"$dist/SHA256SUMS"
if install "$tag" "$work/other" >"$work/out" 2>&1; then fail "install.sh installed an archive without a checksum"; fi
grep -q "does not name" "$work/out" || fail "install.sh did not say why it stopped: $(cat "$work/out")"
[ ! -e "$work/other/runa" ] || fail "install.sh left a runa from an archive without a checksum"
cp "$work/sums" "$dist/SHA256SUMS"

echo "a release that does not exist"
if install v0.0.0-none "$work/other" >"$work/out" 2>&1; then fail "install.sh installed a release that is not there"; fi
grep -q "cannot download" "$work/out" || fail "install.sh did not say what it could not download: $(cat "$work/out")"

echo "formula.sh names every archive with its checksum"
"$root/scripts/formula.sh" "$tag" "$dist/SHA256SUMS" >"$work/runa.rb"
for target in aarch64-apple-darwin x86_64-unknown-linux-gnu aarch64-unknown-linux-gnu; do
  sum="$(cut -d' ' -f1 <"$dist/runa-$tag-$target.tar.gz.sha256")"
  grep -A1 "runa-$tag-$target.tar.gz\"" "$work/runa.rb" | grep -q "sha256 \"$sum\"" ||
    fail "runa.rb does not have the checksum of $target"
done
grep -q "version \"$version\"" "$work/runa.rb" || fail "runa.rb does not have the version"
if command -v ruby >/dev/null; then ruby -c "$work/runa.rb" >/dev/null || fail "runa.rb is not Ruby"; fi
grep -v "aarch64-apple-darwin" "$dist/SHA256SUMS" >"$work/short"
if "$root/scripts/formula.sh" "$tag" "$work/short" >/dev/null 2>&1; then fail "formula.sh made a formula without an archive"; fi

echo "ok"
