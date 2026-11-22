#!/bin/sh
# Installs a released runa: downloads the archive for this machine, checks it against the
# SHA256SUMS of the release and puts `runa` into a directory.
#
#   curl -fsSL https://github.com/pyrlyn/runa/releases/latest/download/install.sh | sh
#
# Environment:
#   RUNA_VERSION        a release (v1.2.3 or 1.2.3) instead of the latest one
#   RUNA_INSTALL_DIR    where `runa` goes (default: ~/.local/bin)
#   RUNA_DOWNLOAD_BASE  where the archives and SHA256SUMS are, instead of the GitHub release
#                       (a mirror)
set -eu

REPOSITORY="https://github.com/pyrlyn/runa"

fail() {
  echo "install.sh: $1" >&2
  exit 1
}

need() {
  command -v "$1" >/dev/null 2>&1 || fail "$1 is needed and was not found"
}

# The triple in the name of the archive that runs on this machine.
target() {
  os="$(uname -s)"
  machine="$(uname -m)"
  case "$os" in
    Darwin)
      case "$machine" in
        arm64) echo aarch64-apple-darwin ;;
        *) fail "runa is built for Apple Silicon only, and this Mac is $machine" ;;
      esac
      ;;
    Linux)
      # The executable is linked against glibc.
      if ldd --version 2>&1 | grep -qi musl; then
        fail "runa needs glibc, and this system uses musl"
      fi
      case "$machine" in
        x86_64 | amd64) echo x86_64-unknown-linux-gnu ;;
        aarch64 | arm64) echo aarch64-unknown-linux-gnu ;;
        *) fail "there is no runa for Linux on $machine" ;;
      esac
      ;;
    *) fail "there is no runa for $os" ;;
  esac
}

# GitHub answers /releases/latest with a redirect to the page of the newest release, so its tag
# is read without the API and its rate limit.
latest() {
  url="$(curl -fsSLI -o /dev/null -w '%{url_effective}' "$REPOSITORY/releases/latest")" ||
    fail "cannot ask $REPOSITORY for its latest release"
  case "$url" in
    */releases/tag/v*) echo "${url##*/}" ;;
    *) fail "$REPOSITORY has no release yet; name one with RUNA_VERSION" ;;
  esac
}

sha256() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d' ' -f1
  else
    shasum -a 256 "$1" | cut -d' ' -f1
  fi
}

linux_hint() {
  cat >&2 <<'EOF'
On Linux runa needs glibc and these system libraries:
  libcrypt.so.1   Fedora: libxcrypt-compat; Debian and Ubuntu have it in libcrypt1
  libz.so.1       zlib (zlib1g, zlib)
  gconv modules   openSUSE minimal images: glibc-locale-base
For downloads it also needs the system CA certificates (ca-certificates).
EOF
}

# Everything runs from here, so a script that arrived cut short does nothing.
main() {
  need curl
  need tar
  need uname
  need awk
  command -v sha256sum >/dev/null 2>&1 || need shasum

  target="$(target)"
  tag="${RUNA_VERSION:-}"
  if [ -z "$tag" ]; then tag="$(latest)"; fi
  case "$tag" in
    v*) ;;
    *) tag="v$tag" ;;
  esac
  base="${RUNA_DOWNLOAD_BASE:-$REPOSITORY/releases/download/$tag}"
  directory="${RUNA_INSTALL_DIR:-$HOME/.local/bin}"
  name="runa-$tag-$target"
  archive="$name.tar.gz"

  work="$(mktemp -d)"
  trap 'rm -rf "$work"' EXIT

  echo "Downloading $base/$archive"
  curl -fsSL --retry 3 -o "$work/$archive" "$base/$archive" ||
    fail "cannot download $base/$archive; is there a release $tag for $target?"
  curl -fsSL --retry 3 -o "$work/SHA256SUMS" "$base/SHA256SUMS" ||
    fail "cannot download $base/SHA256SUMS"

  # sha256sum writes `<sum>  <name>`, and `<sum> *<name>` for a file read as binary.
  expected="$(awk -v name="$archive" '{ sub(/^\*/, "", $2) } $2 == name { print $1 }' "$work/SHA256SUMS")"
  actual="$(sha256 "$work/$archive")"
  [ -n "$expected" ] || fail "the SHA256SUMS of $tag does not name $archive; nothing was installed"
  [ "$expected" = "$actual" ] ||
    fail "the SHA-256 of $archive is $actual, and the release says $expected; nothing was installed"

  tar -xzf "$work/$archive" -C "$work"
  [ -f "$work/$name/runa" ] || fail "$archive has no $name/runa in it"

  mkdir -p "$directory"
  # Written beside the old one and moved over it: a runa that is running keeps its file.
  cp "$work/$name/runa" "$directory/.runa.new.$$"
  chmod 755 "$directory/.runa.new.$$"
  mv -f "$directory/.runa.new.$$" "$directory/runa"

  if ! installed="$("$directory/runa" --version)"; then
    echo "install.sh: $directory/runa was installed and does not start" >&2
    if [ "$(uname -s)" = Linux ]; then linux_hint; fi
    exit 1
  fi
  echo "Installed $installed to $directory/runa"

  case ":$PATH:" in
    *":$directory:"*) ;;
    *) echo "$directory is not in PATH; add it to run \`runa\` by its name" ;;
  esac
}

main "$@"
