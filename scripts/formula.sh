#!/usr/bin/env bash
# Prints the Homebrew formula of a release, which release.yml attaches to it as runa.rb.
#
#   scripts/formula.sh <tag> <SHA256SUMS>
#
# SHA256SUMS is the file of the release: one `<sha256>  <archive>` line for each archive.
set -euo pipefail

tag="${1:?usage: formula.sh <tag> <SHA256SUMS>}"
sums="${2:?usage: formula.sh <tag> <SHA256SUMS>}"
base="https://github.com/pyrlyn/runa/releases/download/$tag"

# The checksum of the archive for one target; a release without that archive has no formula.
sum() {
  local archive="runa-$tag-$1.tar.gz" found
  found="$(awk -v name="$archive" '{ sub(/^\*/, "", $2) } $2 == name { print $1 }' "$sums")"
  if ! [[ "$found" =~ ^[0-9a-f]{64}$ ]]; then
    echo "formula.sh: $sums has no SHA-256 for $archive" >&2
    exit 1
  fi
  echo "$found"
}

mac_arm="$(sum aarch64-apple-darwin)"
linux_arm="$(sum aarch64-unknown-linux-gnu)"
linux_intel="$(sum x86_64-unknown-linux-gnu)"

cat <<EOF
# Written by scripts/formula.sh for the release $tag; do not edit.
class Runa < Formula
  desc "Local models from the terminal: download, load, serve and chat"
  homepage "https://github.com/pyrlyn/runa"
  version "${tag#v}"
  license "GPL-3.0-or-later"

  on_macos do
    depends_on arch: :arm64

    on_arm do
      url "$base/runa-$tag-aarch64-apple-darwin.tar.gz"
      sha256 "$mac_arm"
    end
  end

  on_linux do
    on_arm do
      url "$base/runa-$tag-aarch64-unknown-linux-gnu.tar.gz"
      sha256 "$linux_arm"
    end
    on_intel do
      url "$base/runa-$tag-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "$linux_intel"
    end
  end

  def install
    bin.install "runa"
    generate_completions_from_executable(bin/"runa", "completion")
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/runa --version")
  end
end
EOF
