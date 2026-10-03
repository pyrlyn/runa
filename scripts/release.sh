#!/usr/bin/env bash
# Copyright (c) 2026 Ivan Tugay
# SPDX-License-Identifier: GPL-3.0-or-later
# Licensed under GPL-3.0 or later; see https://www.gnu.org/licenses/gpl-3.0.html

# P13.3. The one place a release version is decided; bump.yml runs it.
#
# The version in Cargo.toml is the version to release. It is raised only when
# that version is already tagged — which is what makes the first run publish
# 0.1.0 instead of skipping to 0.0.2.
#
# Usage: scripts/release.sh [patch|minor|major] [--dry-run|--local]
#   (none)     start the Bump workflow (gh workflow run bump.yml -f level=...)
#   --dry-run  print the version that would be released and change nothing
#   --local    make the version commit and stop (what bump.yml runs)
#
# This script never pushes and never tags. bump.yml (pyrlyn/infra) pushes the
# commit to `release/bump-vX.Y.Z`, opens a pull request, rebase-merges it once
# every required check is green, tags the commit that landed on main, creates a
# draft release and dispatches release.yml and release-variants.yml on the tag.
set -euo pipefail

cd "$(dirname "$0")/.."

level="${1:-patch}"
mode="${2:-}"
case "$level" in
  patch | minor | major) ;;
  *)
    echo "level must be patch, minor or major (got '$level')" >&2
    exit 2
    ;;
esac
case "$mode" in
  "")
    # The only way to a release: the Bump workflow (PR, checks, merge, tag).
    gh workflow run bump.yml -f level="$level"
    echo "Bump and release ($level) started: gh run list --workflow bump.yml"
    exit 0
    ;;
  --dry-run | --local) ;;
  *)
    echo "unknown option: $mode" >&2
    exit 2
    ;;
esac

# Tools come from mise unless the caller says otherwise.
CARGO="${CARGO:-mise exec -- cargo}"
CLIFF="${CLIFF:-mise exec -- git-cliff}"

current=$(grep -m1 '^version = ' Cargo.toml | cut -d'"' -f2)
version="$current"

if git rev-parse -q --verify "refs/tags/v$current" >/dev/null; then
  IFS=. read -r major minor patch <<<"$current"
  case "$level" in
    major) version="$((major + 1)).0.0" ;;
    minor) version="$major.$((minor + 1)).0" ;;
    patch) version="$major.$minor.$((patch + 1))" ;;
  esac
fi

echo "current $current -> release v$version"

if [ "$mode" = "--dry-run" ]; then
  echo "dry run: nothing written"
  exit 0
fi

if [ "$version" != "$current" ]; then
  # -i.bak keeps this working on BSD sed (macOS) as well as GNU.
  sed -i.bak "s|^version = \".*\"|version = \"$version\"|" Cargo.toml && rm -f Cargo.toml.bak
  # Cargo.lock carries the members' own versions too; -w touches workspace
  # members only.
  $CARGO update --workspace --quiet
  # Run before the commit, so the release commit itself is never in the notes
  # it generates.
  $CLIFF --tag "v$version" -o CHANGELOG.md
  git add Cargo.toml Cargo.lock CHANGELOG.md
  git commit --quiet -m "release: v$version"
  committed=yes
else
  # The untagged version already is the version to release: the first release
  # publishes what the workspace carries. Nothing to commit.
  committed=no
  echo "nothing to bump: v$current is untagged and is the version to release"
fi

if [ "$committed" = "yes" ]; then
  echo "local: version commit made, not pushed, not tagged"
else
  echo "local: no version commit (nothing to bump), nothing pushed, nothing tagged"
fi
