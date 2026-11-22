#!/usr/bin/env bash
# The one place a release version is decided; the Bump workflow (.github/workflows/bump.yml)
# runs it with --local.
#
# gradle.properties holds the version. A `-dev` suffix marks a tree that was never released and
# is dropped. The number is raised only when it is tagged already, so a version that was written
# by hand and never released is released as it stands.
#
# Usage: scripts/release.sh [patch|minor|major] [--dry-run|--local]
#   (none)     start the Bump workflow (gh workflow run bump.yml -f level=...)
#   --dry-run  print the version that would be released and change nothing
#   --local    make the version commit and stop
#
# It never pushes and never tags: the Bump workflow pushes the commit to a branch, opens a pull
# request, merges it once the required checks are green, tags the commit that landed on main,
# creates the draft release and starts release.yml on the tag.
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

current="$(sed -n 's/^version=//p' gradle.properties)"
version="${current%-dev}"
if ! [[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "gradle.properties has version '$current'; expected X.Y.Z or X.Y.Z-dev" >&2
  exit 1
fi

if git rev-parse -q --verify "refs/tags/v$version" >/dev/null; then
  IFS=. read -r major minor patch <<<"$version"
  case "$level" in
    major) version="$((major + 1)).0.0" ;;
    minor) version="$major.$((minor + 1)).0" ;;
    patch) version="$major.$minor.$((patch + 1))" ;;
  esac
fi

echo "current $current -> release v$version"
# The Bump workflow reads the version from here; without it it asks cargo, which this
# repository does not have.
if [ -n "${GITHUB_OUTPUT:-}" ]; then echo "version=$version" >>"$GITHUB_OUTPUT"; fi

if [ "$mode" = "--dry-run" ]; then
  echo "dry run: nothing written"
  exit 0
fi

if [ "$version" = "$current" ]; then
  echo "local: v$current is untagged and is the version to release; nothing to commit"
  exit 0
fi

# -i.bak keeps this working with BSD sed (macOS) as well as GNU sed.
sed -i.bak "s|^version=.*|version=$version|" gradle.properties && rm -f gradle.properties.bak
git add gradle.properties
git commit --quiet -m "release: v$version"
echo "local: version commit made, not pushed, not tagged"
