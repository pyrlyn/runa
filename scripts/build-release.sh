#!/usr/bin/env bash
# Builds the release executable of one target, for release.yml.
#
#   TARGET=<triple> [TAG=vX.Y.Z] scripts/build-release.sh
#
# The executable lands in build/release/<triple>/runa. Targets are named as the release archives
# name them, by the triples that installers and package managers recognise. With TAG the
# executable reports that version, whatever gradle.properties says: a test tag (vX.Y.Z-test.N)
# is made by hand, without a version commit.
set -euo pipefail

cd "$(dirname "$0")/.."

case "${TARGET:-}" in
  aarch64-apple-darwin) kotlin=macosArm64 task=MacosArm64 ;;
  x86_64-unknown-linux-gnu) kotlin=linuxX64 task=LinuxX64 ;;
  aarch64-unknown-linux-gnu) kotlin=linuxArm64 task=LinuxArm64 ;;
  *)
    echo "TARGET must be aarch64-apple-darwin, x86_64-unknown-linux-gnu or aarch64-unknown-linux-gnu (got '${TARGET:-}')" >&2
    exit 2
    ;;
esac

version=()
if [ -n "${TAG:-}" ]; then version=("-Pversion=${TAG#v}"); fi

# ${version[@]+...}: an empty array is an unbound variable for the bash 3.2 of macOS.
./gradlew ${version[@]+"${version[@]}"} ":app:linkReleaseExecutable$task" --console=plain

out="build/release/$TARGET"
mkdir -p "$out"
cp "app/build/bin/$kotlin/releaseExecutable/runa.kexe" "$out/runa"
ls -l "$out/runa"
