# shellcheck shell=bash
# Shared by the integration tests in this directory: each runs the real executable against a real
# model in a home of its own, so it neither sees nor disturbs a daemon the user runs.
#
#   source lib.sh; integration_setup <runa executable> <models directory>

integration_setup() {
  runa="$1"
  home="$(mktemp -d)"
  export RUNA_HOME="$home"
  ln -s "$2" "$home/models"
  trap cleanup EXIT
}

pid_of_daemon() { "$runa" status --json | sed -n 's/.*"pid":\([0-9][0-9]*\).*/\1/p'; }

fail() {
  echo "FAILED: $1" >&2
  echo "--- daemon.log" >&2
  # The end of it: a test of the log's rotation fills the file.
  tail -n 200 "$home/logs/daemon.log" >&2 2>/dev/null || true
  exit 1
}

cleanup() {
  pid="$(pid_of_daemon || true)"
  if [ -n "$pid" ]; then kill -9 "$pid" 2>/dev/null || true; fi
  rm -rf "$home"
}

# Waits until the discovery file is gone: the daemon removes it as the last thing it does.
wait_for_stop() {
  for _ in $(seq 50); do
    [ -e "$home/run/daemon.json" ] || return 0
    sleep 0.1
  done
  return 1
}
