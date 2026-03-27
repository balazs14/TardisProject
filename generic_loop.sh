#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'EOF'
Usage:
  ./generic_loop.sh -N <workers> -S <start-date> -E <end-date> <command with DATE placeholder>

Example:
  ./generic_loop.sh -N 4 -S 2025-12-01 -E 2025-12-06 \
    venv/bin/python -m tardis.download_and_convert --start-date DATE --end-date DATE --loglevel DEBUG

Notes:
  - DATE is replaced in every command argument for each day in [S, E] inclusive.
  - Dates must be YYYY-MM-DD.
EOF
}

N=1
START_DATE=""
END_DATE=""

while getopts ":N:S:E:h" opt; do
    case "$opt" in
        N) N="$OPTARG" ;;
        S) START_DATE="$OPTARG" ;;
        E) END_DATE="$OPTARG" ;;
        h) usage; exit 0 ;;
        :) echo "Missing value for -$OPTARG" >&2; usage; exit 2 ;;
        \?) echo "Unknown option: -$OPTARG" >&2; usage; exit 2 ;;
    esac
done
shift $((OPTIND - 1))

[[ -n "$START_DATE" ]] || { echo "-S is required" >&2; usage; exit 2; }
[[ -n "$END_DATE" ]] || { echo "-E is required" >&2; usage; exit 2; }
[[ "$N" =~ ^[1-9][0-9]*$ ]] || { echo "-N must be a positive integer" >&2; exit 2; }
[[ $# -gt 0 ]] || { echo "Command is required" >&2; usage; exit 2; }

if ! python3 - "$START_DATE" "$END_DATE" >/dev/null 2>&1 <<'PY'
import sys
from datetime import date

s = date.fromisoformat(sys.argv[1])
e = date.fromisoformat(sys.argv[2])
if s > e:
    raise SystemExit(1)
PY
then
    echo "Invalid date range: $START_DATE .. $END_DATE" >&2
    exit 2
fi

CMD_TEMPLATE=("$@")

pids=()
FAIL_DIR=$(mktemp -d)

cleanup() {
    trap - EXIT INT TERM HUP QUIT
    prune_pids
    if [[ ${#pids[@]} -gt 0 ]]; then
        echo "Stopping ${#pids[@]} running workers..." >&2
        kill -TERM "${pids[@]}" 2>/dev/null || true
        sleep 0.5
        kill -KILL "${pids[@]}" 2>/dev/null || true
    fi
    wait 2>/dev/null || true
    rm -rf "$FAIL_DIR" 2>/dev/null || true
}

on_int() {
    cleanup
    exit 130
}

on_term() {
    cleanup
    exit 143
}

trap cleanup EXIT
trap on_int INT
trap on_term TERM
trap on_term HUP QUIT

build_cmd_for_day() {
    local day="$1"
    local out=()
    local arg
    for arg in "${CMD_TEMPLATE[@]}"; do
        out+=("${arg//DATE/$day}")
    done
    printf '%s\0' "${out[@]}"
}

run_day() {
    local day="$1"
    local -a cmd=()
    while IFS= read -r -d '' token; do
        cmd+=("$token")
    done < <(build_cmd_for_day "$day")

    echo "[$day] ${cmd[*]}" >&2
    if ! "${cmd[@]}"; then
        echo "[$day] FAILED" >&2
        return 1
    fi
    return 0
}

prune_pids() {
    local -a new_pids=()
    local pid
    for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            new_pids+=("$pid")
        fi
    done
    pids=("${new_pids[@]}")
}

for day in $(python3 - "$START_DATE" "$END_DATE" <<'PY'
import sys
from datetime import date, timedelta

s = date.fromisoformat(sys.argv[1])
e = date.fromisoformat(sys.argv[2])
d = s
while d <= e:
    print(d.isoformat())
    d += timedelta(days=1)
PY
); do
    (
        if ! run_day "$day"; then
            : > "$FAIL_DIR/$day"
            exit 1
        fi
    ) &
    pids+=("$!")

    while [[ ${#pids[@]} -ge $N ]]; do
        wait -n 2>/dev/null || true
        prune_pids
    done
done

wait || true

if compgen -G "$FAIL_DIR/*" > /dev/null; then
    failed_list=$(basename -a "$FAIL_DIR"/* | tr '\n' ' ')
    echo "Some days failed: $failed_list" >&2
    exit 1
fi
