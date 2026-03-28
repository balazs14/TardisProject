#!/bin/bash
set -euo pipefail

N=${N:-4}  # max parallel workers; override with: N=8 ./deribit_chain_loop.sh

cd /my/TardisProject
source venv/bin/activate

# Kill all background children when this script exits or is interrupted.
pids=()
cleanup() {
    trap - EXIT INT TERM HUP QUIT
    if [[ ${#pids[@]} -gt 0 ]]; then
        echo "Killing ${#pids[@]} background workers..." >&2
        kill -TERM "${pids[@]}" 2>/dev/null || true
        sleep 0.5
        kill -KILL "${pids[@]}" 2>/dev/null || true
    fi
    wait 2>/dev/null || true
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

run_day() {
    local day=$1
    local cmd="venv/bin/python -m tardis.download_files \
        --data-type options_chain --exchange deribit --symbol OPTIONS \
        --start-date $day --end-date $day \
        --loglevel DEBUG --data-dir datasets/deribit_chain --resample-freq 5min"
    echo "$cmd" >&2
    if ! $cmd; then
        echo "$day failed, continuing" >&2
    fi
}

for day in $(python3 - << 'EOF'
from datetime import date, timedelta
import pandas as pd

start = date(2024, 1, 1)
d = (pd.Timestamp.now() - pd.Timedelta("1day")).date()
while d > start:
    print(d.isoformat())
    d -= timedelta(days=1)
EOF
); do
    run_day "$day" &
    pids+=($!)

    # Once N jobs are running, wait for one to finish before launching more.
    while [[ ${#pids[@]} -ge $N ]]; do
        wait -n 2>/dev/null || true
        # Remove finished pids from the array.
        new_pids=()
        for pid in "${pids[@]}"; do
            kill -0 "$pid" 2>/dev/null && new_pids+=("$pid")
        done
        pids=("${new_pids[@]}")
    done
done

# Wait for remaining workers.
wait
