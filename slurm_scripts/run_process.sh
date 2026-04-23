#!/usr/bin/env bash
# Run process_to_hdf5.py from the Thesis directory.
#
# Runs on CERN EOS - configure paths
#
# Usage:
#   ./run_process.sh               # process all ROOT files
#   ./run_process.sh zb.root       # process one file only
#   ./run_process.sh zb.root --dry # dry run (just print what would run)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RAW_DIR="Data/raw_root"
OUT_DIR="Data/hdf5_files"
LOGDIR="logs"
mkdir -p "$LOGDIR"

ONLY_ARG=""
if [[ "${1:-}" == *.root ]]; then
    ONLY_ARG="--only $1"
    LOG_NAME="${1%.root}"
    shift
else
    LOG_NAME="all"
fi

if [[ "${1:-}" == "--dry" ]]; then
    echo "Would run:"
    echo "  python3 process_to_hdf5.py --raw-dir $RAW_DIR --out-dir $OUT_DIR $ONLY_ARG"
    exit 0
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOGFILE="$LOGDIR/process_${LOG_NAME}_${TIMESTAMP}.log"

echo "Starting processing — logging to $LOGFILE"
echo "  RAW_DIR : $RAW_DIR"
echo "  OUT_DIR : $OUT_DIR"
[[ -n "$ONLY_ARG" ]] && echo "  File    : ${ONLY_ARG#--only }"

python3 process_to_hdf5.py \
    --raw-dir "$RAW_DIR" \
    --out-dir "$OUT_DIR" \
    $ONLY_ARG \
    2>&1 | tee "$LOGFILE"

echo "Done. Log saved to $LOGFILE"
