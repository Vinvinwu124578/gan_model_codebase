#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
REPO_ROOT="${SCRIPT_DIR:h}"
JOGGER_PYTHON="/Users/vincent/Downloads/tactip_experiment_tactistruct/.venv/bin/python"

if [[ ! -x "$JOGGER_PYTHON" ]]; then
  print -u2 "Required project Python is missing: $JOGGER_PYTHON"
  print -u2 "Restore the tactip_experiment_tactistruct .venv before starting the Jogger."
  exit 1
fi

exec "$JOGGER_PYTHON" "$REPO_ROOT/tools/coverage_board_sampling_jogger.py" "$@"
