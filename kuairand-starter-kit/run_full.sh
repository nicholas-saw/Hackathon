#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
export PYTHONUTF8=1
set -a
. ../.env
set +a
export ANTHROPIC_WORKSPACE_ID=wrkspc_01B1ri56yfTa9r6U2Qr2akSY
echo "FULL RUN  key=${ANTHROPIC_API_KEY:+set}  workspace=$ANTHROPIC_WORKSPACE_ID"
python -c "from agent.llm import MODEL; print('model:', MODEL)"
export MAX_WALL_S="${MAX_WALL_S:-21600}"
echo "wall-clock cap: $((MAX_WALL_S/60)) min"
python -m agent.controller --iterations "${ITERS:-50}" --budget "${BUDGET:-14}"
