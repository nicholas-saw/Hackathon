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
python -m agent.controller --iterations 50 --budget 14
