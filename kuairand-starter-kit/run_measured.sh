#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
export PYTHONUTF8=1
set -a
. ../.env
set +a
export ANTHROPIC_WORKSPACE_ID=wrkspc_01B1ri56yfTa9r6U2Qr2akSY
echo "key set: ${ANTHROPIC_API_KEY:+yes} | workspace: $ANTHROPIC_WORKSPACE_ID"
python -m agent.controller --iterations 2 --budget 3
