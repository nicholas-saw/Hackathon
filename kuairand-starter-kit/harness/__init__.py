"""Harness — deterministic infrastructure around the agent's editable pipeline.

Frozen during the measured run. Nothing here calls an LLM; everything that can be
expressed as a rule lives on this side of the boundary so it costs no tokens.

Importing this package puts the frozen ``kit/`` and the editable ``pipeline/`` on
sys.path, so ``from data import load`` / ``from features import encode`` resolve the
same way they do for ``pipeline/train.py``.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KIT = os.path.join(ROOT, 'kit')
PIPELINE = os.path.join(ROOT, 'pipeline')
DATA_DIR = os.path.join(ROOT, 'KuaiRand-Pure', 'data')
RUNLOGS = os.path.join(ROOT, 'runlogs')
SUBMISSIONS = os.path.join(ROOT, 'submissions')
CONTEXT = os.path.join(ROOT, 'context')

for _p in (KIT, PIPELINE, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Official split sizes. Any loader that disagrees is wrong; assert, never coerce.
SPLIT_SIZES = {'train': 1141112, 'valid': 124909, 'test': 170588}

# The eleven post-impression outcome columns, mirrored from pipeline/features.py so the
# harness can enforce the rule without importing the agent's editable surface.
LEAKY_COLUMNS = frozenset({
    'label', 'long_view', 'is_click', 'is_like', 'is_follow', 'is_comment',
    'is_forward', 'is_hate', 'play_time_ms', 'profile_stay_time',
    'comment_stay_time', 'is_profile_enter',
})

__all__ = ['ROOT', 'KIT', 'PIPELINE', 'DATA_DIR', 'RUNLOGS', 'SUBMISSIONS', 'CONTEXT',
           'SPLIT_SIZES', 'LEAKY_COLUMNS']
