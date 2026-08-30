"""The agent: three LLM roles and three deterministic ones.

Only proposer, coder and reflector call a model. Controller, governor and the arbiter in
`controller.py` are plain code, because Feasibility is scored on tokens and wall-clock and
anything expressible as a rule should not be paying per token to be re-derived.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

__all__ = ['ROOT']
