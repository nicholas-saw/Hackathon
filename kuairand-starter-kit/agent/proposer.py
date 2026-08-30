"""Proposer — the only role that decides WHAT to try.

Emits three hypotheses and picks one, in a single call. That is not decoration: the
convergence rule stops the run after three consecutive iterations that fail to improve
validation by more than eps=0.002, and the measured paired noise floor on this validation
set is sigma ~ 0.0005, so eps is about 4 sigma and a null iteration essentially never
clears it. Best-of-three per iteration is the cheapest way to raise the chance that an
iteration is not null, and it costs one request, not three.

Every hypothesis must cite evidence and state a mechanism and a falsifier. A proposal
that cannot say what would refute it is not a hypothesis, and the harness rejects it.
"""
import json

INSTRUCTIONS = """You are the proposer in an autonomous ML research loop working on
KuaiRand-Pure: within-user ranking of logged impressions, label `long_view`, scored by
GAUC and nDCG@5 with primary = their mean.

Your job is to decide what to try next and why. You do not write code.

Return ONE JSON object, no prose, no code fence:

{
  "candidates": [
    {
      "hypothesis": "one sentence, falsifiable",
      "evidence": ["fact from the packet or the journal", "..."],
      "mechanism": "why this should change ranking quality FOR THIS METRIC",
      "target_metric": "GAUC" | "nDCG@5" | "both",
      "target_segment": "all" | "T1" | "T4" | "short lists" | ...,
      "proposed_change": "what changes in the pipeline, concretely",
      "expected_result": "a number and a direction, e.g. +0.003 valid primary",
      "risk": "what might go wrong",
      "cost": "low" | "medium" | "high",
      "invalid_if": "the observation that would refute this",
      "files_to_modify": ["pipeline/model.py", ...]
    },
    ... exactly 3 candidates ...
  ],
  "chosen": 0,
  "rationale": "why this one first, in terms of expected value, cost, risk and novelty",
  "action": "EXPERIMENT"
}

Rules that matter:
- Only pipeline/features.py, pipeline/model.py and pipeline/train.py may change.
- Post-impression signals (is_click, is_like, play_time_ms, ...) are legal as auxiliary
  TARGETS or as history from strictly earlier rows. They are never inputs for the row
  being predicted. The harness rejects that statically, before your code runs.
- The test split does not exist for you. Selection happens on validation.
- An improvement under about 0.002 validation primary is inside this dataset's noise
  floor. Prefer changes with a mechanism that could plausibly clear it.
- Do not repeat a direction the journal has already refuted. Say so if you are extending
  a refuted direction and explain what is different this time."""

REQUIRED = ('hypothesis', 'evidence', 'mechanism', 'proposed_change', 'invalid_if',
            'files_to_modify')


def validate(obj):
    """Reject malformed proposals with a reason the model can act on."""
    if not isinstance(obj, dict):
        return False, 'reply was not a JSON object'
    cands = obj.get('candidates')
    if not isinstance(cands, list) or len(cands) < 1:
        return False, '"candidates" must be a non-empty list'
    for i, c in enumerate(cands):
        missing = [k for k in REQUIRED if not c.get(k)]
        if missing:
            return False, 'candidate %d is missing %s' % (i, ', '.join(missing))
        if not isinstance(c.get('evidence'), list) or not c['evidence']:
            return False, 'candidate %d cites no evidence' % i
    ch = obj.get('chosen', 0)
    if not isinstance(ch, int) or not (0 <= ch < len(cands)):
        return False, '"chosen" must index into candidates'
    return True, 'ok'


def build_user_message(journal_digest, parent, iteration, budget_note):
    parts = [
        'Iteration %d.' % iteration,
        '',
        'Current best validation primary: %.5f (%s).' % (parent['primary'], parent['label']),
        '',
        'What has been tried so far:',
        journal_digest or '  (nothing yet — this is the first experiment)',
        '',
        budget_note,
        '',
        'Propose three candidates and choose one.',
    ]
    return '\n'.join(parts)


def propose(llm, journal_digest, parent, iteration, budget_note):
    """Returns (proposal_dict, usage_record)."""
    msg = build_user_message(journal_digest, parent, iteration, budget_note)
    obj, rec = llm.ask_json('proposer', INSTRUCTIONS, msg, effort='high')
    ok, why = validate(obj)
    if not ok:
        obj2, rec2 = llm.ask_json(
            'proposer', INSTRUCTIONS,
            msg + '\n\nYour previous proposal was rejected: %s. Fix it.' % why,
            effort='high')
        ok2, why2 = validate(obj2)
        if not ok2:
            raise ValueError('proposer produced an invalid proposal twice: %s' % why2)
        return obj2, rec2
    return obj, rec


def digest(entries, limit=14):
    """Compact the journal for the prompt. Full logs on every call are the fastest way
    to blow the token budget; the proposer needs verdicts, not transcripts."""
    lines = []
    for e in entries[-limit:]:
        p = e.get('payload', {})
        h = (p.get('hypothesis') or '')[:110]
        v = p.get('verdict', '?')
        d = p.get('delta_vs_parent')
        d = ('%+.5f' % d) if isinstance(d, (int, float)) else 'n/a'
        lines.append('  iter %s | %s | delta %s | %s' % (p.get('iteration', '?'), v, d, h))
    return '\n'.join(lines)


if __name__ == '__main__':
    print(json.dumps({'instructions_chars': len(INSTRUCTIONS)}, indent=2))
