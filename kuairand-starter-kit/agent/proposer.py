"""Proposer — the only role that decides WHAT to try.

Three things make a proposal good on this task, and the earlier version enforced none of
them:

  1. It targets a segment that can actually move. 42.2% of validation users have uniform
     labels: nDCG is pinned and GAUC excludes them entirely. A change that only helps
     those users cannot register.
  2. It is not already refuted. The organizers published two dead ends with numbers, and
     two classes of change are provably zero by the geometry of within-user ranking. The
     direction registry carries all of them, and a proposal that re-derives one wastes an
     iteration the convergence rule cannot spare.
  3. Its own predicted effect exceeds the resolution floor. Paired noise on validation is
     sigma ~ 0.0005, so an idea whose author expects +0.0005 is unmeasurable here even if
     it is real. The harness rejects those before they cost a node.

It emits three candidates and picks one, in a single call. Convergence fires after three
consecutive non-improving iterations, so each iteration should be the best of several
ideas rather than the first one — and that costs one request, not three.

It may also decline to experiment and ask for a diagnostic instead. Answering "does this
column vary within a user at all?" takes three seconds; finding out by training takes a
minute and one of roughly ten iterations.
"""
import json

INSTRUCTIONS = """You are the proposer in an autonomous ML research loop on KuaiRand-Pure.

THE TASK, precisely. Within-user ranking over logged impressions. Label `long_view`.
Score = mean(GAUC, nDCG@5), both computed per user. Nothing is ever compared across
users, so only the ORDER inside one user's impression list matters.

WHAT THAT GEOMETRY IMPLIES — these are theorems, not opinions:
- Any score term that is constant across one user's impressions cannot change that user's
  order. A pure user-side feature added additively contributes EXACTLY ZERO.
  User information can only act through a cross with the item side.
- Any per-user monotone transform of the scores at inference is exactly zero: calibration,
  sigmoid, per-user standardisation. Both metrics read only the order.
- 42.2% of validation users have uniform labels (all-0 or all-1). Their nDCG is pinned and
  GAUC excludes them. No model change reaches them. Median list length is 4.

WHERE THE MOVABLE METRIC IS. Longer lists carry most of the headroom and most of the GAUC
weight; short lists are dominated by invariant users. `headroom_by_list_length` gives the
exact split. Ask for it rather than guessing.

THE NOISE FLOOR — this governs what is worth proposing. Five identity seeds give a
validation population std of 0.00032, and paired deltas run sigma ~ 0.0005. The
organizers' convergence threshold is eps = 0.002, roughly 4 sigma. So:
  - an effect below ~0.0015 is not measurable here, however real it is;
  - a proposal whose own `expected_delta_primary` is under 0.0015 will be REJECTED;
  - and three consecutive iterations under eps end the run, with the score locked at the
    best checkpoint so far. You get roughly ten iterations. Spend them on changes with a
    mechanism that could plausibly clear 0.002, not on tuning.

YOU HAVE A DIRECTION REGISTRY. It lists every direction with a status. REFUTED and NO_OP
directions are closed: do not propose them again unless you can state precisely what is
different about your version, in an `extends_refuted` field. Re-deriving a published dead
end is the single most wasteful thing you can do here.

YOU MAY ASK FOR EVIDENCE INSTEAD OF EXPERIMENTING. If a hypothesis rests on a premise you
have not checked, return action REQUEST_ANALYSIS and name one diagnostic. It runs in
seconds and comes back before your next turn. Use it when the answer would change which
candidate you choose.

Return ONE JSON object, no prose, no code fence. Either:

{
  "action": "EXPERIMENT",
  "candidates": [
    {
      "direction_id": "one of the registry ids, or a new short slug",
      "hypothesis": "one falsifiable sentence",
      "evidence": ["a specific fact from the packet, a diagnostic, or the journal"],
      "mechanism": "why this changes WITHIN-USER ORDER for the users that count",
      "target_metric": "GAUC" | "nDCG@5" | "both",
      "target_segment": "all" | "lists of 6+" | "T4" | ...,
      "expected_delta_primary": 0.003,
      "why_it_clears_the_floor": "why you expect more than 0.0015",
      "proposed_change": "concretely, what changes in the pipeline",
      "risk": "what might go wrong",
      "cost": "low" | "medium" | "high",
      "invalid_if": "the observation that would refute this",
      "files_to_modify": ["pipeline/model.py"],
      "extends_refuted": null
    },
    ... exactly 3 ...
  ],
  "chosen": 0,
  "rationale": "why this one first: expected value, cost, risk, and what it rules out"
}

or:

{
  "action": "REQUEST_ANALYSIS",
  "analysis": "no_op_screen",
  "params": {"column": "hourmin"},
  "question": "what you are trying to learn",
  "why_needed": "which candidate this would change, and how"
}

RULES. Only pipeline/features.py, pipeline/model.py and pipeline/train.py may change.
Post-impression signals are legal as auxiliary targets or as history from strictly
earlier rows, never as an input for the row being predicted. Raw columns come from
harness.adapter, which cannot return test rows. The test split does not exist for you."""

REQUIRED = ('direction_id', 'hypothesis', 'evidence', 'mechanism', 'proposed_change',
            'invalid_if', 'expected_delta_primary', 'files_to_modify')

FLOOR = 0.0015          # below this an effect is unmeasurable on this validation set


def validate(obj, closed_ids=()):
    """Reject malformed or self-defeating proposals, with a reason the model can act on."""
    if not isinstance(obj, dict):
        return False, 'reply was not a JSON object'

    action = obj.get('action', 'EXPERIMENT')
    if action == 'REQUEST_ANALYSIS':
        if not obj.get('analysis'):
            return False, 'REQUEST_ANALYSIS needs an "analysis" name'
        if not obj.get('why_needed'):
            return False, 'REQUEST_ANALYSIS needs "why_needed": which candidate it changes'
        return True, 'ok'

    cands = obj.get('candidates')
    if not isinstance(cands, list) or len(cands) < 1:
        return False, '"candidates" must be a non-empty list'
    for i, c in enumerate(cands):
        missing = [k for k in REQUIRED if c.get(k) in (None, '', [])]
        if missing:
            return False, 'candidate %d is missing %s' % (i, ', '.join(missing))
        if not isinstance(c.get('evidence'), list) or not c['evidence']:
            return False, 'candidate %d cites no evidence' % i
        try:
            d = float(c['expected_delta_primary'])
        except (TypeError, ValueError):
            return False, 'candidate %d: expected_delta_primary must be a number' % i
        if d < FLOOR:
            return False, ('candidate %d expects %+.4f, below the %0.4f resolution floor. '
                           'Even if real it could not be measured here — propose something '
                           'with a larger mechanism.' % (i, d, FLOOR))
        did = str(c.get('direction_id', ''))
        if did in closed_ids and not c.get('extends_refuted'):
            return False, ('candidate %d re-proposes %r, which is already refuted or a '
                           'provable no-op. Either choose an open direction or set '
                           '"extends_refuted" explaining what is different.' % (i, did))
    ch = obj.get('chosen', 0)
    if not isinstance(ch, int) or not (0 <= ch < len(cands)):
        return False, '"chosen" must index into candidates'
    return True, 'ok'


def build_user_message(journal_digest, parent, iteration, budget_note,
                       directions, catalogue, analysis_results=None):
    parts = [
        'Iteration %d.' % iteration,
        '',
        'Current best validation primary: %.5f (%s).' % (parent['primary'], parent['label']),
        '',
        '=== DIRECTION REGISTRY ===',
        directions,
        '',
        '=== DIAGNOSTICS YOU MAY REQUEST ===',
        '\n'.join('  %-26s %s' % (k, v) for k, v in catalogue.items()),
        '',
        '=== WHAT THIS RUN HAS TRIED ===',
        journal_digest or '  (nothing yet — this is the first experiment)',
    ]
    if analysis_results:
        parts += ['', '=== DIAGNOSTICS YOU ALREADY REQUESTED ===',
                  json.dumps(analysis_results, indent=1)[:4000]]
    parts += ['', budget_note, '',
              'Propose three candidates and choose one, or request one diagnostic.']
    return '\n'.join(parts)


def propose(llm, journal_digest, parent, iteration, budget_note,
            directions='', catalogue=None, closed_ids=(), analysis_results=None):
    """Returns (proposal, usage_record, rejections).

    `rejections` lists the schema failures that forced a retry. A retry is a whole extra
    call at roughly $0.16, so the reason is returned rather than swallowed — otherwise
    the waste hides inside a doubled proposer bill with nothing to diagnose.
    """
    msg = build_user_message(journal_digest, parent, iteration, budget_note,
                             directions, catalogue or {}, analysis_results)
    obj, rec = llm.ask_json('proposer', INSTRUCTIONS, msg, effort='high')
    ok, why = validate(obj, closed_ids)
    if ok:
        return obj, rec, []
    rejections = [why]
    obj2, rec2 = llm.ask_json(
        'proposer', INSTRUCTIONS,
        msg + '\n\nYour previous proposal was rejected: %s\nFix it.' % why,
        effort='high')
    ok2, why2 = validate(obj2, closed_ids)
    if not ok2:
        rejections.append(why2)
        raise ValueError('proposer produced an invalid proposal twice: %s' % why2)
    return obj2, rec2, rejections


def digest(entries, limit=14):
    """Compact the journal for the prompt. Verdicts and deltas, not transcripts."""
    lines = []
    for e in entries[-limit:]:
        p = e.get('payload', {})
        h = (p.get('hypothesis') or '')[:100]
        d = p.get('delta_vs_parent')
        d = ('%+.5f' % d) if isinstance(d, (int, float)) else 'failed'
        lines.append('  iter %-3s %-12s delta %-9s %s'
                     % (p.get('iteration', '?'), p.get('verdict', '?'), d, h))
    return '\n'.join(lines)


if __name__ == '__main__':
    print(json.dumps({'instructions_chars': len(INSTRUCTIONS),
                      'required_fields': list(REQUIRED),
                      'resolution_floor': FLOOR}, indent=2))
