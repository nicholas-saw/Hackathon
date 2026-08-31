"""Reflector — decides what a result MEANS, not just whether it went up.

The verdict it emits is the artefact judges read for Innovation. A ledger entry that
records a prediction, then a miss, then a mechanism update is worth more than one that
records only outcomes — so the reflector is always shown what was predicted before it is
shown what happened.
"""

INSTRUCTIONS = """You are the reflector in an autonomous ML research loop on KuaiRand-Pure
(within-user ranking, label long_view, primary = mean(GAUC, nDCG@5)).

You are given a hypothesis, what it predicted, the code change, and what actually
happened. Decide what it means. Return ONE JSON object, no prose:

{
  "verdict": "KEEP" | "REVERT" | "INCONCLUSIVE",
  "reason": "what the number tells us, in terms of the mechanism that was claimed",
  "mechanism_update": "what we now believe about why this task behaves as it does",
  "direction_status": "live" | "closed",
  "attribution": "implementation" | "objective" | "unclear",
  "deprioritise": ["directions this result argues against"],
  "next_information_need": "what would most reduce uncertainty next, or null"
}

ATTRIBUTION IS THE FIELD MOST OFTEN GOT WRONG. A single result tells you what THIS CODE
did. It rarely tells you what the OBJECTIVE does, and conflating the two has already cost
this project real ground:

- Nine implementations of "within-user listwise softmax" scored from -0.00318 to
  +0.00162. Reading any one of them as a verdict on listwise would have been wrong seven
  times out of nine, and the two earliest misses did close the direction -- three runs
  before the implementation that became the banked submission arrived.
- The spread across implementations of one label measured 5.2x the spread across seeds of
  one implementation. Implementation detail, not the objective, is the dominant term.

So use "objective" ONLY when the comparison actually isolates the loss -- same model, same
features, same batching, same seeds, one thing changed. Otherwise say "implementation",
or "unclear" if you cannot tell. Set `direction_status: "closed"` only when attribution is
"objective"; a failed implementation does not close a direction, and saying so in
`deprioritise` is how a good direction gets buried under bad code.

Calibration you must apply:
- The paired noise floor on this validation set is sigma ~ 0.0005 primary. A delta under
  about 0.001 is not evidence of anything. Say INCONCLUSIVE rather than inventing a story
  for noise.
- The organizers' convergence rule is eps = 0.002 over N = 3 iterations, so a delta below
  eps does not stop the run on its own, but three in a row do.
- KEEP means the change becomes the new parent. Only keep a change you believe is real.
- A refutation is a result. If the mechanism was sound and the sign was wrong, say so and
  say what that implies — that is more useful than "did not work"."""


def build_user_message(hypothesis, diff, metrics, parent, failure=None):
    parts = ['HYPOTHESIS', '']
    for k in ('hypothesis', 'mechanism', 'expected_result', 'invalid_if'):
        if hypothesis.get(k):
            parts.append('%s: %s' % (k, hypothesis[k]))
    parts += ['', 'CODE CHANGE', diff[:6000] if diff else '(none)', '']
    if failure:
        parts += ['OUTCOME: the experiment FAILED to run.',
                  'failure class: %s' % failure.get('failure'),
                  (failure.get('stderr') or '')[-1500:], '']
    else:
        parts += ['OUTCOME (validation only)',
                  'parent primary : %.5f' % parent['primary'],
                  'this   primary : %.5f' % metrics['primary'],
                  'delta          : %+.5f  (noise floor sigma ~ 0.0005, eps = 0.002)' % (
                      metrics['primary'] - parent['primary']),
                  'GAUC           : %.5f' % metrics['GAUC'],
                  'nDCG@5         : %.5f' % metrics['nDCG@5'], '']
    parts.append('Give your verdict.')
    return '\n'.join(parts)


def validate(obj):
    if not isinstance(obj, dict):
        return False, 'reply was not a JSON object'
    if obj.get('verdict') not in ('KEEP', 'REVERT', 'INCONCLUSIVE'):
        return False, '"verdict" must be KEEP, REVERT or INCONCLUSIVE'
    if not obj.get('reason'):
        return False, '"reason" is required'
    return True, 'ok'


def reflect(llm, hypothesis, diff, metrics, parent, failure=None):
    msg = build_user_message(hypothesis, diff, metrics, parent, failure)
    obj, rec = llm.ask_json('reflector', INSTRUCTIONS, msg, effort='medium')
    ok, why = validate(obj)
    if not ok:
        obj, rec = llm.ask_json('reflector', INSTRUCTIONS,
                                msg + '\n\nRejected: %s' % why, effort='medium')
        ok, why = validate(obj)
        if not ok:
            raise ValueError('reflector returned an invalid verdict twice: %s' % why)
    return obj, rec


def offline_verdict(metrics, parent, failure=None, accept=0.0014):
    """Deterministic fallback used by --dry-run, and if the reflector call fails.

    Not a substitute for the reflector's reasoning — it produces a verdict so the loop
    can continue, and the journal records that no model was consulted for this one.
    """
    if failure:
        return {'verdict': 'REVERT', 'reason': 'experiment failed to run (%s)'
                % failure.get('failure'), 'mechanism_update': '', 'direction_status': 'live', 'attribution': 'unclear',
                'deprioritise': [], 'next_information_need': None, 'offline': True}
    d = metrics['primary'] - parent['primary']
    if d > accept:
        v, why = 'KEEP', 'delta %+.5f clears the 2-sigma accept bar (%.4f)' % (d, accept)
    elif d < -accept:
        v, why = 'REVERT', 'delta %+.5f is a real regression' % d
    else:
        v, why = 'INCONCLUSIVE', 'delta %+.5f is inside the noise floor' % d
    return {'verdict': v, 'reason': why, 'mechanism_update': '',
            'direction_status': 'live', 'attribution': 'unclear', 'deprioritise': [],
            'next_information_need': None, 'offline': True}
