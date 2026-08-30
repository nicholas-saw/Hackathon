"""Stability-tested selection, so the run cannot ship a regression.

No mechanism can make every hypothesis improve the score — a hypothesis that cannot fail
is not research, and the rubric scores refutations as legitimate output. What CAN be
guaranteed is weaker and more useful: the submitted result never goes backwards.

The gap this closes is the winner's curse. Picking `max(nodes, key=validation_primary)`
over N candidates does not select the best model, it selects the luckiest draw. On a
metric whose paired noise is sigma ~ 0.0005, best-of-10 inflates the apparent score by
roughly 1.5 sigma, and that inflation is exactly what fails to transfer to the hidden
test set. Measured on this dataset: the validation-argmax of 21 statistically
indistinguishable runs scored BELOW the official baseline on test.

So selection here is not an argmax. A candidate must:
  1. beat the incumbent on the pooled validation score, AND
  2. beat it on a majority of independent user folds,
and the designated result is whichever of {incumbent, best stable candidate, ensemble of
the stable candidates} wins — with a floor tripwire that refuses to ship anything worse
than the banked floor.

Folds are cheap: each is a fifth of validation, so a five-fold sweep of six candidates is
about thirty evaluator calls on small inputs, seconds at designation time.
"""
import hashlib

import numpy as np

from .score import evaluate_raw, rank_average

N_FOLDS = 5
MIN_FOLD_WINS = 4          # of N_FOLDS; 4/5 is a one-sided sign test at p ~ 0.19 per
                           # candidate, tightened by also requiring the pooled win


def user_folds(users, k=N_FOLDS):
    """Deterministic user-level folds. Splitting by user, never by row, keeps every
    user's impression list intact — the metric is computed within users, so a row-level
    split would silently change what is being measured."""
    users = np.asarray(users)
    uniq = np.unique(users)
    bucket = {u: int(hashlib.md5(str(u).encode()).hexdigest(), 16) % k for u in uniq}
    return np.array([bucket[u] for u in users])


def fold_scores(users, labels, preds, folds=None, k=N_FOLDS):
    """Primary on each user fold. Returns a list of k floats."""
    users = np.asarray(users)
    labels = np.asarray(labels)
    preds = np.asarray(preds, dtype=float)
    if folds is None:
        folds = user_folds(users, k)
    out = []
    for f in range(k):
        m = folds == f
        out.append(float(evaluate_raw(users[m], labels[m], preds[m])['primary']))
    return out


def compare(users, labels, preds_a, preds_b, folds=None, k=N_FOLDS):
    """Is A better than B, robustly? Pooled delta plus a per-fold win count."""
    if folds is None:
        folds = user_folds(users, k)
    pa = float(evaluate_raw(users, labels, preds_a)['primary'])
    pb = float(evaluate_raw(users, labels, preds_b)['primary'])
    fa = fold_scores(users, labels, preds_a, folds, k)
    fb = fold_scores(users, labels, preds_b, folds, k)
    wins = sum(1 for x, y in zip(fa, fb) if x > y)
    diffs = np.array(fa) - np.array(fb)
    return {'pooled_a': round(pa, 5), 'pooled_b': round(pb, 5),
            'pooled_delta': round(pa - pb, 5),
            'fold_wins': int(wins), 'folds': k,
            'fold_deltas': [round(float(d), 5) for d in diffs],
            'fold_delta_mean': round(float(diffs.mean()), 5),
            'fold_delta_sd': round(float(diffs.std(ddof=1)), 5),
            'stable': bool(pa > pb and wins >= MIN_FOLD_WINS)}


def designate(users, labels, candidates, baseline_preds, floor_preds=None,
              max_ensemble=5, log=None):
    """Choose what to submit.

    candidates: [{'iteration': int, 'label': str, 'valid': ndarray, 'test': ndarray}, ...]
    Returns (choice, report) where choice carries 'valid'/'test' vectors to submit.

    The order of preference is deliberate: an ensemble of stable candidates beats a single
    stable candidate beats the baseline. Nothing that fails the stability test is ever
    designated on the strength of a pooled score alone.
    """
    say = log or (lambda *_a, **_k: None)
    users = np.asarray(users)
    labels = np.asarray(labels)
    folds = user_folds(users)
    report = {'n_candidates': len(candidates), 'folds': N_FOLDS,
              'min_fold_wins': MIN_FOLD_WINS, 'evaluated': [], 'decisions': []}

    base_primary = float(evaluate_raw(users, labels, baseline_preds)['primary'])
    report['baseline_primary'] = round(base_primary, 5)

    # 1. rank by pooled score, then stability-test the plausible ones only
    ranked = sorted(candidates, key=lambda c: -float(
        evaluate_raw(users, labels, c['valid'])['primary']))
    stable = []
    for c in ranked[:max_ensemble + 2]:
        cmp = compare(users, labels, c['valid'], baseline_preds, folds)
        rec = {'iteration': c['iteration'], 'label': c.get('label', '')[:70], **cmp}
        report['evaluated'].append(rec)
        say('  cand iter %-3s pooled %+.5f  folds won %d/%d  %s'
            % (c['iteration'], cmp['pooled_delta'], cmp['fold_wins'], N_FOLDS,
               'STABLE' if cmp['stable'] else 'not stable'))
        if cmp['stable']:
            stable.append(c)

    # 2. incumbent is the baseline until something stable beats it
    choice = {'kind': 'baseline', 'iteration': 0, 'label': 'official baseline',
              'valid': baseline_preds,
              'test': next((c['test'] for c in candidates if c['iteration'] == 0), None),
              'primary': base_primary}

    if stable:
        best = stable[0]
        choice = {'kind': 'single', 'iteration': best['iteration'],
                  'label': best.get('label', ''), 'valid': best['valid'],
                  'test': best['test'],
                  'primary': float(evaluate_raw(users, labels, best['valid'])['primary'])}
        report['decisions'].append('best stable single candidate: iteration %s'
                                   % best['iteration'])

        # 3. an ensemble of the stable candidates, if it is itself stable vs the single
        if len(stable) >= 2:
            mem = stable[:max_ensemble]
            ens_valid = rank_average(users, [c['valid'] for c in mem])
            cmp = compare(users, labels, ens_valid, best['valid'], folds)
            report['ensemble'] = {'members': [c['iteration'] for c in mem], **cmp}
            say('  ensemble of %d: pooled %+.5f vs best single, folds won %d/%d  %s'
                % (len(mem), cmp['pooled_delta'], cmp['fold_wins'], N_FOLDS,
                   'STABLE' if cmp['stable'] else 'not stable'))
            if cmp['stable']:
                # test vectors must be combined over TEST users, not validation users
                choice = {'kind': 'ensemble',
                          'iteration': -1,
                          'label': 'rank-average of iterations %s'
                                   % ','.join(str(c['iteration']) for c in mem),
                          'valid': ens_valid,
                          'test_members': [c['test'] for c in mem],
                          'primary': float(evaluate_raw(users, labels, ens_valid)['primary'])}
                report['decisions'].append('ensemble beat the best single, stably')
    else:
        report['decisions'].append(
            'no candidate beat the baseline on both the pooled score and %d/%d folds; '
            'designating the baseline rather than a lucky draw' % (MIN_FOLD_WINS, N_FOLDS))
        say('  no stable improvement over the baseline — designating the baseline')

    # 4. floor tripwire: never ship worse than what was already banked
    if floor_preds is not None:
        cmp = compare(users, labels, choice['valid'], floor_preds, folds)
        report['vs_floor'] = cmp
        if cmp['pooled_delta'] < 0:
            report['decisions'].append(
                'designated result scored %+.5f against the banked floor; shipping the '
                'floor instead' % cmp['pooled_delta'])
            say('  TRIPWIRE: choice is %+.5f vs the banked floor — shipping the floor'
                % cmp['pooled_delta'])
            choice = {'kind': 'floor', 'iteration': -2, 'label': 'banked floor',
                      'valid': floor_preds, 'test': None,
                      'primary': float(evaluate_raw(users, labels, floor_preds)['primary'])}

    report['chosen'] = {'kind': choice['kind'], 'iteration': choice['iteration'],
                        'label': choice['label'],
                        'validation_primary': round(choice['primary'], 5),
                        'delta_vs_baseline': round(choice['primary'] - base_primary, 5)}
    return choice, report
