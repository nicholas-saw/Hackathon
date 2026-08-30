"""Fast invariants. No training, no network — runs in seconds.

    python -m pytest tests/ -q          (or)      python tests/test_harness.py

Covers the four things that would silently invalidate a run: the leak guards, the test
seal, the journal chain, and the row_id contract.
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness import guards, journal as J                                  # noqa: E402
from harness.adapter import TestRowsRequested, raw_columns                # noqa: E402
from harness.score import TestSealError, rank_average, score_split        # noqa: E402


# ---------------- guards ----------------

def test_guard_rejects_leaky_same_row():
    fs = guards.scan_source('pipeline/features.py',
                            text="f = [same_row(x,'user_id'), same_row(x,'is_click')]")
    assert any('is_click' in f['reason'] for f in fs), 'leaky same_row not caught'


def test_guard_rejects_raw_csv_read():
    fs = guards.scan_source('pipeline/features.py',
                            text="fh = open('data/log_standard_4_22_to_5_08_pure.csv')")
    assert fs and 'raw log CSV' in fs[0]['reason']


def test_guard_rejects_test_labels():
    fs = guards.scan_source('pipeline/train.py', text="y = enc['test'][1]")
    assert fs, 'reading enc["test"][1] was not caught'


def test_guard_rejects_edit_outside_pipeline():
    ok, fs = guards.scan_diff('--- a/kit/evaluate.py\n+++ b/kit/evaluate.py\n+x=1\n')
    assert not ok and 'not editable' in fs[0]['reason']


def test_guard_allows_legitimate_diff():
    ok, fs = guards.scan_diff('--- a/pipeline/model.py\n+++ b/pipeline/model.py\n'
                              '+        self.lr = lr * 0.5\n')
    assert ok, 'clean diff rejected: %s' % guards.format_findings(fs)


def test_current_pipeline_is_clean():
    fs = guards.scan_pipeline()
    assert not fs, 'shipped pipeline trips the guards:\n' + guards.format_findings(fs)


# ---------------- test seal ----------------

def test_adapter_refuses_test_split():
    try:
        raw_columns(('date',), splits=('test',))
    except TestRowsRequested:
        return
    raise AssertionError('adapter served test rows')


def test_score_split_refuses_test_without_journal():
    try:
        score_split('test', np.zeros(3), enc={'test': (None, np.zeros(3), ['a'] * 3)})
    except TestSealError:
        return
    raise AssertionError('test was scored with no journal')


def test_score_split_refuses_test_before_designation():
    with tempfile.TemporaryDirectory() as d:
        jr = J.Journal(d)
        jr.append(J.RUN_START, {'x': 1})
        try:
            score_split('test', np.zeros(3), enc={'test': (None, np.zeros(3), ['a'] * 3)},
                        journal=jr)
        except TestSealError:
            return
    raise AssertionError('test was scored before FINAL_DESIGNATION')


# ---------------- journal ----------------

def test_chain_verifies_and_detects_tampering():
    with tempfile.TemporaryDirectory() as d:
        jr = J.Journal(d)
        jr.append(J.RUN_START, {'run': 1})
        jr.append(J.ITERATION, {'iteration': 1, 'metrics': {'primary': 0.6}})
        jr.append(J.FINAL_DESIGNATION, {'chosen_iteration': 1})
        ok, msg = J.verify_chain(jr.path)
        assert ok, msg

        body = open(jr.path, encoding='utf-8').read().replace('0.6', '0.9')
        open(jr.path, 'w', encoding='utf-8').write(body)
        ok2, msg2 = J.verify_chain(jr.path)
        assert not ok2, 'edited payload passed the chain check'


def test_order_requires_designation_before_test_open():
    with tempfile.TemporaryDirectory() as d:
        jr = J.Journal(d)
        jr.append(J.RUN_START, {})
        jr.append(J.TEST_OPEN, {})
        jr.append(J.FINAL_DESIGNATION, {})
        ok, msg = J.verify_order(jr.path)
        assert not ok, 'TEST_OPEN before FINAL_DESIGNATION was accepted'

    with tempfile.TemporaryDirectory() as d:
        jr = J.Journal(d)
        jr.append(J.RUN_START, {})
        jr.append(J.FINAL_DESIGNATION, {})
        jr.append(J.TEST_OPEN, {})
        ok, msg = J.verify_order(jr.path)
        assert ok, msg


def test_intervention_taxonomy_is_enforced():
    with tempfile.TemporaryDirectory() as d:
        jr = J.Journal(d)
        jr.intervention('L1_observe', 'read the log')
        jr.intervention('L4_steer', 'killed a branch')
        try:
            jr.intervention('L9_invented', 'nope')
        except ValueError:
            pass
        else:
            raise AssertionError('unknown intervention class accepted')
        evs = jr.events(J.HUMAN_INTERVENTION)
        assert [e['payload']['counts'] for e in evs] == [False, True]


# ---------------- scoring ----------------

def test_rank_average_is_within_user():
    users = np.array(['a', 'a', 'b', 'b'])
    a = np.array([1.0, 2.0, 5.0, 6.0])
    b = np.array([9.0, 8.0, 1.0, 2.0])       # disagrees with a inside user 'a'
    r = rank_average(users, [a, b])
    # user 'b' agrees in both, so its order must survive
    assert r[3] > r[2]
    # user 'a' disagrees, so the two rows tie
    assert abs(r[0] - r[1]) < 1e-9


def test_rank_average_ignores_cross_user_scale():
    users = np.array(['a', 'a', 'b', 'b'])
    small = np.array([0.1, 0.2, 0.3, 0.4])
    huge = np.array([100.0, 200.0, 300.0, 400.0])
    assert np.allclose(rank_average(users, [small]), rank_average(users, [huge]))


# ---------------- proposal quality gate ----------------

def _cand(**kw):
    base = {'direction_id': 'objective', 'hypothesis': 'h', 'evidence': ['e'],
            'mechanism': 'm', 'proposed_change': 'c', 'invalid_if': 'i',
            'expected_delta_primary': 0.003, 'files_to_modify': ['pipeline/model.py']}
    base.update(kw)
    return {'action': 'EXPERIMENT', 'chosen': 0, 'candidates': [base]}


def test_proposer_rejects_refuted_direction():
    from agent.proposer import validate
    ok, why = validate(_cand(direction_id='capacity'), {'capacity'})
    assert not ok and 'refuted' in why


def test_proposer_rejects_subfloor_expectation():
    from agent.proposer import validate
    ok, why = validate(_cand(expected_delta_primary=0.0004), set())
    assert not ok and 'resolution floor' in why


def test_proposer_allows_extends_refuted_with_reason():
    from agent.proposer import validate
    ok, _ = validate(_cand(direction_id='capacity',
                           extends_refuted='sweep was under a different objective'),
                     {'capacity'})
    assert ok


def test_proposer_requires_evidence_and_falsifier():
    from agent.proposer import validate
    assert not validate(_cand(evidence=[]), set())[0]
    assert not validate(_cand(invalid_if=''), set())[0]


def test_proposer_accepts_open_direction():
    from agent.proposer import validate
    ok, why = validate(_cand(), {'capacity'})
    assert ok, why


def test_request_analysis_needs_a_reason():
    from agent.proposer import validate
    assert not validate({'action': 'REQUEST_ANALYSIS', 'analysis': 'no_op_screen'}, set())[0]
    ok, _ = validate({'action': 'REQUEST_ANALYSIS', 'analysis': 'no_op_screen',
                      'why_needed': 'decides candidate 2'}, set())
    assert ok


# ---------------- knowledge registry ----------------

def test_registry_seeds_published_dead_ends():
    from harness import knowledge as K
    closed = set(K.closed_ids())
    for d in ('capacity', 'static_features', 'user_side_first_order', 'score_transform'):
        assert d in closed, '%s should be closed' % d
    assert 'objective' in set(K.open_ids())


def test_registry_records_and_closes_after_two_misses():
    import tempfile
    from harness import knowledge as K
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'directions.json')
        K.save(K.load(p), p)
        K.record('newidea', 1, 0.0002, 'INCONCLUSIVE', 'noise', path=p)
        assert K.load(p)['directions']['newidea']['status'] == K.LIVE
        K.record('newidea', 2, -0.0009, 'REVERT', 'worse', path=p)
        assert K.load(p)['directions']['newidea']['status'] == K.REFUTED


# ---------------- diagnostics ----------------

def test_diagnostics_refuses_unknown_analysis():
    from harness import diagnostics as D
    r = D.run('rm_minus_rf')
    assert 'error' in r and 'available' in r


def test_diagnostics_catalogue_is_documented():
    from harness import diagnostics as D
    cat = D.catalogue()
    assert cat and all(isinstance(v, str) and v for v in cat.values())


# ---------------- selection: no regression may ship ----------------

def _toy(n_users=400, per_user=6, seed=0):
    """A synthetic within-user ranking problem with a known signal."""
    rng = np.random.default_rng(seed)
    users = np.repeat([('u%04d' % i) for i in range(n_users)], per_user)
    labels = rng.integers(0, 2, n_users * per_user).astype(float)
    return users, labels, rng


def test_selection_rejects_a_lucky_but_unstable_candidate():
    """A candidate that wins overall on noise alone must not be designated."""
    from harness import selection
    users, labels, rng = _toy()
    base = rng.normal(size=len(labels))
    lucky = base + rng.normal(scale=0.01, size=len(labels))   # pure noise around base
    cands = [{'iteration': 0, 'label': 'baseline', 'valid': base, 'test': base},
             {'iteration': 1, 'label': 'lucky', 'valid': lucky, 'test': lucky}]
    choice, rep = selection.designate(users, labels, cands, base)
    assert choice['kind'] in ('baseline', 'single')
    if choice['kind'] == 'single':
        # if it did pass, it must have passed the fold test, not just the pooled score
        ev = [e for e in rep['evaluated'] if e['iteration'] == 1][0]
        assert ev['fold_wins'] >= selection.MIN_FOLD_WINS


def test_selection_accepts_a_genuine_improvement():
    """A candidate carrying real signal must win on the pooled score and the folds."""
    from harness import selection
    users, labels, rng = _toy(seed=3)
    base = rng.normal(size=len(labels))
    good = base + 3.0 * labels                    # strong, consistent, real signal
    cands = [{'iteration': 0, 'label': 'baseline', 'valid': base, 'test': base},
             {'iteration': 1, 'label': 'good', 'valid': good, 'test': good}]
    choice, rep = selection.designate(users, labels, cands, base)
    assert choice['kind'] in ('single', 'ensemble'), rep['decisions']
    assert choice['iteration'] == 1 or choice['kind'] == 'ensemble'


def test_floor_tripwire_blocks_a_regression():
    """Nothing worse than the banked floor may be shipped, whatever the run decided."""
    from harness import selection
    users, labels, rng = _toy(seed=7)
    base = rng.normal(size=len(labels))
    floor = base + 3.0 * labels                   # the floor is genuinely better
    cands = [{'iteration': 0, 'label': 'baseline', 'valid': base, 'test': base}]
    choice, rep = selection.designate(users, labels, cands, base, floor_preds=floor)
    assert choice['kind'] == 'floor', rep['decisions']


def test_user_folds_never_split_a_user():
    from harness import selection
    users, _, _ = _toy(n_users=50, per_user=4)
    f = selection.user_folds(users)
    for u in np.unique(users):
        assert len(set(f[users == u].tolist())) == 1, 'user %s spans folds' % u


def test_guard_rejects_fitting_on_validation():
    fs = guards.scan_source('pipeline/features.py',
                            text="edges = _bucket_edges([x[5] for x in splits['valid']])")
    assert fs and 'validation' in fs[0]['reason']


def test_guard_allows_valid_for_early_stopping_in_train():
    fs = guards.scan_source('pipeline/train.py', text="Xva, yva, uva = enc['valid']")
    assert not fs, 'early stopping on validation was wrongly flagged'


def _run_all():
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith('test_') and callable(f)]
    bad = 0
    for name, fn in fns:
        try:
            fn()
            print('  PASS  %s' % name)
        except Exception as exc:
            bad += 1
            print('  FAIL  %s: %s' % (name, exc))
    print('\n%d/%d passed' % (len(fns) - bad, len(fns)))
    return bad


if __name__ == '__main__':
    raise SystemExit(1 if _run_all() else 0)
