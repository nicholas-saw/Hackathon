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
