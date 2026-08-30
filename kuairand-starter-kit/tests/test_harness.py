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

from harness import guards, journal as J, PIPELINE                        # noqa: E402
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


def test_history_is_strictly_causal():
    """A prior aggregate must never include the row itself or a tie."""
    from harness import history

    # Rows: (date, user, video, author, tab, duration, label). Users 'a' and 'b'.
    # Two of a's rows share timestamp 20 -- they must not see each other.
    rows = [(20220409, 'a', 'v1', 'w1', '1', 1000.0, 1),
            (20220409, 'a', 'v2', 'w1', '1', 1000.0, 1),
            (20220409, 'a', 'v3', 'w1', '1', 1000.0, 0),
            (20220409, 'b', 'v1', 'w1', '1', 1000.0, 0)]
    stamps = [10, 20, 20, 15]

    class FakeAdapter:
        @staticmethod
        def raw_columns(names, dtype=None, **kw):
            return {'train': {'time_ms': np.asarray(stamps, dtype=dtype or np.int64)},
                    'valid': {'time_ms': np.asarray([], dtype=dtype or np.int64)}}

    real, history.adapter = history.adapter, FakeAdapter
    try:
        splits = {'train': rows, 'valid': []}
        rate, count = history.prior_stats(splits, signal='label', key='user_id',
                                          prior_weight=0.0)
    finally:
        history.adapter = real

    # row 0 (t=10): no prior -> count 0
    assert count['train'][0] == 0, count['train'][0]
    # rows 1 and 2 share t=20: each sees only row 0, not each other
    assert count['train'][1] == 1 and count['train'][2] == 1, count['train'][:3]
    assert abs(rate['train'][1] - 1.0) < 1e-6, rate['train'][1]
    assert abs(rate['train'][2] - 1.0) < 1e-6, rate['train'][2]
    # user b is independent of user a
    assert count['train'][3] == 0, count['train'][3]


def test_history_excludes_test_outcomes():
    """Test labels must not reach any aggregate, including the test rows' own."""
    from harness import history

    train = [(20220409, 'a', 'v1', 'w1', '1', 1000.0, 1),
             (20220409, 'a', 'v2', 'w1', '1', 1000.0, 1)]
    test = [(20220429, 'a', 'v9', 'w1', '1', 1000.0, 1),
            (20220429, 'a', 'v8', 'w1', '1', 1000.0, 1)]
    flipped = [tuple(list(r[:6]) + [0]) for r in test]

    class FakeAdapter:
        @staticmethod
        def raw_columns(names, dtype=None, **kw):
            return {'train': {'time_ms': np.asarray([10, 20], dtype=dtype or np.int64)},
                    'valid': {'time_ms': np.asarray([], dtype=dtype or np.int64)}}

    real, history.adapter = history.adapter, FakeAdapter
    try:
        a = history.prior_stats({'train': train, 'valid': [], 'test': test},
                                prior_weight=0.0)
        b = history.prior_stats({'train': train, 'valid': [], 'test': flipped},
                                prior_weight=0.0)
    finally:
        history.adapter = real

    # every test row sees the full train state, and only that
    assert list(a[1]['test']) == [2.0, 2.0], a[1]['test']
    # flipping every test label changes nothing anywhere
    for split in ('train', 'test'):
        assert np.array_equal(a[0][split], b[0][split]), split
        assert np.array_equal(a[1][split], b[1][split]), split


def test_guard_allows_history_helper():
    """The legal route must pass the guard; the direct read must still fail."""
    NL = chr(10)
    legal = NL.join([
        'from harness.history import prior_stats',
        "IDX = {'label': 6}",
        'def encode(splits):',
        "    rate, count = prior_stats(splits, key='user_id')",
        '    return rate, count',
    ])
    illegal = NL.join([
        "IDX = {'label': 6}",
        'def encode(splits):',
        "    hist = [x[IDX['label']] for x in splits['train']]",
        '    return hist',
    ])
    assert not guards.scan_source('pipeline/features.py', text=legal)
    assert guards.scan_source('pipeline/features.py', text=illegal)


def test_guard_scopes_label_rule_to_features_file():
    """train.py legitimately reads labels; the feature rule must not judge it.

    coder.py emits whole files, so every line of train.py reappears as an added line.
    When scan_diff flattened all added lines into one blob, the pristine shipped
    train.py failed its own guard and no training-loop change could ever pass.
    """
    import difflib
    train = open(os.path.join(PIPELINE, "train.py"), encoding="utf-8").read().split(chr(10))
    whole = chr(10).join(difflib.unified_diff(
        [], train, fromfile="a/pipeline/train.py", tofile="b/pipeline/train.py",
        lineterm=""))
    ok, findings = guards.scan_diff(whole)
    assert ok, "pristine train.py rejected by its own guard: %r" % (findings[:1],)

    # The same rule must still fire for features.py.
    feats = open(os.path.join(PIPELINE, "features.py"), encoding="utf-8").read().split(chr(10))
    leaky = list(feats) + ["""    bad = [x[IDX['label']] for x in splits['train']]"""]
    d = chr(10).join(difflib.unified_diff(
        [], leaky, fromfile="a/pipeline/features.py",
        tofile="b/pipeline/features.py", lineterm=""))
    ok2, f2 = guards.scan_diff(d)
    assert not ok2, "direct label read in features.py was allowed through"


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
