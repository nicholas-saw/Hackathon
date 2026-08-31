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


# ---------------- causal history helper ----------------

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


# ---------------- paired per-seed confirmation ----------------

def test_paired_confirmation_flags_a_single_lucky_seed():
    """Ensemble may be up while one seed regressed -- that must not be kept."""
    from agent.controller import paired_confirmation
    c = paired_confirmation({0: 0.610, 1: 0.605, 2: 0.598},
                            {0: 0.602, 1: 0.601, 2: 0.600}, 0.004)
    assert c['worst_delta'] < 0, c
    assert not c['all_paired_positive']


def test_paired_confirmation_accepts_a_consistent_gain():
    from agent.controller import paired_confirmation
    c = paired_confirmation({0: 0.605, 1: 0.604, 2: 0.606},
                            {0: 0.602, 1: 0.601, 2: 0.600}, 0.003)
    assert c['all_paired_positive'] and c['worst_delta'] > 0
    assert abs(c['mean_delta'] - 0.004) < 1e-9


def test_paired_confirmation_needs_two_shared_seeds():
    """An ensemble node trains no seeds; it must fall back, not invent a comparison."""
    from agent.controller import paired_confirmation
    assert paired_confirmation({}, {0: 0.6, 1: 0.6}, 0.001) is None
    assert paired_confirmation({0: 0.61}, {0: 0.60, 1: 0.60}, 0.001) is None



# ---------------- accept bars ----------------

def test_unanimous_bar_is_below_the_single_measurement_bar():
    """A 3-seed paired mean has se sigma/sqrt(3); holding it to ACCEPT is ~3.5 sigma."""
    from agent.controller import ACCEPT, UNANIMOUS_ACCEPT
    assert 0 < UNANIMOUS_ACCEPT < ACCEPT


def test_the_reverted_best_result_would_now_be_kept():
    """run 20260831T011354Z iter 12: mean +0.00112, worst +0.00090, 3/3 seeds up.

    It was the best result of the project and was reverted for missing ACCEPT.
    """
    from agent.controller import UNANIMOUS_ACCEPT, paired_confirmation
    c = paired_confirmation({'0': 0.60724, '1': 0.60700, '2': 0.60712},
                            {'0': 0.60609, '1': 0.60610, '2': 0.60607}, 0.00112)
    assert c['all_paired_positive']
    assert c['mean_delta'] > UNANIMOUS_ACCEPT


def test_a_split_decision_still_fails_the_unanimous_bar():
    from agent.controller import UNANIMOUS_ACCEPT, paired_confirmation
    c = paired_confirmation({'0': 0.6080, '1': 0.6055, '2': 0.6072},
                            {'0': 0.6060, '1': 0.6061, '2': 0.6060}, 0.0012)
    assert not c['all_paired_positive'], 'seed 1 regressed'
    assert c['mean_delta'] > UNANIMOUS_ACCEPT, 'mean alone would have passed'



# ---------------- structural test-label firewall ----------------

def test_run_node_blanks_test_labels_before_agent_code():
    """The guard regex is bypassable by ordinary spellings; this must not be.

    fit_predict needs test FEATURES to build a submission vector and never needs test
    labels, so run_node zeroes them before importing the agent's pipeline. Without this
    the firewall is a regex, and `Xt, yt, ut = enc['test']` walks straight past it.
    """
    import re
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'harness', 'run_node.py'), encoding='utf-8').read()
    blank = src.index('enc[' + chr(39) + 'test' + chr(39) + '] = (Xt,')
    load = src.index('enc, dim = load_encoded()')
    imp = src.index('from train import fit_predict')
    assert load < blank < imp, (
        'test labels must be blanked after load_encoded and BEFORE the agent pipeline '
        'is imported; ordering was load=%d blank=%d import=%d' % (load, blank, imp))


def test_guard_regex_alone_is_not_the_firewall():
    """Documents the hole the blanking closes, so nobody re-relies on the regex."""
    fs = guards.scan_source('pipeline/train.py', text="Xt, yt, ut = enc['test']")
    assert not fs, ('the tuple-unpack spelling is NOT caught by the static guard -- '
                    'that is why harness/run_node.py blanks the labels structurally')


def _fresh_registry(tmp):
    """A registry file seeded from SEED, isolated from the real context/ one."""
    from harness import knowledge as K
    path = os.path.join(tmp, 'directions.json')
    K.save(K.load(path), path)
    return path


def test_registry_closes_a_direction_after_two_misses():
    from harness import knowledge as K
    with tempfile.TemporaryDirectory() as tmp:
        path = _fresh_registry(tmp)
        K.record('made_up', 1, -0.003, 'REVERT', 'miss', path=path)
        d = K.record('made_up', 2, -0.002, 'REVERT', 'miss', path=path)
        assert d['status'] == K.REFUTED
        assert 'made_up' in K.closed_ids(path)


def test_registry_reopens_a_closed_direction_on_a_confirmed_win():
    """The real listwise chronology, replayed.

    A direction id names an intent, not a formulation: nine independently written
    listwise implementations spanned -0.00318 to +0.00162. The two opening misses close
    the direction three runs BEFORE the verified win arrives, so "never close over a
    confirmed win" is not enough on its own -- the win has to reopen it, and the misses
    that follow must not close it again.
    """
    from harness import knowledge as K
    with tempfile.TemporaryDirectory() as tmp:
        path = _fresh_registry(tmp)
        for i, miss in enumerate([-0.00273, -0.00273, -0.00230, -0.00155], start=1):
            d = K.record('listwise_like', i, miss, 'REVERT', 'miss', path=path)
        assert d['status'] == K.REFUTED, 'precondition: misses close it first'

        d = K.record('listwise_like', 5, 0.00197, 'INCONCLUSIVE', 'verified', path=path,
                     confirm={'mean_delta': 0.00162, 'worst_delta': 0.00097})
        assert d['status'] == K.LIVE, 'a confirmed win must reopen a closed direction'

        for i, miss in enumerate([-0.00233, -0.00318, -0.00233], start=6):
            d = K.record('listwise_like', i, miss, 'REVERT', 'miss', path=path)
        assert d['status'] == K.LIVE, 'later misses re-closed a confirmed direction'
        assert 'listwise_like' not in K.closed_ids(path)


def test_registry_confirmation_below_the_bar_does_not_protect():
    """Protection requires clearing the accept bar, not merely being positive."""
    from harness import knowledge as K
    with tempfile.TemporaryDirectory() as tmp:
        path = _fresh_registry(tmp)
        K.record('weak_thing', 1, 0.0005, 'INCONCLUSIVE', 'weak', path=path,
                 confirm={'mean_delta': 0.0005, 'worst_delta': 0.0001})
        d = K.record('weak_thing', 2, -0.002, 'REVERT', 'miss', path=path)
        assert d['status'] == K.REFUTED


def test_impl_id_separates_implementations_that_share_a_label():
    """Prose cannot distinguish implementations; the diff can.

    Every one of the 29 recorded agent iterations produced a unique diff, including the
    nine that all called themselves "within-user listwise softmax".
    """
    from harness import knowledge as K
    a = K.impl_id('--- a/pipeline/model.py\n+    pure softmax, no BCE mix\n')
    b = K.impl_id('--- a/pipeline/model.py\n+    softmax + 0.3 * BCE\n')
    assert a != b, 'different code must not collide'
    assert a == K.impl_id('--- a/pipeline/model.py\n+    pure softmax, no BCE mix\n')
    assert K.impl_id('') == K.impl_id(None)


def test_ablation_candidate_fires_when_implementations_disagree_in_sign():
    from harness import knowledge as K
    with tempfile.TemporaryDirectory() as tmp:
        path = _fresh_registry(tmp)
        K.record('obj_x', 1, -0.0030, 'REVERT', 'lost', diff='DIFF-ONE',
                 objective_family='listwise', path=path)
        assert K.ablation_candidates(path) == [], 'one implementation cannot disagree'

        K.record('obj_x', 2, 0.0025, 'KEEP', 'won', diff='DIFF-TWO',
                 objective_family='listwise', path=path)
        got = K.ablation_candidates(path)
        assert len(got) == 1 and got[0]['direction_id'] == 'obj_x'
        assert got[0]['n_implementations'] == 2
        assert abs(got[0]['spread'] - 0.0055) < 1e-9


def test_ablation_candidate_silent_when_implementations_agree():
    """Consistent losers are a settled direction, not a contradiction to spend on."""
    from harness import knowledge as K
    with tempfile.TemporaryDirectory() as tmp:
        path = _fresh_registry(tmp)
        for i, d in enumerate([-0.0030, -0.0021, -0.0014], start=1):
            K.record('obj_y', i, d, 'REVERT', 'lost', diff='DIFF-%d' % i,
                     objective_family='pairwise', path=path)
        assert K.ablation_candidates(path) == []


def test_traits_accumulate_across_sightings_of_the_same_code():
    from harness import knowledge as K
    with tempfile.TemporaryDirectory() as tmp:
        path = _fresh_registry(tmp)
        K.record('obj_z', 1, 0.001, 'INCONCLUSIVE', 'a', diff='SAME',
                 traits=['no BCE mix'], path=path)
        K.record('obj_z', 2, 0.002, 'KEEP', 'b', diff='SAME',
                 traits=['no BCE mix', 'uncapped lists'], path=path)
        ims = K.implementations_of('obj_z', path)
        assert len(ims) == 1, 'identical diffs must share one implementation id'
        assert ims[0]['traits'] == ['no BCE mix', 'uncapped lists']
        assert len(ims[0]['measured']) == 2


# ---------------- the banked submission's model ----------------

def test_the_banked_model_key_exists_and_is_not_fm_listwise():
    """RESULTS.md's config must select the loss that actually produced the CSV.

    It said `fm_listwise` while the winning loss had never been committed, so that
    config silently trained a different model. Byte-identical regeneration was only
    restored once `fm_listwise_pure` existed.
    """
    import inspect
    sys.path.insert(0, PIPELINE)
    import train, model
    src = inspect.getsource(train.fit_predict)
    assert "model == 'fm_listwise_pure'" in src
    assert "model == 'fm_listwise'" in src, 'the other variant must survive too'
    assert hasattr(model.FM, 'step_listwise') and hasattr(model.FM, 'step_list')

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, 'RESULTS.md'), encoding='utf-8') as fh:
        results = fh.read()
    # The guarantee is that wherever RESULTS.md gives a config for the listwise CSV, it
    # names the key that actually regenerates it. This was originally pinned to the
    # first '| Config |' row because that artifact was the submission; it is now the
    # documented runner-up, so the check is on the document rather than on one row.
    assert 'fm_listwise_pure' in results, 'RESULTS.md no longer names the banked key'
    for line in results.splitlines():
        if line.startswith('| Config |') and 'fm_listwise' in line:
            assert 'fm_listwise_pure' in line, (
                'a Config row names the bare fm_listwise, which trains a different '
                'model than the one that produced the CSV: %s' % line)


def test_pure_listwise_skips_uniform_label_groups():
    """Its defining trait: a group with 0 or ALL positives carries no ordering signal.

    step_list does not share this -- it admits any group with a positive -- which is one
    of the four ways the two implementations differ.
    """
    sys.path.insert(0, PIPELINE)
    from model import FM

    X = np.array([[0, 2], [0, 3], [1, 2], [1, 3]], dtype=np.int64)
    offs = np.array([0, 2, 4], dtype=np.int64)

    m = FM(8, k=4, seed=0)
    before = m.V.copy()
    assert m.step_listwise(X, np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32), offs) == 0.0
    assert np.array_equal(m.V, before), 'uniform-label groups must not move the weights'

    m2 = FM(8, k=4, seed=0)
    loss = m2.step_listwise(X, np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32), offs)
    assert loss > 0.0 and not np.array_equal(m2.V, before), 'mixed groups must train'


def test_the_two_listwise_losses_are_not_interchangeable():
    """If these ever agree, one has been quietly replaced by the other.

    The input carries an ALL-POSITIVE group, where the two differ structurally rather
    than by magnitude: step_listwise skips it (a uniform-label group carries no ordering
    signal), while step_list admits any group holding a positive and still applies its
    BCE term there.

    That asymmetry is what makes the check reliable. On mixed-label-only input the two
    produce weights that are equal to ~1e-9 after one step, because the first Adam
    update is m/(sqrt(v)+eps) = g/|g| -- sign only -- so a purely magnitude-level
    difference cancels exactly. Comparing against the initial weights would not work
    either: `_apply_grad` adds l2 * V over the whole matrix, so every weight moves on
    every step whether or not it received a data gradient.
    """
    sys.path.insert(0, PIPELINE)
    from model import FM

    #        group 0: mixed labels        group 1: all positive
    X = np.array([[0, 4], [0, 5], [1, 6], [1, 7]], dtype=np.int64)
    y = np.array([1.0, 0.0, 1.0, 1.0], dtype=np.float32)

    a = FM(8, k=4, seed=0)
    loss_pure = a.step_listwise(X, y, np.array([0, 2, 4], dtype=np.int64))
    b = FM(8, k=4, seed=0)
    loss_mixed = b.step_list(X, y, np.array([2, 4], dtype=np.int64))

    assert abs(loss_pure - loss_mixed) > 1e-6, (
        'the two objectives report the same loss (%r vs %r); one has replaced the other'
        % (loss_pure, loss_mixed))
    assert not np.allclose(a.V, b.V), (
        'step_listwise and step_list moved the weights identically on an all-positive '
        'group -- they are no longer distinct implementations')

if __name__ == '__main__':
    raise SystemExit(1 if _run_all() else 0)
