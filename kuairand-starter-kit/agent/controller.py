"""Controller — the loop. Plain code, no model call.

Search policy, convergence, checkpointing, budget enforcement and the final designation
all live here, in rules, because the rubric pays for tokens and none of this needs
judgement.

The convergence rule is the load-bearing decision. The official text says a run is
converged when validation primary has not improved by more than eps=0.002 over the last
N=3 consecutive iterations, and that "the submission scored for ranking is the
validation-best checkpoint at that point". The score LOCKS there. So this controller
stops on convergence rather than chasing the 50-iteration cap — running longer cannot
raise the score, and it costs wall-clock, which Feasibility measures.
"""
import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness import PIPELINE, ROOT, RUNLOGS, SUBMISSIONS          # noqa: E402
from harness import diagnostics, executor, guards, journal as J   # noqa: E402
from harness import knowledge                                     # noqa: E402
from harness.cache import load_encoded                            # noqa: E402
from harness.score import evaluate_raw, score_split               # noqa: E402
from agent import coder as coder_mod                              # noqa: E402
from agent import proposer as proposer_mod                        # noqa: E402
from agent import reflector as reflector_mod                      # noqa: E402
from agent.llm import LLM, LLMUnavailable, Meter, BudgetHalt      # noqa: E402

EPS = 0.002              # organizer convergence threshold
N_CONV = 3               # consecutive non-improving iterations
ACCEPT = 0.0014          # 2 sigma on the measured paired noise floor (sigma ~ 0.0007)

# The bar for a change whose paired seeds ALL moved the same way. ACCEPT is 2 sigma on a
# single measurement; the mean of s matched seeds has standard error sigma/sqrt(s), so
# holding a 3-seed paired mean to 0.0014 is ~3.5 sigma -- far stricter than the rule the
# number was derived from, and strictness here is not free. In run 20260831T011354Z
# iteration 12 measured mean +0.00112 with every one of three seeds positive (worst
# +0.00090) and was reverted out of the lineage for missing 0.0014. It was the best
# result of the whole project. Unanimity is itself evidence: p = 0.125 under the null for
# three seeds. Requiring both unanimity and 2 sigma on the paired mean keeps the false
# accept rate low, and designation still stability-tests the winner over 5 user folds.
UNANIMOUS_ACCEPT = 0.0008

# Node failures the coder can plausibly fix from the traceback, and which are therefore
# worth one retry rather than a discarded iteration. executor.classify's own comment says
# "the reason string goes back to the coder"; nothing was sending it. `timeout` and
# `memory` are excluded because a retry would simply repeat them, and `leak` is excluded
# deliberately -- a change that reached the test seal gets no coaching on getting past it.
RETRYABLE_FAILURES = ('syntax', 'import', 'contract', 'nan')
CONFIRM_SEEDS = [0, 1, 2]  # matched seeds for a candidate that might actually be kept
CONFIRM_FLOOR = -0.0007  # -1 sigma: below this a single seed is enough to reject
MAX_ROLE_FAILURES = 3    # consecutive LLM-role failures before giving up on the loop
MAX_ITERS = 50           # official hard cap
MAX_WALL_S = 6 * 3600    # official wall-clock ceiling

# Every node is a rank-average of this many seeds. One seed per node makes each
# measurement carry sigma ~ 0.0006, which is half the accept bar, so a real +0.001
# improvement is indistinguishable from noise and gets rejected. Three seeds cut that
# to ~0.00035 and make the bar mean something. Cost is 3x wall-clock per node, which
# the 6h ceiling absorbs easily at ~60s a seed.
SEEDS_PER_NODE = 3

# A dry run only has to prove the plumbing, so it uses fewer seeds -- but it must use the
# SAME number for the baseline and for every candidate. Measuring candidates on one seed
# against a rank-averaged three-seed baseline handed the baseline the ensemble's variance
# reduction (~+0.0007 here, half the accept bar) and made every idea look worse than it
# was. Two is the minimum that still exercises the paired comparison.
DRY_SEEDS_PER_NODE = 2

# The controller injects a pure-ensemble node this often. It trains nothing and calls no
# model, so it costs zero tokens and seconds of wall-clock — and combining diverse
# candidates is the one move measured to reliably clear the noise floor on this task.
ENSEMBLE_EVERY = 3
ENSEMBLE_MEMBERS = 4
EDITABLE = coder_mod.EDITABLE

# Fixed ideas for --dry-run: config-only changes, no model in the loop. Their job is to
# prove the plumbing survives 3 unattended iterations, not to find anything.
DRY_IDEAS = [
    {'hypothesis': 'A lower learning rate reaches a better validation optimum.',
     'mechanism': 'Adam at lr=1e-3 may overshoot on sparse embedding updates.',
     'expected_result': '+0.001 valid primary', 'invalid_if': 'delta <= 0',
     'evidence': ['baseline lr is 0.001'], 'proposed_change': 'lr=0.0005',
     'files_to_modify': [], 'config': {'lr': 0.0005}},
    {'hypothesis': 'Higher embedding capacity helps.',
     'mechanism': 'k=16 may underfit the user x video cross.',
     'expected_result': '+0.001 valid primary', 'invalid_if': 'delta <= 0',
     'evidence': ['organizers swept k and saw nothing'], 'proposed_change': 'k=32',
     'files_to_modify': [], 'config': {'k': 32}},
    {'hypothesis': 'A 3-seed rank-average beats any single seed.',
     'mechanism': 'Seed variance is real; averaging ranks cancels it.',
     'expected_result': '+0.001 valid primary', 'invalid_if': 'delta <= 0',
     'evidence': ['published 5-seed std 0.0008'], 'proposed_change': 'seeds 0,1,2',
     'files_to_modify': [], 'config': {}, 'seeds': [0, 1, 2]},
]

INJECTIONS = {
    'syntax': ('pipeline/model.py',
               lambda s: s + '\n\ndef broken(:\n    pass\n',
               'a deliberate SyntaxError'),
    # 1e12 is not enough: Adam's step is bounded by lr, so V reaches ~1e11 and the
    # squared terms stay inside float32's 3.4e38. 1e30 overflows to inf on the second
    # step, and inf - inf in the FM interaction term is the NaN we need to detect.
    'nan': ('pipeline/model.py',
            lambda s: s.replace('self.lr, self.l2 = lr, l2',
                                'self.lr, self.l2 = lr * 1e30, l2'),
            'a learning rate that overflows float32 and yields NaN scores'),
    'leak': ('pipeline/features.py',
             lambda s: s.replace("return [same_row(x, 'user_id')",
                                 "return [same_row(x, 'is_click'), same_row(x, 'user_id')"),
             'a post-impression column used as a same-row input'),
    'timeout': ('pipeline/model.py',
                lambda s: s.replace('import numpy as np',
                                    'import numpy as np, time as _t\n_t.sleep(3600)'),
                'a hang that must be killed as a process tree'),
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def paired_confirmation(child_per_seed, parent_per_seed, ensemble_delta):
    """Matched per-seed comparison of a node against its parent, or None.

    Both nodes train the same seed list, so seed s can be compared with seed s. That
    pairing removes the seed as a source of variance and gives the honest effect size
    of the change itself. Contrasting one child seed with the parent's rank-averaged
    ENSEMBLE would not be matched -- the ensemble sits systematically above any single
    seed, so such a test rejects even real improvements.

    Needs at least two shared seeds to say anything; an ensemble node trains none and
    gets None, leaving the caller on the plain ensemble delta.
    """
    child, par = child_per_seed or {}, parent_per_seed or {}
    shared = sorted(set(child) & set(par), key=str)
    if len(shared) < 2:
        return None
    paired = [float(child[k]) - float(par[k]) for k in shared]
    return {'seeds': [int(k) for k in shared],
            'paired_deltas': [round(d, 6) for d in paired],
            'mean_delta': sum(paired) / len(paired),
            'worst_delta': min(paired),
            'all_paired_positive': all(d > 0 for d in paired),
            'ensemble_delta': ensemble_delta,
            'basis': 'paired_per_seed'}


def workspace_hash(root=ROOT):
    """Merkle-ish fingerprint of everything that must not change during a run."""
    h = hashlib.sha256()
    for rel in sorted(EDITABLE) + ['kit/data.py', 'kit/evaluate.py', 'kit/submit.py']:
        p = os.path.join(root, rel.replace('/', os.sep))
        if os.path.exists(p):
            h.update(rel.encode())
            h.update(sha256_file(p).encode())
    return h.hexdigest()


def snapshot(root=ROOT):
    return {rel: open(os.path.join(root, rel.replace('/', os.sep)),
                      encoding='utf-8', newline='').read() for rel in EDITABLE}


def restore(snap, root=ROOT):
    coder_mod.restore(root, snap)


class Controller:
    def __init__(self, run_id=None, dry_run=False, max_iters=MAX_ITERS,
                 budget_usd=14.0, packet_path=None):
        self.run_id = run_id or time.strftime('run_%Y%m%dT%H%M%SZ', time.gmtime())
        self.run_dir = os.path.join(RUNLOGS, self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)
        self.preds_dir = os.path.join(self.run_dir, 'preds')
        os.makedirs(self.preds_dir, exist_ok=True)
        self.jr = J.Journal(self.run_dir)
        self.dry_run = dry_run
        # One seed count governs baseline AND candidates; see DRY_SEEDS_PER_NODE.
        self.seeds_per_node = DRY_SEEDS_PER_NODE if dry_run else SEEDS_PER_NODE
        self.max_iters = max_iters
        self.meter = Meter(ceiling_usd=budget_usd)
        self.llm = None
        self.node_times = []
        self.nodes = []              # every scored candidate
        self.best_history = []       # best-so-far primary after each iteration
        if not dry_run:
            packet = open(packet_path, encoding='utf-8').read() if packet_path else ''
            self.llm = LLM(packet, meter=self.meter)

    # ---------- one experiment ----------
    def run_node(self, iteration, config, seeds, timeout):
        out = os.path.join(self.preds_dir, 'iter%03d.npz' % iteration)
        res = executor.run_python(
            ['-m', 'harness.run_node', '--out', out,
             '--config', json.dumps(config), '--seeds', ','.join(map(str, seeds))],
            timeout=timeout, cwd=ROOT)
        parsed = None
        for line in (res['stdout'] or '').splitlines():
            if line.startswith('RESULT '):
                parsed = json.loads(line[7:])
        return res, parsed

    def ensemble_node(self, it, uv, yv):
        """A node that trains nothing: rank-average the best distinct candidates so far.

        Costs zero tokens and about a second of wall-clock, and on this task combining
        diverse candidates is the only move measured to clear the noise floor reliably —
        a member that is individually worse can still add value, because what an ensemble
        buys is decorrelation, not strength. So this runs on a fixed cadence rather than
        waiting for the proposer to think of it.
        """
        import numpy as np
        from harness.score import rank_average
        pool = [n for n in self.nodes if not n.get('is_ensemble')]
        if len(pool) < 2:
            return None
        mem = sorted(pool, key=lambda n: -n['primary'])[:ENSEMBLE_MEMBERS]
        enc, _ = load_encoded()
        _, _, ut = enc['test']
        vs, ts = [], []
        for n in mem:
            z = np.load(n['preds'])
            vs.append(z['valid']); ts.append(z['test'])
        t0 = time.time()
        ev = rank_average(np.asarray(uv), vs)
        et = rank_average(np.asarray(ut), ts)
        out = os.path.join(self.preds_dir, 'iter%03d.npz' % it)
        np.savez_compressed(out, valid=np.asarray(ev, np.float32),
                            test=np.asarray(et, np.float32))
        r = evaluate_raw(uv, yv, ev)
        return {'metrics': {k: float(v) for k, v in r.items()},
                'preds_path': out, 'seconds': round(time.time() - t0, 2),
                'members': [n['iteration'] for n in mem]}

    # ---------- the loop ----------
    def run(self):
        enc, _ = load_encoded()
        _, yv, uv = enc['valid']

        self.jr.append(J.RUN_START, {
            'run_id': self.run_id, 'dry_run': self.dry_run,
            'workspace_hash': workspace_hash(),
            'limits': {'max_iterations': self.max_iters, 'wall_clock_s': MAX_WALL_S,
                       'eps': EPS, 'N': N_CONV, 'accept_bar': ACCEPT},
            'model': None if self.dry_run else self.llm.model,
            'budget_usd_ceiling': self.meter.ceiling})

        base_snap = snapshot()
        t_start = time.time()

        # Iteration 0 is the untouched baseline: the parent everything is measured against.
        res0, p0 = self.run_node(0, {}, list(range(self.seeds_per_node)), timeout=1800)
        if not p0:
            self.jr.append(J.ERROR_RECOVERY, {'iteration': 0, 'fatal': True, **res0})
            raise RuntimeError('baseline node failed; harness is not trustworthy:\n%s'
                               % res0['stderr'][-1500:])
        parent = {'primary': p0['metrics']['primary'], 'label': 'baseline',
                  'iteration': 0, 'preds': p0['preds_path'],
                  'per_seed': p0.get('per_seed_primary') or {}}
        self.node_times.append(p0['seconds'])
        self.nodes.append({'iteration': 0, 'primary': parent['primary'],
                           'seeds': list(range(self.seeds_per_node)),
                           'artifact': ('rank_average_ensemble'
                                        if self.seeds_per_node > 1 else 'single_seed'),
                           'artifact_source': 'official_baseline',
                           'preds': p0['preds_path'], 'label': 'baseline',
                           'metrics': p0['metrics']})
        self.best_history.append(parent['primary'])
        self.jr.append(J.ITERATION, {
            'iteration': 0, 'hypothesis': 'reproduce the official baseline',
            'verdict': 'KEEP', 'metrics': p0['metrics'], 'delta_vs_parent': 0.0,
            'seconds': p0['seconds'], 'config': {}, 'diff': '', 'provenance': 'harness'})
        print('iter 000  baseline  valid primary %.5f' % parent['primary'], flush=True)

        stop = None
        role_failures = 0          # consecutive LLM-role failures; reset by a good node
        # A designation must exist even if the loop dies. Without this, an unhandled
        # error means no FINAL_DESIGNATION, no submission and a truncated journal --
        # the entire spend of the run is lost. Run 20260830T234014Z ended that way.
        try:
            for it in range(1, self.max_iters + 1):
                if time.time() - t_start > MAX_WALL_S:
                    stop = 'wall_clock'
                    break
                try:
                    self.meter.check()
                except BudgetHalt as exc:
                    self.jr.append(J.BUDGET_HALT, {'iteration': it, 'reason': str(exc)})
                    stop = 'budget'
                    break

                # Zero-token ensembler on a fixed cadence, before spending a proposer call.
                if it % ENSEMBLE_EVERY == 0:
                    ens = self.ensemble_node(it, uv, yv)
                    if ens:
                        m = ens['metrics']
                        delta = m['primary'] - parent['primary']
                        self.nodes.append({'iteration': it, 'primary': m['primary'],
                                           'preds': ens['preds_path'], 'metrics': m,
                                           'is_ensemble': True,
                                           # It trains nothing, so it has no seeds of its
                                           # own. Defaulting to [0]/single_seed would report
                                           # a rank-average of several nodes as one model.
                                           'seeds': [],
                                           'artifact': 'rank_average_of_nodes',
                                           'artifact_source': 'harness_ensemble',
                                           'members': list(ens['members']),
                                           'label': 'rank-average of %s'
                                                    % ','.join(map(str, ens['members']))})
                        self.jr.append(J.ITERATION, {
                            'iteration': it, 'provenance': 'harness',
                            'hypothesis': 'Rank-averaging the %d best distinct candidates '
                                          'decorrelates their errors.' % len(ens['members']),
                            'mechanism': 'An ensemble buys decorrelation, not strength; a '
                                         'member that is individually weaker can still help.',
                            'verdict': 'KEEP' if delta > ACCEPT else 'INCONCLUSIVE',
                            'reason': 'ensemble of iterations %s'
                                      % ','.join(map(str, ens['members'])),
                            'metrics': m, 'delta_vs_parent': delta, 'accept_bar': ACCEPT,
                            'seconds': ens['seconds'], 'config': {}, 'diff': '',
                            'ensemble_members': ens['members'],
                            'cost_so_far_usd': self.meter.totals()['usd']})
                        print('iter %03d  ENSEMBLE of %s  primary %.5f  delta %+.5f'
                              % (it, ens['members'], m['primary'], delta), flush=True)
                        if delta > ACCEPT:
                            parent = {'primary': m['primary'],
                                      'label': 'ensemble of %s'
                                               % ','.join(map(str, ens['members'])),
                                      'iteration': it, 'preds': ens['preds_path']}
                        self.best_history.append(max(self.best_history[-1], m['primary']))
                        if self._converged():
                            stop = 'converged'
                            break
                        continue

                # An LLM role can fail outright -- unparseable JSON twice, a transport
                # error, a budget halt. Before this, any such failure propagated out of
                # run() and killed the process: no FINAL_DESIGNATION, no submission, a
                # truncated journal, and the whole run's spend wasted. Run
                # 20260830T234014Z died exactly this way on "coder produced unparseable
                # JSON twice". A role failure is now an ordinary failed iteration:
                # logged, counted, and skipped.
                try:
                    hyp, cfg, seeds, after_files, diff, note = self._plan(it, parent)
                except BudgetHalt:
                    stop = 'budget'
                    break
                except Exception as exc:
                    self.jr.append(J.ERROR_RECOVERY, {
                        'iteration': it, 'failure': 'agent_role',
                        'timed_out': False, 'orphan_free': True, 'kill_note': '',
                        'seconds': 0.0,
                        'stderr': ('%s: %s' % (type(exc).__name__, exc))[:2000],
                        'hypothesis': '', 'action': 'iteration skipped'})
                    print('iter %03d  ROLE FAILED (%s) -> skipped'
                          % (it, type(exc).__name__), flush=True)
                    role_failures += 1
                    if role_failures >= MAX_ROLE_FAILURES:
                        stop = 'role_failures'
                        break
                    continue
                if hyp == 'SKIP':
                    print('iter %03d  skipped (analysis budget exhausted)' % it, flush=True)
                    continue
                if hyp is None:
                    stop = 'ideas_exhausted'
                    break

                prev = None
                if after_files:
                    ok, findings = guards.scan_diff(diff)
                    if not ok:
                        # Hand the finding back and let the coder fix it, instead of
                        # throwing the iteration away. guards.py is explicit that "a
                        # guard that rejects without explaining trains the agent to
                        # guess", and every finding already carries a `fix` written for
                        # this -- coder.code() has taken a `last_error` since it was
                        # written. Nothing was passing it. In run 20260831T011354Z the
                        # coder tripped the same evaluate-on-test rule at iterations 7,
                        # 8 and 11, each time losing a whole node, because it was never
                        # told. One extra coder call is far cheaper than a lost
                        # iteration: the iteration also costs a convergence slot, and
                        # those are the binding constraint, not tokens.
                        self.jr.append(J.GUARD_REJECT, {
                            'iteration': it, 'findings': findings, 'attempt': 1,
                            'action': 'returned to the coder with the finding',
                            'hypothesis': hyp.get('hypothesis', '')})
                        print('iter %03d  GUARD REJECT: %s -- returning it to the coder'
                              % (it, findings[0]['reason']), flush=True)
                        try:
                            after_files, note, _ = coder_mod.code(
                                self.llm, hyp, ROOT,
                                last_error=('The static guard rejected your change before it ran. '
                                            'This is not a runtime error -- the code never executed.'
                                            + chr(10) + chr(10)
                                            + guards.format_findings(findings)))
                            diff = coder_mod.make_diff(
                                coder_mod.current_files(ROOT), after_files)
                            ok, findings = guards.scan_diff(diff)
                        except BudgetHalt as exc:
                            # Named before Exception: BudgetHalt is a RuntimeError, and
                            # swallowing it here would report the ceiling as a coder
                            # error. The loop's own check stops the run next iteration.
                            self.jr.append(J.BUDGET_HALT,
                                           {'iteration': it, 'reason': str(exc)})
                            print('iter %03d  budget ceiling reached during guard retry'
                                  % it, flush=True)
                            ok = False
                        except Exception as exc:
                            ok, findings = False, [{
                                'file': '<coder>', 'line': 0, 'rule': 'retry failed',
                                'reason': '%s: %s' % (type(exc).__name__, exc),
                                'fix': '', 'snippet': ''}]
                        if ok:
                            print('iter %03d  guard clean on retry' % it, flush=True)
                    if not ok:
                        self.jr.append(J.GUARD_REJECT, {
                            'iteration': it, 'findings': findings, 'attempt': 2,
                            'action': 'iteration abandoned',
                            'hypothesis': hyp.get('hypothesis', '')})
                        print('iter %03d  GUARD REJECT (retry): %s'
                              % (it, findings[0]['reason']), flush=True)
                        continue
                    prev = coder_mod.write_change(ROOT, after_files)

                role_failures = 0          # a good iteration clears the streak
                timeout = executor.adaptive_timeout(self.node_times)
                res, parsed = self.run_node(it, cfg, seeds, timeout)

                # A syntax slip or a broken contract is a typo, not a refuted hypothesis.
                # Give the coder the traceback once before spending the iteration: the
                # node costs a minute of CPU, the iteration costs one of the ~10 that the
                # convergence rule allows.
                if not parsed and after_files and res['failure'] in RETRYABLE_FAILURES:
                    self.jr.append(J.ERROR_RECOVERY, {
                        'iteration': it, 'failure': res['failure'], 'attempt': 1,
                        'timed_out': res['timed_out'], 'orphan_free': res['orphan_free'],
                        'kill_note': res['kill_note'], 'seconds': res['seconds'],
                        'stderr': res['stderr'][-2000:],
                        'hypothesis': hyp.get('hypothesis', ''),
                        'action': 'returned to the coder with the traceback'})
                    print('iter %03d  FAILED (%s) -- returning the traceback to the coder'
                          % (it, res['failure']), flush=True)
                    try:
                        after2, note2, _ = coder_mod.code(
                            self.llm, hyp, ROOT,
                            last_error=('Your change ran and crashed. Failure class: %s.'
                                        % res['failure']) + chr(10) + chr(10)
                                       + (res['stderr'] or '')[-2000:])
                        d2 = coder_mod.make_diff(coder_mod.current_files(ROOT), after2)
                        ok2, f2 = guards.scan_diff(d2)
                        if ok2:
                            if prev:
                                coder_mod.restore(ROOT, prev)
                            after_files, note, diff = after2, note2, d2
                            prev = coder_mod.write_change(ROOT, after_files)
                            res, parsed = self.run_node(it, cfg, seeds, timeout)
                            print('iter %03d  retry %s' % (
                                it, 'ran' if parsed else 'failed again (%s)'
                                % res['failure']), flush=True)
                        else:
                            print('iter %03d  retry rejected by the guard: %s'
                                  % (it, f2[0]['reason']), flush=True)
                    except BudgetHalt as exc:
                        self.jr.append(J.BUDGET_HALT,
                                       {'iteration': it, 'reason': str(exc)})
                        print('iter %03d  budget ceiling reached during node retry'
                              % it, flush=True)
                    except Exception as exc:
                        print('iter %03d  retry could not be produced (%s)'
                              % (it, type(exc).__name__), flush=True)

                if not parsed:
                    self.jr.append(J.ERROR_RECOVERY, {
                        'iteration': it, 'failure': res['failure'],
                        'timed_out': res['timed_out'], 'orphan_free': res['orphan_free'],
                        'kill_note': res['kill_note'], 'seconds': res['seconds'],
                        'stderr': res['stderr'][-2000:],
                        'hypothesis': hyp.get('hypothesis', ''), 'action': 'node pruned'})
                    if prev:
                        coder_mod.restore(ROOT, prev)
                    print('iter %03d  FAILED (%s) -> reverted' % (it, res['failure']), flush=True)
                    verdict = self._verdict(hyp, diff, None, parent, failure=res)
                    self.jr.append(J.ITERATION, {
                        'iteration': it, 'hypothesis': hyp.get('hypothesis', ''),
                        'verdict': verdict['verdict'], 'reason': verdict['reason'],
                        'metrics': None, 'delta_vs_parent': None, 'failure': res['failure'],
                        'seconds': res['seconds'], 'config': cfg, 'diff': diff,
                        'provenance': 'harness' if self.dry_run else 'agent'})
                    continue

                self.node_times.append(parsed['seconds'])
                m = parsed['metrics']
                delta = m['primary'] - parent['primary']

                # A single seed can reject safely -- the deltas that matter are several
                # sigma. It cannot ACCEPT safely: at the 0.0014 bar, a true improvement and
                # a true zero are only ~2 sigma apart on one seed, and a false negative does
                # not merely lose an iteration, it burns one of the three that trigger
                # convergence. So anything not clearly negative is re-measured on matched
                # seeds, and the decision uses the MEAN PER-SEED primary. (run_node
                # rank-averages multiple seeds into an ensemble; that ensemble score is a
                # different quantity and would flatter the candidate.)
                confirm = None
                # Every node already trains SEEDS_PER_NODE seeds, and the parent kept its own
                # per-seed primaries, so the honest effect size is free: pair seed s against
                # seed s. That is a matched comparison. Contrasting one child seed with the
                # parent's rank-averaged ENSEMBLE is not -- the ensemble sits systematically
                # above any single seed, so such a test would reject even real improvements.
                confirm = paired_confirmation(parsed.get('per_seed_primary'),
                                              parent.get('per_seed'), delta)
                if confirm is not None:
                    print('iter %03d  paired seeds: mean %+.5f  worst %+.5f  (%d/%d up)'
                          % (it, confirm['mean_delta'], confirm['worst_delta'],
                             sum(1 for d in confirm['paired_deltas'] if d > 0),
                             len(confirm['paired_deltas'])), flush=True)

                # Fallback: a single-seed node cannot be paired, so re-measure it on matched
                # seeds. Confirmation spends training time, not tokens, and training is not
                # the binding constraint (100 baseline iterations is ~28 min of one CPU core).
                if confirm is None and len(seeds) == 1 and delta > CONFIRM_FLOOR:
                    cres, cparsed = self.run_node(
                        it, cfg, CONFIRM_SEEDS,
                        executor.adaptive_timeout(self.node_times) * len(CONFIRM_SEEDS))
                    if cparsed and cparsed.get('per_seed_primary'):
                        per = sorted(float(v) for v in cparsed['per_seed_primary'].values())
                        mean_primary = sum(per) / len(per)
                        confirm = {'seeds': CONFIRM_SEEDS, 'per_seed_primary': per,
                                   'basis': 'rerun_on_matched_seeds',
                                   'mean_primary': mean_primary,
                                   'mean_delta': mean_primary - parent['primary'],
                                   'worst_delta': per[0] - parent['primary'],
                                   'single_seed_delta': delta,
                                   'ensemble_primary': cparsed['metrics']['primary'],
                                   'seconds': cparsed['seconds']}
                        # The confirmation run is now the node: its metrics and predictions
                        # are the artifact that would actually be submitted.
                        m, parsed = cparsed['metrics'], cparsed
                        delta = confirm['mean_delta']
                        seeds = CONFIRM_SEEDS
                        print('iter %03d  confirming on %d seeds: mean %+.5f  worst %+.5f'
                              % (it, len(CONFIRM_SEEDS), confirm['mean_delta'],
                                 confirm['worst_delta']), flush=True)

                verdict = self._verdict(hyp, diff, m, parent)

                self.nodes.append({'iteration': it, 'primary': m['primary'],
                                   'preds': parsed['preds_path'],
                                   'label': hyp.get('hypothesis', '')[:80], 'metrics': m,
                                   'seeds': list(seeds),
                                   'artifact': ('rank_average_ensemble' if len(seeds) > 1
                                                else 'single_seed'),
                                   # Only the re-run branch REPLACES the node's artifact.
                                   # A paired confirmation just measures the one already
                                   # trained, so it must not relabel its provenance.
                                   'artifact_source': (
                                       'harness_confirmation'
                                       if (confirm or {}).get('basis') == 'rerun_on_matched_seeds'
                                       else 'agent_hypothesis')})

                self.jr.append(J.ITERATION, {
                    'iteration': it, 'hypothesis': hyp.get('hypothesis', ''),
                    'mechanism': hyp.get('mechanism', ''),
                    'predicted': hyp.get('expected_result', ''),
                    'invalid_if': hyp.get('invalid_if', ''),
                    'evidence': hyp.get('evidence', []),
                    'verdict': verdict['verdict'], 'reason': verdict['reason'],
                    'mechanism_update': verdict.get('mechanism_update', ''),
                    'deprioritise': verdict.get('deprioritise', []),
                    'metrics': m, 'delta_vs_parent': delta, 'accept_bar': ACCEPT,
                    'confirmation': confirm,
                    'seconds': parsed['seconds'], 'config': cfg, 'seeds': seeds,
                    'diff': diff, 'note': note,
                    'provenance': 'harness' if self.dry_run else 'agent',
                    'cost_so_far_usd': self.meter.totals()['usd']})

                if not self.dry_run:
                    knowledge.record(hyp.get('direction_id') or 'unlabelled', it, delta,
                                     verdict['verdict'], verdict.get('reason', ''),
                                     hyp.get('hypothesis', ''))

                # KEEP needs the shipped artifact to clear the bar AND every matched seed
                # to improve, so one lucky seed cannot carry a change into the lineage.
                # Under the null, 3/3 paired seeds agreeing has p = 0.125 on its own; with
                # the accept bar on top, a noise iteration essentially never gets kept.
                keep = verdict['verdict'] == 'KEEP' and delta > ACCEPT
                # A confirmed result outranks a narrative verdict. The reflector judges
                # against the hypothesis's own invalid_if, which is often a stricter band
                # (+/-0.002) than the harness accept bar (0.0014). In run 20260830T235541Z
                # a +0.00197 listwise result was called INCONCLUSIVE and reverted, so the
                # lineage never advanced and the next two iterations restarted from the
                # baseline. Hand-verification later measured it at +0.00162 mean over
                # three matched seeds, every seed positive -- a real effect. When
                # confirmation says that, the node joins the lineage regardless of the
                # narrative.
                # Unanimous paired seeds clear a bar set for the paired mean, not for
                # one draw. See UNANIMOUS_ACCEPT.
                if (not keep and confirm is not None
                        and confirm.get('all_paired_positive')
                        and confirm['mean_delta'] > UNANIMOUS_ACCEPT):
                    keep = True
                    print('iter %03d  KEEP: all %d paired seeds up, mean %+.5f (worst '
                          '%+.5f) -- clears the unanimous bar %.4f despite verdict %s'
                          % (it, len(confirm['paired_deltas']), confirm['mean_delta'],
                             confirm['worst_delta'], UNANIMOUS_ACCEPT,
                             verdict['verdict']), flush=True)
                elif (not keep and confirm is not None
                        and confirm['mean_delta'] > ACCEPT
                        and confirm['worst_delta'] > 0):
                    keep = True
                    print('iter %03d  KEEP on confirmed evidence (mean %+.5f, worst %+.5f) '
                          'despite verdict %s'
                          % (it, confirm['mean_delta'], confirm['worst_delta'],
                             verdict['verdict']), flush=True)
                if keep and confirm is not None and confirm['worst_delta'] <= 0:
                    keep = False
                    print('iter %03d  delta cleared the bar but seed delta %+.5f was not '
                          'positive -- not kept' % (it, confirm['worst_delta']), flush=True)
                if keep and confirm is None and len(seeds) == 1:
                    keep = False
                    print('iter %03d  unconfirmed single-seed gain -- not kept' % it,
                          flush=True)
                print('iter %03d  primary %.5f  delta %+.5f  %s%s'
                      % (it, m['primary'], delta, verdict['verdict'],
                         '' if keep else '  (reverted)'), flush=True)
                if keep:
                    parent = {'primary': m['primary'], 'label': hyp.get('hypothesis', '')[:60],
                              'iteration': it, 'preds': parsed['preds_path'],
                              'per_seed': parsed.get('per_seed_primary') or {}}
                elif prev:
                    coder_mod.restore(ROOT, prev)

                self.best_history.append(max(self.best_history[-1], m['primary']))
                if self._converged():
                    stop = 'converged'
                    break
            else:
                stop = 'iteration_cap'

        except BaseException as exc:
            stop = 'crashed'
            self.jr.append(J.ERROR_RECOVERY, {
                'iteration': -1, 'failure': 'controller_crash',
                'timed_out': False, 'orphan_free': True, 'kill_note': '',
                'seconds': round(time.time() - t_start, 1),
                'stderr': ('%s: %s' % (type(exc).__name__, exc))[:2000],
                'hypothesis': '', 'action': 'designating from nodes so far'})
            print('CONTROLLER CRASH (%s) -- designating from %d node(s)'
                  % (type(exc).__name__, len(self.nodes)), flush=True)
        finally:
            # The crash path needs this too: leaving the agent edits in the tree
            # would contaminate designation and the next run baseline.
            restore(base_snap)
        return self._designate(stop, uv, yv, time.time() - t_start)

    # ---------- helpers ----------
    def _converged(self):
        """eps/N over the best-so-far trajectory. The organizers' rule, implemented
        literally, because the scored checkpoint is the validation-best AT this point."""
        h = self.best_history
        if len(h) < N_CONV + 1:
            return False
        return (h[-1] - h[-1 - N_CONV]) <= EPS

    def _plan(self, it, parent, max_analyses=3):
        """Returns (hypothesis, config, seeds, after_files, diff, note).

        The proposer may ask for a diagnostic instead of an experiment. That does NOT
        consume an iteration — answering "does this column vary within a user?" costs
        three seconds, and finding out by training costs a minute and one of the ~10
        iterations the convergence rule allows.
        """
        if self.dry_run:
            if it - 1 >= len(DRY_IDEAS):
                return None, None, None, None, '', ''
            idea = dict(DRY_IDEAS[it - 1])
            return (idea, idea.get('config', {}),
                    idea.get('seeds', list(range(self.seeds_per_node))),
                    None, '', 'fixed idea')

        entries = list(self.jr.events(J.ITERATION))
        digest = proposer_mod.digest(entries)
        budget = 'Budget: $%.2f of $%.2f spent.' % (
            self.meter.totals()['usd'], self.meter.ceiling)
        catalogue = diagnostics.catalogue()
        answered = []

        for attempt in range(max_analyses + 1):
            last = attempt == max_analyses
            note_budget = budget if not last else (
                budget + '\n\nYou have used this iteration\'s analysis budget. Return '
                'action EXPERIMENT now, using what you already know.')
            prop, _, rejections = proposer_mod.propose(
                self.llm, digest, parent, it, note_budget,
                directions=knowledge.summary(), catalogue=catalogue,
                closed_ids=set(knowledge.closed_ids()),
                analysis_results=answered or None)
            # A rejected proposal costs a whole extra call. Record why, so the waste is
            # visible in the report instead of hiding inside a doubled proposer bill.
            for why in rejections:
                self.jr.append(J.PROPOSAL_REJECT, {'iteration': it, 'reason': why})
                print('iter %03d  proposal rejected: %s' % (it, why[:110]), flush=True)

            if prop.get('action') != 'REQUEST_ANALYSIS':
                break
            if last:
                # It asked again after being told it could not. Skip this iteration
                # rather than ending the run — an over-curious proposer is not a reason
                # to throw away the remaining budget.
                self.jr.append(J.ANALYSIS, {
                    'iteration': it, 'analysis': None,
                    'note': 'analysis budget exhausted; iteration skipped'})
                return 'SKIP', None, None, None, '', ''

            name = prop.get('analysis')
            params = prop.get('params') or {}
            result = diagnostics.run(name, **params)
            answered.append({'analysis': name, 'params': params,
                             'question': prop.get('question', ''), 'result': result})
            self.jr.append(J.ANALYSIS, {
                'iteration': it, 'analysis': name, 'params': params,
                'question': prop.get('question', ''),
                'why_needed': prop.get('why_needed', ''), 'result': result})
            print('iter %03d  ANALYSIS %s(%s)' % (it, name, params), flush=True)

        hyp = prop['candidates'][prop['chosen']]
        hyp['_alternatives'] = [c.get('hypothesis') for i, c in
                                enumerate(prop['candidates']) if i != prop['chosen']]
        hyp['_rationale'] = prop.get('rationale', '')
        hyp['_analyses_used'] = [a['analysis'] for a in answered]
        after, note, _ = coder_mod.code(self.llm, hyp, ROOT)
        before = coder_mod.current_files(ROOT)
        diff = coder_mod.make_diff(before, after)
        # The proposer's config is what activates an opt-in code path. Dropping it
        # (as this line used to) makes any change that adds a new branch inert: the
        # node runs the default path and scores exactly its parent.
        cfg = hyp.get('config')
        if not isinstance(cfg, dict):
            cfg = {}
        return hyp, cfg, list(range(self.seeds_per_node)), after, diff, note

    def _verdict(self, hyp, diff, metrics, parent, failure=None):
        if self.dry_run or self.llm is None:
            return reflector_mod.offline_verdict(metrics, parent, failure, ACCEPT)
        try:
            v, _ = reflector_mod.reflect(self.llm, hyp, diff, metrics, parent, failure)
            return v
        except Exception as exc:
            self.jr.append(J.ERROR_RECOVERY,
                           {'stage': 'reflector', 'error': str(exc)[:400],
                            'action': 'fell back to the deterministic verdict'})
            return reflector_mod.offline_verdict(metrics, parent, failure, ACCEPT)

    def _designate(self, stop, uv, yv, elapsed):
        """Pick the validation-best checkpoint, write the submission, seal the decision.

        FINAL_DESIGNATION is written BEFORE anything may read a test label. verify.py
        checks that ordering, which is what turns "we only tuned on validation" from an
        assertion into something a judge can confirm.
        """
        import numpy as np
        from harness import selection
        from harness.submission import write, check
        from harness.score import rank_average

        # Load every node's stored predictions. Selection is NOT an argmax over these:
        # best-of-N on a metric with sigma ~ 0.0005 selects the luckiest draw, and that
        # inflation is exactly what fails to reach the hidden test set.
        cands = []
        for n in self.nodes:
            z = np.load(n['preds'])
            cands.append({'iteration': n['iteration'], 'label': n['label'],
                          'valid': z['valid'], 'test': z['test']})
        base = next(c for c in cands if c['iteration'] == 0)

        floor_valid = None
        floor_path = os.path.join(SUBMISSIONS, 'floor_valid.csv')
        if os.path.exists(floor_path):
            ok_f, _, scores_f = check(floor_path, 'valid')
            if ok_f:
                floor_valid = np.asarray(scores_f, dtype=float)

        print('designating (stability-tested, %d-fold):' % selection.N_FOLDS, flush=True)
        choice, sel_report = selection.designate(
            uv, yv, cands, base['valid'], floor_preds=floor_valid,
            log=lambda s: print(s, flush=True))

        # Build the test-side vector for whatever was chosen.
        if choice['kind'] == 'ensemble':
            _, _, ut = load_encoded()[0]['test']
            test_vec = rank_average(np.asarray(ut), choice['test_members'])
        elif choice['kind'] == 'floor':
            fp = os.path.join(SUBMISSIONS, 'floor_test.csv')
            ok_t, _, s_t = check(fp, 'test')
            if not ok_t:
                raise RuntimeError('floor tripwire fired but floor_test.csv is invalid')
            test_vec = np.asarray(s_t, dtype=float)
        else:
            test_vec = choice['test']

        # Provenance of what is actually being submitted. Selection may land on a node
        # (which carries its own provenance), or on something the harness builds at
        # designation time -- the rank-average ensemble (iteration -1) or the banked
        # floor (-2) -- which has no node and must not borrow one's label.
        chosen_node = next((n for n in self.nodes
                            if n['iteration'] == choice['iteration']), None)
        if chosen_node is not None:
            chosen_artifact = chosen_node.get('artifact', 'single_seed')
            chosen_source = chosen_node.get('artifact_source', 'agent_hypothesis')
            chosen_seeds = chosen_node.get('seeds', [0])
        elif choice['kind'] == 'ensemble':
            chosen_artifact = 'rank_average_ensemble'
            chosen_source = 'harness_selection'
            chosen_seeds = sorted({sd for n in self.nodes
                                   if n['iteration'] in (sel_report.get('ensemble') or {})
                                                        .get('members', [])
                                   for sd in n.get('seeds', [])})
        else:
            chosen_artifact = 'banked_floor'
            chosen_source = 'human_banked_floor'
            chosen_seeds = []

        sub = os.path.join(SUBMISSIONS, '%s_final.csv' % self.run_id)
        write(sub, 'test', test_vec)
        ok, msg, _ = check(sub, 'test')
        payload = {
            'stop_reason': stop,
            'chosen_kind': choice['kind'],
            'chosen_iteration': choice['iteration'],
            'chosen_label': choice['label'],
            'validation_primary': choice['primary'],
            'selection': sel_report,
            'candidates_considered': [
                {'iteration': n['iteration'], 'primary': n['primary'], 'label': n['label'],
                 'seeds': n.get('seeds', [0]),
                 'artifact': n.get('artifact', 'single_seed'),
                 'artifact_source': n.get('artifact_source', 'agent_hypothesis')}
                for n in self.nodes],
            # Provenance of the thing actually submitted. A rank-average ensemble
            # produced by a harness confirmation run scores higher than the single-seed
            # model the agent proposed, purely from variance reduction. That is a real
            # and permitted gain, but it is the harness's doing, not the agent's, and
            # the run log has to say so rather than let it read as a discovery.
            'chosen_artifact': chosen_artifact,
            'chosen_artifact_source': chosen_source,
            'chosen_seeds': chosen_seeds,
            'submission': sub,
            'submission_sha256': sha256_file(sub),
            'submission_check_ok': ok, 'submission_check_msg': msg,
            'iterations_used': len(self.nodes) - 1,
            'iteration_cap': self.max_iters,
            'wall_clock_s': round(elapsed, 1),
            'resources': self.meter.totals(),
            'gpu_hours': 0.0,
            'cache_working': self.meter.cache_working() if not self.dry_run else None,
        }
        self.jr.append(J.FINAL_DESIGNATION, payload)
        if stop == 'converged':
            self.jr.append(J.CONVERGED, {'eps': EPS, 'N': N_CONV,
                                         'best_history': self.best_history})
        return payload


def inject(kind):
    """Write a deliberate fault into the pipeline. Returns the snapshot for restore."""
    rel, mutate, why = INJECTIONS[kind]
    snap = snapshot()
    p = os.path.join(ROOT, rel.replace('/', os.sep))
    src = open(p, encoding='utf-8').read()
    with open(p, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(mutate(src))
    return snap, why


def main():
    ap = argparse.ArgumentParser(description='Autonomous research loop.')
    ap.add_argument('--dry-run', action='store_true',
                    help='fixed idea list, no LLM — proves the plumbing')
    ap.add_argument('--iterations', type=int, default=MAX_ITERS)
    ap.add_argument('--budget', type=float, default=14.0, help='USD ceiling')
    ap.add_argument('--packet', default=os.path.join(ROOT, 'context', 'packet.md'))
    ap.add_argument('--inject', choices=sorted(INJECTIONS),
                    help='inject one fault, run a single node, verify recovery')
    ap.add_argument('--run-id', default=None)
    a = ap.parse_args()

    if a.inject:
        snap, why = inject(a.inject)
        rid = a.run_id or ('inject_%s_%s' % (a.inject, time.strftime('%H%M%S')))
        c = Controller(run_id=rid, dry_run=True, max_iters=1)
        print('injected %s: %s' % (a.inject, why), flush=True)
        try:
            if a.inject == 'leak':
                ok, findings = guards.scan_diff(
                    '--- a/pipeline/features.py\n+++ b/pipeline/features.py\n'
                    + '\n'.join('+' + ln for ln in
                                open(os.path.join(PIPELINE, 'features.py'),
                                     encoding='utf-8').read().split('\n')))
                c.jr.append(J.RUN_START, {'injected': a.inject})
                c.jr.append(J.GUARD_REJECT if not ok else J.ITERATION,
                            {'iteration': 1, 'findings': findings,
                             'recovered': not ok})
                print('RECOVERED: guard rejected before execution' if not ok
                      else 'NOT RECOVERED: guard missed the leak', flush=True)
                ok_final = not ok
            else:
                timeout = 20 if a.inject == 'timeout' else 900
                c.jr.append(J.RUN_START, {'injected': a.inject})
                res, parsed = c.run_node(1, {}, [0], timeout)
                c.jr.append(J.ERROR_RECOVERY, {
                    'iteration': 1, 'injected': a.inject, 'failure': res['failure'],
                    'timed_out': res['timed_out'], 'orphan_free': res['orphan_free'],
                    'kill_note': res['kill_note'], 'seconds': res['seconds'],
                    'recovered': parsed is None})
                ok_final = parsed is None and res['orphan_free']
                print('failure=%s timed_out=%s orphan_free=%s (%.1fs)'
                      % (res['failure'], res['timed_out'], res['orphan_free'],
                         res['seconds']), flush=True)
                print('RECOVERED: node failed safely and was logged' if ok_final
                      else 'NOT RECOVERED', flush=True)
        finally:
            restore(snap)
            print('pipeline restored', flush=True)
        raise SystemExit(0 if ok_final else 1)

    try:
        c = Controller(run_id=a.run_id, dry_run=a.dry_run,
                       max_iters=a.iterations, budget_usd=a.budget,
                       packet_path=None if a.dry_run else a.packet)
    except LLMUnavailable as exc:
        print('LLM unavailable: %s' % exc, file=sys.stderr)
        raise SystemExit(2)
    out = c.run()
    print(json.dumps(out, indent=2, default=float))
    print('\njournal: %s' % c.jr.path)


if __name__ == '__main__':
    main()
