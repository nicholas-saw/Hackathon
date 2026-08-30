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
MAX_ITERS = 50           # official hard cap
MAX_WALL_S = 6 * 3600    # official wall-clock ceiling
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
        res0, p0 = self.run_node(0, {}, [0], timeout=900)
        if not p0:
            self.jr.append(J.ERROR_RECOVERY, {'iteration': 0, 'fatal': True, **res0})
            raise RuntimeError('baseline node failed; harness is not trustworthy:\n%s'
                               % res0['stderr'][-1500:])
        parent = {'primary': p0['metrics']['primary'], 'label': 'baseline',
                  'iteration': 0, 'preds': p0['preds_path']}
        self.node_times.append(p0['seconds'])
        self.nodes.append({'iteration': 0, 'primary': parent['primary'],
                           'preds': p0['preds_path'], 'label': 'baseline',
                           'metrics': p0['metrics']})
        self.best_history.append(parent['primary'])
        self.jr.append(J.ITERATION, {
            'iteration': 0, 'hypothesis': 'reproduce the official baseline',
            'verdict': 'KEEP', 'metrics': p0['metrics'], 'delta_vs_parent': 0.0,
            'seconds': p0['seconds'], 'config': {}, 'diff': '', 'provenance': 'harness'})
        print('iter 000  baseline  valid primary %.5f' % parent['primary'], flush=True)

        stop = None
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

            hyp, cfg, seeds, after_files, diff, note = self._plan(it, parent)
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
                    self.jr.append(J.GUARD_REJECT, {
                        'iteration': it, 'findings': findings,
                        'hypothesis': hyp.get('hypothesis', '')})
                    print('iter %03d  GUARD REJECT: %s' % (it, findings[0]['reason']),
                          flush=True)
                    continue
                prev = coder_mod.write_change(ROOT, after_files)

            timeout = executor.adaptive_timeout(self.node_times)
            res, parsed = self.run_node(it, cfg, seeds, timeout)

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
            verdict = self._verdict(hyp, diff, m, parent)

            self.nodes.append({'iteration': it, 'primary': m['primary'],
                               'preds': parsed['preds_path'],
                               'label': hyp.get('hypothesis', '')[:80], 'metrics': m})

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
                'seconds': parsed['seconds'], 'config': cfg, 'seeds': seeds,
                'diff': diff, 'note': note,
                'provenance': 'harness' if self.dry_run else 'agent',
                'cost_so_far_usd': self.meter.totals()['usd']})

            if not self.dry_run:
                knowledge.record(hyp.get('direction_id') or 'unlabelled', it, delta,
                                 verdict['verdict'], verdict.get('reason', ''),
                                 hyp.get('hypothesis', ''))

            keep = verdict['verdict'] == 'KEEP' and delta > ACCEPT
            print('iter %03d  primary %.5f  delta %+.5f  %s%s'
                  % (it, m['primary'], delta, verdict['verdict'],
                     '' if keep else '  (reverted)'), flush=True)
            if keep:
                parent = {'primary': m['primary'], 'label': hyp.get('hypothesis', '')[:60],
                          'iteration': it, 'preds': parsed['preds_path']}
            elif prev:
                coder_mod.restore(ROOT, prev)

            self.best_history.append(max(self.best_history[-1], m['primary']))
            if self._converged():
                stop = 'converged'
                break
        else:
            stop = 'iteration_cap'

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
            return idea, idea.get('config', {}), idea.get('seeds', [0]), None, '', 'fixed idea'

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
            prop, _ = proposer_mod.propose(
                self.llm, digest, parent, it, note_budget,
                directions=knowledge.summary(), catalogue=catalogue,
                closed_ids=set(knowledge.closed_ids()),
                analysis_results=answered or None)

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
        return hyp, {}, [0], after, diff, note

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
                {'iteration': n['iteration'], 'primary': n['primary'], 'label': n['label']}
                for n in self.nodes],
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
