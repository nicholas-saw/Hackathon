"""Integrity checker a judge can run in five minutes.

The test labels ship in this repository. `kit/baseline.py` prints test metrics with no
flag at all, and `kit/submit.py --score` defaults to `--split test`. So "we developed on
validation only" is, on its own, an unfalsifiable claim about a repo where the answer key
is sitting on disk.

This makes it checkable. The journal is hash-chained, so a later edit is detectable, and
FINAL_DESIGNATION must appear before TEST_OPEN — meaning the submission was named before
anything read a test label.

    python verify.py --chain --order runlogs/<run_id>/journal.jsonl
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harness.journal import (Journal, FINAL_DESIGNATION, HUMAN_INTERVENTION,   # noqa: E402
                             INTERVENTION_CLASSES, ITERATION, TEST_OPEN,
                             verify_chain, verify_order)


def intervention_report(path):
    counted, logged = [], []
    for e in Journal.read(path):
        if e['type'] != HUMAN_INTERVENTION:
            continue
        p = e['payload']
        (counted if p.get('counts') else logged).append(p)
    return counted, logged


def summary(path):
    evs = Journal.read(path)
    iters = [e for e in evs if e['type'] == ITERATION]
    fd = next((e for e in evs if e['type'] == FINAL_DESIGNATION), None)
    counted, logged = intervention_report(path)
    return {'events': len(evs), 'iterations': len(iters),
            'interventions_counted': len(counted), 'interventions_logged': len(logged),
            'final': fd['payload'] if fd else None}


def main():
    ap = argparse.ArgumentParser(description='Verify a run journal.')
    ap.add_argument('journal')
    ap.add_argument('--chain', action='store_true', help='recompute the hash chain')
    ap.add_argument('--order', action='store_true',
                    help='FINAL_DESIGNATION must precede TEST_OPEN')
    ap.add_argument('--json', action='store_true')
    a = ap.parse_args()
    if not (a.chain or a.order):
        a.chain = a.order = True

    results, ok_all = [], True
    if a.chain:
        ok, msg = verify_chain(a.journal)
        ok_all &= ok
        results.append(('hash chain', ok, msg))
    if a.order:
        ok, msg = verify_order(a.journal)
        ok_all &= ok
        results.append(('designation order', ok, msg))

    s = summary(a.journal)
    if a.json:
        print(json.dumps({'ok': ok_all,
                          'checks': [{'name': n, 'ok': o, 'detail': m} for n, o, m in results],
                          'summary': s}, indent=2, default=float))
    else:
        for name, ok, msg in results:
            print('%-20s %-6s %s' % (name, 'PASS' if ok else 'FAIL', msg))
        print('%-20s %-6s %d events, %d iterations'
              % ('journal', 'INFO', s['events'], s['iterations']))
        print('%-20s %-6s %d counted (L2-L5), %d logged-only (L0/L1)'
              % ('interventions', 'INFO', s['interventions_counted'],
                 s['interventions_logged']))
        f = s['final']
        if f:
            print('%-20s %-6s iteration %s, validation primary %.5f'
                  % ('designated', 'INFO', f.get('chosen_iteration'),
                     f.get('validation_primary', float('nan'))))
            print('%-20s %-6s %s' % ('submission', 'INFO', f.get('submission')))
            print('%-20s %-6s %s' % ('sha256', 'INFO', (f.get('submission_sha256') or '')[:32]))
            r = f.get('resources') or {}
            print('%-20s %-6s %s tokens, $%.2f, %s iterations of %s, %.0fs wall, %s GPU-h'
                  % ('resources', 'INFO', r.get('tokens_total', 0), r.get('usd', 0.0),
                     f.get('iterations_used'), f.get('iteration_cap'),
                     f.get('wall_clock_s', 0), f.get('gpu_hours')))
        print()
        print('OVERALL: ' + ('PASS' if ok_all else 'FAIL'))
    raise SystemExit(0 if ok_all else 1)


if __name__ == '__main__':
    main()
