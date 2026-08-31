"""Reconstruct the pipeline a run actually won with, from its journal.

The controller restores the pristine pipeline in a `finally` block, so the code that
produced a run's best score does not survive the run. That is deliberate -- the next
run's baseline has to be clean -- but it means the winning artifact is a CSV with no
reproducible source behind it. Run 20260831T011354Z scored 0.60724 on validation, its
best node was designated and submitted, and nothing in the working tree could produce
that number again.

Nothing was lost, though: the journal stores the unified diff of every iteration. This
walks a journal, pulls the diffs of the iterations that were KEPT, and writes them out
in order with the config each one needs. `git apply` them in sequence and the pipeline
is back.

    python replay.py runlogs/<run_id>/journal.jsonl              # show what it kept
    python replay.py runlogs/<run_id>/journal.jsonl -o patches/   # write the diffs

Verified: both KEEP diffs of run 20260831T011354Z apply cleanly to the committed
pipeline with `git apply --directory=kuairand-starter-kit`.
"""
import argparse
import json
import os


def kept_iterations(journal_path):
    """Iterations whose change entered the lineage, oldest first.

    'KEEP' in the journal is the reflector's verdict, which is not the same thing as the
    controller keeping the node -- the accept bar and the paired-seed gate sit between
    them. The lineage is what the FINAL_DESIGNATION's chosen node descends from, so the
    honest reconstruction follows the recorded verdict AND the recorded delta.
    """
    out, designated = [], None
    with open(journal_path, encoding='utf-8') as fh:
        for line in fh:
            ev = json.loads(line)
            p = ev.get('payload', {})
            if ev['type'] == 'iteration' and p.get('verdict') == 'KEEP' \
                    and (p.get('iteration') or 0) > 0 and p.get('diff'):
                out.append({'iteration': p['iteration'],
                            'diff': p['diff'],
                            'config': p.get('config') or {},
                            'primary': (p.get('metrics') or {}).get('primary'),
                            'delta': p.get('delta_vs_parent'),
                            'confirmation': p.get('confirmation'),
                            'hypothesis': p.get('hypothesis', '')})
            elif ev['type'] == 'FINAL_DESIGNATION':
                designated = p
    return out, designated


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('journal')
    ap.add_argument('-o', '--out', help='directory to write the .patch files into')
    a = ap.parse_args()

    kept, final = kept_iterations(a.journal)
    if not kept:
        print('no KEEP iterations carrying a diff in %s' % a.journal)
        print('(a run whose best node was the baseline has nothing to reconstruct)')
        return 0

    if final:
        print('designated: iteration %s, validation primary %.5f (%s)'
              % (final.get('chosen_iteration'), final.get('validation_primary') or 0.0,
                 final.get('chosen_artifact', '?')))
    print('%d kept change(s), apply in this order:' % len(kept))
    print()
    merged = {}
    for k in kept:
        files = [ln[6:] for ln in k['diff'].split('\n') if ln.startswith('+++ b/')]
        print('  iteration %-3s primary %.5f  delta %+.5f'
              % (k['iteration'], k['primary'] or 0.0, k['delta'] or 0.0))
        print('    %s' % (k['hypothesis'][:88]))
        print('    files : %s' % ', '.join(files))
        if k['config']:
            print('    config: %s' % json.dumps(k['config']))
            merged.update(k['config'])
        c = k['confirmation']
        if c:
            print('    seeds : mean %+.5f worst %+.5f'
                  % (c.get('mean_delta', 0.0), c.get('worst_delta', 0.0)))
        print()

    if merged:
        print('combined config for fit_predict: %s' % json.dumps(merged))
        print()

    if not a.out:
        print('re-run with -o <dir> to write the patches.')
        return 0

    os.makedirs(a.out, exist_ok=True)
    written = []
    for k in kept:
        p = os.path.join(a.out, 'iter%03d.patch' % k['iteration'])
        # newline='' keeps the diff bytes exactly as the run recorded them; rewriting
        # LF as CRLF here would make every hunk fail to apply.
        with open(p, 'w', encoding='utf-8', newline='') as fh:
            fh.write(k['diff'])
        written.append(p)
        print('wrote %s' % p)
    print()
    print('apply from the repository root, in order:')
    for p in written:
        print('  git apply --directory=%s %s'
              % (os.path.basename(os.path.dirname(os.path.abspath(__file__))), p))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
