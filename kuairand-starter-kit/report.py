"""journal.jsonl -> a single self-contained report.html.

The run report is the deliverable judges actually read: Innovation is scored on what the
agent chose to try and why, Autonomy on the intervention count, Feasibility on tokens and
wall-clock. All of that lives in the journal; this turns it into something a person can
scan in two minutes without a server, a build step or a network connection.
"""
import argparse
import html
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harness.journal import (Journal, ERROR_RECOVERY, FINAL_DESIGNATION,     # noqa: E402
                             GUARD_REJECT, HUMAN_INTERVENTION, ITERATION,
                             RUN_START, verify_chain, verify_order)

CSS = """
:root{--bg:#eef1f4;--card:#fff;--ink:#101820;--ink2:#46525e;--ink3:#7c8894;
--rule:#cdd5dd;--ok:#2d6a4f;--bad:#a31621;--warn:#b45309;--code:#2a6f97}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 "Segoe UI",system-ui,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:28px 24px 72px}
h1{font-size:30px;letter-spacing:-.02em;margin:0 0 4px}
h2{font-size:17px;margin:34px 0 10px;padding-bottom:6px;border-bottom:2px solid var(--ink)}
.sub{color:var(--ink2);margin:0 0 20px}
.panel{display:grid;grid-template-columns:repeat(auto-fit,minmax(132px,1fr));gap:1px;
background:var(--rule);border:1px solid var(--rule);margin:18px 0}
.panel div{background:var(--card);padding:11px 13px}
.panel dt{font:11px ui-monospace,Consolas,monospace;letter-spacing:.09em;text-transform:uppercase;color:var(--ink3);margin-bottom:4px}
.panel dd{margin:0;font-size:20px;font-weight:700;letter-spacing:-.01em;font-variant-numeric:tabular-nums}
.panel dd small{display:block;font-size:11px;font-weight:400;color:var(--ink3);margin-top:2px}
table{border-collapse:collapse;width:100%;background:var(--card);border:1px solid var(--rule);font-size:13px}
th{text-align:left;font:11px ui-monospace,Consolas,monospace;letter-spacing:.08em;text-transform:uppercase;
color:var(--ink3);background:#e7ebef;padding:8px 10px;border-bottom:1px solid var(--rule)}
td{padding:9px 10px;border-bottom:1px solid #e6eaee;vertical-align:top}
tr:last-child td{border-bottom:none}
.n{text-align:right;font-family:ui-monospace,Consolas,monospace;font-variant-numeric:tabular-nums;white-space:nowrap}
.pill{display:inline-block;font:10px ui-monospace,Consolas,monospace;letter-spacing:.06em;
text-transform:uppercase;padding:2px 7px;border-radius:2px;border:1px solid}
.keep{color:var(--ok);background:#eaf3ee;border-color:var(--ok)}
.revert{color:var(--bad);background:#fbeced;border-color:var(--bad)}
.inconclusive{color:var(--warn);background:#fdf3e7;border-color:var(--warn)}
.pos{color:var(--ok);font-weight:600}.neg{color:var(--bad);font-weight:600}
.card{background:var(--card);border:1px solid var(--rule);padding:15px 17px;margin:12px 0}
.card.good{border-top:3px solid var(--ok)}.card.bad{border-top:3px solid var(--bad)}
details{margin-top:7px}summary{cursor:pointer;color:var(--code);font-size:12px}
pre{font:12px/1.45 ui-monospace,Consolas,monospace;background:#f5f7f9;border:1px solid var(--rule);
padding:9px 11px;overflow-x:auto;white-space:pre-wrap;margin:7px 0 0}
.mono{font-family:ui-monospace,Consolas,monospace}
.muted{color:var(--ink3)}
"""


def esc(x):
    return html.escape(str(x), quote=False)


def fmt(v, spec='%.5f'):
    try:
        return spec % float(v)
    except Exception:
        return '—'


def build(journal_path):
    evs = Journal.read(journal_path)
    start = next((e for e in evs if e['type'] == RUN_START), None)
    final = next((e for e in evs if e['type'] == FINAL_DESIGNATION), None)
    iters = [e for e in evs if e['type'] == ITERATION]
    recov = [e for e in evs if e['type'] == ERROR_RECOVERY]
    rejects = [e for e in evs if e['type'] == GUARD_REJECT]
    inters = [e for e in evs if e['type'] == HUMAN_INTERVENTION]
    counted = [e for e in inters if e['payload'].get('counts')]

    chain_ok, chain_msg = verify_chain(journal_path)
    order_ok, order_msg = verify_order(journal_path)

    f = final['payload'] if final else {}
    res = f.get('resources') or {}
    sp = start['payload'] if start else {}

    o = ['<!doctype html><html><head><meta charset="utf-8">',
         '<title>Run report — %s</title>' % esc(sp.get('run_id', 'run')),
         '<style>%s</style></head><body><div class="wrap">' % CSS]

    o.append('<h1>Autonomous run report</h1>')
    o.append('<p class="sub mono">%s &middot; KuaiRand-Pure &middot; stopped: <b>%s</b>%s</p>'
             % (esc(sp.get('run_id', '?')), esc(f.get('stop_reason', 'in progress')),
                ' &middot; <b>DRY RUN (no model in the loop)</b>' if sp.get('dry_run') else ''))

    # ---- cost panel: this IS the Feasibility submission ----
    o.append('<div class="panel">')
    for label, val, sub in [
        ('Validation primary', fmt(f.get('validation_primary')), 'designated checkpoint'),
        ('Iterations', '%s<small>of %s</small>' % (f.get('iterations_used', len(iters) - 1),
                                                   f.get('iteration_cap', 50)), None),
        ('Wall-clock', '%.0f s' % (f.get('wall_clock_s') or 0), 'of 21600 s ceiling'),
        ('LLM tokens', '{:,}'.format(res.get('tokens_total', 0)), 'input + output'),
        ('Cost', '$%.2f' % res.get('usd', 0.0), 'Anthropic API'),
        ('GPU-hours', '%.1f' % (f.get('gpu_hours') or 0.0), 'CPU only by design'),
        ('Interventions', str(len(counted)), 'L2-L5, counted'),
    ]:
        o.append('<div><dt>%s</dt><dd>%s%s</dd></div>'
                 % (esc(label), val, '<small>%s</small>' % esc(sub) if sub else ''))
    o.append('</div>')

    # ---- integrity ----
    o.append('<h2>Integrity</h2>')
    o.append('<div class="card %s">' % ('good' if chain_ok and order_ok else 'bad'))
    for name, ok, msg in (('Hash chain', chain_ok, chain_msg),
                          ('Designation before test read', order_ok, order_msg)):
        o.append('<div><span class="pill %s">%s</span> <b>%s</b> — <span class="muted">%s</span></div>'
                 % ('keep' if ok else 'revert', 'pass' if ok else 'fail', esc(name), esc(msg)))
    if f.get('submission'):
        o.append('<div style="margin-top:9px" class="mono muted">submission %s<br>sha256 %s<br>%s</div>'
                 % (esc(os.path.basename(f['submission'])),
                    esc((f.get('submission_sha256') or '')[:48]),
                    esc(f.get('submission_check_msg', ''))))
    try:
        shown = os.path.relpath(journal_path)
    except ValueError:
        shown = journal_path        # different drive on Windows; relpath refuses
    o.append('<div style="margin-top:9px" class="muted">Reproduce: '
             '<span class="mono">python verify.py --chain --order %s</span></div>'
             % esc(shown))
    o.append('</div>')

    # ---- iterations ----
    o.append('<h2>Iterations</h2>')
    o.append('<table><thead><tr><th>#</th><th>Hypothesis</th><th class="n">Primary</th>'
             '<th class="n">&Delta; vs parent</th><th>Verdict</th><th class="n">Sec</th>'
             '</tr></thead><tbody>')
    for e in iters:
        p = e['payload']
        m = p.get('metrics') or {}
        d = p.get('delta_vs_parent')
        v = (p.get('verdict') or '?').lower()
        dcls = 'pos' if isinstance(d, (int, float)) and d > 0 else (
            'neg' if isinstance(d, (int, float)) and d < 0 else 'muted')
        detail = []
        for k, lbl in (('mechanism', 'Mechanism'), ('predicted', 'Predicted'),
                       ('invalid_if', 'Falsifier'), ('reason', 'Verdict reason'),
                       ('mechanism_update', 'Mechanism update')):
            if p.get(k):
                detail.append('<b>%s:</b> %s' % (lbl, esc(p[k])))
        if p.get('evidence'):
            detail.append('<b>Evidence:</b> ' + esc('; '.join(map(str, p['evidence']))))
        if p.get('failure'):
            detail.append('<b>Failure:</b> ' + esc(p['failure']))
        blk = '<details><summary>detail</summary><div style="margin-top:6px">%s</div>%s</details>' % (
            '<br>'.join(detail) or '<span class="muted">none recorded</span>',
            '<pre>%s</pre>' % esc(p['diff'][:4000]) if p.get('diff') else '')
        o.append('<tr><td class="n">%s</td><td>%s%s</td><td class="n">%s</td>'
                 '<td class="n %s">%s</td><td><span class="pill %s">%s</span></td>'
                 '<td class="n">%s</td></tr>'
                 % (p.get('iteration', '?'), esc(p.get('hypothesis', '')), blk,
                    fmt(m.get('primary')) if m else '—',
                    dcls, ('%+.5f' % d) if isinstance(d, (int, float)) else '—',
                    v if v in ('keep', 'revert') else 'inconclusive',
                    esc(p.get('verdict', '?')), fmt(p.get('seconds'), '%.0f')))
    o.append('</tbody></table>')

    # ---- evidence the agent asked for ----
    analyses = [e for e in evs if e['type'] == 'analysis' and e['payload'].get('analysis')]
    if analyses:
        o.append('<h2>Evidence the agent requested</h2>')
        o.append('<p class="sub">Diagnostics run on train and validation before spending '
                 'an iteration. Asking costs seconds; finding out by training costs a '
                 'minute and one of roughly ten iterations.</p>')
        o.append('<table><thead><tr><th class="n">Iter</th><th>Analysis</th>'
                 '<th>Question</th><th>Result</th></tr></thead><tbody>')
        for e in analyses:
            p = e['payload']
            # NOT `res` — that name holds the run's resource totals, and rebinding it
            # here silently deleted the cost panel further down.
            ares = p.get('result') or {}
            key = ares.get('verdict') or ares.get('primary') or ares.get('coverage_pct')
            o.append('<tr><td class="n">%s</td><td class="mono">%s(%s)</td><td>%s</td>'
                     '<td>%s<details><summary>full</summary><pre>%s</pre></details></td></tr>'
                     % (p.get('iteration', '?'), esc(p.get('analysis')),
                        esc(json.dumps(p.get('params') or {})[1:-1]),
                        esc(p.get('question', '')),
                        esc(key if key is not None else ''),
                        esc(json.dumps(ares, indent=1)[:1800])))
        o.append('</tbody></table>')

    # ---- robustness ----
    o.append('<h2>Robustness</h2>')
    if not recov and not rejects:
        o.append('<p class="muted">No failures or guard rejections in this run.</p>')
    else:
        o.append('<table><thead><tr><th>Seq</th><th>Kind</th><th>Detail</th>'
                 '<th>Recovered</th></tr></thead><tbody>')
        for e in rejects:
            p = e['payload']
            first = (p.get('findings') or [{}])[0]
            o.append('<tr><td class="n">%d</td><td>guard reject</td><td>%s</td>'
                     '<td><span class="pill keep">yes</span> before execution</td></tr>'
                     % (e['seq'], esc(first.get('reason', ''))))
        for e in recov:
            p = e['payload']
            o.append('<tr><td class="n">%d</td><td>%s</td><td>%s</td>'
                     '<td><span class="pill %s">%s</span> %s</td></tr>'
                     % (e['seq'], esc(p.get('failure') or p.get('stage', 'error')),
                        esc((p.get('kill_note') or p.get('error') or p.get('action') or ''))[:160],
                        'keep' if p.get('orphan_free', True) else 'revert',
                        'yes' if p.get('orphan_free', True) else 'orphan',
                        'node pruned' if p.get('action') else ''))
        o.append('</tbody></table>')

    # ---- interventions ----
    o.append('<h2>Manual interventions</h2>')
    o.append('<p class="sub">Counted classes are L2 environment repair, L3 code edit, '
             'L4 steering and L5 human selection. L0 setup (before RUN_START) and L1 '
             'observation are logged but do not count against autonomy.</p>')
    if not inters:
        o.append('<div class="card good"><b>0 counted interventions.</b> '
                 'No human touched the run between RUN_START and FINAL_DESIGNATION.</div>')
    else:
        o.append('<table><thead><tr><th>Seq</th><th>Class</th><th>Counts</th>'
                 '<th>Note</th></tr></thead><tbody>')
        for e in inters:
            p = e['payload']
            o.append('<tr><td class="n">%d</td><td class="mono">%s</td><td>%s</td>'
                     '<td>%s</td></tr>' % (e['seq'], esc(p.get('class')),
                                           'yes' if p.get('counts') else 'no',
                                           esc(p.get('note', ''))))
        o.append('</tbody></table>')

    # ---- selection: why this checkpoint and not the argmax ----
    sel = f.get('selection') or {}
    if sel.get('evaluated'):
        o.append('<h2>Why this checkpoint</h2>')
        o.append('<p class="sub">Selection is not an argmax. Best-of-N on a metric whose '
                 'paired noise is sigma ~ 0.0005 picks the luckiest draw, and that '
                 'inflation is exactly what does not survive to the hidden test set. A '
                 'candidate must beat the incumbent on the pooled score <b>and</b> on '
                 '%s of %s independent user folds.</p>'
                 % (sel.get('min_fold_wins'), sel.get('folds')))
        o.append('<table><thead><tr><th class="n">Iter</th><th>Candidate</th>'
                 '<th class="n">Pooled &Delta;</th><th class="n">Folds won</th>'
                 '<th>Stable?</th></tr></thead><tbody>')
        for e in sel['evaluated']:
            d = e.get('pooled_delta', 0)
            o.append('<tr><td class="n">%s</td><td>%s</td><td class="n %s">%+.5f</td>'
                     '<td class="n">%s/%s</td><td><span class="pill %s">%s</span></td></tr>'
                     % (e.get('iteration'), esc(e.get('label', '')),
                        'pos' if d > 0 else ('neg' if d < 0 else 'muted'), d,
                        e.get('fold_wins'), sel.get('folds'),
                        'keep' if e.get('stable') else 'revert',
                        'stable' if e.get('stable') else 'not stable'))
        o.append('</tbody></table>')
        if sel.get('ensemble'):
            en = sel['ensemble']
            o.append('<p class="sub">Ensemble of iterations %s vs the best single: '
                     'pooled %+.5f, %s/%s folds — %s.</p>'
                     % (esc(', '.join(map(str, en.get('members', [])))),
                        en.get('pooled_delta', 0), en.get('fold_wins'), sel.get('folds'),
                        'adopted' if en.get('stable') else 'rejected'))
        for d in sel.get('decisions', []):
            o.append('<div class="card %s">%s</div>'
                     % ('bad' if 'floor' in d or 'no candidate' in d else 'good', esc(d)))
        if sel.get('vs_floor'):
            vf = sel['vs_floor']
            o.append('<p class="sub mono">vs banked floor: pooled %+.5f, %s/%s folds</p>'
                     % (vf.get('pooled_delta', 0), vf.get('fold_wins'), sel.get('folds')))

    # ---- candidates ----
    if f.get('candidates_considered'):
        o.append('<h2>All candidates</h2>')
        o.append('<p class="sub">Every scored node. The frozen list was written to the '
                 'journal before any test label was read.</p>')
        o.append('<table><thead><tr><th class="n">Iter</th><th>Candidate</th>'
                 '<th class="n">Validation primary</th><th></th></tr></thead><tbody>')
        best = f.get('chosen_iteration')
        for c in sorted(f['candidates_considered'], key=lambda x: -x['primary']):
            o.append('<tr><td class="n">%s</td><td>%s</td><td class="n">%s</td><td>%s</td></tr>'
                     % (c['iteration'], esc(c['label']), fmt(c['primary']),
                        '<span class="pill keep">designated</span>'
                        if c['iteration'] == best else ''))
        o.append('</tbody></table>')

    # ---- resources by role ----
    if res.get('by_role'):
        o.append('<h2>Where the tokens went</h2>')
        o.append('<table><thead><tr><th>Role</th><th class="n">Calls</th>'
                 '<th class="n">Input</th><th class="n">Output</th><th class="n">USD</th>'
                 '</tr></thead><tbody>')
        for role, r in sorted(res['by_role'].items()):
            o.append('<tr><td class="mono">%s</td><td class="n">%d</td><td class="n">%s</td>'
                     '<td class="n">%s</td><td class="n">$%.2f</td></tr>'
                     % (esc(role), r['calls'], '{:,}'.format(r['input']),
                        '{:,}'.format(r['output']), r['usd']))
        o.append('</tbody></table>')
        o.append('<p class="sub">Controller, governor, guards, executor, cache, scorer '
                 'and the submission builder make no model calls at all.</p>')

    o.append('</div></body></html>')
    return '\n'.join(o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('journal')
    ap.add_argument('-o', '--out', default='report.html')
    a = ap.parse_args()
    doc = build(a.journal)
    with open(a.out, 'w', encoding='utf-8') as fh:
        fh.write(doc)
    print('wrote %s (%d bytes) from %d events'
          % (a.out, len(doc), len(Journal.read(a.journal))))


if __name__ == '__main__':
    main()
