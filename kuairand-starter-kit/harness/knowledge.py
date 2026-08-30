"""Persistent memory of research directions: what has been tried, and what it cost.

Why not a vector database. The whole corpus here is a few dozen method cards and at most
fifty journal entries — well under 10k tokens, and it already rides in the cached system
prompt at ~$0.0005 a call. Embedding it and retrieving nearest neighbours would add a
dependency, a build step and per-call latency to solve a recall problem that does not
exist at this scale, and it would return *similar* directions when what the proposer
actually needs is the *exhaustive* list of what is already refuted. Similarity search is
the wrong primitive for "have I tried this?" — set membership is.

So this is a structured registry, not RAG. It answers three questions exactly:
  - which directions exist, and what is known about each
  - which are refuted, with the measured number that refuted them
  - which have never been tried

It persists across runs, so a second run does not re-derive the first run's dead ends.

Seeded only with evidence the agent is allowed to have: organizer-published results, and
measurements taken on train/validation. Nothing test-derived.
"""
import json
import os
import time

from . import CONTEXT

PATH = os.path.join(CONTEXT, 'directions.json')

UNTRIED, LIVE, REFUTED, NO_OP = 'untried', 'live', 'refuted', 'no_op'

SEED = [
    {'id': 'objective', 'name': 'Change the training objective',
     'summary': 'Pointwise BCE optimises a different thing from a within-user ranking '
                'metric. Pairwise (BPR) or listwise (softmax over the user\'s list) '
                'targets the metric directly.',
     'status': UNTRIED, 'source': 'organizer_ranked_1',
     'notes': ['The organizers rank this their most likely direction. Untested here.']},
    {'id': 'sequence', 'name': 'User behaviour sequences / target attention',
     'summary': 'No behaviour sequence is used at all. DIN/SIM-style interest modelling '
                'over the user history is unexplored.',
     'status': UNTRIED, 'source': 'organizer_ranked_2',
     'notes': ['Check history_coverage first: attention needs the candidate or its '
               'author to actually appear in the user history.']},
    {'id': 'multitask', 'name': 'Multi-task auxiliary heads',
     'summary': 'is_click, is_like, play_time_ms and others are available as auxiliary '
                'targets while long_view stays the scored task.',
     'status': UNTRIED, 'source': 'organizer_ranked_3',
     'notes': ['Reach them with harness.adapter.auxiliary_targets(). Never same-row '
               'inputs.']},
    {'id': 'watchtime', 'name': 'Censored watch-time regression',
     'summary': 'A completed play truncates true watch time, so a one-sided loss rather '
                'than squared error.',
     'status': UNTRIED, 'source': 'organizer_ranked_4',
     'notes': ['Needs play_time_ms and duration_ms via the adapter.']},
    {'id': 'model_family', 'name': 'Different model family',
     'summary': 'DeepFM / DCN / xDeepFM / GBDT over engineered features.',
     'status': UNTRIED, 'source': 'organizer_ranked_5',
     'notes': ['The organizers deprioritise this because capacity measurably is not the '
               'bottleneck — see the capacity dead end.']},
    {'id': 'temporal', 'name': 'Time features and drift',
     'summary': 'hourmin, date, and the train-to-validation regime change.',
     'status': UNTRIED, 'source': 'organizer_ranked_6',
     'notes': ['Daily volume collapses 13.6x INSIDE the training window (4/11 peak, '
               '4/19 trough), not at the train/validation boundary.']},
    {'id': 'debias', 'name': 'Exposure debiasing / unbiased validation',
     'summary': 'The random-exposure log as an unbiased check on a model fitted to '
                'biased traffic.',
     'status': UNTRIED, 'source': 'organizer_ranked_7',
     'notes': ['is_rand is 0 on every standard-log row, so it carries no propensity '
               'signal there. The random log has no rows before 20220422.']},
    {'id': 'ensembling', 'name': 'Variance reduction / ensembling',
     'summary': 'Combine several models\' within-user ranks to cancel seed variance.',
     'status': UNTRIED, 'source': 'inferable_from_seed_variance',
     'notes': ['Five identity seeds give a validation population std of 0.00032.']},

    # --- already settled. Re-deriving these wastes an iteration the run cannot spare. ---
    {'id': 'static_features', 'name': 'Add more static categorical fields',
     'summary': 'Extending the FM from 5 fields to the 13 CWM fields.',
     'status': REFUTED, 'source': 'organizer_published',
     'measured': [{'what': '13 fields vs 5 fields', 'primary': 0.5940,
                   'baseline': 0.5950, 'delta': -0.0010}],
     'notes': ['Organizer-published. Within noise and slightly worse. Reproducible via '
               'ablation_features.py — do not run it, it scores test.']},
    {'id': 'capacity', 'name': 'Increase embedding dimension',
     'summary': 'Larger k for the factorization machine.',
     'status': REFUTED, 'source': 'organizer_published',
     'measured': [{'what': 'k=8/16/32', 'primary': '0.5895 / 0.5902 / 0.5887'}],
     'notes': ['Organizer-published. Capacity is not the bottleneck; 1.14M rows do not '
               'support more of it.']},
    {'id': 'user_side_first_order', 'name': 'Pure user-side first-order features',
     'summary': 'A user-level feature added as an additive term.',
     'status': NO_OP, 'source': 'mathematical',
     'notes': ['Ranking is within-user, so any term constant across a user\'s '
               'impressions cannot change that user\'s order. Contributes exactly zero. '
               'User information can only act through a cross with the item side.']},
    {'id': 'score_transform', 'name': 'Per-user monotone score transforms',
     'summary': 'Calibration, sigmoid, per-user normalisation at inference.',
     'status': NO_OP, 'source': 'mathematical',
     'notes': ['Both GAUC and nDCG@5 depend only on within-user order, which a monotone '
               'transform preserves. Exactly zero effect.']},
]


def _now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def load(path=PATH):
    if os.path.exists(path):
        with open(path, encoding='utf-8') as fh:
            return json.load(fh)
    return {'directions': {d['id']: dict(d) for d in SEED}, 'updated': _now()}


def save(store, path=PATH):
    store['updated'] = _now()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(store, fh, indent=1, ensure_ascii=False)
    return path


def record(direction_id, iteration, delta, verdict, note, hypothesis='', path=PATH):
    """Write one experiment's outcome back into the registry."""
    store = load(path)
    d = store['directions'].setdefault(direction_id, {
        'id': direction_id, 'name': direction_id, 'summary': '(proposed by the agent)',
        'status': UNTRIED, 'source': 'agent', 'notes': []})
    d.setdefault('measured', []).append({
        'iteration': iteration, 'hypothesis': hypothesis[:180],
        'delta': None if delta is None else round(float(delta), 5),
        'verdict': verdict, 'note': note[:240]})
    tried = [m for m in d['measured'] if m.get('delta') is not None]
    if verdict == 'KEEP':
        d['status'] = LIVE
    elif len(tried) >= 2 and all((m['delta'] or 0) <= 0.0014 for m in tried):
        d['status'] = REFUTED          # two independent misses is enough to close it
    elif d['status'] == UNTRIED:
        d['status'] = LIVE
    save(store, path)
    return d


def summary(path=PATH, max_notes=2):
    """Compact text for the proposer. Refuted directions lead, so they are unmissable."""
    store = load(path)
    ds = list(store['directions'].values())
    order = {REFUTED: 0, NO_OP: 1, LIVE: 2, UNTRIED: 3}
    ds.sort(key=lambda d: (order.get(d.get('status'), 4), d['id']))
    lines = []
    for d in ds:
        head = '[%s] %s — %s' % (d.get('status', '?').upper(), d['id'], d['name'])
        lines.append(head)
        if d.get('summary'):
            lines.append('    %s' % d['summary'])
        for m in (d.get('measured') or [])[-2:]:
            if 'primary' in m:
                lines.append('    measured: %s -> %s' % (m['what'], m['primary']))
            else:
                lines.append('    iter %s: delta %s, %s'
                             % (m.get('iteration'), m.get('delta'), m.get('verdict')))
        for n in (d.get('notes') or [])[:max_notes]:
            lines.append('    note: %s' % n)
    return '\n'.join(lines)


def open_ids(path=PATH):
    store = load(path)
    return sorted(d['id'] for d in store['directions'].values()
                  if d.get('status') in (UNTRIED, LIVE))


def closed_ids(path=PATH):
    store = load(path)
    return sorted(d['id'] for d in store['directions'].values()
                  if d.get('status') in (REFUTED, NO_OP))


if __name__ == '__main__':
    p = save(load())
    print('wrote %s' % p)
    print()
    print(summary())
