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

THREE LEVELS, NOT ONE. A direction id names an INTENT; it does not name the code that
was run. Nine implementations of "within-user listwise softmax" scored between -0.00318
and +0.00162 (constraints.md C25), so a single flat status over that id is a lie in
whichever direction you round it. The registry therefore separates:

    model_family              FM
    objective_family          listwise
    objective_implementation  ListNet-style softmax / group by user /
                              normalise by total positives / full impression list

`directions` carries the coarse level, and is what gates the proposer. `implementations`
carries the fine level, keyed by the SHA-1 of the diff that produced it -- byte-identical
code gets one id, and differing code cannot collide however it is described in prose.
When implementations under one direction disagree in sign, `ablation_candidates()` says
so, and the answer is a controlled objective ablation
(`research/objective_ablation/ablate.py`), not another guess.

It persists across runs, so a second run does not re-derive the first run's dead ends.

Seeded only with evidence the agent is allowed to have: organizer-published results, and
measurements taken on train/validation. Nothing test-derived.
"""
import hashlib
import json
import os
import time

from . import CONTEXT

PATH = os.path.join(CONTEXT, 'directions.json')

UNTRIED, LIVE, REFUTED, NO_OP = 'untried', 'live', 'refuted', 'no_op'

ACCEPT = 0.0014   # mirrors controller.ACCEPT: 2 sigma on the measured paired noise floor
# Mirrors controller.UNANIMOUS_ACCEPT. ACCEPT is 2 sigma on ONE measurement; the
# mean of s matched seeds has standard error sigma/sqrt(s), so holding a unanimous
# 3-seed mean to ACCEPT is ~3.5 sigma. The registry has to use the same bar the
# controller keeps a node on, or the two disagree about what a confirmed win is.
UNANIMOUS_ACCEPT = 0.0008

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
            store = json.load(fh)
    else:
        store = {'directions': {d['id']: dict(d) for d in SEED}, 'updated': _now()}
    store.setdefault('implementations', {})       # added after the first runs; may be absent
    return store


def save(store, path=PATH):
    store['updated'] = _now()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(store, fh, indent=1, ensure_ascii=False)
    return path


def impl_id(diff):
    """Identity of one implementation: the SHA-1 of the diff that produced it.

    Prose cannot be trusted to distinguish implementations -- eight runs all called
    their code "within-user listwise softmax". The diff can: identical code collides by
    construction, different code cannot.
    """
    return hashlib.sha1((diff or '').encode('utf-8')).hexdigest()[:12]


def record(direction_id, iteration, delta, verdict, note, hypothesis='',
           confirm=None, diff='', traits=None, model_family='FM',
           objective_family=None, path=PATH):
    """Write one experiment's outcome back into the registry, at both levels.

    `confirm` is the controller's matched-seed confirmation block, if the iteration
    earned one. It is the only evidence strong enough to make a direction permanently
    safe from the auto-close rule below.

    `diff`, `traits`, `model_family` and `objective_family` record WHICH implementation
    this was, so that a later disagreement between two implementations of the same
    direction is visible as a disagreement rather than averaged into one status.
    """
    store = load(path)
    d = store['directions'].setdefault(direction_id, {
        'id': direction_id, 'name': direction_id, 'summary': '(proposed by the agent)',
        'status': UNTRIED, 'source': 'agent', 'notes': []})
    entry = {
        'iteration': iteration, 'hypothesis': hypothesis[:180],
        'delta': None if delta is None else round(float(delta), 5),
        'verdict': verdict, 'note': note[:240]}
    # Unanimity is what buys the lower bar: a positive worst seed means every matched
    # seed agreed, which is the condition controller.UNANIMOUS_ACCEPT was calibrated for.
    if (confirm and confirm.get('worst_delta', 0) > 0
            and confirm.get('mean_delta', 0) > UNANIMOUS_ACCEPT):
        entry['confirmed_positive'] = True
    d.setdefault('measured', []).append(entry)
    tried = [m for m in d['measured'] if m.get('delta') is not None]
    if verdict == 'KEEP':
        d['status'] = LIVE
    elif any(m.get('confirmed_positive') for m in d['measured']):
        # A direction that has produced a confirmed matched-seed gain is never closed by
        # later misses, and a confirmed gain REOPENS one that earlier misses had closed.
        # A direction id names an INTENT, not a formulation: nine independently written
        # listwise-softmax implementations spanned -0.00318 to +0.00162. Replaying the
        # real chronology, the two opening misses closed the direction on run
        # 20260830T164210Z -- three runs BEFORE the verified +0.00162 that became the
        # banked submission arrived. Without the reopen, that win lands on a direction
        # the proposer has already been told is dead. See constraints.md C25.
        if d['status'] in (UNTRIED, REFUTED):
            d['status'] = LIVE
    elif len(tried) >= 2 and all((m['delta'] or 0) <= ACCEPT for m in tried):
        d['status'] = REFUTED          # two independent misses is enough to close it
    elif d['status'] == UNTRIED:
        d['status'] = LIVE
    if diff:
        iid = impl_id(diff)
        im = store['implementations'].setdefault(iid, {
            'id': iid, 'direction_id': direction_id, 'model_family': model_family,
            'objective_family': objective_family or direction_id,
            'traits': list(traits or []), 'measured': []})
        im['measured'].append(dict(entry))
        # Traits are declarative and may arrive late or partially; union rather than
        # overwrite, so a second sighting of the same code cannot lose what we knew.
        for t in (traits or []):
            if t not in im['traits']:
                im['traits'].append(t)

    save(store, path)
    return d


def implementations_of(direction_id, path=PATH):
    """Every distinct implementation recorded under one direction."""
    store = load(path)
    return [im for im in store['implementations'].values()
            if im.get('direction_id') == direction_id]


def ablation_candidates(path=PATH, floor=ACCEPT):
    """Directions whose implementations disagree in sign, worst first.

    This is the trigger for a controlled objective ablation. Running the full
    objective x implementation matrix for every idea would cost most of the research
    budget (8 implementations x 3 objectives x 3 seeds = 72 trainings), so it is worth
    paying only where the evidence is actually contradictory -- which is exactly where
    another guess is least informative.
    """
    out = []
    store = load(path)
    for did in {im.get('direction_id') for im in store['implementations'].values()}:
        # An ablation already on record answers the question. Without this the proposer
        # is told to run an experiment that has been run, every iteration, forever.
        if (store['directions'].get(did) or {}).get('ablation'):
            continue
        ims = [im for im in store['implementations'].values()
               if im.get('direction_id') == did]
        best = {}
        for im in ims:
            ds = [m['delta'] for m in im['measured'] if m.get('delta') is not None]
            if ds:
                best[im['id']] = max(ds)
        if len(best) < 2:
            continue
        hi, lo = max(best.values()), min(best.values())
        if hi > floor and lo <= 0:
            out.append({
                'direction_id': did, 'n_implementations': len(best),
                'best': hi, 'worst': lo, 'spread': hi - lo,
                'best_impl': max(best, key=best.get),
                'worst_impl': min(best, key=best.get),
                'reason': ('%d implementations of %r disagree in sign: best %+.5f, '
                           'worst %+.5f, spread %.5f. The label is not identifying the '
                           'experiment. Run the controlled objective ablation before '
                           'proposing another one.'
                           % (len(best), did, hi, lo, hi - lo))})
    return sorted(out, key=lambda r: -r['spread'])


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

    cands = ablation_candidates(path)
    if cands:
        lines.append('')
        lines.append('CONTRADICTORY DIRECTIONS -- the label is not identifying the experiment:')
        for c in cands:
            lines.append('  ! %s' % c['reason'])

    settled = [d for d in store['directions'].values() if d.get('ablation')]
    if settled:
        lines.append('')
        lines.append('CONTROLLED ABLATIONS ALREADY RUN -- do not re-derive these:')
        for d in settled:
            a = d['ablation']
            lines.append('  = %s: %s' % (d['id'], a.get('conclusion', '')))
            for arm, val in (a.get('arms') or {}).items():
                lines.append('      %-12s %s' % (arm, val))

    winners = [im for im in store.get('implementations', {}).values()
               if any(m.get('confirmed_positive') for m in im.get('measured', []))
               and im.get('traits')]
    if winners:
        lines.append('')
        lines.append('WHAT THE CONFIRMED IMPLEMENTATIONS ACTUALLY DID:')
        for im in winners:
            lines.append('  [%s] %s / %s' % (im['id'], im.get('model_family'),
                                             im.get('objective_family')))
            for t in im['traits']:
                lines.append('      - %s' % t)
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
