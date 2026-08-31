"""Generate the research packet the proposer sees.

Two rules govern what goes in here, and they are the difference between an autonomous
agent and an executor:

  1. Observations, not conclusions. "T4 nDCG@5 is 0.5112" belongs in the packet.
     "therefore use sequence modelling" does not — that is the prioritisation the agent
     is scored on doing itself.
  2. Validation only. The test-split composition (27.1% all-negative, oracle 0.7289) is
     structure of the hidden split. The agent reasons with the validation equivalents
     (30.32%, oracle 0.6968), which are materially different numbers.

Run: python context/build_packet.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def build_data_profile():
    b = json.load(open(os.path.join(ROOT, 'analysis_output',
                                    'bucket_analysis_summary.json'), encoding='utf-8'))
    audit = b['fixed_score_audit']
    prof = {
        '_note': ('Validation-split observations. Test-split structure is deliberately '
                  'absent: it describes the hidden set and is not a development input.'),
        'split_sizes': {'train': 1141112, 'valid': 124909, 'test': 170588},
        'tier_definition': ('Validation users binned by their TRAIN impression count. '
                            'Cold = 0 train impressions; the remaining 21,955 warm users '
                            'are quartiled at edges 17 / 36 / 65. Two other tier schemes '
                            'exist in this repo under the same names T1-T4 — this is the '
                            'one these numbers come from.'),
        'impressions_per_user_valid': b['impressions_per_user_valid'],
        'metric_invariance_valid': {
            'note': ('A user whose validation labels are all 0 or all 1 has an nDCG@5 '
                     'pinned at 0 or 1 and is excluded from GAUC entirely. No model can '
                     'move them.'),
            'invariant_users_pct': round(audit['overall']['fixed_users_pct'], 4),
            'invariant_rows_pct': round(audit['overall']['fixed_rows_pct'], 4),
            'all_negative_users': audit['overall']['all_negative_users'],
            'all_positive_users': audit['overall']['all_positive_users'],
            'single_impression_users': audit['overall']['single_impression_users'],
            'by_tier': {k: {'n_users': v['n_users'], 'n_rows': v['n_rows'],
                            'invariant_users_pct': round(v['fixed_users_pct'], 4)}
                        for k, v in audit['by_tier'].items()},
        },
        'baseline_metrics_by_tier_valid': b['bucket_metrics'],
        'current_representation': {
            'fields': ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket'],
            'encoded_dim': 40260,
            'row_tuple_from_kit_data_load': ['date', 'user_id', 'video_id', 'author_id',
                                            'tab', 'duration_ms', 'label'],
            'note': ('kit/data.py exposes only these seven fields. The other 12 log '
                     'columns and all user/video side tables are reachable only through '
                     'harness.adapter, which date-filters to train+valid.'),
        },
        'available_but_unloaded': {
            'log_columns': ['hourmin', 'time_ms', 'is_click', 'is_like', 'is_follow',
                            'is_comment', 'is_forward', 'is_hate', 'play_time_ms',
                            'profile_stay_time', 'comment_stay_time', 'is_profile_enter',
                            'is_rand'],
            'side_tables': {'user_features_pure.csv': 30, 'video_features_basic_pure.csv': 12,
                            'video_features_statistic_pure.csv': 52},
            'access': 'harness.adapter.raw_columns() / entity_table()',
        },
    }
    return prof


def build_baseline():
    s = json.load(open(os.path.join(ROOT, 'kit', 'baseline_scores.json'), encoding='utf-8'))
    v = s['scores']
    return {
        '_note': ('Validation reference rungs. The published hidden-test numbers exist in '
                  'kit/baseline_scores.json and are the competition target, but they are '
                  'not a development signal and are not repeated here.'),
        'target_to_beat_hidden_test_primary': v['fm_official']['test']['primary'],
        'validation': {
            'random': v['random']['valid'],
            'item_popularity': v['item_popularity']['valid'],
            'fm_official': v['fm_official']['valid'],
            'oracle_ceiling': v['oracle_ceiling']['valid'],
        },
        'noise': {
            'published_seed_std_test': v['fm_official']['std_over_5_seeds']['test_primary'],
            'measured_seed_std_valid_population': 0.00032,
            'measured_paired_delta_sigma_valid': 0.0005,
            'note': ('The widely-quoted 0.0008 is the TEST seed std. On validation, five '
                     'identity seeds give a population std of 0.00032 and paired deltas '
                     'run about 0.0005. So eps = 0.002 is roughly 4 sigma: a null '
                     'iteration essentially never clears it.'),
        },
        'convergence': s['convergence_rule'],
        'run_limits': {'max_iterations': 50, 'wall_clock_hours': 6},
        'baseline_config': v['fm_official']['config'],
        'reproduced_here': {'valid_primary': 0.60147, 'best_epoch': 7, 'early_stop_epoch': 11},
    }


CONSTRAINTS = """# Constraints — measured facts only

Everything here was measured, by the organizers or by this team, on train or validation.
None of it tells you what to try next; that is your call.

## Structural, provable

C1. Ranking is WITHIN user. Any score term that is constant across one user's impressions
cannot change that user's ordering. A pure user-side first-order feature therefore
contributes exactly zero. User information can only act through a cross with the item
side. (Mathematical consequence of the metric, confirmed by the organizers: `item_pop x
user_bias` scores bit-identically to plain `item_pop`.)

C2. Any per-user monotone transform of the scores at inference is a no-op for both GAUC
and nDCG@5. So is any global calibration.

C3. 42.2% of validation users are metric-invariant: their labels are all-0 or all-1, so
nDCG@5 is pinned and GAUC excludes them. 17.5% have a single impression. No model change
reaches these users.

## Measured negative results — do not re-derive these

C4. Static feature stuffing does not help. The organizers extended the FM from 5 fields
to the 13 CWM fields: primary 0.5940 versus 0.5950 for 5 fields. Within noise, slightly
worse. Reproducible via `ablation_features.py` (do not run it — it scores test).

C5. Embedding capacity is not the bottleneck. k = 8 / 16 / 32 gives 0.5895 / 0.5902 /
0.5887.

C6. Removing fields measured slightly POSITIVE on validation: dropping `author_id` gives
+0.00157 and dropping `video_id` +0.00136, each 5/5 positive across paired seeds. The
organizers' stated reason for C4 and C5 is that the `user_id x video_id` cross already
absorbs most of the learnable signal.

C7. There are 6,510 authors for 7,583 videos and 87% of authors have exactly one video,
so `author_id` is close to a duplicate of `video_id`.

C8. Only 1.63% of validation rows have their (user, video) pair present in train, and
3.38% their (user, author) pair. 98.1% of validation users have some train history.

## Rules that are enforced in code, not by discipline

C9. Post-impression signals (is_click, is_like, is_follow, is_comment, is_forward,
is_hate, play_time_ms, profile_stay_time, comment_stay_time, is_profile_enter, long_view)
are never same-row inputs. They are legal as auxiliary targets and as history aggregated
from strictly earlier rows. `harness/guards.py` rejects violations statically, before the
code runs.

C10. The three editable files are `pipeline/features.py`, `pipeline/model.py`,
`pipeline/train.py`. `kit/` is read-only at the filesystem level.

C11. Raw log columns are reachable only through `harness.adapter`, which date-filters to
train+valid before returning anything and aligns positionally with `kit.data.load()`.
Joining on (user_id, video_id) is wrong: that pair is not unique — 3.06% of evaluation
rows repeat it, up to 12 times.
"""

REFERENCES = """# Method index

Short cards. What a method is, what it assumes, what it costs. Deliberately unranked and
without recommendations: choosing among these is the research decision you are here to
make, and a pre-ranked list would make you an executor.

**Pairwise ranking (BPR)** — optimise P(positive ranked above negative) within a user
instead of a pointwise probability. Assumes usable positive/negative pairs per user;
degenerate users contribute nothing. Cost: a sampler plus a change to the gradient; same
order of wall-clock as the baseline. Rendle et al., UAI 2009.

**Listwise softmax / ListNet** — a softmax over each user's impression list, cross-entropy
against the normalised label vector. Assumes lists are meaningful units. Note the
validation median list length is 4. Cao et al., ICML 2007.

**LambdaRank / LambdaMART** — weight pairwise gradients by the nDCG change from swapping
the pair. Directly targets a truncated metric. Needs grouped data and a working
delta-nDCG. Burges, 2010.

**Multi-task / ESMM-style** — auxiliary heads on other feedback signals sharing a
representation with the scored task. Assumes the auxiliary signal correlates with the
target and that shared capacity is not the binding constraint. Ma et al., SIGIR 2018.

**Censored watch-time regression (CWM)** — a completed play truncates the true watch time,
so a one-sided loss rather than squared error. Requires play_time_ms and duration_ms.
Zhao et al., KDD 2024.

**Target attention (DIN)** — score a candidate by attending over the user's history.
Assumes the candidate or its attributes recur in that history.

**Field-aware and deep factorisation (FFM, DeepFM, DCN)** — richer interaction structure
over the same sparse fields.

**Inverse propensity weighting** — reweight by exposure probability to debias a logged
policy. Requires propensities; note `is_rand` is 0 on every standard-log row, and the
random-exposure log has no rows before 20220422.

**Seed ensembling / rank averaging** — combine several models' within-user ranks. Reduces
variance rather than bias.
"""


def main():
    prof = build_data_profile()
    base = build_baseline()
    with open(os.path.join(HERE, 'data_profile.json'), 'w', encoding='utf-8') as fh:
        json.dump(prof, fh, indent=1, ensure_ascii=False)
    with open(os.path.join(HERE, 'baseline.json'), 'w', encoding='utf-8') as fh:
        json.dump(base, fh, indent=1, ensure_ascii=False)
    for name, body in (('constraints.md', CONSTRAINTS), ('references.md', REFERENCES)):
        with open(os.path.join(HERE, name), 'w', encoding='utf-8') as fh:
            fh.write(body)

    spec = open(os.path.join(HERE, 'problem_spec.md'), encoding='utf-8').read()
    rules = open(os.path.join(ROOT, 'AGENT_RULES.md'), encoding='utf-8').read()
    packet = '\n\n'.join([
        spec,
        '# Agent rules\n\n' + rules,
        CONSTRAINTS,
        REFERENCES,
        '# Data profile (validation)\n\n```json\n' + json.dumps(prof, indent=1, ensure_ascii=False) + '\n```',
        '# Baseline and noise\n\n```json\n' + json.dumps(base, indent=1, ensure_ascii=False) + '\n```',
    ])
    with open(os.path.join(HERE, 'packet.md'), 'w', encoding='utf-8') as fh:
        fh.write(packet)
    print('packet.md: %d chars (~%d tokens, cached as one system block)'
          % (len(packet), len(packet) // 4))
    for n in ('data_profile.json', 'baseline.json', 'constraints.md', 'references.md'):
        print('  %s: %d bytes' % (n, os.path.getsize(os.path.join(HERE, n))))


if __name__ == '__main__':
    main()
