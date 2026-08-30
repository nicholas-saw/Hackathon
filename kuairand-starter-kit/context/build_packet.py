"""Generate the research packet the proposer sees.

Two rules govern what goes in here, and they are the difference between an autonomous
agent and an executor:

  1. Observations, not conclusions. "T4 nDCG@5 is 0.5112" belongs in the packet.
     "therefore use sequence modelling" does not — that is the prioritisation the agent
     is scored on doing itself.
  2. Validation only. The test-split composition (27.1% all-negative, oracle 0.7289) is
     structure of the hidden split. The agent reasons with the validation equivalents
     (30.32%, oracle 0.6968), which are materially different numbers.

  3. Single source of truth. `constraints.md` and `references.md` are no longer inlined
     here — they are the human-reviewed files on disk, authored from the consolidated
     pre-audit and its review. This script reads them; it must never rewrite them, or a
     rebuild would silently discard the reviewed evidence. See
     `context/CONTEXT_UPDATE_REPORT.md` for what the review changed and why.

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
            'reference_seed_std': v['fm_official']['std_over_5_seeds']['test_primary'],
            'measured_seed_std_valid_population': 0.00032,
            'note': ("Use the published 0.0008 as the noise reference: it is the "
                     "organizer-reported TEST seed std. The locally measured validation "
                     "population std of 0.00032 is narrower, and the pre-audit review "
                     "ruled that treating the narrower figure as the reference is "
                     "over-confident. Against 0.0008, eps = 0.002 is about 2.5 sigma, "
                     "which is the derivation the organizers give. Read a delta under "
                     "roughly 0.0008 as indistinguishable from a seed, and one under "
                     "0.002 as below the practical threshold set by the competition."),
        },
        'convergence': s['convergence_rule'],
        'run_limits': {'max_iterations': 50, 'wall_clock_hours': 6},
        'baseline_config': v['fm_official']['config'],
        'reproduced_here': {'valid_primary': 0.60147, 'best_epoch': 7, 'early_stop_epoch': 11},
    }


def _read(name, base=None):
    """Read a human-reviewed context file. These are inputs to the packet, not outputs."""
    with open(os.path.join(base or HERE, name), encoding='utf-8') as fh:
        return fh.read()


PROFILE_JSON_HEADER = """# Repo representation and column access

Only the fields that are unique to this repo. The tier, invariance, split-size and
baseline-by-tier figures that `data_profile.json` also carries are omitted here because
the reviewed profile above already states them, cell for cell, and duplicating ~9,000
characters of agreeing numbers costs tokens and invites the model to treat two copies as
two sources. The full generated file remains at `context/data_profile.json`.

```json
"""

# The keys worth sending: everything else in `prof` duplicates the reviewed profile.
PACKET_PROFILE_KEYS = ('current_representation', 'available_but_unloaded')

FENCE_END = """
```"""


def _slim_constraints(text):
    """Drop what only a human maintainer needs, keep every fact.

    Two things go: the per-entry Provenance blocks (PRE_AUDIT / REVIEW_REPORT citations
    -- an audit trail the agent cannot act on), and the trailing "What Must NOT Appear
    In This File" section, which instructs whoever edits the file, not whoever reads it.
    The file on disk keeps both and stays frozen; only this packet copy is slimmed.
    """
    out, skip = [], False
    for line in text.split(chr(10)):
        if line.startswith('**Provenance:**'):
            skip = True
            continue
        if skip:
            if line.startswith('---') or line.startswith('#'):
                skip = False
            else:
                continue
        if line.startswith('## 9. What Must NOT Appear'):
            break
        out.append(line)
    body = chr(10).join(out).rstrip() + chr(10)
    return body + (
        chr(10) + '> Provenance citations (PRE_AUDIT / REVIEW_REPORT per entry) and the '
        'file-maintenance' + chr(10) + '> section are omitted from this packet copy. '
        'They are intact in `context/constraints.md`.' + chr(10))


def main():
    prof = build_data_profile()
    base = build_baseline()
    constraints = _slim_constraints(_read('constraints.md'))
    references = _read('references.md')
    # The reviewed measurement profile. The proposer has no file-read tool, so anything
    # not in the packet is invisible to it -- including measurements deliberately demoted
    # out of constraints.md to stop them reading as directives.
    profile_md = _read('data_profile.md', os.path.join(ROOT, 'research'))
    with open(os.path.join(HERE, 'data_profile.json'), 'w', encoding='utf-8') as fh:
        json.dump(prof, fh, indent=1, ensure_ascii=False)
    with open(os.path.join(HERE, 'baseline.json'), 'w', encoding='utf-8') as fh:
        json.dump(base, fh, indent=1, ensure_ascii=False)
    spec = open(os.path.join(HERE, 'problem_spec.md'), encoding='utf-8').read()
    rules = open(os.path.join(ROOT, 'AGENT_RULES.md'), encoding='utf-8').read()
    packet = '\n\n'.join([
        spec,
        '# Agent rules\n\n' + rules,
        constraints,
        references,
        profile_md,
        PROFILE_JSON_HEADER + json.dumps(
            {k: prof[k] for k in PACKET_PROFILE_KEYS if k in prof},
            indent=1, ensure_ascii=False) + FENCE_END,
        '# Baseline and noise\n\n```json\n' + json.dumps(base, indent=1, ensure_ascii=False) + '\n```',
    ])
    with open(os.path.join(HERE, 'packet.md'), 'w', encoding='utf-8') as fh:
        fh.write(packet)
    # chars//4 understated this content by ~60%; 2.5 is calibrated against the API's
    # own count for this packet. Still an estimate -- count_tokens is authoritative.
    print('packet.md: %d chars (~%d tokens est., cached as one system block)'
          % (len(packet), int(len(packet) / 2.5)))
    for n in ('data_profile.json', 'baseline.json'):
        print('  wrote  %s: %d bytes' % (n, os.path.getsize(os.path.join(HERE, n))))
    print('  read   ../research/data_profile.md: %d bytes (human-reviewed)'
          % os.path.getsize(os.path.join(ROOT, 'research', 'data_profile.md')))
    for n in ('constraints.md', 'references.md'):
        print('  read   %s: %d bytes (human-reviewed, not regenerated)'
              % (n, os.path.getsize(os.path.join(HERE, n))))


if __name__ == '__main__':
    main()
