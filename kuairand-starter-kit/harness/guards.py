"""Static guards run BEFORE any generated code executes.

`pipeline/features.py` ships a runtime guard, `same_row()`, which raises on the eleven
post-impression columns. It is good but narrow: it only fires on the string literal
passed to it, so direct tuple indexing (`x[6]`), a renamed derived column
("engagement_score"), or a history aggregate that folds in row t all walk straight past
it. And it lives in the file the agent is allowed to rewrite.

So these checks live on the harness side of the boundary and run on the diff before the
executor is allowed to start. Every rejection returns a reason, and the reason goes back
to the coder — a guard that rejects without explaining trains the agent to guess.
"""
import os
import re

from . import PIPELINE, ROOT

EDITABLE = {'pipeline/features.py', 'pipeline/model.py', 'pipeline/train.py'}

# (regex, short reason, what to do instead)
FORBIDDEN = [
    (r'^\s*(?:from\s+baseline\s+import|import\s+baseline)\b',
     'imports kit/baseline.py',
     'baseline.py scores the test split unconditionally — it has no report_test flag. '
     'Use pipeline/model.py and pipeline/train.py instead.'),
    (r'ablation_features',
     'touches ablation_features.py',
     'that script evaluates test labels. It is a historical record, not a tool.'),
    (r'enc\[[\'"]test[\'"]\]\s*\[\s*1\s*\]',
     'reads test labels out of the encoded arrays',
     'enc["test"][1] is y for the hidden split. Predicting on test features is fine; '
     'reading its labels is not.'),
    (r'splits\[[\'"]test[\'"]\]',
     'reads raw test rows',
     'raw rows carry long_view. Go through harness.adapter, which date-filters to '
     'train+valid before returning anything.'),
    (r'evaluate\s*\([^)]*\btest\b',
     'calls the official evaluator on test',
     'scoring test is the harness gate\'s job, once, after FINAL_DESIGNATION.'),
    (r'open\s*\(\s*[^)]*log_(?:standard|random)[^)]*\)',
     'opens a raw log CSV directly',
     'log_standard_4_22_to_5_08_pure.csv spans validation AND test and carries '
     'long_view. Use harness.adapter.raw_columns(), which cannot return test rows.'),
    (r'os\.remove|shutil\.rmtree|os\.unlink',
     'deletes files',
     'the pipeline has no reason to delete anything.'),
]

LEAKY_SAME_ROW = ('is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward',
                  'is_hate', 'play_time_ms', 'profile_stay_time', 'comment_stay_time',
                  'is_profile_enter', 'long_view', 'label')

# The one legitimate read of the label inside features.py: building the target vector.
_LABEL_OK = re.compile(r'^\s*y\s*\[\s*\w+\s*\]\s*=\s*\w+\s*\[\s*IDX\[[\'"]label[\'"]\]\s*\]')


def _strip_comments(src):
    out = []
    for ln in src.split('\n'):
        s = ln.split('#', 1)[0] if '#' in ln else ln
        out.append(s)
    return '\n'.join(out)


def scan_source(path, text=None):
    """Scan one pipeline file. Returns a list of {line, rule, reason, fix, snippet}."""
    rel = os.path.relpath(os.path.abspath(path), ROOT).replace('\\', '/')
    src = text if text is not None else open(path, encoding='utf-8').read()
    body = _strip_comments(src)
    findings = []
    for i, line in enumerate(body.split('\n'), 1):
        if not line.strip():
            continue
        for pat, reason, fix in FORBIDDEN:
            if re.search(pat, line):
                findings.append({'file': rel, 'line': i, 'rule': pat,
                                 'reason': reason, 'fix': fix,
                                 'snippet': line.strip()[:160]})
        # Same-row use of a post-impression column, by any spelling. finditer, not
        # search: one line can hold several same_row() calls and only a later one may
        # be leaky, e.g. [same_row(x,'user_id'), same_row(x,'is_click')].
        for m in re.finditer(r'same_row\s*\(\s*\w+\s*,\s*[\'"](\w+)[\'"]\s*\)', line):
            if m.group(1) in LEAKY_SAME_ROW:
                findings.append({'file': rel, 'line': i, 'rule': 'same_row leaky column',
                                 'reason': 'uses %r as a same-row input feature' % m.group(1),
                                 'fix': 'post-impression signals are legal as auxiliary '
                                        'targets or as history from strictly earlier rows, '
                                        'never as an input for the row being predicted.',
                                 'snippet': line.strip()[:160]})
        # Direct index around the label, inside the FEATURE builder only.
        # train.py legitimately reads labels as training targets and as the ground truth
        # handed to evaluate(); the hazard is the label reaching raw()/X in features.py,
        # where same_row() would have refused it by name.
        if rel.endswith('features.py') or rel.endswith('<diff>'):
            if re.search(r'IDX\[[\'"]label[\'"]\]', line) and not _LABEL_OK.match(line):
                findings.append({'file': rel, 'line': i, 'rule': 'direct label index',
                                 'reason': 'reads IDX["label"] in the feature builder, '
                                           'outside the target assignment',
                                 'fix': 'same_row() refuses the label by name; reaching '
                                        'it through IDX bypasses that guard. y is built '
                                        'once, in encode(); features never see it.',
                                 'snippet': line.strip()[:160]})
    return findings


def scan_pipeline():
    """Scan all three editable files as they currently stand on disk."""
    findings = []
    for name in ('features.py', 'model.py', 'train.py'):
        findings += scan_source(os.path.join(PIPELINE, name))
    return findings


def scan_diff(diff_text):
    """Check a unified diff: which files it touches, and its added lines.

    Returns (ok, findings). A diff that touches anything outside the three editable
    files is rejected outright — that boundary is the whole anti-gaming design.
    """
    findings = []
    touched = set()
    for m in re.finditer(r'^\+\+\+ [ab]/(.+)$', diff_text, re.M):
        touched.add(m.group(1).strip().replace('\\', '/'))
    for m in re.finditer(r'^--- [ab]/(.+)$', diff_text, re.M):
        p = m.group(1).strip().replace('\\', '/')
        if p != '/dev/null':
            touched.add(p)
    for t in sorted(touched):
        if t not in EDITABLE:
            findings.append({'file': t, 'line': 0, 'rule': 'edit surface',
                             'reason': 'diff touches %r, which is not editable' % t,
                             'fix': 'only %s may change. If a hypothesis genuinely needs '
                                    'a new module, emit a capability request instead of '
                                    'widening the diff.' % ', '.join(sorted(EDITABLE)),
                             'snippet': ''})
    added = '\n'.join(ln[1:] for ln in diff_text.split('\n')
                      if ln.startswith('+') and not ln.startswith('+++'))
    if added.strip():
        findings += scan_source('pipeline/<diff>', text=added)
    return (not findings), findings


def check_rowid(path, split, data_dir=None):
    """The row_id contract, via the frozen checker. Returns (ok, message)."""
    from .submission import check
    from . import DATA_DIR
    ok, msg, _ = check(path, split, data_dir or DATA_DIR)
    return ok, msg


def format_findings(findings):
    if not findings:
        return 'clean'
    out = []
    for f in findings:
        loc = '%s:%d' % (f['file'], f['line']) if f['line'] else f['file']
        out.append('%s  %s\n    %s\n    -> %s' % (loc, f['reason'], f['snippet'], f['fix']))
    return '\n'.join(out)


if __name__ == '__main__':
    fs = scan_pipeline()
    print(format_findings(fs))
    raise SystemExit(1 if fs else 0)
