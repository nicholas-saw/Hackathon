"""Coder — turns one hypothesis into one code change.

Deliberately starved of context. It sees the chosen hypothesis, the current contents of
the three editable files, and the rules. It does not see the research packet, the journal,
or the other candidates, because the judgement already happened upstream and re-sending
it per node is most of the token bill.

It returns whole files rather than a unified diff. Applying a model-written patch is a
second failure mode on top of the code itself; whole files always apply, and the harness
computes the real diff with difflib for the journal.
"""
import difflib
import os

INSTRUCTIONS = """You are the coder in an autonomous ML research loop on KuaiRand-Pure.

You are given one hypothesis and the current contents of the three editable files. Make
the smallest change that tests that hypothesis. Return ONE JSON object, no prose:

{
  "files": {"pipeline/model.py": "<COMPLETE new file contents>", ...},
  "note": "what you changed and why, two sentences"
}

Include a file only if you changed it. Return its COMPLETE contents, not a fragment and
not a diff.

Hard constraints — a violation is rejected before your code runs, and you lose the node:
- Only pipeline/features.py, pipeline/model.py, pipeline/train.py may appear in "files".
- Never import kit/baseline.py or reference ablation_features.py: both score test.
- Never read splits['test'], enc['test'][1], or call evaluate() on test.
- Never open a raw CSV. log_standard_4_22_to_5_08_pure.csv spans validation AND test.
  Use harness.adapter.raw_columns(), which cannot return test rows.
- Post-impression columns are never same-row inputs.

Contracts the harness depends on — preserve them or the node cannot be scored:
  features.py: encode(splits) -> (enc, dim);  enc[split] = (X, y, users)
  model.py:    FM(dim, k, lr, l2, seed) with .step(X, y) -> loss, .predict(X) -> ndarray,
               and readable/writable .V, .W, .b
  train.py:    fit_predict(enc, dim, model=..., seed=..., **cfg)
                 -> {'train': ndarray, 'valid': ndarray, 'test': ndarray}
               one score per row, aligned to enc[split] row order.

If the hypothesis needs a batch shape the current step() cannot express (a pairwise or
listwise loss, say), change model.py and train.py together — both are yours."""

EDITABLE = ('pipeline/features.py', 'pipeline/model.py', 'pipeline/train.py')


def current_files(root):
    """Read with newline='' so line endings survive the round trip.

    Universal-newline mode would normalise CRLF to LF, and then a restore would rewrite
    every line of a file it was supposed to leave alone — changing the workspace hash and
    making the "no unattributed writes" check noisy for no reason.
    """
    out = {}
    for rel in EDITABLE:
        with open(os.path.join(root, rel), encoding='utf-8', newline='') as fh:
            out[rel] = fh.read()
    return out


def build_user_message(hypothesis, files, last_error=None):
    parts = ['HYPOTHESIS', '']
    for k in ('hypothesis', 'mechanism', 'proposed_change', 'expected_result', 'invalid_if'):
        if hypothesis.get(k):
            parts.append('%s: %s' % (k, hypothesis[k]))
    parts += ['', 'FILES TO MODIFY: %s' % ', '.join(hypothesis.get('files_to_modify') or []), '']
    for rel, src in files.items():
        parts += ['--- %s ---' % rel, src, '']
    if last_error:
        parts += ['YOUR PREVIOUS ATTEMPT FAILED:', last_error,
                  'Fix the cause. Do not restate the hypothesis.', '']
    return '\n'.join(parts)


def validate(obj):
    if not isinstance(obj, dict) or not isinstance(obj.get('files'), dict):
        return False, 'reply must be {"files": {path: contents}, "note": str}'
    if not obj['files']:
        return False, '"files" is empty — no change was proposed'
    for path, body in obj['files'].items():
        p = path.replace('\\', '/')
        if p not in EDITABLE:
            return False, '%r is not editable; only %s' % (path, ', '.join(EDITABLE))
        if not isinstance(body, str) or len(body) < 200:
            return False, '%r looks truncated (%d chars); return the COMPLETE file' % (
                path, len(body) if isinstance(body, str) else -1)
    return True, 'ok'


def unified_diff(before, after, path):
    return ''.join(difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True),
        fromfile='a/' + path, tofile='b/' + path))


def make_diff(before_files, after_files):
    """Real unified diff for the journal and for the static guard."""
    chunks = []
    for path, after in after_files.items():
        p = path.replace('\\', '/')
        chunks.append(unified_diff(before_files.get(p, ''), after, p))
    return ''.join(chunks)


def write_change(root, after_files):
    """Apply whole-file replacements. Returns the previous contents for rollback."""
    prev = {}
    for path, body in after_files.items():
        p = os.path.join(root, path.replace('/', os.sep))
        with open(p, encoding='utf-8') as fh:
            prev[path] = fh.read()
        with open(p, 'w', encoding='utf-8', newline='\n') as fh:
            fh.write(body)
    return prev


def restore(root, prev_files):
    for path, body in prev_files.items():
        p = os.path.join(root, path.replace('/', os.sep))
        with open(p, 'w', encoding='utf-8', newline='\n') as fh:
            fh.write(body)


def code(llm, hypothesis, root, last_error=None):
    """Returns (after_files, note, usage_record)."""
    files = current_files(root)
    msg = build_user_message(hypothesis, files, last_error)
    obj, rec = llm.ask_json('coder', INSTRUCTIONS, msg, effort='medium')
    ok, why = validate(obj)
    if not ok:
        obj, rec = llm.ask_json(
            'coder', INSTRUCTIONS,
            msg + '\n\nYour previous reply was rejected: %s' % why, effort='medium')
        ok, why = validate(obj)
        if not ok:
            raise ValueError('coder returned an invalid change twice: %s' % why)
    return {k.replace('\\', '/'): v for k, v in obj['files'].items()}, obj.get('note', ''), rec
