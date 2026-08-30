"""Hash-chained, append-only run journal.

The journal is a scored deliverable, not debug exhaust: judges read it for Innovation
(what the agent tried and why) and for Autonomy (how many times a human touched it).
It is hash-chained so that a later edit is detectable, and so that `verify.py` can
prove the submission was designated before any test label was ever read.

    hash_n = sha256(hash_{n-1} || seq || type || canonical_json(payload))

Canonical JSON = sorted keys, no whitespace, ensure_ascii=False. Any re-serialisation
must reproduce the same bytes or the chain check fails, which is the point.
"""
import hashlib
import json
import os
import time

GENESIS = '0' * 64

# Event vocabulary. Anything outside this set is still accepted, but these are the ones
# verify.py and report.py understand.
RUN_START = 'RUN_START'
ITERATION = 'iteration'
ANALYSIS = 'analysis'
GUARD_REJECT = 'guard_reject'
ERROR_RECOVERY = 'error_recovery'
HUMAN_INTERVENTION = 'human_intervention'
BUDGET_HALT = 'budget_halt'
CONVERGED = 'converged'
FINAL_DESIGNATION = 'FINAL_DESIGNATION'
TEST_OPEN = 'TEST_OPEN'

# Manual-intervention taxonomy. L0/L1 are recorded but do not count against autonomy.
INTERVENTION_CLASSES = {
    'L0_setup': False,            # before RUN_START, bounded by the workspace hash
    'L1_observe': False,          # reading logs; no effect on the run
    'L2_env_repair': True,        # restart with no code or state change
    'L3_code_edit': True,         # edit inside the hashed tree while the run is live
    'L4_steer': True,             # injecting an idea, changing config, killing a branch
    'L5_select': True,            # a human chose the final submission
}


def canonical(payload):
    return json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def event_hash(prev_hash, seq, etype, payload):
    h = hashlib.sha256()
    h.update(prev_hash.encode())
    h.update(str(seq).encode())
    h.update(etype.encode())
    h.update(canonical(payload).encode('utf-8'))
    return h.hexdigest()


class Journal:
    """Append-only JSONL with a hash chain. One instance per run."""

    def __init__(self, run_dir):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.path = os.path.join(run_dir, 'journal.jsonl')
        self._seq = 0
        self._prev = GENESIS
        if os.path.exists(self.path):
            for ev in self.read(self.path):
                self._seq = ev['seq'] + 1
                self._prev = ev['hash']

    def append(self, etype, payload=None):
        payload = payload or {}
        ev = {'seq': self._seq,
              'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'type': etype,
              'prev_hash': self._prev,
              'payload': payload}
        ev['hash'] = event_hash(self._prev, ev['seq'], etype, payload)
        with open(self.path, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(ev, ensure_ascii=False) + '\n')
        self._seq += 1
        self._prev = ev['hash']
        return ev

    def intervention(self, cls, note):
        if cls not in INTERVENTION_CLASSES:
            raise ValueError('unknown intervention class %r; expected one of %s'
                             % (cls, sorted(INTERVENTION_CLASSES)))
        return self.append(HUMAN_INTERVENTION,
                           {'class': cls, 'counts': INTERVENTION_CLASSES[cls], 'note': note})

    @staticmethod
    def read(path):
        if not os.path.exists(path):
            return []
        out = []
        with open(path, encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    def events(self, etype=None):
        evs = self.read(self.path)
        return [e for e in evs if etype is None or e['type'] == etype]

    def has(self, etype):
        return any(e['type'] == etype for e in self.read(self.path))

    def seq_of(self, etype):
        """Sequence number of the first event of this type, or None."""
        for e in self.read(self.path):
            if e['type'] == etype:
                return e['seq']
        return None


def verify_chain(path):
    """Recompute the chain. Returns (ok, message)."""
    evs = Journal.read(path)
    if not evs:
        return False, 'journal is empty'
    prev = GENESIS
    for i, ev in enumerate(evs):
        if ev['seq'] != i:
            return False, 'seq gap at index %d: found %r' % (i, ev['seq'])
        if ev['prev_hash'] != prev:
            return False, 'chain break at seq %d: prev_hash mismatch' % ev['seq']
        want = event_hash(prev, ev['seq'], ev['type'], ev['payload'])
        if ev['hash'] != want:
            return False, 'hash mismatch at seq %d (payload edited after the fact)' % ev['seq']
        prev = ev['hash']
    return True, '%d events, chain intact' % len(evs)


def verify_order(path):
    """FINAL_DESIGNATION must precede TEST_OPEN. Returns (ok, message)."""
    evs = Journal.read(path)
    fd = next((e['seq'] for e in evs if e['type'] == FINAL_DESIGNATION), None)
    to = next((e['seq'] for e in evs if e['type'] == TEST_OPEN), None)
    if fd is None:
        return False, 'no FINAL_DESIGNATION event: the run never named a submission'
    if to is None:
        return True, 'FINAL_DESIGNATION at seq %d; test never opened' % fd
    if fd < to:
        return True, 'FINAL_DESIGNATION seq %d precedes TEST_OPEN seq %d' % (fd, to)
    return False, 'TEST_OPEN seq %d precedes FINAL_DESIGNATION seq %d' % (to, fd)
