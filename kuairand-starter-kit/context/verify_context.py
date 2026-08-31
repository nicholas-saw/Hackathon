"""Verify the frozen context package has not been altered.

Hashes are taken over LF-normalised bytes. The raw bytes of a text file are not stable
across platforms: git converts LF to CRLF on checkout under the default Windows
autocrlf, so a file that is byte-identical in the repository hashes differently on two
machines. Hashing the normalised form makes the check answer the question actually being
asked -- "is the content the reviewed content?" -- rather than "was this checked out on
the same OS as the person who recorded the hash?".

    python context/verify_context.py          # exit 0 if every file matches

The freeze exists because these files are what the agent is told. Changing one mid-project
silently changes the evidence behind every hypothesis that followed it.
"""
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# path -> sha256 of the file with CRLF normalised to LF
FROZEN = {
    'context/problem_spec.md':  '577413213b38a3739f46251825a627fe33f12e3c14735854b8a48d9739f8b116',
    'context/PROBLEM.md':       '57c4773346a04e78efb88d3b70081d39e303b64cb4fba32583881f518110e655',
    'context/RULES.md':         'a678c8ba6b85412cdf737a7ecbdfc6a0472e4158b559b0091ad395685e397c89',
    'context/DATA_GUIDE.md':    '3ed7b8d8f78ec7a9faa2efdb7a802d4221632b26cd516b6e840b7374b93ffe5f',
    'context/constraints.md':   '4026d5d29a8ac453d760db835ebc5657f589367d74e760b3d1b421e009874f87',
    'context/references.md':    '7540da6998740689c134261e6b034768b6a2076b7b5eea7000089a9e02daefeb',
    'research/data_profile.md': '0789e5f9a1504ef9eadef2d2a753b3b378880bf337132360addb714da0e8d2f6',
    'AGENT_RULES.md':           '78904478a8d883da444608740bce2693038ca8a335e797ae3b538d90614cac45',
}


def digest(path):
    with open(path, 'rb') as fh:
        return hashlib.sha256(fh.read().replace(b'\r\n', b'\n')).hexdigest()


def main():
    bad = 0
    for rel, want in sorted(FROZEN.items()):
        p = os.path.join(ROOT, rel.replace('/', os.sep))
        if not os.path.exists(p):
            print('MISSING   %s' % rel)
            bad += 1
            continue
        got = digest(p)
        if got == want:
            print('ok        %s' % rel)
        else:
            print('ALTERED   %s\n            recorded %s\n            actual   %s'
                  % (rel, want, got))
            bad += 1
    print('\n%s: %d of %d files match the freeze record'
          % ('FAIL' if bad else 'PASS', len(FROZEN) - bad, len(FROZEN)))
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
