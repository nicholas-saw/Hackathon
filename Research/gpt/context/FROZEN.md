# Context Freeze Record

> Frozen 2026-08-30. The files below are the human-reviewed context package. They are
> read-only on disk (Windows read-only attribute) so that an accidental write
> fails loudly instead of silently changing what the agent was told.

## Frozen files

| File | Bytes | SHA-256 |
|---|---:|---|
| `context/PROBLEM.md` | 13196 | `57c4773346a04e78efb88d3b70081d39e303b64cb4fba32583881f518110e655` |
| `context/RULES.md` | 13004 | `a678c8ba6b85412cdf737a7ecbdfc6a0472e4158b559b0091ad395685e397c89` |
| `context/DATA_GUIDE.md` | 12882 | `3ed7b8d8f78ec7a9faa2efdb7a802d4221632b26cd516b6e840b7374b93ffe5f` |
| `context/constraints.md` | 31902 | `4026d5d29a8ac453d760db835ebc5657f589367d74e760b3d1b421e009874f87` |
| `context/references.md` | 17352 | `7540da6998740689c134261e6b034768b6a2076b7b5eea7000089a9e02daefeb` |
| `research/data_profile.md` | 13140 | `0789e5f9a1504ef9eadef2d2a753b3b378880bf337132360addb714da0e8d2f6` |

Verify at any time:

```bash
python -c "import hashlib,sys;[print(hashlib.sha256(open(p,'rb').read()).hexdigest(),p) for p in sys.argv[1:]]" context/PROBLEM.md context/RULES.md context/DATA_GUIDE.md context/constraints.md context/references.md research/data_profile.md
```

## Not frozen, and why

- `context/CONTEXT_UPDATE_REPORT.md` — a record of the update, not an input.
- `backup/context_before_audit_update/` — the pre-update versions.
- `research/PRE_AUDIT.md`, `research/REVIEW_REPORT.md`, `research/consolidated/` —
  source audit material, unmodified throughout.

## Unfreezing

The freeze is the Windows read-only attribute, nothing more. To edit a file:

```bash
attrib -R "<path>"
```

Re-freeze with `attrib +R "<path>"`. If you change a frozen file, update its
fingerprint in this table — a stale fingerprint is worse than none, because it
reads as verification that did not happen.
