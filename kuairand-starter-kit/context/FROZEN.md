# Context Freeze Record

> Frozen 2026-08-30. The files below are the human-reviewed context package. They are
> read-only on disk (Windows read-only attribute) so that an accidental write
> fails loudly instead of silently changing what the agent was told.

## Frozen files

> `context/constraints.md` was re-frozen on 2026-08-31 after adding C24-C30
> (agent run evidence from `runlogs/*/journal.jsonl`) and marking C22 superseded.
> Its row below carries the post-edit size and hash. No other frozen file changed.

| File | Bytes | SHA-256 |
|---|---:|---|
| `context/problem_spec.md` | 4268 | `2dcb13f41d6988a1019f5f874580b363a9d5eb7375f09571a76885a966abed4c` |
| `context/PROBLEM.md` | 13196 | `57c4773346a04e78efb88d3b70081d39e303b64cb4fba32583881f518110e655` |
| `context/RULES.md` | 13004 | `a678c8ba6b85412cdf737a7ecbdfc6a0472e4158b559b0091ad395685e397c89` |
| `context/DATA_GUIDE.md` | 12882 | `3ed7b8d8f78ec7a9faa2efdb7a802d4221632b26cd516b6e840b7374b93ffe5f` |
| `context/constraints.md` | 43258 | `d91bc9353934e69131d40cc52ed87f968248f0df9b644a34a959caea953878d4` |
| `context/references.md` | 17352 | `7540da6998740689c134261e6b034768b6a2076b7b5eea7000089a9e02daefeb` |
| `research/data_profile.md` | 13140 | `0789e5f9a1504ef9eadef2d2a753b3b378880bf337132360addb714da0e8d2f6` |
| `AGENT_RULES.md` | 18346 | `78904478a8d883da444608740bce2693038ca8a335e797ae3b538d90614cac45` |

Verify at any time:

```bash
python -c "import hashlib,sys;[print(hashlib.sha256(open(p,'rb').read()).hexdigest(),p) for p in sys.argv[1:]]" context/problem_spec.md context/PROBLEM.md context/RULES.md context/DATA_GUIDE.md context/constraints.md context/references.md research/data_profile.md AGENT_RULES.md
```

## Not frozen, and why

- `context/packet.md`, `context/data_profile.json`, `context/baseline.json` —
  **generated**. `context/build_packet.py` rewrites them on every run. It reads the
  frozen files and never writes them, so a rebuild cannot alter the reviewed
  evidence; it only re-assembles the packet around it.
- `context/build_packet.py` — the generator itself stays writable so the packet can
  be rebuilt during testing.
- `context/CONTEXT_UPDATE_REPORT.md` — a record of the update, not an input the
  agent reads.
- `pipeline/`, `harness/`, `agent/` — the working surfaces under test.

Rebuilding the packet is safe while frozen:

```bash
python context/build_packet.py
```

## Unfreezing

The freeze is the Windows read-only attribute, nothing more. To edit a file:

```bash
attrib -R "<path>"
```

Re-freeze with `attrib +R "<path>"`. If you change a frozen file, update its
fingerprint in this table — a stale fingerprint is worse than none, because it
reads as verification that did not happen.
