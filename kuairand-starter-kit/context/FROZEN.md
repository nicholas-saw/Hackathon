# Context Freeze Record

> Frozen 2026-08-30. The files below are the human-reviewed context package. They are
> read-only on disk (Windows read-only attribute) so that an accidental write
> fails loudly instead of silently changing what the agent was told.

## Frozen files

| File | Bytes (as recorded) | SHA-256 (LF-normalised) |
|---|---:|---|
| `context/problem_spec.md` | 4268 | `577413213b38a3739f46251825a627fe33f12e3c14735854b8a48d9739f8b116` |
| `context/PROBLEM.md` | 13196 | `57c4773346a04e78efb88d3b70081d39e303b64cb4fba32583881f518110e655` |
| `context/RULES.md` | 13004 | `a678c8ba6b85412cdf737a7ecbdfc6a0472e4158b559b0091ad395685e397c89` |
| `context/DATA_GUIDE.md` | 12882 | `3ed7b8d8f78ec7a9faa2efdb7a802d4221632b26cd516b6e840b7374b93ffe5f` |
| `context/constraints.md` | 31902 | `4026d5d29a8ac453d760db835ebc5657f589367d74e760b3d1b421e009874f87` |
| `context/references.md` | 17352 | `7540da6998740689c134261e6b034768b6a2076b7b5eea7000089a9e02daefeb` |
| `research/data_profile.md` | 13140 | `0789e5f9a1504ef9eadef2d2a753b3b378880bf337132360addb714da0e8d2f6` |
| `AGENT_RULES.md` | 18346 | `78904478a8d883da444608740bce2693038ca8a335e797ae3b538d90614cac45` |

Verify at any time:

```bash
python context/verify_context.py        # exit 0 when all eight match
```

> **The hashes above are over LF-normalised bytes, and that is deliberate.** Under
> git's default Windows checkout every LF becomes CRLF, so the raw bytes of a text
> file differ between machines while its content is identical. An earlier version of
> this record published raw-byte hashes and a one-line `hashlib` command; on a Windows
> checkout that command reported MISMATCH for seven of the eight files, each byte delta
> equal to the file's CRLF count exactly (constraints.md: 32,657 - 31,902 = 755 = its
> 755 CRLF line endings). Nothing had been altered. `FROZEN.md` warns that "a stale
> fingerprint is worse than none, because it reads as verification that did not
> happen" -- a fingerprint that fails on an untouched file is the same problem, so the
> check now normalises before hashing and answers the question actually being asked.

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
