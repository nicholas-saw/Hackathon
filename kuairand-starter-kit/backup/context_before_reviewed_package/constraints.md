# Constraints — measured facts only

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
