"""Phase C -- baseline mechanism: seed variance, field ablations, lr sensitivity,
organizer static-feature / embedding-dim reproduction. VALIDATION ONLY."""
import sys, os, json, time, statistics
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import common as C
import fm_utils as F

def main():
    out = {}
    train = C.load_train_log()
    valid = C.load_valid_log()
    vbasic = C.load_video_basic()
    ufeat = C.load_user_features()

    fields5 = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket']

    # ---- C02 seed variance (5 seeds, official fields) ----
    print("== C02 seed variance ==")
    seed_scores = []
    Xtr, ytr, Xva, yva, uva, dim = F.encode_fields(train, valid, vbasic, fields5)
    cached_scores_seed0 = None
    for seed in range(5):
        t0 = time.time()
        m, scores, metrics, hist = F.train_fm(Xtr, ytr, Xva, yva, uva, dim, seed=seed)
        dt = time.time() - t0
        print(f"  seed {seed}: GAUC {metrics['GAUC']:.4f} nDCG@5 {metrics['nDCG@5']:.4f} "
              f"primary {metrics['primary']:.4f} ({dt:.1f}s, {len(hist)} epochs)")
        seed_scores.append(metrics)
        if seed == 0:
            cached_scores_seed0 = scores
            np.save(os.path.join(C.RESULTS_DIR, 'baseline_seed0_valid_scores.npy'), scores)
            np.save(os.path.join(C.RESULTS_DIR, 'baseline_seed0_valid_users.npy'), uva)
            np.save(os.path.join(C.RESULTS_DIR, 'baseline_seed0_valid_labels.npy'), yva)
    out['seed_variance'] = {
        'per_seed': seed_scores,
        'mean_primary': statistics.mean(s['primary'] for s in seed_scores),
        'std_primary': statistics.pstdev(s['primary'] for s in seed_scores),
        'mean_GAUC': statistics.mean(s['GAUC'] for s in seed_scores),
        'std_GAUC': statistics.pstdev(s['GAUC'] for s in seed_scores),
        'mean_nDCG5': statistics.mean(s['nDCG@5'] for s in seed_scores),
        'std_nDCG5': statistics.pstdev(s['nDCG@5'] for s in seed_scores),
    }

    # ---- C01 field ablations (leave-one-out from the 5 official fields), seed avg over 3 ----
    print("== C01 field ablations ==")
    ablation_results = {}
    for drop in [None] + fields5:
        flds = fields5 if drop is None else [f for f in fields5 if f != drop]
        runs = []
        for seed in range(3):
            Xtr2, ytr2, Xva2, yva2, uva2, dim2 = F.encode_fields(train, valid, vbasic, flds)
            m, scores, metrics, hist = F.train_fm(Xtr2, ytr2, Xva2, yva2, uva2, dim2, seed=seed)
            runs.append(metrics)
        label = 'full_5field' if drop is None else f'drop_{drop}'
        ablation_results[label] = {
            'fields': flds,
            'mean_primary': statistics.mean(r['primary'] for r in runs),
            'std_primary': statistics.pstdev(r['primary'] for r in runs),
            'mean_GAUC': statistics.mean(r['GAUC'] for r in runs),
            'mean_nDCG5': statistics.mean(r['nDCG@5'] for r in runs),
        }
        print(f"  {label:20s} primary {ablation_results[label]['mean_primary']:.4f} "
              f"+/- {ablation_results[label]['std_primary']:.4f}")
    out['field_ablations'] = ablation_results

    # ---- C03 learning-rate sensitivity (5-field, 3 seeds each) ----
    print("== C03 lr sensitivity ==")
    lr_results = {}
    for lr in [0.0003, 0.001, 0.003, 0.01]:
        runs = []
        for seed in range(3):
            m, scores, metrics, hist = F.train_fm(Xtr, ytr, Xva, yva, uva, dim, lr=lr, seed=seed)
            runs.append(metrics)
        lr_results[str(lr)] = {
            'mean_primary': statistics.mean(r['primary'] for r in runs),
            'std_primary': statistics.pstdev(r['primary'] for r in runs),
        }
        print(f"  lr={lr}: primary {lr_results[str(lr)]['mean_primary']:.4f} "
              f"+/- {lr_results[str(lr)]['std_primary']:.4f}")
    out['lr_sensitivity'] = lr_results

    # ---- C05 FM embedding dimension check (k=8/16/32/64), 3 seeds ----
    print("== C05 embedding dimension ==")
    dim_results = {}
    for k in [8, 16, 32, 64]:
        runs = []
        for seed in range(3):
            m, scores, metrics, hist = F.train_fm(Xtr, ytr, Xva, yva, uva, dim, k=k, seed=seed)
            runs.append(metrics)
        dim_results[str(k)] = {
            'mean_primary': statistics.mean(r['primary'] for r in runs),
            'std_primary': statistics.pstdev(r['primary'] for r in runs),
        }
        print(f"  k={k}: primary {dim_results[str(k)]['mean_primary']:.4f} "
              f"+/- {dim_results[str(k)]['std_primary']:.4f}")
    out['embedding_dim'] = dim_results

    # ---- C04 organizer static-feature expansion reproduction (VALID ONLY) ----
    print("== C04 static feature expansion (validation only) ==")
    USER_FE = ['follow_user_num_range', 'register_days_range', 'fans_user_num_range',
               'friend_user_num_range', 'user_active_degree']
    VID_FE_EXTRA = ['music_id', 'video_type', 'upload_type']  # author_id already in base

    u_maps = {f: ufeat.set_index('user_id')[f] for f in USER_FE}
    v_maps = {f: vbasic.set_index('video_id')[f] for f in VID_FE_EXTRA}

    static_results = {}
    configs = {
        'base_5field': (fields5, {}),
        'item_8field': (fields5 + VID_FE_EXTRA, {**v_maps}),
        'cwm_13field': (fields5 + VID_FE_EXTRA + USER_FE, {**v_maps, **u_maps}),
    }
    for label, (flds, extra_maps) in configs.items():
        runs = []
        for seed in range(3):
            Xtr2, ytr2, Xva2, yva2, uva2, dim2 = F.encode_fields(train, valid, vbasic, flds,
                                                                    extra_train_maps=extra_maps)
            m, scores, metrics, hist = F.train_fm(Xtr2, ytr2, Xva2, yva2, uva2, dim2, seed=seed)
            runs.append(metrics)
        static_results[label] = {
            'fields': flds,
            'mean_primary': statistics.mean(r['primary'] for r in runs),
            'std_primary': statistics.pstdev(r['primary'] for r in runs),
            'mean_GAUC': statistics.mean(r['GAUC'] for r in runs),
            'mean_nDCG5': statistics.mean(r['nDCG@5'] for r in runs),
        }
        print(f"  {label}: primary {static_results[label]['mean_primary']:.4f} "
              f"+/- {static_results[label]['std_primary']:.4f}")
    out['static_feature_expansion'] = static_results

    C.save_json(out, 'phase_c_baseline.json')

if __name__ == '__main__':
    main()
