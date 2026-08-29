"""Phase J -- engineering profile: load/encode/train/eval timing, cache
speedup, Windows subprocess timeout/tree-kill behavior, syntax-error and
NaN/Inf recovery checks. All timing uses train+valid only."""
import sys, os, json, time, subprocess, textwrap
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import common as C
import fm_utils as F

def timed(fn, *a, **kw):
    t0 = time.time()
    r = fn(*a, **kw)
    return r, time.time() - t0

def main():
    out = {}
    proc = None
    try:
        import psutil
        proc = psutil.Process(os.getpid())
    except Exception:
        pass

    def rss_mb():
        return float(proc.memory_info().rss / 1e6) if proc else None

    out['memory_start_mb'] = rss_mb()

    # ---- CSV load timing ----
    train, t_train_load = timed(C.load_train_log)
    valid, t_valid_load = timed(C.load_valid_log)
    vbasic, t_vbasic_load = timed(C.load_video_basic)
    vstat, t_vstat_load = timed(C.load_video_stat)
    ufeat, t_ufeat_load = timed(C.load_user_features)
    out['csv_load_seconds'] = {
        'train_log': t_train_load, 'valid_log': t_valid_load,
        'video_basic': t_vbasic_load, 'video_stat': t_vstat_load, 'user_features': t_ufeat_load,
        'total': t_train_load + t_valid_load + t_vbasic_load + t_vstat_load + t_ufeat_load,
    }
    out['memory_after_load_mb'] = rss_mb()

    # ---- encoding timing (5-field baseline) ----
    fields5 = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket']
    (Xtr, ytr, Xva, yva, uva, dim), t_encode = timed(F.encode_fields, train, valid, vbasic, fields5)
    out['encode_seconds'] = t_encode
    out['encoded_dim'] = dim
    out['memory_after_encode_mb'] = rss_mb()

    # ---- training + eval timing (single seed, matches official config) ----
    t0 = time.time()
    m, scores, metrics, history = F.train_fm(Xtr, ytr, Xva, yva, uva, dim, seed=0, verbose=False)
    t_train_total = time.time() - t0
    out['fm_train_total_seconds'] = t_train_total
    out['fm_epochs_run'] = len(history)
    out['fm_seconds_per_epoch_mean'] = float(np.mean([h['time_s'] for h in history]))
    out['fm_final_metrics'] = metrics
    out['memory_after_train_mb'] = rss_mb()

    t0 = time.time()
    _ = m.predict(Xva)
    out['predict_only_seconds'] = time.time() - t0

    out['cold_run_total_seconds_load_encode_train'] = (
        out['csv_load_seconds']['total'] + t_encode + t_train_total)

    # ---- G02 cache speedup: pickle encoded arrays, reload ----
    import pickle
    cache_path = os.path.join(C.RESULTS_DIR, '_cache_test.pkl')
    t0 = time.time()
    with open(cache_path, 'wb') as fh:
        pickle.dump((Xtr, ytr, Xva, yva, uva, dim), fh, protocol=4)
    t_cache_write = time.time() - t0
    t0 = time.time()
    with open(cache_path, 'rb') as fh:
        Xtr2, ytr2, Xva2, yva2, uva2, dim2 = pickle.load(fh)
    t_cache_read = time.time() - t0
    identical = (np.array_equal(Xtr, Xtr2) and np.array_equal(ytr, ytr2) and
                 np.array_equal(Xva, Xva2) and np.array_equal(yva, yva2) and dim == dim2)
    out['cache_test'] = {
        'write_seconds': t_cache_write, 'read_seconds': t_cache_read,
        'reload_vs_recompute_speedup_x': (t_encode / t_cache_read) if t_cache_read > 0 else None,
        'reproduces_identical_arrays': bool(identical),
        'no_labels_used_beyond_train_valid': True,
    }
    os.remove(cache_path)

    # ---- Windows subprocess timeout / process-tree kill test ----
    windows_tests = {}
    script = textwrap.dedent("""
        import subprocess, sys, time
        # spawn a child that itself sleeps, to test tree termination
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
        time.sleep(30)
    """)
    script_path = os.path.join(C.RESULTS_DIR, '_sleep_tree_test.py')
    with open(script_path, 'w') as fh:
        fh.write(script)
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, script_path], timeout=3,
                            capture_output=True, text=True)
        windows_tests['timeout_triggered'] = False
    except subprocess.TimeoutExpired as e:
        windows_tests['timeout_triggered'] = True
        windows_tests['elapsed_seconds'] = time.time() - t0
    windows_tests['note'] = ('subprocess.run(timeout=N) raises TimeoutExpired and Python kills the direct '
                              'child, but on Windows this does NOT reliably kill grandchild processes spawned '
                              'by that child (no process-group signal semantics like POSIX). A harness that '
                              'needs true tree-kill on Windows should launch with CREATE_NEW_PROCESS_GROUP and '
                              'use taskkill /T /F on the PID, or use psutil.Process(pid).children(recursive=True).')
    os.remove(script_path)
    out['windows_subprocess_timeout'] = windows_tests

    # verify orphan grandchild reality: check for leftover python processes sleeping
    try:
        import psutil as _ps
        time.sleep(0.5)
        leftover = [p2.pid for p2 in _ps.process_iter(['pid', 'cmdline'])
                    if p2.info['cmdline'] and 'time.sleep(30)' in ' '.join(p2.info['cmdline'])]
        windows_tests['orphaned_grandchild_pids_still_running'] = leftover
        for pid in leftover:
            try:
                _ps.Process(pid).kill()
            except Exception:
                pass
    except Exception as e:
        windows_tests['orphan_check_error'] = str(e)

    # ---- syntax-error recovery check ----
    bad_script = os.path.join(C.RESULTS_DIR, '_bad_syntax_test.py')
    with open(bad_script, 'w') as fh:
        fh.write("def f(:\n    pass\n")
    r = subprocess.run([sys.executable, bad_script], capture_output=True, text=True)
    out['syntax_error_recovery'] = {
        'returncode': r.returncode,
        'stderr_contains_SyntaxError': 'SyntaxError' in r.stderr,
        'recoverable_signal': 'nonzero returncode + SyntaxError in stderr is a clean, detectable failure mode',
    }
    os.remove(bad_script)

    # ---- NaN/Inf submission rejection check (uses submit.py's own validator, VALID split only) ----
    sys.path.insert(0, F.STARTER_KIT)
    import importlib
    submit_mod = importlib.import_module('submit')
    import data as official_data
    splits_for_check = {'valid': [tuple(x) for x in zip(
        valid['user_id'], valid['video_id'], valid['date'], valid['hourmin'], valid['time_ms'],
        valid['is_click'], valid['is_like'], valid['is_follow'], valid['is_comment'], valid['is_forward'],
        valid['is_hate'], valid['long_view'], valid['play_time_ms'], valid['duration_ms'],
        valid['profile_stay_time'], valid['comment_stay_time'], valid['is_profile_enter'], valid['is_rand'],
        valid['tab'])]}
    # Build a minimal rows structure matching data.load()'s tuple shape subset actually used by submit.py:
    # submit.py only needs rows[n][1], rows[n][2] (user_id, video_id) for alignment checks.
    rows_like = list(zip(valid['user_id'].astype(str), valid['video_id'].astype(str)))

    nan_csv = os.path.join(C.RESULTS_DIR, '_nan_submission_test.csv')
    with open(nan_csv, 'w', newline='') as fh:
        import csv as csvmod
        w = csvmod.writer(fh)
        w.writerow(['row_id', 'user_id', 'video_id', 'score'])
        for i, (uid, vid) in enumerate(rows_like):
            score = 'nan' if i == 0 else '0.0'
            w.writerow([i, uid, vid, score])

    class FakeRow:
        def __init__(self, uid, vid):
            self._d = {1: uid, 2: vid}
        def __getitem__(self, k):
            return self._d[k]

    fake_rows = [FakeRow(uid, vid) for uid, vid in rows_like]
    try:
        submit_mod.read_submission(nan_csv, fake_rows)
        out['nan_inf_recovery'] = {'rejected': False, 'note': 'UNEXPECTED: NaN score was not rejected'}
    except ValueError as e:
        out['nan_inf_recovery'] = {'rejected': True, 'error_message': str(e)}
    os.remove(nan_csv)

    out['memory_end_mb'] = rss_mb()
    try:
        vm = __import__('psutil').virtual_memory()
        out['system_memory'] = {'total_gb': round(vm.total / 1e9, 2), 'available_gb': round(vm.available / 1e9, 2),
                                 'percent_used': vm.percent}
    except Exception:
        pass

    import platform
    out['environment'] = {'os': platform.platform(), 'python': platform.python_version(),
                            'cpu_count': os.cpu_count()}

    C.save_json(out, 'phase_j_engineering.json')
    print(json.dumps(out, indent=2, default=str))

if __name__ == '__main__':
    main()
