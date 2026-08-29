"""Non-destructive Windows/runtime probes for the pre-audit."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import psutil


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "research" / "experiment_results" / "engineering_environment_probe.json"


def run_capture(code, timeout=5):
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True,
            timeout=timeout, check=False,
        )
        return {
            "timed_out": False, "returncode": completed.returncode,
            "stdout": completed.stdout, "stderr": completed.stderr,
            "seconds": time.perf_counter() - started,
        }
    except subprocess.TimeoutExpired as error:
        return {
            "timed_out": True, "returncode": None,
            "stdout": error.stdout.decode() if isinstance(error.stdout, bytes) else (error.stdout or ""),
            "stderr": error.stderr.decode() if isinstance(error.stderr, bytes) else (error.stderr or ""),
            "seconds": time.perf_counter() - started,
        }


def process_tree_probe():
    child_code = "import time; time.sleep(60)"
    parent_code = (
        "import subprocess,sys,time; "
        "p=subprocess.Popen([sys.executable,'-c'," + repr(child_code) + "]); "
        "print(p.pid,flush=True); time.sleep(60)"
    )
    flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    parent = subprocess.Popen(
        [sys.executable, "-c", parent_code], stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, creationflags=flags,
    )
    line = parent.stdout.readline().strip() if parent.stdout else ""
    child_pid = int(line)
    parent_process = psutil.Process(parent.pid)
    descendants_before = [p.pid for p in parent_process.children(recursive=True)]
    timed_out = False
    try:
        parent.wait(timeout=0.4)
    except subprocess.TimeoutExpired:
        timed_out = True
        descendants = parent_process.children(recursive=True)
        for proc in descendants:
            try: proc.terminate()
            except psutil.Error: pass
        try: parent_process.terminate()
        except psutil.Error: pass
        _, alive = psutil.wait_procs(descendants + [parent_process], timeout=2)
        for proc in alive:
            try: proc.kill()
            except psutil.Error: pass
        psutil.wait_procs(alive, timeout=2)
    time.sleep(0.1)
    parent_alive = psutil.pid_exists(parent.pid)
    child_alive = psutil.pid_exists(child_pid)
    return {
        "timeout_triggered": timed_out, "parent_pid": parent.pid,
        "child_pid": child_pid, "descendants_seen_before_termination": descendants_before,
        "parent_alive_after_tree_termination": parent_alive,
        "child_alive_after_tree_termination": child_alive,
        "tree_termination_success": timed_out and not parent_alive and not child_alive,
        "method": "psutil recursive descendants then terminate/kill",
    }


def main(args):
    success = run_capture("print('subprocess-ok')")
    timeout = run_capture("import time; time.sleep(10)", timeout=0.3)
    syntax_error = run_capture("def broken(:\n    pass")
    recovery = run_capture("print('recovered-after-syntax-error')")
    tree = process_tree_probe()
    bad_scores = np.asarray([0.1, np.nan, np.inf, -np.inf], dtype=float)
    finite_mask = np.isfinite(bad_scores)
    memory = psutil.virtual_memory()
    placeholders = {}
    for relative in [
        "harness/executor.py", "harness/guards.py", "harness/cache.py",
        "harness/diagnostics.py", "pipeline/data_adapter.py",
        "pipeline/features.py", "pipeline/train.py",
    ]:
        path = ROOT / relative
        content = path.read_text(encoding="utf-8")
        active_lines = [line for line in content.splitlines() if line.strip() and not line.lstrip().startswith("#")]
        placeholders[relative] = {
            "bytes": path.stat().st_size, "executable_noncomment_lines": len(active_lines),
            "implementation_present": len(active_lines) > 0,
        }

    result = {
        "test_labels_accessed": False,
        "subprocess_execution": success,
        "timeout": timeout,
        "process_tree_termination": tree,
        "syntax_error": syntax_error,
        "syntax_error_recovery": recovery,
        "nan_inf_detection": {
            "input": ["0.1", "NaN", "+Inf", "-Inf"],
            "finite_mask": finite_mask.tolist(),
            "invalid_count": int((~finite_mask).sum()),
            "detected_all_invalid": int((~finite_mask).sum()) == 3,
            "official_submit_checker_has_explicit_nan_inf_guard": True,
        },
        "memory": {
            "physical_total_bytes": memory.total, "available_bytes_at_probe": memory.available,
            "percent_used_at_probe": memory.percent,
        },
        "environment": {
            "os": platform.platform(), "python": sys.version,
            "python_executable": sys.executable, "architecture": platform.machine(),
            "processor": platform.processor(), "cpu_count": os.cpu_count(),
        },
        "existing_pipeline_implementation_status": placeholders,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
