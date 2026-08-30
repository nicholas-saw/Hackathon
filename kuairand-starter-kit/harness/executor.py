"""Run one experiment as a subprocess, and be able to actually kill it on Windows.

Windows has no `os.killpg`, no `os.setsid`, no `os.fork`, no `SIGKILL`, no `SIGALRM`, and
no `resource` module. `Popen.kill()` terminates only the direct child, so a training run
that spawned anything of its own leaves an orphan holding a core for the rest of the
night. The repo's own probe measured a **30.1 s overrun on a 3 s timeout** doing exactly
that.

So: `taskkill /F /T /PID` to take the whole process tree, then psutil to confirm the pid
is really gone before the node is recorded, and an `error_recovery` event either way.
"""
import os
import subprocess
import sys
import time

IS_WINDOWS = os.name == 'nt'

# Recognisable failure signatures, in priority order. The reason string goes back to the
# coder, so it has to name the thing that broke.
_SIGNATURES = (
    ('nan', ('nan', 'inf', 'overflow encountered', 'invalid value encountered',
             'divide by zero')),
    ('syntax', ('SyntaxError', 'IndentationError', 'TabError')),
    ('import', ('ModuleNotFoundError', 'ImportError')),
    ('memory', ('MemoryError', 'Unable to allocate', 'cannot allocate')),
    ('leak', ('TestSealError', 'TestRowsRequested', '是曝光后结果列')),
    ('contract', ('TypeError', 'AttributeError', 'KeyError', 'ValueError',
                  'IndexError', 'AssertionError')),
)


def classify(stdout, stderr, returncode, timed_out):
    """Name the failure so the reflector and the coder get something actionable."""
    if timed_out:
        return 'timeout'
    if returncode == 0:
        return 'ok'
    blob = (stdout or '') + '\n' + (stderr or '')
    for name, needles in _SIGNATURES:
        for nd in needles:
            if nd in blob:
                return name
    return 'unknown'


def _kill_tree(proc):
    """Kill the process and everything it spawned. Returns (killed_ok, note)."""
    if proc.poll() is not None:
        return True, 'already exited'
    if IS_WINDOWS:
        subprocess.run(['taskkill', '/F', '/T', '/PID', str(proc.pid)],
                       capture_output=True)
    else:
        try:
            os.killpg(os.getpgid(proc.pid), 9)
        except Exception:
            proc.kill()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    return _confirm_gone(proc.pid)


def _confirm_gone(pid):
    """Never report a kill you have not verified. An orphan eats a core silently."""
    try:
        import psutil
    except ImportError:
        return True, 'psutil unavailable; kill not verified'
    try:
        p = psutil.Process(pid)
        if p.is_running() and p.status() != psutil.STATUS_ZOMBIE:
            kids = [c.pid for c in p.children(recursive=True)]
            return False, 'pid %d survived taskkill (children: %r)' % (pid, kids)
        return True, 'pid %d gone' % pid
    except psutil.NoSuchProcess:
        return True, 'pid %d gone' % pid
    except Exception as exc:
        return True, 'kill unverified: %s' % exc


def run(cmd, timeout=600, cwd=None, env=None, capture=True):
    """Run a command with a hard timeout and a real process-tree kill.

    Returns a dict the journal can store verbatim.
    """
    e = dict(os.environ)
    # Without this the child crashes printing any non-cp1252 character to a pipe —
    # which is how kit/submit.py fails on a submission it has just validated.
    e['PYTHONUTF8'] = '1'
    e['PYTHONIOENCODING'] = 'utf-8'
    if env:
        e.update(env)
    kw = dict(cwd=cwd, env=e, text=True, encoding='utf-8', errors='replace')
    if capture:
        kw.update(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if not IS_WINDOWS:
        kw['start_new_session'] = True

    t0 = time.time()
    proc = subprocess.Popen(cmd, **kw)
    timed_out = False
    kill_ok, kill_note = True, ''
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        kill_ok, kill_note = _kill_tree(proc)
        try:
            out, err = proc.communicate(timeout=10)
        except Exception:
            out, err = '', ''
    secs = time.time() - t0

    return {'cmd': cmd if isinstance(cmd, str) else ' '.join(map(str, cmd)),
            'returncode': proc.returncode,
            'timed_out': timed_out,
            'seconds': round(secs, 2),
            'failure': classify(out, err, proc.returncode, timed_out),
            'orphan_free': kill_ok,
            'kill_note': kill_note,
            'stdout': (out or '')[-8000:],
            'stderr': (err or '')[-8000:]}


def run_python(args, timeout=600, cwd=None, env=None):
    """Run `python <args...>` with the current interpreter."""
    return run([sys.executable] + list(args), timeout=timeout, cwd=cwd, env=env)


def adaptive_timeout(history, floor_s=180, factor=8.0):
    """Kill a node at `factor` x the running median, never below `floor_s`.

    A fixed timeout is either too tight for an honest slow idea or too loose to stop a
    runaway. Scale it to what this run has actually been costing.
    """
    xs = sorted(h for h in history if h and h > 0)
    if not xs:
        return floor_s * 4
    mid = xs[len(xs) // 2] if len(xs) % 2 else 0.5 * (xs[len(xs) // 2 - 1] + xs[len(xs) // 2])
    return max(floor_s, factor * mid)
