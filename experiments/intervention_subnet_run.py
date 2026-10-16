"""Scheduling of experiments/intervention_subnet.py as used for the run of the paper: most of the time of a sub-network point is
single-threaded annealing, so holding 1-4 CPU slots per point from start to end would leave them idle.
Each sub-network point takes 1 slot only (scheduler MAX_W = 1). Only before the CP-SAT steps (subnet_exact.compute and the
same-dispatch polish intervention_subnet.polish) does the child process take, without blocking, up to LEASE - 1 more slots under
its own tag (subnet-<pid>-c); the number of CP-SAT threads = 1 + the slots taken, given back as soon as the step is done (with a
heartbeat every 5 s meanwhile). The actual number of threads of every step is added to the result of the point (cpsat_threads;
threads = the largest of them).
intervention_subnet.py is unchanged: instances, steps and time limits are those of intervention_subnet; only the number of
CP-SAT threads differs (as in the exact sub-network study, the number of threads is not part of the configuration and is
recorded per point). One scheduler computes all remaining instances (one progress.json); the sub-network of road099-4 (SKIP)
is not scheduled: it has no comparable instance (experiments/intervention_diag.py).

Run:   python experiments/intervention_subnet_run.py --out <dir> --bounds <points.csv of intervention_bounds.py> [--only INSTANCE ...]
           [--max-minutes M] [--lease N] [--slots DIR] [--aggregate-only]
--lease (or the environment variable CPSAT_LEASE; default 4): at most this many CP-SAT threads in one step, the slot of the point
included. The slots come from --slots or the environment variable CPU_SLOTS (a directory with the module cpu_slots, as in
experiments/runner.py); without either, the slot interface is off and every CP-SAT step runs on the one thread the scheduler
gives the point. Outputs as intervention_subnet.py.
"""
import json
import os
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import runner as F                        # noqa: E402
import subnet_exact as SUB                # noqa: E402
import intervention_subnet as S           # noqa: E402

LEASE = int(os.environ.get("CPSAT_LEASE", "4"))     # at most this many threads in one CP-SAT step (the slot of the point included)
USED = []                                           # threads actually used by each CP-SAT step of this point


def leased(fn):
    """Wraps a CP-SAT step whose last positional argument is the number of threads: takes slots, runs on the number taken,
    gives them back. Without a slot directory (environment variable CPU_SLOTS) it runs on the number of threads passed."""
    def run(*a, **kw):
        sd = os.environ.get("CPU_SLOTS")
        if not sd:
            USED.append(a[-1])
            return fn(*a, **kw)
        sys.path.insert(0, sd)
        from cpu_slots import Slots
        s = Slots(f"subnet-{os.getpid()}-c")
        s.want(LEASE - 1)
        while len(s.held) < LEASE - 1 and s.take():
            pass
        w = 1 + len(s.held)
        stop = threading.Event()

        def beat():
            while not stop.wait(5.0):
                s.want(len(s.held))
        threading.Thread(target=beat, daemon=True).start()
        try:
            USED.append(w)
            return fn(*a[:-1], w, **kw)
        finally:
            stop.set()
            s.close()
    return run


_ORIG = getattr(S, "_originals", None) or dict(compute=SUB.compute, polish=S.polish, run=S.run_point, keys=S.keys)
S._originals = _ORIG                        # the original functions are kept on intervention_subnet: a reload wraps only once
# intervention_subnet.run_point calls it as a module attribute: replaced on import of this module (child processes too)
SUB.compute = leased(_ORIG["compute"])
S.polish = leased(_ORIG["polish"])
_RUN = _ORIG["run"]


def run_point(key, minutes, w, out):
    """Child process: one point (the point itself holds 1 slot); afterwards the actual threads of each CP-SAT step are added
    to the result of the point."""
    USED.clear()
    _RUN(key, minutes, 1, out)
    f = Path(out) / "points" / F.fname(key)
    if f.exists():
        x = json.loads(f.read_text(encoding="utf-8"))
        x.update(threads=max(USED, default=1), cpsat_threads=list(USED), slot_lease=LEASE)
        F.write_json(f, x)


S.run_point = run_point                     # intervention_subnet.main hands it to the scheduler by its global name
_KEYS = _ORIG["keys"]
SKIP = S.SKIP                               # no comparable instance: not scheduled, not counted


def keys(only=None):
    return [k for k in _KEYS(only) if k.split("|")[0] not in SKIP]


S.keys = keys                               # intervention_subnet.main: progress and points.csv cover only the scheduled points


def main():
    args = sys.argv[1:]
    if "--lease" in args:
        i = args.index("--lease")
        os.environ["CPSAT_LEASE"] = args[i + 1]
        del args[i:i + 2]
    if "--slots" in args:
        os.environ["CPU_SLOTS"] = args[args.index("--slots") + 1]    # inherited by the child processes, which lease the CP-SAT threads
    sys.argv = [sys.argv[0]] + args + ["--max-threads", "1"]          # scheduler: one slot per point
    S.main()


if __name__ == "__main__":
    main()
