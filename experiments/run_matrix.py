"""Run the variation matrix (Step 3): variations x robots x orientation modes, in parallel processes.

    python -B experiments/run_matrix.py --workers 6                 # everything
    python -B experiments/run_matrix.py --combos FR3:hold UR5e:hold # priority subset
    python -B experiments/run_matrix.py --configs var_alpha var_placement

Every (robot, orientation, config) is one `run_experiment.py` call, i.e. one batch folder under
results/runs/matrix_<stamp>/<robot>_<orientation>/ . The alpha variation is additionally run for five
start points drawn from the verified initial set (task.start = {index: 0..4}, needs S2NNDS_DIR with the
S2NNDS config files, see experiments/prepare_s2nnds.py), without per-run plots.
Base condition (see results/summary/stall_diagnosis.md): friction x0, paper gains, LASA/Sine.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
CONFIGS = ["var_alpha", "var_placement", "var_com", "var_com_truecomp", "var_mass", "var_mass_truecomp",
           "var_inertia"]
COMBOS = ["FR3:hold", "UR5e:hold", "FR3:none", "UR5e:none"]
N_STARTS = 5


def job(cmd, log):
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", MPLCONFIGDIR=os.environ.get("MPLCONFIGDIR", "/tmp/mpl"))
    with open(log, "w", encoding="utf-8") as fh:
        p = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env)
    return p.returncode, log, cmd


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--combos", nargs="+", default=COMBOS)
    ap.add_argument("--configs", nargs="+", default=CONFIGS)
    ap.add_argument("--no-starts", action="store_true")
    ap.add_argument("--root", default=None, help="output root (default results/runs/matrix_<stamp>)")
    a = ap.parse_args(argv)
    root = a.root or os.path.join(ROOT, "results", "runs", f"matrix_{dt.datetime.now():%Y%m%d-%H%M%S}")
    os.makedirs(os.path.join(root, "_logs"), exist_ok=False)
    jobs = []
    for combo in a.combos:
        robot, ori = combo.split(":")
        out = os.path.join(root, f"{robot}_{ori}")
        common = ["--set", f"robot={robot}", "--set", f"task.orientation={ori}", "--out-root", out]
        for c in a.configs:
            jobs.append([PY, "-B", "experiments/run_experiment.py", c, *common, "--tag", c])
        if not a.no_starts:
            for i in range(N_STARTS):
                jobs.append([PY, "-B", "experiments/run_experiment.py", "var_alpha", *common,
                             "--set", f"task.start={{index: {i}}}", "--tag", f"var_alpha_start{i}", "--no-plots"])
    print(f"{len(jobs)} jobs, root {root}", flush=True)
    bad = 0
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(job, j, os.path.join(root, "_logs", f"{n:03d}.log")) for n, j in enumerate(jobs)]
        for f in cf.as_completed(futs):
            rc, log, cmd = f.result()
            tag = cmd[cmd.index("--tag") + 1]
            robot = [x for x in cmd if x.startswith("robot=")][0]
            ori = [x for x in cmd if x.startswith("task.orientation=")][0]
            print(f"exit {rc}  {robot} {ori} {tag}  ({os.path.basename(log)})", flush=True)
            bad += rc != 0
    print(f"done, {bad} jobs with non-zero exit; root {root}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
