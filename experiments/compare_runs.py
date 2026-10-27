"""Compare existing runs (overlay plots + metric table) without re-simulating.

    python experiments/compare_runs.py results/runs/<batch>/nominal__servo results/runs/<batch>/nominal__pic
    python experiments/compare_runs.py results/runs/<batchA>/* --name my_comparison

Writes to results/compare/<timestamp>_<name>/ (a new folder every time). The legend shows the
parameters that differ between the runs; identical parameters are listed in the figure titles.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.dont_write_bytecode = True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run directories (containing manifest.json)")
    ap.add_argument("--name", default="comparison")
    ap.add_argument("--out-root", default=os.path.join(ROOT, "results", "compare"))
    a = ap.parse_args(argv)

    from pic import plots
    runs = []
    for p in a.runs:
        if not os.path.isfile(os.path.join(p, "manifest.json")):
            print(f"not a run directory (no manifest.json): {p}", file=sys.stderr)
            return 2
        r = plots.RunData(p)
        if r.manifest.get("status") != "complete":
            print(f"run is not complete (status {r.manifest.get('status')}): {p}", file=sys.stderr)
            return 2
        runs.append(r)
    if len(runs) < 2:
        print("need at least two runs", file=sys.stderr)
        return 2
    out = os.path.join(a.out_root, f"{dt.datetime.now():%Y%m%d-%H%M%S}_{a.name}")
    os.makedirs(out, exist_ok=False)
    paths = plots.make_compare_plots(runs, out, title=a.name)
    labels, common = plots.legend_labels(runs)
    rows = ["# Comparison " + a.name, "", common, "",
            "| legend | run | class | final error [mm] | t_reach [s] | RMSE sync [mm] | peak τ app/lim | saturated [%] |",
            "|---|---|---|---|---|---|---|---|"]
    for lb, r in zip(labels, runs):
        m = r.metrics
        tr = "-" if m["t_reach_s"] is None else f"{m['t_reach_s']:.3f}"
        rows.append(f"| {lb} | {r.name} | {m['failure_class']} | {1e3 * m['final_dist_m']:.4g} | {tr} | "
                    f"{1e3 * m['rmse_sync_m']:.3g} | {m['peak_applied_torque_ratio']:.3f} | "
                    f"{100 * m['saturated_fraction']:.2f} |")
    rows += ["", "Runs: " + ", ".join(os.path.abspath(r.path) for r in runs), ""]
    with open(os.path.join(out, "comparison.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(rows) + "\n")
    print("\n".join(rows[:2 + len(runs) + 4]))
    print("plots:", *paths, sep="\n  ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
