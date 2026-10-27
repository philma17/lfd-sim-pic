"""Run a configured experiment: one command -> simulate, save data, plot, report.

    python experiments/run_experiment.py configs/experiments/nominal_fr3.yaml
    python experiments/run_experiment.py speed_scaling                      # short name works too
    python experiments/run_experiment.py nominal_fr3 --set plant.torque_limit_scale=0.3
    python experiments/run_experiment.py nominal_fr3 --controllers servo    # only the servo
    python experiments/run_experiment.py nominal_fr3 --strict               # exit 3 if not converged

Exit code 0 = every run finished and was saved (a non-converged run is a result, not an error),
1 = at least one run failed with an error, 2 = bad command line / configuration,
3 = --strict and at least one run did not converge.
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.dont_write_bytecode = True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("config", help="experiment YAML (path or name in configs/experiments/)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config value in ALL conditions, e.g. ds_speed_factor=2.0 "
                         "or plant.dof_frictionloss_scale=0 (repeatable)")
    ap.add_argument("--controllers", nargs="+", choices=["servo", "pic", "pic_pulse"])
    ap.add_argument("--out-root", help="parent directory of the batch folder "
                                       "(default results/runs)")
    ap.add_argument("--tag", help="name used in the batch folder instead of the experiment name")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--strict", action="store_true", help="exit code 3 if a run did not converge")
    a = ap.parse_args(argv)

    from pic.experiment import parse_set, run_experiment
    try:
        sets = parse_set(a.set)
        summary = run_experiment(a.config, sets=sets, controllers=a.controllers,
                                 out_root=a.out_root, make_plots=not a.no_plots, tag=a.tag,
                                 argv=sys.argv)
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"configuration error: {e}", file=sys.stderr)
        return 2
    print(f"\nbatch report : {os.path.join(summary['batch_dir'], 'batch_report.md')}")
    print(f"runs {summary['n_runs']}, errors {summary['n_errors']}, "
          f"finished but not converged {summary['n_not_converged']}")
    if summary["n_errors"] or summary.get("compare_failed"):
        return 1
    if a.strict and summary["n_not_converged"]:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
