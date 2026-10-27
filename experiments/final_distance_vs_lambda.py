"""Final distance to the goal over lambda1 at friction x1 (log-log), both robots.

    python -B experiments/final_distance_vs_lambda.py [--out-dir results/summary]

Question: is the final error of a friction-stalled PIC-DS run ~ 1/lambda1 ("Coulomb error at standstill")?
The success criterion (|p(T_max) - goal| < 1 % L at T_max = 10 s) is NOT changed; the final distance is
reported in addition, for T_max = 10 s (as configured) and 40 s (extra information). Only lambda1 changes;
lambda2, lambda3 and all other parameters stay at configs/picds.yaml. UR5e friction is SYNTHETIC (relative).
Also reported per run: |f| at the end, lambda1 * |f_end| (the Coulomb balance would make this ~ constant if the
arm stopped where lambda1 |f| equals a friction level) and whether the TCP is at rest.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import copy
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.dont_write_bytecode = True

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LAMBDAS = [round(20.0 * 2 ** (k / 2.0), 1) for k in range(13)]        # 20 ... 1280, factor sqrt(2)


def near_goal_x0(norm: float) -> list:
    """First point of the ideal rollout from the demo-mean start with ||x_n|| < norm (on the DS path)."""
    from common.scenario import load_config
    from pic.sim_pic import build_case
    cfg = load_config("configs/picds.yaml")
    case = build_case(cfg)
    x = case.x0_n.copy()
    for _ in range(int(60 / case.sp.dt)):
        if np.linalg.norm(x) < norm:
            return [float(v) for v in x]
        x = x + case.sp.dt * case.rate * case.ds.f_n(x)
    raise RuntimeError("ideal rollout never gets that close")


def one(args):
    robot, lam1, tmax, x0 = args
    from common.scenario import load_config
    from experiments.diagnose_stall import f_at
    from pic.controller import PICParams
    from pic.sim_pic import build_case, run
    cfg = load_config("configs/picds.yaml")
    cfg["robot"] = robot
    cfg["plant"] = {"dof_frictionloss_scale": 1.0}
    cfg["sim"]["T_max"] = float(tmax)
    if x0 is not None:
        cfg["task"]["start"] = {"x0": x0}
    params = PICParams.from_cfg(cfg["pic"])
    params.lam = (float(lam1), params.lam[1], params.lam[2])
    case = build_case(cfg)
    r = run(case, params, null_d_q=float(cfg["pic"]["null_space"]["d_q"]))
    tr = r["traj"]
    f_end = float(np.linalg.norm(f_at(case, tr["P"][-1])))
    v_end = float(np.linalg.norm(tr["xd"][-250:], axis=1).max())
    return {"robot": robot, "lambda1": lam1, "T_max_s": tmax, "success": bool(r["success"]),
            "final_dist_m": r["final_dist_m"], "f_end_m_s": f_end, "lambda1_x_f_end": lam1 * f_end,
            "tcp_speed_last_0.5s_max_m_s": v_end, "start_dist_m": float(tr["dist_goal"][0])}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "results", "summary"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--start-norm", type=float, default=None,
                    help="start on the ideal DS path where ||x_n|| first drops below this value (e.g. 0.3 = 30 mm) "
                         "instead of the demo-mean start; files get the suffix _near<norm>")
    a = ap.parse_args(argv)
    x0 = near_goal_x0(a.start_norm) if a.start_norm else None
    suffix = f"_near{a.start_norm:g}" if a.start_norm else ""
    jobs = [(rb, lam, t, x0) for rb in ("FR3", "UR5e") for t in (10.0, 40.0) for lam in LAMBDAS]
    with cf.ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(one, jobs))
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, f"stall_final_distance{suffix}.json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=1)

    fig, axs = plt.subplots(1, 2, figsize=(13, 5), sharey=True, layout="constrained")
    lam_star = {}
    with open(os.path.join(a.out_dir, "stall_diagnosis.json"), encoding="utf-8") as fh:
        lam_star["FR3"] = json.load(fh)["start"]["lam1_min"]
    with open(os.path.join(a.out_dir, "stall_diagnosis_UR5e.json"), encoding="utf-8") as fh:
        lam_star["UR5e"] = json.load(fh)["start"]["lam1_min"]
    for ax, rb in zip(axs, ("FR3", "UR5e")):
        for t, mk in ((10.0, "o-"), (40.0, "s--")):
            sel = sorted([r for r in rows if r["robot"] == rb and r["T_max_s"] == t], key=lambda r: r["lambda1"])
            ax.loglog([r["lambda1"] for r in sel], [1e3 * r["final_dist_m"] for r in sel], mk,
                      label=f"final distance, T_max = {t:g} s" + (" (criterion)" if t == 10 else " (extra)"))
        s10 = sorted([r for r in rows if r["robot"] == rb and r["T_max_s"] == 10.0], key=lambda r: r["lambda1"])
        d0 = 1e3 * s10[0]["start_dist_m"] if not a.start_norm else 1e3 * s10[0]["final_dist_m"]
        lam = np.array(LAMBDAS)
        ax.loglog(lam, d0 * LAMBDAS[0] / lam, "k:", label=r"$\propto 1/\lambda_1$ (anchored at the $\lambda_1$=20 value)")
        ax.axhline(1.0, color="gray", ls="-.", lw=0.8, label="success threshold 1 mm")
        ax.axvline(lam_star[rb], color="tab:red", ls="--", lw=0.9, label=rf"predicted breakaway $\lambda_1^*$={lam_star[rb]:.0f}")
        ax.set(xlabel=r"$\lambda_1$ [N s/m]", title=f"{rb}, friction x1" + (" (SYNTHETIC friction)" if rb == "UR5e" else ""))
        ax.grid(alpha=0.3, which="both")
    axs[0].set_ylabel("distance to goal at the end [mm]")
    axs[0].legend(fontsize=7)
    ncv = {rb: sum(r["success"] for r in rows if r["robot"] == rb and r["T_max_s"] == 10.0) for rb in ("FR3", "UR5e")}
    fig.suptitle("PIC-DS, friction x1: final distance vs lambda1 (single start" + (f" ON the ideal path at ||x_n||={a.start_norm:g}" if a.start_norm else ", demo-mean start") +
                 f"; success criterion unchanged: converged runs at T_max=10 s: FR3 {ncv['FR3']}/13, UR5e {ncv['UR5e']}/13)", fontsize=9)
    fig.savefig(os.path.join(a.out_dir, f"stall_final_distance{suffix}.png"), dpi=120)
    plt.close(fig)

    # log-log slope of the final distance (least squares) for lambda1 >= 20 .. 1280
    out = {}
    for rb in ("FR3", "UR5e"):
        for t in (10.0, 40.0):
            sel = sorted([r for r in rows if r["robot"] == rb and r["T_max_s"] == t], key=lambda r: r["lambda1"])
            x = np.log([r["lambda1"] for r in sel])
            y = np.log([r["final_dist_m"] for r in sel])
            slope = float(np.polyfit(x, y, 1)[0])
            out[f"{rb}_T{t:g}"] = {"loglog_slope": slope, "n_success": int(sum(r["success"] for r in sel)),
                                   "min_mm": 1e3 * float(min(r["final_dist_m"] for r in sel)),
                                   "max_mm": 1e3 * float(max(r["final_dist_m"] for r in sel))}
            print(rb, t, out[f"{rb}_T{t:g}"])
    with open(os.path.join(a.out_dir, f"stall_final_distance_slopes{suffix}.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
