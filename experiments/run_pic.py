"""PIC-DS vs. the baseline position servo on the FR3 (LASA Sine, 3D position task).

    ~/lfd_env/bin/python experiments/run_pic.py                    # both plants, ~1 min
    ~/lfd_env/bin/python experiments/run_pic.py --friction-scales 1.0
    ~/lfd_env/bin/python experiments/run_pic.py --no-figure

Per plant (dof_frictionloss scale; 1.0 = unmodified Menagerie model, 0.0 = DIAGNOSTIC):
  servo        common.sim_dynamic.run (baseline, unchanged): joint position servo, ctrl = q_des
  pic          pic.sim_pic.run, undisturbed
  pic+pulse    pic.sim_pic.run with the TCP force pulses from configs/picds.yaml
All runs start from the same q_start (baseline start procedure) and use the same T_max and
convergence threshold from the config. Outputs: results/pic/metrics.json, traj_*.npz,
figures/pic/fig_overview.png
"""
from __future__ import annotations

import argparse
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

from common import sim_dynamic
from common.scenario import load_config
from pic.controller import PICParams
from pic.sim_pic import build_case, run

OUT = os.path.join(ROOT, "results", "pic")
FIG = os.path.join(ROOT, "figures", "pic")


def to_jsonable(x):
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    return x


def servo_summary(res, case) -> dict:
    """Baseline servo run reduced to the same fields as the PIC runs."""
    sp, emb = case.sp, case.emb
    goal = emb.to_world(np.zeros(2))
    mt, _, _ = sim_dynamic.tracking_metrics(res["traj"]["P"], res["traj"]["P_ideal"], goal,
                                            emb.L, sp.dt, sp.success_frac)
    return {"controller": "position servo (baseline)", "success": res["success"],
            "final_dist_m": res["final_dist_m"], "t_reach_s": res["t_reach_s"], "tracking": mt,
            "torque": {"limit_Nm": res["tau_doc_Nm"],
                       "max_abs_cmd_Nm": res["max_abs_tau_cmd_Nm"],
                       "max_abs_applied_Nm": res["max_abs_tau_applied_Nm"],
                       "peak_applied_ratio": res["peak_tau_ratio"],
                       "n_exceed_documented_limit_steps": res["n_tau_doc_exceed_steps"]},
            "n_pos_limit_viol_steps": res["n_pos_limit_viol_steps"],
            "n_vel_limit_viol_steps": res["n_vel_limit_viol_steps"],
            "max_vel_ratio": res["max_vel_ratio"], "n_contact_steps": res["n_contact_steps"],
            "max_out_of_plane_m": res["max_out_of_plane_m"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "picds.yaml"))
    ap.add_argument("--friction-scales", type=float, nargs="+", default=[1.0, 0.0])
    ap.add_argument("--no-figure", action="store_true")
    a = ap.parse_args()

    cfg = load_config(a.config)
    pc = cfg["pic"]
    params = PICParams.from_cfg(pc)
    ns = pc["null_space"]
    null_d_q = float(ns["d_q"]) if ns["enabled"] else 0.0
    clip = bool(pc["torque_limit"]["clip"])
    pulses = cfg["disturbance"]["pulses"] if cfg["disturbance"]["enabled"] else []

    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    report = {"config": a.config, "pic_params": to_jsonable(pc), "pulses": to_jsonable(pulses),
              "plants": {}}
    trajs = {}
    for fs in a.friction_scales:
        case = build_case(cfg, friction_scale=fs)
        if "q_start" not in report:
            report["q_start_rad"] = case.q_start.tolist()
            report["start_ik"] = {"pos_err_m": case.ik.pos_err, "sigma_min": case.ik.sigma_min,
                                  "margin_pct": case.ik.margin_pct}
            report["x0_n"] = case.x0_n.tolist()
            report["T_max_s"] = case.sp.T_max
            report["success_threshold_m"] = case.sp.success_frac * case.emb.L
        key = f"friction_{fs:g}"
        entry = {"dof_frictionloss_scale": fs,
                 "plant": "unmodified Menagerie model" if fs == 1.0 else "DIAGNOSTIC plant"}
        servo = sim_dynamic.run(case.robot, case.ds, case.emb, case.x0_n, case.rate, case.tgt,
                                case.sp, q_start=case.q_start, n_substeps=1)
        entry["servo"] = servo_summary(servo, case)
        pic = run(case, params, null_d_q=null_d_q, clip=clip)
        picp = run(case, params, null_d_q=null_d_q, clip=clip, pulses=pulses)
        dev = np.linalg.norm(picp["traj"]["P"] - pic["traj"]["P"], axis=1)
        picp["pulse_response"] = {
            "peak_dev_vs_undisturbed_m": float(dev.max()),
            "t_peak_dev_s": float(dev.argmax() * case.sp.dt),
            "peak_ext_force_N": float(np.linalg.norm(picp["traj"]["F_ext"], axis=1).max())}
        strip = lambda r: {k: v for k, v in r.items() if k != "traj"}  # noqa: E731
        entry["pic"], entry["pic_pulse"] = strip(pic), strip(picp)
        report["plants"][key] = entry
        trajs[key] = {"servo": servo["traj"], "pic": pic["traj"], "pic_pulse": picp["traj"]}
        for name, tr in trajs[key].items():
            np.savez_compressed(os.path.join(OUT, f"traj_{key}_{name}.npz"),
                                **{k: v for k, v in tr.items() if isinstance(v, np.ndarray)})
        print(f"[{key}] ({entry['plant']})")
        for name, r in (("servo", entry["servo"]), ("pic", entry["pic"]),
                        ("pic+pulse", entry["pic_pulse"])):
            print(f"  {name:10s} final {1e3 * r['final_dist_m']:8.3f} mm  success {r['success']!s:5}"
                  f"  t_reach {r['t_reach_s']}  geom_dev_max {1e3 * r['tracking']['geom_dev_max_m']:.2f} mm"
                  f"  peak tau/lim {r['torque']['peak_applied_ratio']:.3f}")
    with open(os.path.join(OUT, "metrics.json"), "w") as fh:
        json.dump(to_jsonable(report), fh, indent=1)
    if not a.no_figure:
        figure(trajs, case, os.path.join(FIG, "fig_overview.png"))


def figure(trajs, case, path) -> None:
    emb = case.emb
    col = {"servo": "tab:blue", "pic": "tab:orange", "pic_pulse": "tab:red"}
    lab = {"servo": "position servo", "pic": "PIC-DS", "pic_pulse": "PIC-DS + pulse"}
    keys = list(trajs)
    fig, axes = plt.subplots(len(keys), 4, figsize=(15, 3.4 * len(keys)), squeeze=False,
                             layout="constrained")
    thr = 1e3 * case.sp.success_frac * emb.L
    for r, key in enumerate(keys):
        fs = key.split("_")[1]
        for name, tr in trajs[key].items():
            xy = 1e3 * emb.L * (emb.to_plane(tr["P"])[0] - emb.anchor_n)
            axes[r, 0].plot(xy[:, 0], xy[:, 1], color=col[name], label=lab[name])
            axes[r, 1].semilogy(tr["t"], 1e3 * tr["dist_goal"] if "dist_goal" in tr else
                                1e3 * np.linalg.norm(tr["P"] - emb.to_world(np.zeros(2)), axis=1),
                                color=col[name])
            n_ctrl = len(tr["t"]) - 1                      # last PIC row is the final state only
            ratio = np.max(np.abs(tr["tau_applied"][:n_ctrl]) / tr["tau_limit"], axis=1)
            axes[r, 3].plot(tr["t"][:n_ctrl], ratio, color=col[name])
            if "s" in tr and name != "servo":
                axes[r, 2].plot(tr["t"], tr["s"], color=col[name])
        xy_id = 1e3 * emb.L * (emb.to_plane(trajs[key]["pic"]["P_ideal"])[0] - emb.anchor_n)
        axes[r, 0].plot(xy_id[:, 0], xy_id[:, 1], "k--", lw=1, label="ideal DS")
        axes[r, 0].plot(0, 0, "k*", ms=9)
        axes[r, 0].set(title=f"path in DS plane [mm], friction scale {fs}", aspect="equal")
        axes[r, 1].axhline(thr, color="gray", ls=":")
        axes[r, 1].set(title="distance to goal [mm] (dotted: success threshold)", xlabel="t [s]")
        axes[r, 2].set(title="tank s [J] (PIC)", xlabel="t [s]")
        axes[r, 3].set(title="max_j |tau_applied| / limit", xlabel="t [s]")
        for ax in axes[r]:
            ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("figure:", path)


if __name__ == "__main__":
    main()
