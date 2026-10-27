"""Stall diagnosis of PIC-DS on the nominal FR3 (friction x1): why does the arm not move, and
which lambda1 is needed?  Writes results/summary/stall_diagnosis.{json,md}.

    python -B experiments/diagnose_stall.py [--robot FR3] [--friction 1.0] [--out-dir DIR]

Nothing is tuned here: lambda2, lambda3 and all other parameters stay at configs/picds.yaml;
only lambda1 is swept, as requested. Predicted threshold (stiction at t=0, see report):
    tau_drive = Jp^T F,  F = lambda1 * beta_R * f  (xd = 0)   (gravity is compensated)
    joint i breaks away if |tau_drive_i| > frictionloss_i
    lambda1_min = min_i  frictionloss_i / |(Jp^T f)_i|   (first joint breaks away)
    lambda1_all = max_i  frictionloss_i / |(Jp^T f)_i|   (every joint with a component breaks away)
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.dont_write_bytecode = True

import mujoco
import numpy as np

from common.scenario import load_config
from pic.controller import PICController, PICParams
from pic.sim_pic import build_case, run

OUT = os.path.join(ROOT, "results", "summary")


def start_state(case, params):
    """Quantities at t = 0 (qd = 0): f, Jp, F, drive torque, stiction ratio, tank."""
    r, emb, sp = case.robot, case.emb, case.sp
    q0 = case.q_start
    p, _ = r.fk(q0)
    Jp = r.jac()[:3]
    x_n, dp = emb.to_plane(p)
    v_peak = case.rate * emb.L
    f = v_peak * emb.vel_to_world(case.ds.f_n(x_n)) - sp.k_perp * dp * emb.normal
    ctrl = PICController(params)
    out = ctrl.step(np.zeros(3), f, sp.dt)
    fl = r.model.dof_frictionloss[r.dadr]
    tau_unit = Jp.T @ f                                   # per unit lambda1 (beta_R = 1)
    tau_drive = Jp.T @ out["F"]
    comp = np.abs(tau_unit) > 1e-12
    thr = np.where(comp, fl / np.maximum(np.abs(tau_unit), 1e-300), np.inf)
    return {"q0": q0, "f": f, "f_norm": float(np.linalg.norm(f)), "F": out["F"], "beta_R": out["beta_R"],
            "s": out["s"], "tau_drive": tau_drive, "frictionloss": fl, "ratio": np.abs(tau_drive) / np.maximum(fl, 1e-300),
            "lam1_joint": thr, "lam1_min": float(np.min(thr[np.isfinite(thr)])) if np.any(comp) else None,
            "lam1_all": float(np.max(thr[np.isfinite(thr)])) if np.any(comp) else None,
            "tau_limit": case.tau_limit, "Jp": Jp}


def f_at(case, P):
    """DS value f (world, m/s) at TCP position P, exactly as in pic.sim_pic.run."""
    emb, sp = case.emb, case.sp
    x_n, dp = emb.to_plane(P)
    return case.rate * emb.L * emb.vel_to_world(case.ds.f_n(x_n)) - sp.k_perp * dp * emb.normal


def stability_along(case, res, params, stride=25):
    """Explicit-damping stability. The controller applies -D xd computed from the previous state,
    i.e. explicitly, so the one-step map of the task-space velocity is
        xd+ = (I - dt * Lambda_x^-1 D) xd + ...,   Lambda_x^-1 = Jp M^-1 Jp^T
    which is stable if rho = max |eig(dt * Lambda_x^-1 D)| < 2 (necessary and sufficient for the
    scalar/symmetric-similar case; a necessary-condition check here, not a proof). Evaluated along
    the measured path with the damping basis oriented by the true f."""
    r = case.robot
    m = r.model
    Q, P = res["traj"]["Q"], res["traj"]["P"]
    d = mujoco.MjData(m)
    ctrl = PICController(params)
    out = []
    for k in range(0, len(Q) - 1, stride):
        d.qpos[r.qadr] = Q[k]
        mujoco.mj_forward(m, d)
        M = np.zeros((m.nv, m.nv))
        mujoco.mj_fullM(m, d, M)
        r.fk(Q[k])
        Jp = r.jac()[:3]
        Minv_x = Jp @ np.linalg.solve(M, Jp.T)
        f = f_at(case, P[k])
        Qb = ctrl.basis.update(f)
        D = (Qb * np.asarray(params.lam, float)) @ Qb.T
        out.append(float(np.max(np.abs(np.linalg.eigvals(case.sp.dt * Minv_x @ D)))))
    return float(np.max(out)), float(np.median(out))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="FR3")
    ap.add_argument("--friction", type=float, default=1.0)
    ap.add_argument("--lambda1", type=float, nargs="+", default=[20, 40, 80, 160, 320, 640, 1280])   # 640, 1280: EXTENSION beyond the requested sweep
    ap.add_argument("--out-dir", default=OUT)
    a = ap.parse_args(argv)

    cfg = load_config("configs/picds.yaml")
    cfg["robot"] = a.robot
    cfg["plant"] = {"dof_frictionloss_scale": a.friction}
    pc = cfg["pic"]
    null_d_q = float(pc["null_space"]["d_q"])
    base = PICParams.from_cfg(pc)

    report = {"robot": a.robot, "friction_scale": a.friction, "paper_lambda": list(base.lam), "s0": base.s0,
              "s_max": base.s_max, "delta_s": base.delta_s_frac * base.s_max, "delta_z": base.delta_z,
              "eta": base.eta}
    case0 = build_case(cfg)
    st = start_state(case0, base)
    report["start"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in st.items() if k != "Jp"}
    print(f"|f| at start {st['f_norm']:.4g} m/s; lambda1* min {st['lam1_min']:.4g}, all {st['lam1_all']:.4g}")
    print("tau_drive/frictionloss per joint:", np.round(st["ratio"], 3).tolist())

    rows = []
    for lam1 in a.lambda1:
        params = copy.deepcopy(base)
        params.lam = (float(lam1), base.lam[1], base.lam[2])
        case = build_case(cfg)
        res = run(case, params, null_d_q=null_d_q)
        tr = res["traj"]
        travel = float(np.sum(np.linalg.norm(np.diff(tr["P"], axis=0), axis=1)))
        qd = np.abs(tr["QD"])
        moved1s = (qd[: int(1.0 / case.sp.dt)].max(axis=0) > 1e-4).tolist()
        rho_max, rho_med = stability_along(case, res, params)
        # creep vs sliding: |xd| / |f| over the first 3 s (1 = DS tracked freely, << 1 = held by friction)
        t3 = int(3.0 / case.sp.dt)
        f3 = np.array([np.linalg.norm(f_at(case, p)) for p in tr["P"][:t3:10]])
        v3 = np.linalg.norm(tr["xd"][:t3:10], axis=1)
        track_ratio = float(np.median(v3 / np.maximum(f3, 1e-12)))
        v_end = float(np.linalg.norm(tr["xd"][-250:], axis=1).max())
        rows.append({
            "lambda1": float(lam1), "success": res["success"], "final_dist_m": res["final_dist_m"],
            "t_reach_s": res["t_reach_s"], "tcp_travel_m": travel,
            "joints_moving_in_first_s": moved1s,
            "max_abs_qd_rad_s": float(qd.max()),
            "tank_s_min": res["tank"]["s_min"], "tank_s_max": res["tank"]["s_max_reached"],
            "tank_clamp_steps": res["tank"]["n_clamp_steps"],
            "min_beta_R": res["tank"]["min_beta_R"],
            "n_clip_steps": res["torque"]["n_clip_steps"],
            "peak_cmd_ratio": res["torque"]["peak_cmd_ratio"],
            "n_dW_positive_steps": res["energy"]["n_dW_positive_steps"],
            "sum_positive_dW_J": res["energy"]["sum_positive_dW_J"], "W_dW_max_J": res["energy"]["W_dW_max_J"],
            "stability_rho_max": rho_max, "stability_rho_median": rho_med,
            "tracking_ratio_first3s": track_ratio, "tcp_speed_last_0.5s_max_m_s": v_end,
        })
        print(f"lambda1 {lam1:6.0f}: success {res['success']!s:5} final {1e3 * res['final_dist_m']:9.3f} mm  "
              f"travel {1e3 * travel:8.2f} mm  clip {res['torque']['n_clip_steps']:5d}  rho_max {rho_max:.3f}  "
              f"dW+ steps {res['energy']['n_dW_positive_steps']}  |xd|/|f| {track_ratio:.3f}  "
              f"v_end {v_end:.2e}  min beta_R {res['tank']['min_beta_R']:.3f}")
    report["sweep"] = rows

    # (d) energy check for the friction-free reference (paper lambda)
    cfg0 = copy.deepcopy(cfg)
    cfg0["plant"] = {"dof_frictionloss_scale": 0.0}
    case = build_case(cfg0)
    res = run(case, base, null_d_q=null_d_q)
    report["frictionless_reference"] = {
        "success": res["success"], "final_dist_m": res["final_dist_m"], "t_reach_s": res["t_reach_s"],
        "n_dW_positive_steps": res["energy"]["n_dW_positive_steps"],
        "sum_positive_dW_J": res["energy"]["sum_positive_dW_J"], "W_dW_max_J": res["energy"]["W_dW_max_J"],
        "tank_s_min": res["tank"]["s_min"], "tank_s_max": res["tank"]["s_max_reached"]}
    print("frictionless reference:", report["frictionless_reference"])

    # (c2) same sweep with T_max = 40 s: separates "too slow for 10 s" from "does not converge".
    # DIAGNOSTIC duration change only; no controller parameter is touched.
    long_rows = []
    cfg40 = copy.deepcopy(cfg)
    cfg40["sim"]["T_max"] = 40.0
    for lam1 in a.lambda1:
        if lam1 < 160:
            continue
        params = copy.deepcopy(base)
        params.lam = (float(lam1), base.lam[1], base.lam[2])
        res = run(build_case(cfg40), params, null_d_q=null_d_q)
        v_end = float(np.linalg.norm(res["traj"]["xd"][-1000:], axis=1).max())
        long_rows.append({"lambda1": float(lam1), "T_max_s": 40.0, "success": res["success"],
                          "final_dist_m": res["final_dist_m"], "t_reach_s": res["t_reach_s"],
                          "tcp_speed_last_2s_max_m_s": v_end, "n_clip_steps": res["torque"]["n_clip_steps"],
                          "tank_s_min": res["tank"]["s_min"], "min_beta_R": res["tank"]["min_beta_R"]})
        print(f"T_max 40 lambda1 {lam1:6.0f}: success {res['success']!s:5} final {1e3 * res['final_dist_m']:8.2f} mm "
              f"v_last2s {v_end:.2e}")
    report["sweep_T40"] = long_rows

    os.makedirs(a.out_dir, exist_ok=True)
    fn = "stall_diagnosis.json" if a.robot == "FR3" else f"stall_diagnosis_{a.robot}.json"
    with open(os.path.join(a.out_dir, fn), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print("wrote", os.path.join(a.out_dir, fn))
    return 0


if __name__ == "__main__":
    sys.exit(main())
