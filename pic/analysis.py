"""Run standardisation and derived quantities.

Both execution modes (position servo, PIC-DS) return different trajectory dictionaries. This
module maps them to ONE schema (all arrays have N+1 rows, one per sampled state) so that plots,
CSV export and run comparison do not need to know the controller.

MEASURED (taken from the MuJoCo state, no numerical differentiation):
    t, q, qd, tcp, tcp_ref, goal, err_sync, err_geom, dist_goal, tau_cmd, tau_app, sat_any,
    controller variables (PIC: s, z, beta_R, ...; servo: q_des)
COMPUTED EXACTLY from measured values:
    tcp_vel = J_p(q) qd, sigma_min/sigma_max/manipulability of J_p(q) (kinematics only)
DERIVED BY NUMERICAL DIFFERENTIATION (prefix `d_`, NOT measured):
    acceleration and jerk, via a Savitzky-Golay polynomial filter applied to the (measured)
    velocity: acceleration = 1st SG derivative of velocity, jerk = 2nd SG derivative. Window
    and polynomial order are in cfg `analysis` (default 0.05 s, order 3). The filter smooths
    numerical noise but also attenuates real fast transients and is least reliable near the
    first/last samples and at stick-slip events; treat jerk as a qualitative indicator.
Torque rows exist for the N control steps only; the final state row has no torque and is NaN
(never zero), so a missing value cannot be mistaken for a measured one.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

ANALYSIS_DEFAULTS = {"sg_window_s": 0.05, "sg_polyorder": 3, "stall_speed_m_s": 1.0e-3,
                     "stall_window_s": 0.5,
                     "startup_exclude_s": 0.1,
                     "csv_stride": 5}


def analysis_params(cfg: dict) -> dict:
    a = {**ANALYSIS_DEFAULTS, **(cfg.get("analysis") or {})}
    unknown = sorted(set(cfg.get("analysis") or {}) - set(ANALYSIS_DEFAULTS))
    if unknown:
        raise ValueError(f"unknown analysis keys {unknown}; allowed {sorted(ANALYSIS_DEFAULTS)}")
    return a


def sg_derivative(x: np.ndarray, dt: float, deriv: int, window_s: float, order: int) -> np.ndarray:
    """deriv-th time derivative of x (rows = time) by Savitzky-Golay filtering."""
    w = int(round(window_s / dt))
    w += (w % 2 == 0)                       # odd window length
    w = max(w, order + 2 + ((order + 2) % 2 == 0))
    if x.shape[0] < w:
        raise ValueError(f"series ({x.shape[0]} samples) shorter than SG window ({w})")
    return savgol_filter(x, w, order, deriv=deriv, delta=dt, axis=0, mode="interp")


def jacobian_conditioning(robot, Q: np.ndarray):
    """Singular values of the 3xn position Jacobian along the measured joint trajectory.
    Kinematics only, so any Robot instance with the same geometry gives the same numbers."""
    n = Q.shape[0]
    smin, smax, manip = np.empty(n), np.empty(n), np.empty(n)
    J = np.empty((n, 3, Q.shape[1]))
    for k in range(n):
        robot.fk(Q[k])
        Jp = robot.jac()[:3]
        J[k] = Jp
        s = np.linalg.svd(Jp, compute_uv=False)
        smin[k], smax[k], manip[k] = s[-1], s[0], float(np.prod(s))
    return J, smin, smax, manip


def standardize(controller: str, res: dict, case, tau_limit: np.ndarray, an: dict) -> dict:
    """Map a run result to the common schema (+ derived quantities). `res` is the dict returned
    by pic.sim_pic.run or common.sim_dynamic.run."""
    tr = res["traj"]
    t = np.asarray(tr["t"])
    N = len(t) - 1
    dt = float(case.sp.dt)
    goal = case.emb.to_world(np.zeros(len(case.x0_n)))
    P, P_ref = np.asarray(tr["P"]), np.asarray(tr["P_ideal"])
    Q, QD = np.asarray(tr["Q"]), np.asarray(tr["QD"])
    out = {"t": t, "q": Q, "qd": QD, "tcp": P, "tcp_ref": P_ref, "goal": goal,
           "tau_limit": np.asarray(tau_limit, float)}

    if controller == "servo":
        from common import sim_dynamic
        _, e_sync, g_dev = sim_dynamic.tracking_metrics(P, P_ref, goal, case.emb.L, dt,
                                                        case.sp.success_frac)
        tau_c, tau_a = tr["tau_cmd"], tr["tau_applied"]
        out["q_des"] = np.asarray(tr["Q_des"])
        out["q_gap"] = out["q_des"] - Q
    else:
        e_sync, g_dev = np.asarray(tr["err_sync"]), np.asarray(tr["geom_dev"])
        tau_c, tau_a = tr["tau_cmd"][:N], tr["tau_applied"][:N]
        # s, KE and F_ext are evaluated for the final state too; the controller outputs are not
        # (no control step is executed there), so their last row is NaN instead of a fake 0
        for k in ("s", "KE"):
            out[f"pic_{k}"] = np.asarray(tr[k])
        for k in ("z", "alpha", "beta_s", "beta_R", "p_diss"):
            out[f"pic_{k}"] = _pad(np.asarray(tr[k])[:N, None], N)[:, 0]
        out["F_task"] = _pad(np.asarray(tr["F"])[:N], N)
        out["F_ext"] = np.asarray(tr["F_ext"])
        out["tau_grav"] = _pad(tr["tau_grav"][:N], N)
    out["err_sync"], out["err_geom"] = e_sync, g_dev
    out["dist_goal"] = np.linalg.norm(P - goal, axis=1)
    out["tau_cmd"], out["tau_app"] = _pad(tau_c, N), _pad(tau_a, N)
    lim = out["tau_limit"]
    # saturation: the commanded torque exceeded the limit (applied is then clamped to the limit)
    sat = np.abs(out["tau_cmd"]) > lim + 1e-9
    sat[N] = False
    out["sat_joint"] = sat.astype(np.int8)
    out["sat_any"] = sat.any(axis=1).astype(np.int8)

    J, smin, smax, manip = jacobian_conditioning(case.robot_nominal or case.robot, Q)
    out["sigma_min_Jp"], out["sigma_max_Jp"], out["manipulability_Jp"] = smin, smax, manip
    out["tcp_vel"] = np.einsum("kij,kj->ki", J, QD)         # exact: J_p(q) qd

    w, o = an["sg_window_s"], int(an["sg_polyorder"])
    out["d_tcp_acc"] = sg_derivative(out["tcp_vel"], dt, 1, w, o)
    out["d_tcp_jerk"] = sg_derivative(out["tcp_vel"], dt, 2, w, o)
    out["d_qdd"] = sg_derivative(QD, dt, 1, w, o)
    out["d_qddd"] = sg_derivative(QD, dt, 2, w, o)
    return out


def max_ratio(tau: np.ndarray, lim: np.ndarray) -> np.ndarray:
    """max_j |tau_j| / limit_j per row; rows without torque (all NaN) stay NaN."""
    r = np.abs(tau) / lim
    out = np.full(r.shape[0], np.nan)
    ok = ~np.isnan(r).all(axis=1)
    out[ok] = r[ok].max(axis=1)
    return out


def _pad(a: np.ndarray, N: int) -> np.ndarray:
    """(N, n) -> (N+1, n) with a NaN row for the final state (no torque is computed there)."""
    a = np.asarray(a, float)
    out = np.full((N + 1, a.shape[1]), np.nan)
    out[:N] = a[:N]
    return out


def failure_class(res: dict, arr: dict, an: dict, dt: float) -> str:
    """converged | stalled | timeout_moving | nonfinite. `stalled` = not converged and the TCP
    speed stayed below an['stall_speed_m_s'] during the last an['stall_window_s'] (a fixed,
    documented threshold, not tuned per run)."""
    if not res.get("finite", True) or not np.all(np.isfinite(arr["q"])):
        return "nonfinite"
    if res["success"]:
        return "converged"
    n = max(1, int(round(an["stall_window_s"] / dt)))
    v = np.linalg.norm(arr["tcp_vel"][-n:], axis=1)
    return "stalled" if v.max() < an["stall_speed_m_s"] else "timeout_moving"


def summarize(controller: str, res: dict, arr: dict, case, an: dict) -> dict:
    """Curated, flat metrics for report/CSV comparison (all from saved arrays)."""
    dt = float(case.sp.dt)
    N = len(arr["t"]) - 1
    cmd, app, lim = arr["tau_cmd"][:N], arr["tau_app"][:N], arr["tau_limit"]
    speed = np.linalg.norm(arr["tcp_vel"], axis=1)
    acc = np.linalg.norm(arr["d_tcp_acc"], axis=1)
    jerk = np.linalg.norm(arr["d_tcp_jerk"], axis=1)
    sat_steps = int(arr["sat_any"].sum())
    k0 = int(round(an["startup_exclude_s"] / dt))
    s = {
        "failure_class": failure_class(res, arr, an, dt),
        "converged": bool(res["success"]),
        "final_dist_m": float(res["final_dist_m"]),
        "success_threshold_m": float(case.sp.success_frac * case.emb.L),
        "t_reach_s": res["t_reach_s"],
        "sim_duration_s": float(arr["t"][-1]),
        "n_steps": N, "dt_s": dt,
        "rmse_sync_m": float(np.sqrt(np.mean(arr["err_sync"] ** 2))),
        "max_err_sync_m": float(arr["err_sync"].max()),
        "geom_dev_max_m": float(arr["err_geom"].max()),
        "max_out_of_plane_m": float(res["max_out_of_plane_m"]),
        "tcp_speed_peak_m_s": float(speed.max()),
        "d_tcp_acc_peak_m_s2": float(acc.max()),
        "d_tcp_jerk_peak_m_s3": float(jerk.max()),
        # the first samples contain the start-up transient (servo sag step, SG edge effects)
        "startup_exclude_s": an["startup_exclude_s"],
        "d_tcp_acc_peak_after_startup_m_s2": float(acc[k0:].max()),
        "d_tcp_jerk_peak_after_startup_m_s3": float(jerk[k0:].max()),
        "peak_qd_ratio_to_limit": float(res["max_vel_ratio"]),
        "peak_cmd_torque_ratio": float(np.max(np.abs(cmd) / lim)),
        "peak_applied_torque_ratio": float(np.max(np.abs(app) / lim)),
        "n_saturated_steps": sat_steps,
        "saturated_fraction": sat_steps / N,
        "saturated_steps_per_joint": arr["sat_joint"].sum(axis=0).tolist(),
        "n_pos_limit_viol_steps": int(res["n_pos_limit_viol_steps"]),
        "n_vel_limit_viol_steps": int(res["n_vel_limit_viol_steps"]),
        "sigma_min_Jp_min": float(arr["sigma_min_Jp"].min()),
        "manipulability_Jp_min": float(arr["manipulability_Jp"].min()),
        "contact_detected": res.get("first_contact") is not None,
    }
    if controller == "servo":
        s["max_abs_servo_gap_rad"] = float(res["max_abs_gap_rad"])
    else:
        s.update({"tank_s_min_J": res["tank"]["s_min"], "tank_s_max_J": res["tank"]["s_max_reached"],
                  "tank_clamp_steps": res["tank"]["n_clamp_steps"],
                  "min_beta_R": res["tank"]["min_beta_R"],
                  "W_dW_max_J": res["energy"]["W_dW_max_J"]})
    return s
