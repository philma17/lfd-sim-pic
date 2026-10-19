"""Kinematische Ausfuehrung eines DS auf einem Roboter (geschlossener Regelkreis).

Pro Schritt k (Zeit t = k*dt):
  p, R    = FK(q)
  x_n     = Projektion von p in die Normebene
  v_lin   = v_peak * E f(x_n) - k_perp * d_perp * n   (DS in der Ebene, Korrektur senkrecht)
  omega   = k_rot * Orientierungsfehler               (nur axis5/full6)
  q_dot   = IK(J_task[:, frei], [v_lin; omega])       (Lock = Spaltenentfernung)
  q      += q_dot * dt, danach Clamp an Gelenkpositionslimits (gezaehlt)

Referenz: ideales DS = Euler-Rollout in der Ebene, gleicher Start, gleiches dt.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import metrics as M
from .ik import (IKParams, IKResult, Target, multistart_ik, rot_error, solve_qdot, task_rows,
                 task_sigma_min)


@dataclass
class SimParams:
    dt: float = 0.002
    T_max: float = 30.0
    k_perp: float = 5.0
    k_rot: float = 5.0
    success_frac: float = 0.01      # Erfolg: Endabstand < success_frac * L
    n_starts: int = 32
    seed: int = 0
    start_criterion: str = "nearest_ref"
    start_min_margin_pct: float = 5.0
    clear_every: int = 10           # Hindernisabstand alle n Schritte pruefen
    ik: IKParams = field(default_factory=IKParams)


def run(robot, ds, emb, x0_n, rate: float, tgt: Target, sp: SimParams,
        free=None, q_ref=None, keep_traj: bool = False, q_start=None, q_null=None) -> dict:
    """q_start: feste Startkonfiguration (z. B. aus einer Roboteraufnahme) statt Multi-Start-IK.
    q_null: Ziel der Nullraumaufgabe (Standard: Gelenkmitte)."""
    n = robot.n
    free = np.ones(n, bool) if free is None else np.asarray(free, bool)
    N = int(round(sp.T_max / sp.dt))
    X_id = ds.rollout_ideal(x0_n, sp.dt, N, rate)
    P_id = emb.to_world(X_id)
    v_peak = rate * emb.L
    goal = emb.to_world(np.zeros(2))

    q_ref = robot.q_home if q_ref is None else np.asarray(q_ref, float)
    if q_start is None:
        ikr = multistart_ik(robot, P_id[0], tgt, q_ref, sp.n_starts, sp.seed,
                            sp.start_criterion, free=free, min_margin_pct=sp.start_min_margin_pct)
    else:
        q0 = np.asarray(q_start, float)
        p0, R0s = robot.fk(q0)
        _, _, oe0 = task_rows(robot.jac(), R0s, tgt, P_id[0] - p0, 1.0)
        ikr = IKResult(q0, float(np.linalg.norm(P_id[0] - p0)), float(oe0), True, 1, 0,
                       task_sigma_min(robot, q0, tgt, free),
                       float(robot.limit_margin_pct(q0).min()), True)
    q = ikr.q.copy()
    q_mid, q_rng = 0.5 * (robot.q_min + robot.q_max), robot.q_max - robot.q_min
    if q_null is not None:
        q_mid = np.asarray(q_null, float)
    check_obst = bool(getattr(robot, "obstacle_geoms", None))
    tags = sorted(robot.obstacle_tags) if check_obst else []
    clr = {t: {"links_m": np.inf, "link": None, "tool_m": np.inf, "t_first_link_s": None}
           for t in tags}

    P = np.empty((N + 1, 3))
    Xn = np.empty((N + 1, 2))
    Q = np.empty((N + 1, n))
    ori = np.zeros(N + 1)
    dperp = np.zeros(N + 1)
    smin = np.full(N, np.inf)
    manip = np.full(N, np.inf)
    n_vel_clip = n_pos_clip = n_damped = 0
    k_vel_clip = k_pos_clip = None          # erster Schritt mit Clip
    pos_clip_joints = set()
    vel_scale_max = 0.0
    R0 = None
    for k in range(N + 1):
        p, R = robot.fk(q)
        if R0 is None:
            R0 = R
        x_n, dp = emb.to_plane(p)
        P[k], Xn[k], Q[k], dperp[k] = p, x_n, q, dp
        if check_obst and (k % sp.clear_every == 0 or k == N):
            for t in tags:
                c = clr[t]
                cl = robot.clearance(robot.link_geoms, tag=t)
                j = int(np.argmin(cl))
                if cl[j] < c["links_m"]:
                    c["links_m"] = float(cl[j])
                    c["link"] = robot.model.body(int(robot.model.geom_bodyid[
                        robot.link_geoms[j]])).name
                if cl[j] < 0 and c["t_first_link_s"] is None:
                    c["t_first_link_s"] = k * sp.dt
                if robot.tool_geoms:
                    c["tool_m"] = min(c["tool_m"],
                                      float(robot.clearance(robot.tool_geoms, tag=t).min()))
            robot.fk(q)  # TCP-Puffer fuer den Jacobian wiederherstellen
        v_lin = v_peak * emb.vel_to_world(ds.f_n(x_n)) - sp.k_perp * dp * emb.normal
        J6 = robot.jac()
        Jt, vt, oe = task_rows(J6, R, tgt, v_lin, sp.k_rot)
        ori[k] = oe if tgt.mode != "pos" else float(np.linalg.norm(rot_error(R, R0)))
        if k == N:
            break
        qd, info = solve_qdot(Jt, vt, free, robot.qd_max, sp.ik, q, q_mid, q_rng)
        smin[k], manip[k] = info["sigma_min"], info["manip"]
        vel_scale_max = max(vel_scale_max, info["vel_scale"])
        if info["vel_scale"] > 1.0:
            n_vel_clip += 1
            k_vel_clip = k if k_vel_clip is None else k_vel_clip
        n_damped += info["lam2"] > 0.0
        q_new = q + qd * sp.dt
        q_cl = np.clip(q_new, robot.q_min, robot.q_max)
        hit = q_cl != q_new
        if np.any(hit):
            n_pos_clip += 1
            k_pos_clip = k if k_pos_clip is None else k_pos_clip
            pos_clip_joints.update(int(j) for j in np.flatnonzero(hit))
        q = q_cl

    L = emb.L
    dist_goal = np.linalg.norm(P - goal, axis=1)
    thr = sp.success_frac * L
    reached = np.flatnonzero(dist_goal < thr)
    dist_goal_id = np.linalg.norm(P_id - goal, axis=1)
    reached_id = np.flatnonzero(dist_goal_id < thr)
    ch_mean, ch_max = M.chamfer(P, P_id)
    margin = np.array([robot.limit_margin_pct(qq).min() for qq in Q])
    res = {
        "robot": robot.spec.name, "mode": tgt.mode,
        "locked": [int(i) for i in np.flatnonzero(~free)],
        "L_m": L, "v_peak_m_s": v_peak, "dt": sp.dt, "T_max": sp.T_max,
        "success": bool(dist_goal[-1] < thr),
        "final_dist_m": float(dist_goal[-1]),
        "t_reach_s": float(reached[0] * sp.dt) if len(reached) else None,
        "ideal_success": bool(dist_goal_id[-1] < thr),
        "ideal_t_reach_s": float(reached_id[0] * sp.dt) if len(reached_id) else None,
        "domain_viol_steps": int(np.sum(np.any(np.abs(Xn) > 1.0, axis=1))),
        "ideal_domain_viol_steps": int(np.sum(np.any(np.abs(X_id) > 1.0, axis=1))),
        "chamfer_mean_m": float(ch_mean), "chamfer_max_m": float(ch_max),
        "frechet_m": M.frechet_discrete(P, P_id),
        "dtw_mean_m": M.dtw_mean(P, P_id),
        "rmse_sync_m": M.rmse_time_sync(P, P_id),
        "maxdev_sync_m": M.max_dev_time_sync(P, P_id),
        "max_out_of_plane_m": float(np.max(np.abs(dperp))),
        "max_ori_err_rad": float(np.max(ori)),
        "min_sigma_min": float(np.min(smin)),
        "min_manip": float(np.min(manip)),
        "min_limit_margin_pct": float(np.min(margin)),
        "n_vel_clips": int(n_vel_clip), "n_pos_clips": int(n_pos_clip),
        "n_damped_steps": int(n_damped), "n_steps": N,
        "ik_start_pos_err_m": ikr.pos_err, "ik_start_ori_err_rad": ikr.ori_err,
        "ik_start_converged": ikr.converged, "ik_n_converged": ikr.n_converged,
        "ik_n_starts": ikr.n_starts,
        "start_sigma_min": ikr.sigma_min, "start_margin_pct": ikr.margin_pct,
        "start_margin_ok": ikr.margin_ok,
        "k_null": sp.ik.k_null,
        "vel_scale_max": float(vel_scale_max),
        "pos_clip_joints": sorted(pos_clip_joints),
        # Ort der ersten Ereignisse (Welt [m] und Normebene); None = trat nicht auf
        "t_first_vel_clip_s": None if k_vel_clip is None else k_vel_clip * sp.dt,
        "t_first_pos_clip_s": None if k_pos_clip is None else k_pos_clip * sp.dt,
        "p_first_clip": None if (k_vel_clip is None and k_pos_clip is None)
        else P[min(k for k in (k_vel_clip, k_pos_clip) if k is not None)].tolist(),
        "p_min_sigma": P[int(np.argmin(smin))].tolist(),
        "xn_min_sigma": Xn[int(np.argmin(smin))].tolist(),
    }
    if check_obst:
        res["clearance"] = clr     # je Hindernisgruppe: min. Abstand Armkoerper / Werkzeug
    if keep_traj:
        res["traj"] = {"P": P, "P_ideal": P_id, "Q": Q, "Xn": Xn, "X_ideal": X_id}
    return res
