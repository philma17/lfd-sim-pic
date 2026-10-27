"""MuJoCo execution of PIC-DS on the FR3 (3D TCP position task, direct joint-torque interface).

Interface (nothing is written to the MJCF; all changes are in memory and restored afterwards):
  * actuators are switched off with mjDSBL_ACTUATION, torques go to data.qfrc_applied
  * dt = model.opt.timestep (1 control step = 1 mj_step), state is the MEASURED qpos/qvel
  * manual torque clipping; unclipped (tau_cmd) and clipped (tau_applied) are logged separately
  * external TCP force pulses are applied through data.xfrc_applied

Per step k (t = k dt):
    q, qd  = measured joint state                    x = FK(q), Jp = position Jacobian
    xd     = Jp qd                                   f = v_peak E f_n(x_n) - k_perp d_perp n
    F      = PIC-DS force law (pic.controller)       [P] Eq. 13, tank [P] Eq. 10
    tau    = g(q) + Jp^T F - d_q (I - Jp^+ Jp) qd    (null-space term only if null(Jp) != {})
    g(q)   = common.sim_dynamic.gravity_torque (qfrc_bias at qd = 0 of the NOMINAL model; here
             the nominal model equals the simulated model, so the compensation is exact)

No orientation control (position task only): the null space of Jp (4 dim for the 7-DoF FR3)
contains the TCP orientation and is only damped.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

from dataclasses import dataclass

import mujoco
import numpy as np

from common.ds import PlanarEmbedding
from common.ik import IKResult, Target, multistart_ik, task_rows, task_sigma_min
from common.robots import ROBOTS, Robot
from common.scenario import build_ds, sim_params
from common.sim_dynamic import (TAU_MAX_DOC, check_dims, gravity_torque, ideal_rollout,
                                tracking_metrics)

from .controller import PICController, PICParams
from .plant import apply_plant, resolve_plant, torque_limit


@dataclass
class Case:
    """Everything that defines the baseline FR3 / LASA Sine / pos cell."""
    cfg: dict
    robot: Robot
    ds: object
    sc: object
    rate: float
    sp: object
    emb: PlanarEmbedding
    tgt: Target
    x0_n: np.ndarray
    q_start: np.ndarray
    ik: object
    # --- added for configurable experiments (all have defaults, older callers unaffected) ---
    robot_nominal: Robot | None = None      # unmodified model: source of the g(q) compensation
    tau_limit: np.ndarray | None = None     # torque limit used for clipping/reporting [Nm]
    speed_factor: float = 1.0               # alpha in  x_dot = alpha * f(x)
    plant_report: dict | None = None        # requested + applied plant parameters


def build_case(cfg: dict, friction_scale: float | None = None,
               speed_factor: float | None = None) -> Case:
    """Same start procedure as common.sim.run (multi-start IK on the ideal-rollout start point),
    so q_start is the baseline start configuration of the FR3 pos cell.

    Configurable (all optional, defaults reproduce the historical cell):
      cfg["ds_speed_factor"]  alpha > 0: x_dot = alpha * f(x). Applied to the DS time rate, which
                              scales BOTH the commanded velocity (v_peak = rate*L) and the ideal
                              reference rollout, so geometry is unchanged and only timing scales.
      cfg["plant"]            see pic.plant (friction, damping, gravity, mass, inertia, torque limit)
      cfg["task"]["q_start_rad"]  explicit start configuration instead of multi-start IK
    friction_scale / speed_factor arguments override the cfg values (kept for older callers).
    1.0 friction = unmodified Menagerie plant (the baseline); anything else is a DIAGNOSTIC plant."""
    if cfg["robot"] != "FR3":
        raise NotImplementedError("first PIC step supports the FR3 only")
    if cfg["task"]["mode"] != "pos":
        raise NotImplementedError("first PIC step: position task only (no orientation control)")
    plant = dict(cfg.get("plant") or {})
    if friction_scale is not None:
        plant["dof_frictionloss_scale"] = float(friction_scale)
    plant = resolve_plant(plant)
    robot = Robot(ROBOTS["FR3"])
    plant_report = apply_plant(robot, plant)
    robot_nominal = Robot(ROBOTS["FR3"])               # never modified
    alpha = float(cfg.get("ds_speed_factor", 1.0) if speed_factor is None else speed_factor)
    if not (np.isfinite(alpha) and alpha > 0):
        raise ValueError(f"ds_speed_factor must be finite and > 0, got {alpha}")
    ds, sc, rate = build_ds(cfg)
    rate = rate * alpha
    sp = sim_params(cfg)
    if not np.isclose(sp.dt, robot.model.opt.timestep):
        raise ValueError(f"sim.dt {sp.dt} != model timestep {robot.model.opt.timestep}")
    t = cfg["task"]
    emb = PlanarEmbedding(np.array(t["center"], float), t["plane"]["e1"], t["plane"]["e2"],
                          float(t["L_m"]), t["anchor"])
    if t.get("start", "demo_mean") != "demo_mean":
        raise NotImplementedError("only start: demo_mean (x0 = mean demonstration start)")
    x0 = np.asarray(sc.x0_n, float)
    tgt = Target("pos")
    p0 = emb.to_world(x0)
    if t.get("q_start_rad") is not None:
        q0 = np.asarray(t["q_start_rad"], float)
        if q0.shape != (robot.n,):
            raise ValueError(f"task.q_start_rad needs {robot.n} values")
        if np.any(q0 < robot.q_min) or np.any(q0 > robot.q_max):
            raise ValueError("task.q_start_rad violates the joint position limits")
        p_fk, R_fk = robot.fk(q0)
        _, _, oe0 = task_rows(robot.jac(), R_fk, tgt, p0 - p_fk, 1.0)
        ikr = IKResult(q0, float(np.linalg.norm(p0 - p_fk)), float(oe0), True, 1, 0,
                       task_sigma_min(robot, q0, tgt, np.ones(robot.n, bool)),
                       float(robot.limit_margin_pct(q0).min()), True)
    else:
        ikr = multistart_ik(robot, p0, tgt, robot.q_home, sp.n_starts, sp.seed,
                            sp.start_criterion, free=None,
                            min_margin_pct=sp.start_min_margin_pct)
        if not ikr.converged:
            raise RuntimeError(f"start IK did not converge (pos err {ikr.pos_err:.3e} m)")
    return Case(cfg, robot, ds, sc, rate, sp, emb, tgt, x0, ikr.q.copy(), ikr,
                robot_nominal=robot_nominal,
                tau_limit=torque_limit(robot.spec.name, plant["torque_limit_scale"]),
                speed_factor=alpha, plant_report=plant_report)


def null_projector(Jp: np.ndarray, rtol: float = 1e-6):
    """Orthogonal projector onto null(Jp) and its dimension. Dimension 0 -> zero matrix (the
    null-space term is blocked, as required for an empty null space)."""
    _, S, Vt = np.linalg.svd(Jp)
    rank = int(np.sum(S > rtol * S[0]))
    n = Jp.shape[1]
    if rank >= n:
        return np.zeros((n, n)), 0
    V0 = Vt[rank:].T
    return V0 @ V0.T, n - rank


def apply_tcp_force(m, d, body: int, p_tcp: np.ndarray, force: np.ndarray) -> None:
    """Point force at the TCP through xfrc_applied. MuJoCo applies xfrc_applied at the body's
    centre of mass, so the moment (p_tcp - xipos) x F is added to make it a TCP point force
    (equivalent to the joint torque Jp^T F, verified in tests/test_sim_pic.py)."""
    d.xfrc_applied[body, :3] = force
    d.xfrc_applied[body, 3:] = np.cross(p_tcp - d.xipos[body], force)


def run(case: Case, params: PICParams, *, null_d_q: float = 0.0, tau_limit=None,
        clip: bool = True, pulses=(), use_f: bool = True, v0=None, T_max=None,
        q_start=None) -> dict:
    """Closed-loop PIC-DS run. use_f=False sets f = 0 (damping + gravity compensation only).
    v0: initial TCP velocity [m/s] (world), realised with the pseudo-inverse of Jp.
    pulses: iterable of dicts {t_start_s, duration_s, force_N[3]} (world frame, at the TCP)."""
    robot, ds, emb, sp = case.robot, case.ds, case.emb, case.sp
    m = robot.model
    n = robot.n
    if m.nv != n:
        raise NotImplementedError("model has DoF outside the arm")
    dt = sp.dt
    if not np.isclose(dt, m.opt.timestep):
        raise ValueError(f"controller dt {dt} != model timestep {m.opt.timestep}")
    T = sp.T_max if T_max is None else float(T_max)
    N = int(round(T / dt))
    x0_n = case.x0_n
    dim = check_dims(ds, emb, x0_n)
    X_id = ideal_rollout(ds, x0_n, dt, N, case.rate)
    P_id = emb.to_world(X_id)
    v_peak = case.rate * emb.L
    goal = emb.to_world(np.zeros(dim))
    if tau_limit is not None:
        tau_lim = np.asarray(tau_limit, float)
    elif case.tau_limit is not None:
        tau_lim = np.asarray(case.tau_limit, float)
    else:
        tau_lim = TAU_MAX_DOC[robot.spec.name][0]
    g_robot = case.robot_nominal if case.robot_nominal is not None else robot
    pulses = list(pulses)
    q0 = np.asarray(case.q_start if q_start is None else q_start, float)

    ctrl = PICController(params)
    d = mujoco.MjData(m)
    d.qpos[robot.qadr] = q0
    d.qvel[:] = 0.0
    if v0 is not None:
        robot.fk(q0)
        d.qvel[robot.dadr] = np.linalg.pinv(robot.jac()[:3]) @ np.asarray(v0, float)

    keys = ("P", "Xn", "Q", "QD", "XD", "F", "F_ext", "TAU_CMD", "TAU_APP", "TAU_G")
    shapes = {"P": 3, "Xn": dim, "Q": n, "QD": n, "XD": 3, "F": 3, "F_ext": 3, "TAU_CMD": n,
              "TAU_APP": n, "TAU_G": n}
    L = {k: np.zeros((N + 1, shapes[k])) for k in keys}
    sc_keys = ("s", "z", "alpha", "beta_s", "beta_R", "p_diss", "KE", "P_ext", "dperp",
               "f_norm", "null_dim")
    S = {k: np.zeros(N + 1) for k in sc_keys}

    old_flags = int(m.opt.disableflags)
    m.opt.disableflags = old_flags | int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
    tmp = np.zeros(m.nv)
    first_contact = None
    try:
        for k in range(N + 1):
            t = k * dt
            mujoco.mj_forward(m, d)                      # fresh M(q), xipos at the measured state
            q = d.qpos[robot.qadr].copy()
            qd = d.qvel[robot.dadr].copy()
            p, _ = robot.fk(q)
            Jp = robot.jac()[:3]
            xd = Jp @ qd
            x_n, dp = emb.to_plane(p)
            f = (v_peak * emb.vel_to_world(ds.f_n(x_n)) - sp.k_perp * dp * emb.normal
                 if use_f else np.zeros(3))
            mujoco.mj_mulM(m, d, tmp, d.qvel)
            F_ext = np.zeros(3)
            for pu in pulses:
                if pu["t_start_s"] <= t < pu["t_start_s"] + pu["duration_s"]:
                    F_ext = F_ext + np.asarray(pu["force_N"], float)
            L["P"][k], L["Xn"][k], L["Q"][k], L["QD"][k], L["XD"][k] = p, x_n, q, qd, xd
            L["F_ext"][k] = F_ext
            S["KE"][k] = 0.5 * float(d.qvel @ tmp)
            S["P_ext"][k] = float(F_ext @ xd)
            S["dperp"][k] = dp
            S["f_norm"][k] = float(np.linalg.norm(f))
            if k == N:
                S["s"][k] = ctrl.s
                break
            out = ctrl.step(xd, f, dt)
            g = gravity_torque(g_robot, q)       # compensation uses the NOMINAL model
            Nproj, nd = null_projector(Jp)
            tau_null = -null_d_q * (Nproj @ qd) if (null_d_q > 0 and nd > 0) else np.zeros(n)
            tau_cmd = g + Jp.T @ out["F"] + tau_null
            tau_app = np.clip(tau_cmd, -tau_lim, tau_lim) if clip else tau_cmd.copy()
            d.qfrc_applied[robot.dadr] = tau_app
            d.xfrc_applied[:] = 0.0
            if np.any(F_ext):
                apply_tcp_force(m, d, robot.site_body, p, F_ext)
            L["F"][k], L["TAU_CMD"][k], L["TAU_APP"][k], L["TAU_G"][k] = out["F"], tau_cmd, tau_app, g
            for key in ("s", "z", "alpha", "beta_s", "beta_R", "p_diss"):
                S[key][k] = out[key]
            S["null_dim"][k] = nd
            mujoco.mj_step(m, d)
            if d.ncon and first_contact is None:
                c = d.contact[0]
                first_contact = {"t_s": (k + 1) * dt,
                                 "geoms": [m.body(int(m.geom_bodyid[c.geom1])).name,
                                           m.body(int(m.geom_bodyid[c.geom2])).name]}
            if not np.all(np.isfinite(d.qpos)):
                raise FloatingPointError(f"non-finite qpos at step {k}")
    finally:
        m.opt.disableflags = old_flags                   # leave the shared model as we found it
        d.qfrc_applied[:] = 0.0
        d.xfrc_applied[:] = 0.0

    # ---- metrics ---------------------------------------------------------
    P, Q, QD = L["P"], L["Q"], L["QD"]
    mt, e_sync, g_dev = tracking_metrics(P, P_id, goal, emb.L, dt, sp.success_frac)
    dist = np.linalg.norm(P - goal, axis=1)
    reached = np.flatnonzero(dist < sp.success_frac * emb.L)
    TC, TA = L["TAU_CMD"][:N], L["TAU_APP"][:N]
    clipped = np.abs(TC) > tau_lim + 1e-12
    pos_exc = np.maximum(robot.q_min - Q, Q - robot.q_max)
    vel_ratio = np.abs(QD) / robot.qd_max
    W = S["KE"] + S["s"]                                 # storage W = KE + s (f_C = 0, [P] Thm 1)
    dW = W[1:] - W[:-1] - S["P_ext"][:-1] * dt           # per-step change minus external work
    res = {
        "robot": robot.spec.name, "controller": "PIC-DS", "mode": "pos", "n_steps": N, "dt": dt,
        "T_max": T, "L_m": emb.L, "v_peak_m_s": v_peak, "use_f": bool(use_f),
        "interface": "mjDSBL_ACTUATION + data.qfrc_applied (joint torque)",
        "params": {"lambda": list(params.lam), "s_max": params.s_max, "s0": params.s0,
                   "delta_s": params.delta_s_frac * params.s_max, "delta_z": params.delta_z,
                   "eta": params.eta, "step": params.step, "decomposition": params.decomposition,
                   "null_d_q": null_d_q, "clip": bool(clip)},
        "success": bool(dist[-1] < sp.success_frac * emb.L), "final_dist_m": float(dist[-1]),
        "success_threshold_m": float(sp.success_frac * emb.L),
        "t_reach_s": float(reached[0] * dt) if len(reached) else None,
        "max_dist_after_pulse_m": None,
        "tracking": mt,
        "max_out_of_plane_m": float(np.max(np.abs(S["dperp"]))),
        # tank
        "tank": {"s_min": float(S["s"].min()), "s_max_reached": float(S["s"].max()),
                 "s_final": float(S["s"][-1]), "n_clamp_steps": int(ctrl.n_clamp),
                 "max_excursion_J": float(ctrl.max_excursion),
                 "min_beta_R": float(S["beta_R"][:N].min()),
                 "n_beta_R_below_1_steps": int(np.sum(S["beta_R"][:N] < 1.0 - 1e-12)),
                 "n_basis_kept_steps": int(ctrl.basis.n_kept)},
        # torques: unclipped and clipped are separate
        "torque": {"limit_Nm": tau_lim.tolist(), "clip_enabled": bool(clip),
                   "max_abs_cmd_Nm": np.abs(TC).max(0).tolist(),
                   "max_abs_applied_Nm": np.abs(TA).max(0).tolist(),
                   "peak_cmd_ratio": float(np.max(np.abs(TC) / tau_lim)),
                   "peak_applied_ratio": float(np.max(np.abs(TA) / tau_lim)),
                   "n_clip_steps": int(np.sum(np.any(clipped, axis=1))),
                   "n_clip_per_joint": np.sum(clipped, axis=0).tolist()},
        # actual state vs limits (soft constraints in MuJoCo: measured, not enforced)
        "n_pos_limit_viol_steps": int(np.sum(np.any(pos_exc > 0, axis=1))),
        "max_pos_limit_exceed_rad": float(max(0.0, pos_exc.max())),
        "n_vel_limit_viol_steps": int(np.sum(np.any(vel_ratio > 1.0, axis=1))),
        "max_vel_ratio": float(vel_ratio.max()),
        "min_limit_margin_pct": float(min(robot.limit_margin_pct(qq).min() for qq in Q)),
        "null_dim_min_max": [int(S["null_dim"][:N].min()), int(S["null_dim"][:N].max())],
        "first_contact": first_contact,
        # energy bookkeeping (diagnostic of Eq. 17 along the discrete trajectory, NOT a proof)
        "energy": {"KE_max_J": float(S["KE"].max()), "KE_final_J": float(S["KE"][-1]),
                   "W_dW_max_J": float(dW.max()),
                   "n_dW_positive_steps": int(np.sum(dW > 1e-9)),
                   "sum_positive_dW_J": float(dW[dW > 0].sum())},
        "q_start_rad": q0.tolist(), "q_final_rad": Q[-1].tolist(),
        "finite": bool(np.all(np.isfinite(Q)) and np.all(np.isfinite(QD))),
    }
    res["traj"] = {"t": np.arange(N + 1) * dt, "P": P, "P_ideal": P_id, "Xn": L["Xn"],
                   "X_ideal": X_id, "Q": Q, "QD": QD, "xd": L["XD"], "F": L["F"],
                   "F_ext": L["F_ext"], "tau_cmd": L["TAU_CMD"], "tau_applied": L["TAU_APP"],
                   "tau_grav": L["TAU_G"], "tau_limit": tau_lim, "err_sync": e_sync,
                   "geom_dev": g_dev, "dist_goal": dist, "dW": dW, **S}
    return res
