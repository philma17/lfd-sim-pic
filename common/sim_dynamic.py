"""Dynamics-based execution of a learned DS on a MuJoCo robot (position-servo interface).

The kinematic reference (common/sim.py) integrates q += q_dot * dt in Python. Here the
joint state is advanced exclusively by mujoco.mj_step(); Python only writes data.ctrl.

Per controller step k (t = k * dt, dt = n_substeps * model.opt.timestep):
  q, q_dot = measured joint state from the physics MjData
  p, R     = FK(q)                                 Robot.fk, same TCP as sim.py
  x_n, d   = projection of p into the DS plane
  v_lin    = v_peak * E f(x_n) - k_perp * d * n    identical to sim.py
  omega    = k_rot * orientation error             identical to sim.py (axis5/full6)
  qd_cmd   = DLS-IK(J_task(q), [v_lin; omega])     common.ik.solve_qdot, same parameters
  q_des   += qd_cmd * dt, clipped to joint range and ctrlrange (internal target state)
  ctrl     = q_des                                 actuator index mapping read from the model
  mj_step  (n_substeps times)

Actuator model (verified in position_servo_map, not assumed): MJCF <general> actuators with
gain FIXED and bias AFFINE (0, -kp, -kv) on a joint transmission, i.e. a joint-space PD
position servo  tau = kp (ctrl - q) - kv q_dot, clamped to forcerange if forcelimited.
This is position control through the simulated servo, NOT direct torque control.

q_des is an internal integrator state and is never rebuilt from the measured q. Rebuilding
(q_des = q_meas + qd_cmd dt) would leave the servo with only kp * qd_cmd * dt of force, so
the static gravity offset (sag) q - q_des = -g(q) / kp would be lost and re-accumulated every
step. Because the DS and the Jacobian are evaluated at the MEASURED state, the task-space
loop acts as an integrator on the end-effector error: sag is compensated at the task level
and shows up as a joint-space gap q_des - q, which is reported (windup check).
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from . import metrics as M
from .ik import rot_error, solve_qdot, task_rows

# Joint torque limits used for REPORTING (the FR3 MJCF does not enforce any forcerange).
TAU_MAX_DOC = {
    "FR3": (np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0]),
            "frankarobotics/franka_description robots/fr3/joint_limits.yaml, 'effort' "
            "(retrieved 2026-10-18); NOT enforced by the Menagerie FR3 MJCF"),
    "UR5e": (np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0]),
             "Menagerie ur5e.xml actuator forcerange (enforced by the model); not "
             "cross-checked against the UR datasheet here"),
}


@dataclass
class ServoMap:
    """Position-servo actuators, indexed in Robot joint order (robot.spec.joints)."""
    act: np.ndarray          # actuator index driving robot joint i
    kp: np.ndarray
    kv: np.ndarray
    ctrl_lo: np.ndarray
    ctrl_hi: np.ndarray
    force_limited: np.ndarray
    force_lo: np.ndarray
    force_hi: np.ndarray

    def as_dict(self) -> dict:
        return {k: getattr(self, k).tolist() for k in self.__dataclass_fields__}


def position_servo_map(robot) -> ServoMap:
    """Read and verify the actuator model. Raises NotImplementedError for anything that is
    not a 1:1 joint-space PD position servo (then the controller design must be revisited)."""
    m = robot.model
    jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in robot.spec.joints]
    act = np.full(robot.n, -1)
    for a in range(m.nu):
        name = m.actuator(a).name
        if m.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT:
            raise NotImplementedError(f"{name}: transmission is not a joint")
        j = int(m.actuator_trnid[a, 0])
        if j not in jids:
            raise NotImplementedError(f"{name}: drives joint {m.joint(j).name} outside the arm")
        i = jids.index(j)
        if act[i] >= 0:
            raise NotImplementedError(f"joint {robot.spec.joints[i]} has several actuators")
        act[i] = a
        kp, b = m.actuator_gainprm[a, 0], m.actuator_biasprm[a, :3]
        ok = (m.actuator_gaintype[a] == mujoco.mjtGain.mjGAIN_FIXED
              and m.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_AFFINE
              and m.actuator_dyntype[a] == mujoco.mjtDyn.mjDYN_NONE
              and m.actuator_gear[a, 0] == 1.0 and b[0] == 0.0 and b[1] == -kp and b[2] <= 0.0
              and kp > 0)
        if not ok:
            raise NotImplementedError(f"{name}: not a PD position servo (gain/bias/gear differ)")
    if np.any(act < 0) or m.nu != robot.n:
        raise NotImplementedError("actuators do not cover every arm joint exactly once")
    return ServoMap(
        act=act, kp=m.actuator_gainprm[act, 0].copy(), kv=-m.actuator_biasprm[act, 2].copy(),
        ctrl_lo=np.where(m.actuator_ctrllimited[act], m.actuator_ctrlrange[act, 0], -np.inf),
        ctrl_hi=np.where(m.actuator_ctrllimited[act], m.actuator_ctrlrange[act, 1], np.inf),
        force_limited=m.actuator_forcelimited[act].astype(bool),
        force_lo=m.actuator_forcerange[act, 0].copy(), force_hi=m.actuator_forcerange[act, 1].copy())


def apply_servo_override(robot, tau_servo: float) -> ServoMap:
    """Set kv = kp * tau_servo for every arm actuator, IN MEMORY ONLY (robot.model was compiled
    from the MJCF for this Robot instance; nothing is written back). kp is kept unchanged.

    For an overdamped joint servo the velocity tracking behaves like a first-order lag with
    time constant kv / kp, so equal tau_servo gives both robots the same command lag."""
    sv = position_servo_map(robot)
    robot.model.actuator_biasprm[sv.act, 2] = -sv.kp * float(tau_servo)
    return position_servo_map(robot)   # re-verified after the change


def gravity_torque(robot, q) -> np.ndarray:
    """Generalized gravity force g(q) (qfrc_bias at q_dot = 0) in robot joint order."""
    d = mujoco.MjData(robot.model)
    d.qpos[robot.qadr] = q
    mujoco.mj_forward(robot.model, d)
    return d.qfrc_bias[robot.dadr].copy()


def static_sag(robot, q0, hold_s: float = 3.0, gravity_ff: bool = False) -> dict:
    """Hold ctrl = q0 (or q0 + g(q0)/kp with gravity_ff) from rest at q0 and let MuJoCo
    settle (no DS, no IK).

    Predicted equilibrium without feed-forward (first order, friction neglected):
    q - q0 = -g(q0) / kp. With feed-forward the prediction is q - q0 = 0 (to first order)."""
    sv = position_servo_map(robot)
    m = robot.model
    d = mujoco.MjData(m)
    d.qpos[robot.qadr] = q0
    d.ctrl[sv.act] = q0 + (gravity_torque(robot, q0) / sv.kp if gravity_ff else 0.0)
    mujoco.mj_forward(m, d)
    for _ in range(int(round(hold_s / m.opt.timestep))):
        mujoco.mj_step(m, d)
    q = d.qpos[robot.qadr].copy()
    p0, R0 = robot.fk(q0)
    p, R = robot.fk(q)
    g = gravity_torque(robot, q0)
    return {
        "hold_s": hold_s, "gravity_ff": bool(gravity_ff),
        "dq_rad": (q - q0).tolist(),
        "dq_pred_rad": (np.zeros_like(g) if gravity_ff else -g / sv.kp).tolist(),
        "max_abs_dq_rad": float(np.max(np.abs(q - q0))),
        "tcp_disp_mm": float(1e3 * np.linalg.norm(p - p0)),
        "tcp_disp_vec_mm": (1e3 * (p - p0)).tolist(),
        "tcp_rot_deg": float(np.degrees(np.linalg.norm(rot_error(R0, R)))),
        "gravity_torque_Nm": g.tolist(),
        "residual_qvel_max": float(np.max(np.abs(d.qvel[robot.dadr]))),
    }


def servo_stability(robot, q, step_rad: float = 0.01, hold_s: float = 1.5) -> dict:
    """Per-joint servo check at configuration q.

    Analytic (decoupled joint, diagonal of the joint-space inertia incl. armature):
      I_ii s^2 + kv s + kp = 0  ->  w_n = sqrt(kp / I_ii), zeta = kv / (2 sqrt(kp I_ii)),
      w_n * dt (explicit position stiffness; implicitfast treats the kv term implicitly).
    Empirical (coupled, MuJoCo): from rest at q with gravity feed-forward, step every ctrl by
    +step_rad and record 63 % rise time, overshoot and residual error after hold_s."""
    sv = position_servo_map(robot)
    m = robot.model
    d = mujoco.MjData(m)
    d.qpos[robot.qadr] = q
    mujoco.mj_forward(m, d)
    M = np.zeros((m.nv, m.nv))
    try:                                    # MuJoCo >= 3.x (this env: 3.16): (m, d, dst)
        mujoco.mj_fullM(m, d, M)
    except TypeError:                       # older API: (m, dst, qM)
        mujoco.mj_fullM(m, M, d.qM)
    I = np.diag(M)[robot.dadr]
    wn = np.sqrt(sv.kp / I)
    zeta = sv.kv / (2.0 * np.sqrt(sv.kp * I))
    poles = [np.roots([Ii, kv, kp]) for Ii, kv, kp in zip(I, sv.kv, sv.kp)]
    # empirical step response
    g = d.qfrc_bias[robot.dadr].copy()
    q_hold = q + g / sv.kp
    d.ctrl[sv.act] = q_hold + step_rad
    n = int(round(hold_s / m.opt.timestep))
    Q = np.empty((n + 1, robot.n))
    Q[0] = d.qpos[robot.qadr]
    for k in range(n):
        mujoco.mj_step(m, d)
        Q[k + 1] = d.qpos[robot.qadr]
    resp = (Q - Q[0]) / step_rad
    t = np.arange(n + 1) * m.opt.timestep
    t63 = [float(t[np.argmax(r >= 0.63)]) if np.any(r >= 0.63) else None for r in resp.T]
    return {
        "kp": sv.kp.tolist(), "kv": sv.kv.tolist(), "tau_kv_over_kp_s": (sv.kv / sv.kp).tolist(),
        "inertia_diag_kgm2": I.tolist(), "w_n_rad_s": wn.tolist(), "zeta": zeta.tolist(),
        "w_n_dt": (wn * m.opt.timestep).tolist(),
        "max_real_pole": float(max(np.max(p.real) for p in poles)),
        "step_rad": step_rad, "step_t63_s": t63,
        "step_overshoot_pct": (100.0 * np.maximum(resp.max(0) - 1.0, 0.0)).tolist(),
        "step_final_frac": resp[-1].tolist(),
        "finite": bool(np.all(np.isfinite(Q))),
    }


def check_dims(ds, emb, x0_n) -> int:
    """Model input dimension == embedding dimension == len(x0_n); returns it."""
    d_model = int(ds.f.dim)
    d_emb = int(np.asarray(emb.E).shape[1])
    if not (d_model == d_emb == len(x0_n)):
        raise ValueError(f"dimension mismatch: model input {d_model}, embedding {d_emb}, "
                         f"x0 {len(x0_n)}")
    return d_model


def ideal_rollout(ds, x0_n, dt: float, n_steps: int, rate: float) -> np.ndarray:
    """Explicit Euler of the ideal DS, (n_steps+1, dim). For 2D this calls
    LearnedDS.rollout_ideal (unchanged reference); the same scheme for other dimensions."""
    x0_n = np.asarray(x0_n, float)
    if len(x0_n) == 2:
        return ds.rollout_ideal(x0_n, dt, n_steps, rate)
    X = np.empty((n_steps + 1, len(x0_n)))
    X[0] = x0_n
    for k in range(n_steps):
        X[k + 1] = X[k] + dt * rate * ds.f_n(X[k])
    return X


def run(robot, ds, emb, x0_n, rate: float, tgt, sp, q_start, free=None, q_null=None,
        n_substeps: int = 1, contact_check: bool = True, gravity_ff: bool = False,
        tau_limit=None) -> dict:
    """Closed-loop DS execution with MuJoCo dynamics. Same arguments as sim.run (q_start is
    mandatory so that both execution modes start from the identical configuration).

    gravity_ff: start the servo target at q_start + g(q_start)/kp (static equilibrium of the
    PD servo at q_start), so the robot does not sag at t = 0. The robot itself still starts
    at q_start with zero velocity. Default False = previous baseline behavior.
    tau_limit: joint torque limits used for REPORTING only [Nm]; default TAU_MAX_DOC."""
    m = robot.model
    n = robot.n
    if not np.isclose(sp.dt, n_substeps * m.opt.timestep):
        raise ValueError(f"controller dt {sp.dt} != n_substeps * timestep {m.opt.timestep}")
    sv = position_servo_map(robot)
    free = np.ones(n, bool) if free is None else np.asarray(free, bool)
    N = int(round(sp.T_max / sp.dt))
    x0_n = np.asarray(x0_n, float)
    dim = check_dims(ds, emb, x0_n)                  # 2 (plane) or 3 (spatial embedding)
    X_id = ideal_rollout(ds, x0_n, sp.dt, N, rate)   # same reference as sim.run for 2D
    P_id = emb.to_world(X_id)
    v_peak = rate * emb.L
    goal = emb.to_world(np.zeros(dim))
    q_mid = 0.5 * (robot.q_min + robot.q_max) if q_null is None else np.asarray(q_null, float)
    q_rng = robot.q_max - robot.q_min
    lo = np.maximum(robot.q_min, sv.ctrl_lo)
    hi = np.minimum(robot.q_max, sv.ctrl_hi)
    tau_doc = (TAU_MAX_DOC.get(robot.spec.name, (np.full(n, np.inf), "none"))[0]
               if tau_limit is None else np.asarray(tau_limit, float))

    # Physics state lives in its own MjData; robot.data is only used for FK/Jacobian
    # evaluation at the measured q, so the existing kinematic helpers are reused unchanged.
    d = mujoco.MjData(m)
    q0 = np.asarray(q_start, float)
    d.qpos[robot.qadr] = q0
    d.qvel[:] = 0.0
    q_des = q0.copy()                      # windup guard: start at the measured start pose
    if gravity_ff:
        q_des = q_des + gravity_torque(robot, q0) / sv.kp
    q_des = np.clip(q_des, lo, hi)
    qdes0 = q_des.copy()
    d.ctrl[sv.act] = q_des                 # set before the first mj_step (no initial jump)
    mujoco.mj_forward(m, d)

    P = np.empty((N + 1, 3))
    Xn = np.empty((N + 1, dim))
    Q = np.empty((N + 1, n))
    QD = np.empty((N + 1, n))
    QDES = np.empty((N + 1, n))            # target active while Q[k] was measured
    QD_CMD = np.zeros((N, n))
    TAU_CMD = np.zeros((N, n))             # unclamped servo law kp (ctrl - q) - kv q_dot
    TAU_APP = np.zeros((N, n))             # data.actuator_force after clamping (applied)
    ori = np.zeros(N + 1)
    dperp = np.zeros(N + 1)
    smin = np.full(N, np.inf)
    vscale = np.zeros(N)                   # IK velocity scaling factor (> 1 = command scaled down)
    ncon = np.zeros(N, int)
    n_vel_scale = n_qdes_clip = n_damped = 0
    qdes_clip_joints = set()
    first_contact = None
    R0 = None
    for k in range(N + 1):
        q = d.qpos[robot.qadr].copy()
        qd_meas = d.qvel[robot.dadr].copy()
        p, R = robot.fk(q)                 # evaluated at the MEASURED state
        R0 = R if R0 is None else R0
        x_n, dp = emb.to_plane(p)
        P[k], Xn[k], Q[k], QD[k], QDES[k], dperp[k] = p, x_n, q, qd_meas, q_des, dp
        v_lin = v_peak * emb.vel_to_world(ds.f_n(x_n)) - sp.k_perp * dp * emb.normal
        Jt, vt, oe = task_rows(robot.jac(), R, tgt, v_lin, sp.k_rot)
        ori[k] = oe if tgt.mode != "pos" else float(np.linalg.norm(rot_error(R, R0)))
        if k == N:
            break
        qd_cmd, info = solve_qdot(Jt, vt, free, robot.qd_max, sp.ik, q, q_mid, q_rng)
        smin[k] = info["sigma_min"]
        vscale[k] = info["vel_scale"]
        n_vel_scale += info["vel_scale"] > 1.0
        n_damped += info["lam2"] > 0.0
        q_new = q_des + qd_cmd * sp.dt
        q_des = np.clip(q_new, lo, hi)
        hit = q_des != q_new
        if np.any(hit):
            n_qdes_clip += 1
            qdes_clip_joints.update(int(j) for j in np.flatnonzero(hit))
        d.ctrl[sv.act] = q_des
        QD_CMD[k] = qd_cmd
        TAU_CMD[k] = sv.kp * (q_des - q) - sv.kv * qd_meas
        for _ in range(n_substeps):
            mujoco.mj_step(m, d)
        TAU_APP[k] = d.actuator_force[sv.act]
        if contact_check:
            ncon[k] = d.ncon
            if d.ncon and first_contact is None:
                c = d.contact[0]
                first_contact = {"t_s": (k + 1) * sp.dt,
                                 "geoms": [m.body(int(m.geom_bodyid[c.geom1])).name,
                                           m.body(int(m.geom_bodyid[c.geom2])).name]}
        if not np.all(np.isfinite(d.qpos)):
            raise FloatingPointError(f"{robot.spec.name}: non-finite qpos at step {k}")

    t = np.arange(N + 1) * sp.dt
    L = emb.L
    thr = sp.success_frac * L
    dist_goal = np.linalg.norm(P - goal, axis=1)
    reached = np.flatnonzero(dist_goal < thr)
    gap = QDES - Q
    k_gap = int(np.argmax(np.max(np.abs(gap), axis=1)))
    # actual-state limit checks (MuJoCo joint limits are soft constraints)
    pos_exc = np.maximum(robot.q_min - Q, Q - robot.q_max)          # > 0 = outside range
    vel_ratio = np.abs(QD) / robot.qd_max
    sat = sv.force_limited & ((TAU_CMD < sv.force_lo) | (TAU_CMD > sv.force_hi))
    warn = {mujoco.mjtWarning(i).name: int(d.warning[i].number)
            for i in range(len(d.warning)) if d.warning[i].number}
    res = {
        "robot": robot.spec.name, "mode": tgt.mode, "execution": "dynamic",
        "control_interface": "MuJoCo joint position servo (ctrl = q_des), "
                             "tau = kp (ctrl - q) - kv q_dot",
        "servo": sv.as_dict(), "n_substeps": n_substeps, "timestep": m.opt.timestep,
        "L_m": L, "v_peak_m_s": v_peak, "dt": sp.dt, "T_max": sp.T_max, "n_steps": N,
        "success": bool(dist_goal[-1] < thr), "final_dist_m": float(dist_goal[-1]),
        "t_reach_s": float(reached[0] * sp.dt) if len(reached) else None,
        "max_out_of_plane_m": float(np.max(np.abs(dperp))),
        "max_ori_err_rad": float(np.max(ori)),
        "min_sigma_min": float(np.min(smin)), "n_damped_steps": int(n_damped),
        # command interventions (NOT limit violations)
        "n_vel_scale_steps": int(n_vel_scale),
        "n_qdes_clip_steps": int(n_qdes_clip), "qdes_clip_joints": sorted(qdes_clip_joints),
        "n_force_sat_steps": int(np.sum(np.any(sat, axis=1))),
        "force_sat_steps_per_joint": np.sum(sat, axis=0).tolist(),
        # actual state
        "n_pos_limit_viol_steps": int(np.sum(np.any(pos_exc > 0, axis=1))),
        "max_pos_limit_exceed_rad": float(max(0.0, pos_exc.max())),
        "n_vel_limit_viol_steps": int(np.sum(np.any(vel_ratio > 1.0, axis=1))),
        "max_vel_ratio": float(vel_ratio.max()),
        "max_abs_qd_rad_s": np.abs(QD).max(0).tolist(),
        # torques
        "max_abs_tau_cmd_Nm": np.abs(TAU_CMD).max(0).tolist(),
        "max_abs_tau_applied_Nm": np.abs(TAU_APP).max(0).tolist(),
        "tau_doc_Nm": tau_doc.tolist(),
        "n_tau_doc_exceed_steps": int(np.sum(np.any(np.abs(TAU_APP) > tau_doc, axis=1))),
        "peak_tau_ratio": float(np.max(np.abs(TAU_APP) / tau_doc)),
        # start / configuration
        "gravity_ff": bool(gravity_ff),
        "qdes0_minus_q0_rad": (qdes0 - q0).tolist(),
        "min_limit_margin_pct": float(min(robot.limit_margin_pct(qq).min() for qq in Q)),
        "q_final_rad": Q[-1].tolist(),
        # windup / servo gap
        "max_abs_gap_rad": float(np.abs(gap).max()),
        "max_abs_gap_per_joint_rad": np.abs(gap).max(0).tolist(),
        "t_max_gap_s": float(k_gap * sp.dt),
        "final_gap_rad": gap[-1].tolist(),
        # numerics / contacts
        "n_contact_steps": int(np.sum(ncon > 0)), "first_contact": first_contact,
        "mujoco_warnings": warn,
        "finite": bool(np.all(np.isfinite(Q)) and np.all(np.isfinite(QD))),
    }
    res["traj"] = {"t": t, "P": P, "P_ideal": P_id, "Xn": Xn, "X_ideal": X_id, "Q": Q,
                   "QD": QD, "Q_des": QDES, "QD_cmd": QD_CMD, "tau_cmd": TAU_CMD,
                   "tau_applied": TAU_APP, "vel_scale": vscale, "ncon": ncon,
                   "tau_limit": tau_doc}
    return res


def tracking_metrics(P, P_id, goal, L, dt, success_frac) -> dict:
    """Shared task-space metrics for both execution modes (world frame, metres).

    time-synchronized error e_k = ||P_k - P_id,k||, P_id,k = ideal DS Euler rollout at the same
    t_k = k dt from the same start x0 (open-loop reference, NOT re-anchored at the robot).
    geometric deviation g_k = distance of P_k to the ideal path polyline (timing-invariant)."""
    e = np.linalg.norm(P - P_id, axis=1)
    g = M.point_to_polyline(P, P_id)
    ch_mean, ch_max = M.chamfer(P, P_id)
    dist_goal = np.linalg.norm(P - goal, axis=1)
    thr = success_frac * L
    reached = np.flatnonzero(dist_goal < thr)
    return {
        "final_goal_err_m": float(dist_goal[-1]),
        "mse_sync_m2": float(np.mean(e ** 2)),
        "rmse_sync_m": float(np.sqrt(np.mean(e ** 2))), "max_err_sync_m": float(e.max()),
        "t_max_err_sync_s": float(np.argmax(e) * dt),
        "geom_dev_mean_m": float(g.mean()), "geom_dev_max_m": float(g.max()),
        "chamfer_mean_m": float(ch_mean), "hausdorff_m": float(ch_max),
        "frechet_m": M.frechet_discrete(P, P_id),
        "t_reach_s": float(reached[0] * dt) if len(reached) else None,
    }, e, g
