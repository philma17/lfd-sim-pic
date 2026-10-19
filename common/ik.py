"""Differentielle IK.

Grundsaetze (Fehlerliste des Altversuchs):
  * Gesperrte Gelenke: Spalte aus J entfernen und mit den freien Gelenken neu
    loesen. NIE q_dot[j] = 0 nach der Pseudoinversen setzen.
  * Gedaempfte LS mit adaptiver Daempfung: lambda = 0, solange sigma_min >= eps.
    Abseits von Singularitaeten ist die Loesung damit exakt (J q_dot = v).
  * Nullraum ueber SVD-Pseudoinverse, nur wenn die freien Gelenke Redundanz haben.
  * Geschwindigkeitslimits: gleichmaessiges Herunterskalieren von q_dot
    (Richtung bleibt erhalten), jeder Eingriff wird gezaehlt.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

TASK_DIMS = {"pos": 3, "axis5": 5, "full6": 6}


# --------------------------------------------------------------------------
# Lineare Algebra
# --------------------------------------------------------------------------
def svd_pinv(J: np.ndarray, rcond: float = 1e-10) -> np.ndarray:
    U, S, Vt = np.linalg.svd(J, full_matrices=False)
    keep = S > rcond * max(S.max(initial=0.0), 1e-300)
    Sinv = np.where(keep, 1.0 / np.where(keep, S, 1.0), 0.0)
    return (Vt.T * Sinv) @ U.T


def nullspace_projector(J: np.ndarray) -> np.ndarray:
    return np.eye(J.shape[1]) - svd_pinv(J) @ J


def dls(J: np.ndarray, v: np.ndarray, eps: float, lam_max: float):
    """Adaptiv gedaempfte LS. Rueckgabe (q_dot, sigma_min, lambda^2)."""
    U, S, Vt = np.linalg.svd(J, full_matrices=False)
    smin = float(S[-1])
    lam2 = 0.0 if smin >= eps else (1.0 - (smin / eps) ** 2) * lam_max ** 2
    denom = S ** 2 + lam2
    coef = np.where(denom > 0, S / np.where(denom > 0, denom, 1.0), 0.0)
    return Vt.T @ (coef * (U.T @ v)), smin, lam2


def manipulability(J: np.ndarray) -> float:
    S = np.linalg.svd(J, compute_uv=False)
    return float(np.prod(S))  # = sqrt(det(J J^T)) fuer m <= n


# --------------------------------------------------------------------------
# Aufgaben: Position (3D), Werkzeugachse (5D), volle Pose (6D)
# --------------------------------------------------------------------------
def perp_basis(a: np.ndarray) -> np.ndarray:
    """Zwei orthonormale Vektoren senkrecht zu a, als Spalten (3,2). Deterministisch."""
    a = a / np.linalg.norm(a)
    h = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    b1 = np.cross(a, h)
    b1 /= np.linalg.norm(b1)
    return np.stack([b1, np.cross(a, b1)], axis=1)


def rot_error(R: np.ndarray, R_des: np.ndarray) -> np.ndarray:
    """Rotationsvektor (Welt) von R nach R_des."""
    return Rotation.from_matrix(R_des @ R.T).as_rotvec()


def axis_error(R: np.ndarray, a_des: np.ndarray, axis_local=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Rotationsvektor (Welt), der die Werkzeugachse auf a_des dreht (minimaler Winkel)."""
    a = R @ np.asarray(axis_local, float)
    c = np.cross(a, a_des)
    s = np.linalg.norm(c)
    th = np.arctan2(s, float(a @ a_des))
    if s < 1e-15:
        return np.zeros(3) if th < 1.0 else perp_basis(a)[:, 0] * th
    return c / s * th


@dataclass
class Target:
    """Orientierungsziel. mode: 'pos' | 'axis5' | 'full6'."""
    mode: str = "pos"
    R_des: np.ndarray | None = None
    a_des: np.ndarray | None = None
    axis_local: tuple = (0.0, 0.0, 1.0)


def task_rows(J6: np.ndarray, R: np.ndarray, tgt: Target, v_lin: np.ndarray, k_rot: float):
    """Aufgaben-Jacobian und Sollgeschwindigkeit. Rueckgabe (J_task, v_task, ori_err_rad)."""
    if tgt.mode == "pos":
        return J6[:3], v_lin, 0.0
    if tgt.mode == "full6":
        e = rot_error(R, tgt.R_des)
        return J6, np.concatenate([v_lin, k_rot * e]), float(np.linalg.norm(e))
    if tgt.mode == "axis5":
        e = axis_error(R, tgt.a_des, tgt.axis_local)
        B = perp_basis(R @ np.asarray(tgt.axis_local, float))
        Jt = np.vstack([J6[:3], B.T @ J6[3:]])
        return Jt, np.concatenate([v_lin, k_rot * (B.T @ e)]), float(np.linalg.norm(e))
    raise ValueError(tgt.mode)


# --------------------------------------------------------------------------
# Geschwindigkeits-IK mit Lock (Spaltenentfernung) und Limits
# --------------------------------------------------------------------------
@dataclass
class IKParams:
    eps: float = 0.02         # sigma_min-Schwelle fuer Daempfung
    lam_max: float = 0.05     # maximale Daempfung
    k_null: float = 0.0       # Gain Gelenkmitten-Nullraumaufgabe [rad/s]


def solve_qdot(J_task: np.ndarray, v_task: np.ndarray, free: np.ndarray, qd_max: np.ndarray,
               p: IKParams, q: np.ndarray | None = None, q_mid: np.ndarray | None = None,
               q_rng: np.ndarray | None = None):
    """q_dot fuer alle n Gelenke; gesperrte Gelenke haben q_dot = 0 per Konstruktion.

    Rueckgabe (q_dot, info) mit sigma_min, lam2, vel_scale (>1 = geclippt).
    """
    n = J_task.shape[1]
    idx = np.flatnonzero(free)
    Jf = J_task[:, idx]
    qd_f, smin, lam2 = dls(Jf, v_task, p.eps, p.lam_max)
    if p.k_null > 0 and len(idx) > Jf.shape[0] and q is not None:
        # Zur Gelenkmitte ziehen, normiert auf den halben Bereich: [-1, 1] * k_null [rad/s]
        grad = -p.k_null * (q[idx] - q_mid[idx]) / (0.5 * q_rng[idx])
        qd_f = qd_f + nullspace_projector(Jf) @ grad
    qd = np.zeros(n)
    qd[idx] = qd_f
    scale = float(np.max(np.abs(qd) / qd_max))
    if scale > 1.0:
        qd = qd / scale
    return qd, {"sigma_min": smin, "lam2": lam2, "vel_scale": scale,
                "manip": manipulability(Jf) if Jf.shape[0] <= Jf.shape[1] else 0.0}


# --------------------------------------------------------------------------
# Startpose: Multi-Start-IK
# --------------------------------------------------------------------------
@dataclass
class IKResult:
    q: np.ndarray
    pos_err: float
    ori_err: float
    converged: bool
    n_converged: int
    n_starts: int
    sigma_min: float = float("nan")
    margin_pct: float = float("nan")
    margin_ok: bool = True
    solutions: list = field(default_factory=list)   # alle konvergierten Startloesungen


def ik_pose(robot, p_des, tgt: Target, q_init, free=None, iters: int = 400,
            tol: float = 1e-9, lam: float = 1e-3, max_step: float = 0.2):
    """Levenberg-Marquardt auf Position (+ Orientierung). Clipping an Gelenklimits."""
    q = np.clip(np.array(q_init, float), robot.q_min, robot.q_max)
    free = np.ones(robot.n, bool) if free is None else free
    idx = np.flatnonzero(free)
    for _ in range(iters):
        pc, R = robot.fk(q)
        J6 = robot.jac()
        e_p = p_des - pc
        Jt, err, ori = task_rows(J6, R, tgt, e_p, 1.0)
        if np.linalg.norm(e_p) < tol and ori < tol:
            break
        A = Jt[:, idx]
        dq = A.T @ np.linalg.solve(A @ A.T + lam ** 2 * np.eye(A.shape[0]), err)
        nrm = np.linalg.norm(dq)
        if nrm > max_step:
            dq *= max_step / nrm
        q[idx] = np.clip(q[idx] + dq, robot.q_min[idx], robot.q_max[idx])
    pc, R = robot.fk(q)
    _, _, ori = task_rows(robot.jac(), R, tgt, p_des - pc, 1.0)
    pe = float(np.linalg.norm(p_des - pc))
    return q, pe, ori


def task_sigma_min(robot, q, tgt: Target, free) -> float:
    """Kleinster Singulaerwert des Aufgaben-Jacobians (nur freie Spalten) in q."""
    _, R = robot.fk(q)
    Jt, _, _ = task_rows(robot.jac(), R, tgt, np.zeros(3), 1.0)
    return float(np.linalg.svd(Jt[:, free], compute_uv=False)[-1])


def multistart_ik(robot, p_des, tgt: Target, q_ref, n_starts: int, seed: int,
                  criterion: str = "nearest_ref", free=None, tol_conv: float = 1e-6,
                  min_margin_pct: float = 5.0) -> IKResult:
    """Start 0 = q_ref, danach gleichverteilte Zufallsstarts im Gelenkbereich (seed fix).

    Auswahl unter den konvergierten Loesungen nach festem Kriterium:
      nearest_ref       : min ||q - q_ref||
      max_margin        : max min. Gelenklimit-Abstand
      max_sigma_margin  : max sigma_min(J_task) unter allen Loesungen mit
                          Limitabstand > min_margin_pct (F7). Erfuellt keine
                          Loesung den Abstand, max sigma_min ueber alle
                          konvergierten (margin_ok = False wird berichtet).
    """
    rng = np.random.default_rng(seed)
    q_ref = np.asarray(q_ref, float)
    free = np.ones(robot.n, bool) if free is None else np.asarray(free, bool)
    starts = [q_ref] + list(rng.uniform(robot.q_min, robot.q_max, size=(n_starts - 1, robot.n)))
    starts = [np.where(free, s, q_ref) for s in starts]  # gesperrte Gelenke bleiben auf q_ref
    sols = []
    for s in starts:
        q, pe, oe = ik_pose(robot, p_des, tgt, s, free=free)
        sols.append((q, pe, oe))
    conv = [s for s in sols if s[1] < tol_conv and s[2] < tol_conv]
    margin_ok = True
    if conv:
        if criterion == "nearest_ref":
            best = min(conv, key=lambda s: np.linalg.norm(s[0] - q_ref))
        elif criterion == "max_margin":
            best = max(conv, key=lambda s: robot.limit_margin_pct(s[0]).min())
        elif criterion == "max_sigma_margin":
            ok = [s for s in conv if robot.limit_margin_pct(s[0]).min() > min_margin_pct]
            margin_ok = bool(ok)
            pool = ok if ok else conv
            best = max(pool, key=lambda s: task_sigma_min(robot, s[0], tgt, free))
        else:
            raise ValueError(criterion)
    else:
        best = min(sols, key=lambda s: s[1] + s[2])
        margin_ok = False
    res = IKResult(best[0], best[1], best[2], bool(conv), len(conv), n_starts)
    res.sigma_min = task_sigma_min(robot, best[0], tgt, free)
    res.margin_pct = float(robot.limit_margin_pct(best[0]).min())
    res.margin_ok = margin_ok
    res.solutions = [s[0].copy() for s in conv]
    return res


def distinct_configs(robot, qs, tol: float = 0.05):
    """Konfigurationen bis auf 2*pi-Mehrdeutigkeit zusammenfassen (Winkel auf (-pi, pi])."""
    out, keys = [], []
    for q in qs:
        w = (np.asarray(q) + np.pi) % (2 * np.pi) - np.pi
        if all(np.max(np.abs((w - k + np.pi) % (2 * np.pi) - np.pi)) > tol for k in keys):
            keys.append(w)
            out.append(np.asarray(q))
    return out
