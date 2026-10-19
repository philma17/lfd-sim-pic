"""Passive Interaction Control with Dynamical Systems (PIC-DS), pure numpy (no MuJoCo here).

Reference: K. Kronander and A. Billard, "Passive Interaction Control With Dynamical Systems",
IEEE RA-L 1(1), 2016 (cited below as [P], equation numbers as in the paper) and the reference
code github.com/epfl-lasa/passive-ds-control (cited as [R]).

Task-space law for the 3D TCP position x (xi = x in [P]):

    D(x)  = Q Lambda Q^T,  e1 = f / |f|,  e2, e3 orthonormal                          [P] Eq. 3
    F     = -D xdot + lambda1 (f_C + beta_R(z, s) f_R)                                [P] Eq. 13
    z     = xdot . f_R                                                                [P] Eq. 10
    sdot  = alpha(s) xdot^T D xdot - beta_s(z, s) lambda1 z                           [P] Eq. 10
    tau   = g(q) + J_p^T F + tau_null                                  (joint space, sim_pic.py)

Deviations from / choices beyond the paper (each one is listed in README.md):
  * Coupling functions alpha, beta_s, beta_R follow [P] Appendix A (Eq. 25-28, quintic step).
    [R] uses different functions (cosine step, a simpler beta_R, a dead band |z| < 0.01, and
    alpha(0) = 0). The cosine step is available as step="cosine", the rest is NOT taken from [R].
  * Underlined s in Eq. 27/28 is read as the lower bound 0 (the text extraction of the PDF
    drops the underline; [R] uses 0 as well and Eq. 12 requires beta_s = 0 at s <= 0).
  * The tank is integrated with explicit Euler (as [R]); the state at step k is used for both F
    and sdot. Discrete-time passivity is NOT proven by [P] (continuous-time Theorem 1).
  * [P] leaves e2, e3 arbitrary. Like [R], the previous basis is Gram-Schmidt-orthonormalized
    around the new e1 so that the basis varies continuously (this matters only if
    lambda2 != lambda3). The initial basis is the identity (deterministic; [R] uses random).
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------
# Smooth steps and coupling functions ([P] Appendix A)
# --------------------------------------------------------------------------
def smooth_step(x, a: float, b: float, kind: str = "quintic"):
    """H_{a,b}(x): 0 for x < a, 1 for x > b, smooth in between. [P] Eq. 25a (quintic).
    kind="cosine" is the step used by [R] (smooth_rise in smooth_truncation.h)."""
    if not b > a:
        raise ValueError(f"smooth step needs b > a, got a={a}, b={b}")
    u = np.clip((np.asarray(x, float) - a) / (b - a), 0.0, 1.0)
    if kind == "quintic":
        return 6.0 * u**5 - 15.0 * u**4 + 10.0 * u**3
    if kind == "cosine":
        return 0.5 - 0.5 * np.cos(np.pi * u)
    raise ValueError(f"unknown smooth step '{kind}'")


def smooth_step_inv(x, a: float, b: float, kind: str = "quintic"):
    """H^-_{a,b}(x) = 1 - H_{a,b}(x), [P] Eq. 25b."""
    return 1.0 - smooth_step(x, a, b, kind)


@dataclass(frozen=True)
class Coupling:
    """alpha, beta_s, beta_R of [P] Eq. 26-28 for tank bounds [0, s_max]."""
    s_max: float
    delta_s: float
    delta_z: float
    kind: str = "quintic"

    def __post_init__(self):
        if not (self.s_max > 0 and self.delta_z > 0 and self.delta_s > 0):
            raise ValueError("s_max, delta_s, delta_z must be > 0")
        if self.delta_s > 0.5 * self.s_max:
            # beta_R = beta_s for z >= 0 (Eq. 14) needs the two s-transition bands
            # [0, delta_s] and [s_max - delta_s, s_max] to be disjoint
            raise ValueError("delta_s must be <= s_max / 2")

    def alpha(self, s):
        """[P] Eq. 26: alpha(s) = H^-_{s_max - delta_s, s_max}(s)."""
        return smooth_step_inv(s, self.s_max - self.delta_s, self.s_max, self.kind)

    def beta_s(self, z, s):
        """[P] Eq. 27 (lower tank bound = 0)."""
        k = self.kind
        t1 = smooth_step(z, -self.delta_z, 0.0, k) * smooth_step_inv(s, 0.0, self.delta_s, k)
        t2 = (smooth_step_inv(z, 0.0, self.delta_z, k)
              * smooth_step(s, self.s_max - self.delta_s, self.s_max, k))
        return 1.0 - t1 - t2

    def beta_R(self, z, s):
        """[P] Eq. 28 (lower tank bound = 0)."""
        k = self.kind
        a = smooth_step(z, -self.delta_z, 0.0, k)
        A = smooth_step_inv(s, 0.0, self.delta_s, k)
        b = smooth_step_inv(z, 0.0, self.delta_z, k)
        B = smooth_step(s, self.s_max - self.delta_s, self.s_max, k)
        return (1.0 - a * A) * (1.0 - a * b * B)


# --------------------------------------------------------------------------
# Damping matrix ([P] Eq. 3)
# --------------------------------------------------------------------------
def _gram_schmidt(Q: np.ndarray) -> np.ndarray:
    """Orthonormalize the columns of Q in order (column 0 is kept in direction). A column that
    degenerates (parallel to earlier ones) is replaced by the standard basis vector that is
    most orthogonal to the earlier columns: deterministic, no random numbers."""
    n = Q.shape[1]
    out = np.zeros_like(Q)
    for i in range(n):
        v = Q[:, i].copy()
        for j in range(i):
            v -= (out[:, j] @ v) * out[:, j]
        nv = np.linalg.norm(v)
        if nv < 1e-8:
            best = None
            for c in np.eye(Q.shape[0]):
                w = c - sum((out[:, j] @ c) * out[:, j] for j in range(i))
                if best is None or np.linalg.norm(w) > np.linalg.norm(best):
                    best = w
            v, nv = best, np.linalg.norm(best)
        out[:, i] = v / nv
    return out


class DampingBasis:
    """Orthonormal Q(x) with first column e1 = f / |f|. If |f| < eta the previous basis is
    kept ([P] Sec. III-A, [R] MINSPEED)."""

    def __init__(self, dim: int = 3, eta: float = 1e-6):
        self.Q = np.eye(dim)
        self.eta = float(eta)
        self.n_kept = 0                     # steps where |f| < eta and the basis was kept

    def update(self, f: np.ndarray) -> np.ndarray:
        nf = float(np.linalg.norm(f))
        if nf >= self.eta:
            Q = self.Q.copy()
            Q[:, 0] = f / nf
            self.Q = _gram_schmidt(Q)
        else:
            self.n_kept += 1
        return self.Q


def damping_matrix(Q: np.ndarray, lam) -> np.ndarray:
    """D = Q diag(lam) Q^T (symmetric positive semi-definite for lam >= 0)."""
    return (Q * np.asarray(lam, float)) @ Q.T


# --------------------------------------------------------------------------
# Controller with energy tank
# --------------------------------------------------------------------------
@dataclass
class PICParams:
    lam: tuple = (20.0, 200.0, 200.0)
    s_max: float = 10.0
    s0: float = 10.0
    delta_s_frac: float = 0.1
    delta_z: float = 0.01
    eta: float = 1e-6
    step: str = "quintic"
    decomposition: str = "none"

    def validate(self) -> None:
        if len(self.lam) != 3 or min(self.lam) < 0:
            raise ValueError("lambda must be 3 non-negative values")
        if not 0.0 < self.s0 <= self.s_max:           # Theorem 1 of [P]: 0 < s(0) <= s_bar
            raise ValueError(f"need 0 < s0 <= s_max, got s0={self.s0}, s_max={self.s_max}")
        if self.decomposition != "none":
            raise NotImplementedError(
                f"decomposition '{self.decomposition}' is not implemented in this step; "
                "only 'none' (f_C = 0, f_R = f)")

    @classmethod
    def from_cfg(cls, c: dict) -> "PICParams":
        p = cls(lam=tuple(float(x) for x in c["lambda"]), s_max=float(c["s_max"]),
                s0=float(c["s0"]), delta_s_frac=float(c["delta_s_frac"]),
                delta_z=float(c["delta_z"]), eta=float(c["eta"]), step=str(c["step"]),
                decomposition=str(c["decomposition"]))
        p.validate()
        return p


class PICController:
    """PIC-DS force law and energy tank in task space. State: tank level s and damping basis."""

    def __init__(self, p: PICParams, dim: int = 3):
        p.validate()
        self.p = p
        self.cp = Coupling(p.s_max, p.delta_s_frac * p.s_max, p.delta_z, p.step)
        self.basis = DampingBasis(dim, p.eta)
        self.s = float(p.s0)
        self.n_clamp = 0                    # steps where Euler left [0, s_max] and was clamped
        self.max_excursion = 0.0            # largest violation before clamping [J]

    def step(self, xd: np.ndarray, f: np.ndarray, dt: float) -> dict:
        """One control step at the measured TCP velocity xd and DS value f(x) (both world, 3D).

        decomposition "none": f_C = 0, f_R = f. Returns the task force F and diagnostics; the
        tank is advanced to the next step after F has been computed from the CURRENT s."""
        p, cp = self.p, self.cp
        Q = self.basis.update(f)
        D = damping_matrix(Q, p.lam)
        lam1 = p.lam[0]
        f_C = np.zeros_like(f)
        f_R = f
        z = float(xd @ f_R)
        s = self.s
        beta_R = float(cp.beta_R(z, s))
        beta_s = float(cp.beta_s(z, s))
        alpha = float(cp.alpha(s))
        F = -D @ xd + lam1 * (f_C + beta_R * f_R)
        p_diss = float(xd @ D @ xd)         # dissipated power xdot^T D xdot  [W]
        sdot = alpha * p_diss - beta_s * lam1 * z
        s_new = s + sdot * dt
        exc = max(s_new - p.s_max, -s_new, 0.0)
        if exc > 0.0:
            # explicit Euler left the invariant interval of the continuous system: clamp and
            # COUNT it (reported, never hidden)
            self.n_clamp += 1
            self.max_excursion = max(self.max_excursion, exc)
            s_new = min(max(s_new, 0.0), p.s_max)
        self.s = s_new
        return {"F": F, "D": D, "s": s, "s_next": s_new, "z": z, "alpha": alpha,
                "beta_s": beta_s, "beta_R": beta_R, "sdot": sdot, "p_diss": p_diss,
                "e1": Q[:, 0].copy()}
