"""Automatic plots. They are built ONLY from saved run files (traj.npz + manifest/metrics JSON),
so any run can be re-plotted later (`python experiments/plot_run.py <run_dir>`) and the figures
always correspond to the stored data.

Per run (make_run_plots):
    01_tcp_3d.png          TCP path in 3D + top view, start, goal, ideal reference rollout
    02_tracking_error.png  time-synchronous error, geometric deviation, distance to goal
    03_joints.png          joint positions / velocities with limits
    04_kinematics.png      TCP speed, acceleration, jerk; joint acceleration, jerk (DERIVED)
    05_torques.png         commanded vs applied torque per joint with limits, saturation shading
    06_controller.png      PIC-DS: tank, coupling, forces; servo: target-vs-measured gap
Across runs (make_compare_plots): overlays with a legend built from the parameters that differ.
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .analysis import max_ratio  # noqa: E402

JOINT_COLORS = plt.cm.tab10(np.arange(7))
CTRL_LABEL = {"servo": "position servo", "pic": "PIC-DS", "pic_pulse": "PIC-DS + pulse"}


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
class RunData:
    def __init__(self, path: str):
        self.path = path
        with open(os.path.join(path, "manifest.json"), encoding="utf-8") as fh:
            self.manifest = json.load(fh)
        with open(os.path.join(path, "metrics.json"), encoding="utf-8") as fh:
            self.metrics = json.load(fh)
        with np.load(os.path.join(path, "traj.npz")) as z:
            self.a = {k: z[k] for k in z.files}
        self.controller = self.manifest["controller"]
        self.name = self.manifest["run_id"]

    @property
    def params(self) -> dict:
        """Flat dict of the experiment parameters that define the condition."""
        m = self.manifest
        pl = m["plant"]["requested"]
        return {
            "controller": CTRL_LABEL.get(self.controller, self.controller),
            "alpha": m["ds_speed_factor"],
            "friction": pl["dof_frictionloss_scale"],
            "damping": pl["dof_damping_scale"],
            "gravity": pl["gravity_scale"],
            "mass": pl["mass_scale"],
            "inertia": pl["inertia_scale"],
            "tau_lim": pl["torque_limit_scale"],
            "q_start": m.get("start_label", "default"),
        }


_PARAM_FMT = {"alpha": "α={}", "friction": "fric×{}", "damping": "damp×{}", "gravity": "g×{}",
              "mass": "m×{}", "inertia": "I×{}", "tau_lim": "τlim×{}", "q_start": "start={}",
              "controller": "{}"}


def param_text(p: dict, keys=None) -> str:
    keys = list(p) if keys is None else keys
    return ", ".join(_PARAM_FMT[k].format(f"{p[k]:g}" if isinstance(p[k], float) else p[k])
                     for k in keys)


def legend_labels(runs: list) -> tuple[list, str]:
    """Per-run legend text from the parameters that DIFFER between the runs, and a text with the
    parameters that are identical in all runs (for the figure title)."""
    ps = [r.params for r in runs]
    diff = [k for k in ps[0] if len({str(p[k]) for p in ps}) > 1]
    same = [k for k in ps[0] if k not in diff and k != "controller"]
    labels = [param_text(p, diff) if diff else param_text(p, ["controller"]) for p in ps]
    return labels, "common: " + param_text(ps[0], same)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _save(fig, path):
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _title(run: RunData) -> str:
    return f"{run.name}\n{param_text(run.params)}"


def _status_line(run: RunData) -> str:
    m = run.metrics
    tr = m["t_reach_s"]
    return (f"{m['failure_class']}: final error {1e3 * m['final_dist_m']:.3g} mm"
            f" (threshold {1e3 * m['success_threshold_m']:.3g} mm)"
            + (f", reached at {tr:.2f} s" if tr is not None else ""))


def _equal_3d(ax, pts):
    lo, hi = pts.min(0), pts.max(0)
    # nearly planar paths: give every axis at least 25 % of the largest span (and 2 cm) so that
    # tick labels do not collapse; the box aspect stays metric (equal scale on all axes)
    span = np.maximum(hi - lo, max(0.25 * float((hi - lo).max()), 0.02))
    mid = 0.5 * (hi + lo)
    # limits FIRST, box aspect afterwards (the reverse order re-expands the limits in matplotlib 3.11)
    ax.set_xlim(mid[0] - span[0] / 2, mid[0] + span[0] / 2)
    ax.set_ylim(mid[1] - span[1] / 2, mid[1] + span[1] / 2)
    ax.set_zlim(mid[2] - span[2] / 2, mid[2] + span[2] / 2)
    ax.set_box_aspect(span)


# --------------------------------------------------------------------------
# per-run plots
# --------------------------------------------------------------------------
def plot_tcp_3d(run: RunData, path: str) -> None:
    a = run.a
    P, R, goal = a["tcp"], a["tcp_ref"], a["goal"]
    fig = plt.figure(figsize=(12, 5.5), layout="constrained")
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot(*R.T, "k--", lw=1, label="ideal DS rollout (reference)")
    ax.plot(*P.T, color="tab:orange", lw=1.8, label="TCP (simulated)")
    ax.scatter(*P[0], color="tab:green", s=50, marker="o", label="TCP start", zorder=5)
    ax.scatter(*goal, color="tab:red", s=90, marker="*", label="goal", zorder=5)
    ax.scatter(*P[-1], color="tab:blue", s=30, marker="x", label="TCP end", zorder=5)
    _equal_3d(ax, np.vstack([P, R, goal[None]]))
    ax.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]", title="3D TCP path")
    ax.legend(fontsize=8, loc="upper left")
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(R[:, 0], R[:, 1], "k--", lw=1, label="ideal DS rollout")
    ax2.plot(P[:, 0], P[:, 1], color="tab:orange", lw=1.8, label="TCP")
    ax2.plot(*P[0, :2], "o", color="tab:green", ms=8, label="start")
    ax2.plot(*goal[:2], "*", color="tab:red", ms=14, label="goal")
    ax2.plot(*P[-1, :2], "x", color="tab:blue", ms=8, label="end")
    ax2.set(xlabel="x [m]", ylabel="y [m]", title="top view (DS plane)", aspect="equal")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)
    fig.suptitle(_title(run) + "\n" + _status_line(run), fontsize=10)
    _save(fig, path)


def plot_tracking(run: RunData, path: str) -> None:
    a, t = run.a, run.a["t"]
    thr = 1e3 * run.metrics["success_threshold_m"]
    fig, axs = plt.subplots(3, 1, figsize=(9, 9), sharex=True, layout="constrained")
    axs[0].plot(t, 1e3 * a["err_sync"], color="tab:red")
    axs[0].set(ylabel="time-synchronous error [mm]",
               title="||p_TCP(t) − p_ref(t)||, p_ref = ideal rollout at the same time")
    axs[1].plot(t, 1e3 * a["err_geom"], color="tab:purple")
    axs[1].set(ylabel="geometric deviation [mm]",
               title="distance of p_TCP(t) to the ideal PATH (timing-independent)")
    axs[2].semilogy(t, np.maximum(1e3 * a["dist_goal"], 1e-9), color="tab:blue")
    axs[2].axhline(thr, color="gray", ls=":", label=f"success threshold {thr:g} mm")
    axs[2].set(ylabel="distance to goal [mm]", xlabel="time [s]", title="convergence")
    axs[2].legend(fontsize=8)
    for ax in axs:
        ax.grid(alpha=0.3)
    fig.suptitle(_title(run) + "\n" + _status_line(run), fontsize=10)
    _save(fig, path)


def plot_joints(run: RunData, path: str) -> None:
    a, t = run.a, run.a["t"]
    lim = run.manifest["robot_limits"]
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True, layout="constrained")
    for j in range(a["q"].shape[1]):
        c = JOINT_COLORS[j]
        axs[0].plot(t, a["q"][:, j], color=c, label=f"q{j + 1}")
        axs[1].plot(t, a["qd"][:, j], color=c, label=f"q̇{j + 1}")
        axs[0].axhline(lim["q_min_rad"][j], color=c, ls=":", lw=0.7)
        axs[0].axhline(lim["q_max_rad"][j], color=c, ls=":", lw=0.7)
        axs[1].axhline(lim["qd_max_rad_s"][j], color=c, ls=":", lw=0.7)
        axs[1].axhline(-lim["qd_max_rad_s"][j], color=c, ls=":", lw=0.7)
    axs[0].set(ylabel="joint position [rad]", title="measured joint state (dotted: joint limits)")
    axs[1].set(ylabel="joint velocity [rad/s]", xlabel="time [s]")
    for ax in axs:
        ax.grid(alpha=0.3)
        ax.legend(ncol=7, fontsize=8, loc="upper right")
    fig.suptitle(_title(run), fontsize=10)
    _save(fig, path)


def plot_kinematics(run: RunData, path: str) -> None:
    a, t = run.a, run.a["t"]
    an = run.manifest["analysis"]
    note = (f"acceleration/jerk are DERIVED (Savitzky-Golay, window {an['sg_window_s']} s, "
            f"order {an['sg_polyorder']}), not measured")
    fig, axs = plt.subplots(3, 2, figsize=(13, 9), sharex=True, layout="constrained")
    axs[0, 0].plot(t, np.linalg.norm(a["tcp_vel"], axis=1), color="tab:orange")
    axs[0, 0].set(ylabel="TCP speed [m/s]", title="TCP speed = ||J_p(q) q̇|| (exact)")
    axs[1, 0].semilogy(t, np.maximum(np.linalg.norm(a["d_tcp_acc"], axis=1), 1e-9),
                       color="tab:orange")
    axs[1, 0].set(ylabel="TCP acceleration norm [m/s²]", title="derived (log scale)")
    axs[2, 0].semilogy(t, np.maximum(np.linalg.norm(a["d_tcp_jerk"], axis=1), 1e-9),
                       color="tab:orange")
    axs[2, 0].set(ylabel="TCP jerk norm [m/s³]", xlabel="time [s]", title="derived (log scale)")
    for j in range(a["qd"].shape[1]):
        c = JOINT_COLORS[j]
        axs[0, 1].plot(t, a["qd"][:, j], color=c, label=f"q̇{j + 1}")
        axs[1, 1].semilogy(t, np.maximum(np.abs(a["d_qdd"][:, j]), 1e-9), color=c)
        axs[2, 1].semilogy(t, np.maximum(np.abs(a["d_qddd"][:, j]), 1e-9), color=c)
    axs[0, 1].set(ylabel="joint velocity [rad/s]", title="measured")
    axs[1, 1].set(ylabel="|joint acceleration| [rad/s²]", title="derived (log scale)")
    axs[2, 1].set(ylabel="|joint jerk| [rad/s³]", xlabel="time [s]", title="derived (log scale)")
    axs[0, 1].legend(ncol=4, fontsize=7)
    for ax in axs.flat:
        ax.grid(alpha=0.3)
    fig.suptitle(_title(run) + "\n" + note, fontsize=10)
    _save(fig, path)


def plot_torques(run: RunData, path: str) -> None:
    a, t = run.a, run.a["t"]
    lim = a["tau_limit"]
    n = a["tau_cmd"].shape[1]
    fig, axs = plt.subplots(4, 2, figsize=(14, 11), sharex=True, layout="constrained")
    for j in range(n):
        ax = axs.flat[j]
        ax.plot(t, a["tau_cmd"][:, j], color="tab:gray", lw=1.0, label="commanded")
        ax.plot(t, a["tau_app"][:, j], color=JOINT_COLORS[j], lw=1.4, label="applied")
        ax.axhline(lim[j], color="k", ls="--", lw=0.8, label="± limit")
        ax.axhline(-lim[j], color="k", ls="--", lw=0.8)
        sat = a["sat_joint"][:, j].astype(bool)
        if sat.any():
            ax.fill_between(t, -lim[j], lim[j], where=sat, color="red", alpha=0.15,
                            step="post", label="saturated")
        ax.set(ylabel=f"τ{j + 1} [Nm]")
        ax.grid(alpha=0.3)
    ax = axs.flat[n]
    ratio_c = max_ratio(a["tau_cmd"], lim)
    ratio_a = max_ratio(a["tau_app"], lim)
    ax.plot(t, ratio_c, color="tab:gray", label="max_j |commanded| / limit")
    ax.plot(t, ratio_a, color="tab:orange", label="max_j |applied| / limit")
    ax.axhline(1.0, color="k", ls="--", lw=0.8)
    ax.set(ylabel="torque / limit [-]")
    for ax in axs[-1]:
        ax.set_xlabel("time [s]")
    axs.flat[0].legend(fontsize=7, loc="upper right")
    axs.flat[n].legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)
    m = run.metrics
    fig.suptitle(_title(run) + f"\nsaturated steps: {m['n_saturated_steps']} "
                 f"({100 * m['saturated_fraction']:.2f} %), peak commanded/limit "
                 f"{m['peak_cmd_torque_ratio']:.2f}", fontsize=10)
    _save(fig, path)


def plot_controller(run: RunData, path: str) -> None:
    a, t = run.a, run.a["t"]
    if run.controller == "servo":
        fig, axs = plt.subplots(2, 1, figsize=(9, 7), sharex=True, layout="constrained")
        for j in range(a["q_gap"].shape[1]):
            axs[0].plot(t, a["q_gap"][:, j], color=JOINT_COLORS[j], label=f"j{j + 1}")
        axs[0].set(ylabel="q_des − q [rad]", title="servo target minus measured position")
        axs[0].legend(ncol=7, fontsize=7)
        axs[1].plot(t, a["sigma_min_Jp"], color="tab:green")
        axs[1].set(ylabel="σ_min(J_p) [m/rad]", xlabel="time [s]",
                   title="smallest singular value of the position Jacobian")
        for ax in axs:
            ax.grid(alpha=0.3)
    else:
        fig, axs = plt.subplots(3, 2, figsize=(13, 9), sharex=True, layout="constrained")
        axs[0, 0].plot(t, a["pic_s"], color="tab:purple")
        axs[0, 0].set(ylabel="tank s [J]", title="energy tank")
        axs[0, 1].plot(t, a["pic_beta_R"], label="β_R")
        axs[0, 1].plot(t, a["pic_beta_s"], label="β_s")
        axs[0, 1].plot(t, a["pic_alpha"], label="α(s)")
        axs[0, 1].set(ylabel="coupling [-]", title="coupling functions")
        axs[0, 1].legend(fontsize=8)
        axs[1, 0].plot(t, a["pic_z"], color="tab:brown")
        axs[1, 0].set(ylabel="z = ẋ·f_R [(m/s)²]", title="power-like term z")
        axs[1, 1].plot(t, np.linalg.norm(a["F_task"], axis=1), label="|F| task force")
        axs[1, 1].plot(t, np.linalg.norm(a["F_ext"], axis=1), label="|F_ext| pulse")
        axs[1, 1].set(ylabel="force [N]", title="TCP forces")
        axs[1, 1].legend(fontsize=8)
        axs[2, 0].plot(t, a["pic_KE"], color="tab:red")
        axs[2, 0].set(ylabel="kinetic energy [J]", xlabel="time [s]")
        axs[2, 1].plot(t, a["sigma_min_Jp"], color="tab:green")
        axs[2, 1].set(ylabel="σ_min(J_p) [m/rad]", xlabel="time [s]",
                      title="smallest singular value of the position Jacobian")
        for ax in axs.flat:
            ax.grid(alpha=0.3)
    fig.suptitle(_title(run), fontsize=10)
    _save(fig, path)


def make_run_plots(run: RunData, outdir: str | None = None) -> list:
    outdir = outdir or os.path.join(run.path, "plots")
    os.makedirs(outdir, exist_ok=True)
    jobs = [("01_tcp_3d.png", plot_tcp_3d), ("02_tracking_error.png", plot_tracking),
            ("03_joints.png", plot_joints), ("04_kinematics.png", plot_kinematics),
            ("05_torques.png", plot_torques), ("06_controller.png", plot_controller)]
    paths = []
    for name, fn in jobs:
        p = os.path.join(outdir, name)
        fn(run, p)
        paths.append(p)
    return paths


# --------------------------------------------------------------------------
# comparison
# --------------------------------------------------------------------------
def make_compare_plots(runs: list, outdir: str, title: str = "") -> list:
    """Overlay several runs. Legend = parameters that differ between the runs."""
    os.makedirs(outdir, exist_ok=True)
    labels, common = legend_labels(runs)
    cols = plt.cm.tab10(np.arange(len(runs)) % 10)
    multi_ctrl = len({r.controller for r in runs}) > 1
    if multi_ctrl and not any("position" in lb or "PIC" in lb for lb in labels):
        labels = [f"{CTRL_LABEL.get(r.controller, r.controller)} | {lb}"
                  for r, lb in zip(runs, labels)]
    ls = ["-", "--", ":", "-."]
    styles = {c: ls[i % 4] for i, c in enumerate(sorted({r.controller for r in runs}))}
    head = (title + "\n" if title else "") + common
    paths = []

    # 1) path in the DS plane
    fig, ax = plt.subplots(figsize=(8, 7), layout="constrained")
    ref = runs[0].a["tcp_ref"]
    ax.plot(ref[:, 0], ref[:, 1], "k--", lw=1, label="ideal DS rollout (first run)")
    for r, lb, c in zip(runs, labels, cols):
        ax.plot(r.a["tcp"][:, 0], r.a["tcp"][:, 1], color=c, ls=styles[r.controller], label=lb)
    ax.plot(*runs[0].a["tcp"][0, :2], "o", color="tab:green", ms=9, label="start")
    ax.plot(*runs[0].a["goal"][:2], "*", color="tab:red", ms=15, label="goal")
    ax.set(xlabel="x [m]", ylabel="y [m]", aspect="equal", title="TCP path (top view)\n" + head)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    paths.append(os.path.join(outdir, "cmp_01_path.png"))
    _save(fig, paths[-1])

    # 2) 3D
    fig = plt.figure(figsize=(9, 7), layout="constrained")
    ax = fig.add_subplot(projection="3d")
    ax.plot(*ref.T, "k--", lw=1, label="ideal DS rollout (first run)")
    allp = [ref]
    for r, lb, c in zip(runs, labels, cols):
        ax.plot(*r.a["tcp"].T, color=c, ls=styles[r.controller], label=lb)
        allp.append(r.a["tcp"])
    ax.scatter(*runs[0].a["tcp"][0], color="tab:green", s=50, label="start")
    ax.scatter(*runs[0].a["goal"], color="tab:red", s=90, marker="*", label="goal")
    _equal_3d(ax, np.vstack(allp))
    ax.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]", title="TCP path (3D)\n" + head)
    ax.legend(fontsize=7)
    paths.append(os.path.join(outdir, "cmp_02_path3d.png"))
    _save(fig, paths[-1])

    # 3) time series
    series = [("dist_goal", 1e3, "distance to goal [mm]", True),
              ("err_sync", 1e3, "time-synchronous error [mm]", False),
              ("err_geom", 1e3, "geometric deviation [mm]", False),
              ("tcp_speed", 1.0, "TCP speed [m/s]", False),
              ("d_tcp_acc", 1.0, "TCP acceleration norm [m/s²] (derived)", "log"),
              ("d_tcp_jerk", 1.0, "TCP jerk norm [m/s³] (derived)", "log"),
              ("tau_ratio_app", 1.0, "max_j |applied τ| / limit [-]", False),
              ("tau_ratio_cmd", 1.0, "max_j |commanded τ| / limit [-]", False)]
    fig, axs = plt.subplots(4, 2, figsize=(14, 13), sharex=True, layout="constrained")
    for ax, (key, sc, yl, logy) in zip(axs.flat, series):
        for r, lb, c in zip(runs, labels, cols):
            a = r.a
            if key == "tcp_speed":
                y = np.linalg.norm(a["tcp_vel"], axis=1)
            elif key in ("d_tcp_acc", "d_tcp_jerk"):
                y = np.maximum(np.linalg.norm(a[key], axis=1), 1e-9)
            elif key == "tau_ratio_app":
                y = max_ratio(a["tau_app"], a["tau_limit"])
            elif key == "tau_ratio_cmd":
                y = max_ratio(a["tau_cmd"], a["tau_limit"])
            else:
                y = a[key]
            ax.plot(a["t"], sc * y if not logy else np.maximum(sc * y, 1e-9), color=c,
                    ls=styles[r.controller], lw=1.3, label=lb)
        if logy:
            ax.set_yscale("log")
        if key == "dist_goal":
            thr = 1e3 * runs[0].metrics["success_threshold_m"]
            ax.axhline(thr, color="gray", ls=":", label="success threshold")
        if key.startswith("tau_ratio"):
            ax.axhline(1.0, color="k", ls="--", lw=0.8)
        ax.set(ylabel=yl)
        ax.grid(alpha=0.3)
    for ax in axs[-1]:
        ax.set_xlabel("time [s]")
    axs.flat[0].legend(fontsize=7)
    fig.suptitle(head, fontsize=10)
    paths.append(os.path.join(outdir, "cmp_03_timeseries.png"))
    _save(fig, paths[-1])

    # 4) joint states
    fig, axs = plt.subplots(2, 1, figsize=(11, 8), sharex=True, layout="constrained")
    for r, lb, c in zip(runs, labels, cols):
        for j in range(r.a["q"].shape[1]):
            axs[0].plot(r.a["t"], r.a["q"][:, j], color=c, ls=styles[r.controller], lw=0.9,
                        label=lb if j == 0 else None)
            axs[1].plot(r.a["t"], r.a["qd"][:, j], color=c, ls=styles[r.controller], lw=0.9)
    axs[0].set(ylabel="joint positions q1..q7 [rad]", title="joint states, all joints per run\n" + head)
    axs[1].set(ylabel="joint velocities [rad/s]", xlabel="time [s]")
    axs[0].legend(fontsize=7)
    for ax in axs:
        ax.grid(alpha=0.3)
    paths.append(os.path.join(outdir, "cmp_04_joints.png"))
    _save(fig, paths[-1])
    return paths
