"""Collect a run_matrix.py result tree into the break matrix (CSV + figure), a long metrics table, the
multi-start alpha table and the direct FR3 vs UR5e comparison plots.

    python -B experiments/make_break_matrix.py results/runs/matrix_main --out results/summary

Cell text = convergence class + the first violated limit (earliest of: torque saturation, joint position
limit, joint velocity limit, obstacle penetration; acceleration/jerk limits are NOT evaluated because no
verified documented values exist in this repository). Nothing here is a generalization: every cell is one run
(one start, seed 0) unless the multi-start table says otherwise.
"""
from __future__ import annotations

import argparse
import csv
import glob
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

VARIATIONS = [  # (tag, display name, level order)
    ("var_alpha", "alpha", ["alpha0.5", "alpha1", "alpha2", "alpha4"]),
    ("var_placement", "placement", ["r0.3_nearbase", "r0.5_nominal", "r0.8_extended"]),
    ("var_com", "COM, nominal g-comp", ["comx_plus5cm", "comx_minus5cm", "comz_plus5cm"]),
    ("var_com_truecomp", "COM, true g-comp", ["comx_plus5cm", "comx_minus5cm", "comz_plus5cm"]),
    ("var_mass", "mass, nominal g-comp", ["mass0.5", "mass1.5", "mass2"]),
    ("var_mass_truecomp", "mass, true g-comp", ["mass0.5", "mass1.5", "mass2"]),
    ("var_inertia", "inertia", ["inertia0.5", "inertia1.5", "inertia2"]),
    ("torque_limit", "torque limit", ["tau1.0", "tau0.5", "tau0.25"]),
]
COMBOS = [("FR3", "none"), ("FR3", "hold"), ("UR5e", "none"), ("UR5e", "hold")]
CLASS_COLOR = {"converged": "#b7e4c7", "slow": "#ffe8a3", "stalled": "#ffd6a5", "saturated": "#ffadad",
               "diverged": "#e5989b", "nonfinite": "#c77dff", "ERROR": "#999999", "-": "#ffffff"}


def scan(root):
    rows = []
    for mpath in sorted(glob.glob(os.path.join(root, "*", "*", "*", "metrics.json"))):
        rd = os.path.dirname(mpath)
        with open(mpath, encoding="utf-8") as fh:
            m = json.load(fh)
        with open(os.path.join(rd, "manifest.json"), encoding="utf-8") as fh:
            man = json.load(fh)
        batch = os.path.basename(os.path.dirname(rd))
        tag = batch.split("_", 1)[1] if "_" in batch else batch
        rows.append({"dir": rd, "tag": tag, "label": man["label"], "robot": man["robot"],
                     "orientation": man["orientation_mode"], "controller": man["controller"], "m": m, "man": man})
    return rows


def cell_text(r):
    if r is None:
        return "-", "-"
    m = r["m"]
    cls = m["failure_class"]
    fv = m["first_violation"]
    if fv["limit"] == "none":
        return cls, cls
    short = {"torque_saturation": "tau-sat", "joint_position_limit": "q-lim", "joint_velocity_limit": "qd-lim",
             "obstacle_penetration": "obstacle"}[fv["limit"]]
    j = f" j{fv['joint']}" if fv["joint"] else ""
    return cls, f"{cls}\n{short}{j} @{fv['t_s']:.1f}s"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "summary"))
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    rows = scan(a.root)
    if not rows:
        print("no runs found", file=sys.stderr)
        return 2
    idx = {(r["tag"], r["robot"], r["orientation"], r["label"], r["controller"]): r for r in rows}

    # ---- long table -----------------------------------------------------
    keys = ["failure_class", "final_dist_m", "t_reach_s", "rmse_sync_m", "geom_dev_max_m", "tcp_speed_peak_m_s",
            "d_tcp_acc_peak_after_startup_m_s2", "d_tcp_jerk_peak_after_startup_m_s3", "qacc_direct_peak_rad_s2",
            "peak_cmd_torque_ratio", "peak_applied_torque_ratio", "saturated_fraction", "peak_qd_ratio_to_limit",
            "min_limit_margin_pct", "sigma_min_Jp_start", "sigma_min_Jp_min", "cond_Jp_start", "cond_Jp_max",
            "sigma_min_J6_min", "cond_J6_max", "max_ori_err_rad", "null_dim", "null_drift_max_rad",
            "clearance_min_m", "tank_s_min_J", "tank_s_max_J", "W_dW_max_J",
            # Savitzky-Golay derived jerk (window 0.05 s, order 3) and acceleration; direct MuJoCo qacc (PIC and servo)
            "d_tcp_jerk_peak_m_s3", "d_qddd_peak_rad_s3", "d_qddd_peak_after_startup_rad_s3", "d_qdd_peak_rad_s2",
            "qacc_direct_peak_after_startup_rad_s2"]
    with open(os.path.join(a.out, "matrix_long.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["variation_tag", "level", "robot", "orientation", "controller", "first_limit", "first_limit_t_s",
                    "first_limit_joint", *keys])
        for r in rows:
            if "_start" in r["tag"]:
                continue
            fv = r["m"]["first_violation"]
            w.writerow([r["tag"], r["label"], r["robot"], r["orientation"], r["controller"], fv["limit"],
                        fv["t_s"], fv["joint"], *[r["m"].get(k) for k in keys]])

    # ---- break matrix ---------------------------------------------------
    table_rows = []                                  # (variation, level, controller)
    for tag, name, levels in VARIATIONS:
        ctrls = sorted({r["controller"] for r in rows if r["tag"] == tag}, reverse=True)   # servo before pic
        for lv in levels:
            for c in ctrls:
                table_rows.append((tag, name, lv, c))
    cells = []
    with open(os.path.join(a.out, "break_matrix.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["variation", "level", "controller"] + [f"{rb}_{o}_class" for rb, o in COMBOS] +
                   [f"{rb}_{o}_first_limit" for rb, o in COMBOS])
        for tag, name, lv, c in table_rows:
            rr = [idx.get((tag, rb, o, lv, c)) for rb, o in COMBOS]
            cl = [cell_text(x) for x in rr]
            fl = [("-" if x is None else (f"{x['m']['first_violation']['limit']}"
                   + (f" j{x['m']['first_violation']['joint']}" if x['m']['first_violation']['joint'] else "")
                   + (f" @{x['m']['first_violation']['t_s']:.2f}s" if x['m']['first_violation']['t_s'] is not None else "")))
                  for x in rr]
            w.writerow([name, lv, c] + [x[0] for x in cl] + fl)
            cells.append([x[1] for x in cl] + [x[0] for x in cl])
    n = len(table_rows)
    fig, ax = plt.subplots(figsize=(15, 0.55 * n + 3), layout="constrained")
    ax.axis("off")
    col_labels = ["variation", "level", "ctrl"] + [f"{rb}\norientation={o}" for rb, o in COMBOS]
    cell_txt = [[nm, lv, c] + [x for x in cells[i][:4]] for i, (tag, nm, lv, c) in enumerate(table_rows)]
    tb = ax.table(cellText=cell_txt, colLabels=col_labels, loc="upper center", cellLoc="center")
    tb.auto_set_font_size(False)
    tb.set_fontsize(7)
    tb.scale(1, 1.9)
    for i in range(n):
        for j in range(4):
            tb[(i + 1, 3 + j)].set_facecolor(CLASS_COLOR.get(cells[i][4 + j], "#ffffff"))
    for j in range(len(col_labels)):
        tb[(0, j)].set_facecolor("#dddddd")
    ax.set_title("Break matrix: convergence class and first violated limit (single start, seed 0, friction x0 diagnostic "
                 "plant, LASA/Sine, paper gains)", fontsize=10)
    fig.savefig(os.path.join(a.out, "break_matrix.png"), dpi=110)
    plt.close(fig)

    # ---- multi-start alpha table ------------------------------------------
    srows = []
    for rb, o in COMBOS:
        for al in ("alpha0.5", "alpha1", "alpha2", "alpha4"):
            for c in ("servo", "pic"):
                sel = [r for r in rows if r["robot"] == rb and r["orientation"] == o and r["controller"] == c and
                       r["label"] == al and "_start" in r["tag"]]
                if not sel:
                    continue
                cls = [r["m"]["failure_class"] for r in sel]
                srows.append({"robot": rb, "orientation": o, "alpha": al, "controller": c, "n_starts_run": len(sel),
                              "n_converged": cls.count("converged"),
                              "classes": ";".join(sorted(set(cls))),
                              "median_t_reach_s": (float(np.median([r["m"]["t_reach_s"] for r in sel if r["m"]["t_reach_s"] is not None]))
                                                   if any(r["m"]["t_reach_s"] is not None for r in sel) else None),
                              "max_final_dist_mm": 1e3 * max(r["m"]["final_dist_m"] for r in sel)})
    if srows:
        with open(os.path.join(a.out, "alpha_starts.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(srows[0]))
            w.writeheader()
            w.writerows(srows)

    # ---- torque-limit sweep (x1, x0.5, x0.25 of the documented limits) ---------
    tl = [r for r in rows if r["tag"] == "torque_limit"]
    if tl:
        with open(os.path.join(a.out, "torque_limit_sweep.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["robot", "orientation", "controller", "limit_scale", "class", "first_saturation_joint",
                        "first_saturation_t_s", "saturated_fraction", "saturated_steps_per_joint",
                        "peak_cmd_torque_ratio_of_scaled_limit", "peak_applied_torque_ratio_of_scaled_limit",
                        "final_dist_m", "first_limit_overall"])
            for r in sorted(tl, key=lambda r: (r["robot"], r["orientation"], r["controller"], r["label"])):
                m = r["m"]
                w.writerow([r["robot"], r["orientation"], r["controller"], r["label"].replace("tau", ""),
                            m["failure_class"], m["first_saturation"]["joint"], m["first_saturation"]["t_s"],
                            m["saturated_fraction"], ";".join(str(x) for x in m["saturated_steps_per_joint"]),
                            m["peak_cmd_torque_ratio"], m["peak_applied_torque_ratio"], m["final_dist_m"],
                            m["first_violation"]["limit"]])

    # ---- alpha x L: where does the torque limit bind first? -------------------
    sl = [r for r in rows if r["tag"] == "var_speed_L"]
    if sl:
        out = []
        for r in sl:
            m = r["m"]
            al, L = r["label"].split("_L")
            fs = m["first_saturation"]
            out.append({"robot": r["robot"], "orientation": r["orientation"], "controller": r["controller"],
                        "alpha": float(al[1:]), "L_m": float(L), "class": m["failure_class"],
                        "first_saturation_joint": fs["joint"], "first_saturation_t_s": fs["t_s"],
                        "first_limit_overall": m["first_violation"]["limit"],
                        "first_limit_overall_t_s": m["first_violation"]["t_s"],
                        "first_limit_overall_joint": m["first_violation"]["joint"],
                        "saturated_fraction": m["saturated_fraction"],
                        "peak_cmd_torque_ratio": m["peak_cmd_torque_ratio"],
                        "peak_applied_torque_ratio": m["peak_applied_torque_ratio"],
                        "saturated_steps_per_joint": ";".join(str(x) for x in m["saturated_steps_per_joint"]),
                        "tcp_speed_peak_m_s": m["tcp_speed_peak_m_s"], "final_dist_m": m["final_dist_m"],
                        "d_tcp_jerk_peak_after_startup_m_s3": m["d_tcp_jerk_peak_after_startup_m_s3"],
                        "qacc_direct_peak_rad_s2": m.get("qacc_direct_peak_rad_s2")})
        out.sort(key=lambda x: (x["robot"], x["orientation"], x["controller"], x["L_m"], x["alpha"]))
        with open(os.path.join(a.out, "speed_L_torque_binding.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out[0]))
            w.writeheader()
            for o in out:
                w.writerow(o)
        fig, axs = plt.subplots(2, 4, figsize=(17, 7.5), layout="constrained")
        alphas, Ls = [1, 2, 4, 8, 16], [0.1, 0.2, 0.3]
        k = 0
        for c in ("pic", "servo"):
            for rb in ("FR3", "UR5e"):
                for o in ("none", "hold"):
                    ax = axs.flat[k]
                    k += 1
                    ax.set_xticks(range(len(alphas)), [f"{x:g}" for x in alphas])
                    ax.set_yticks(range(len(Ls)), [f"{x:g}" for x in Ls])
                    ax.set_xlim(-0.5, len(alphas) - 0.5)
                    ax.set_ylim(-0.5, len(Ls) - 0.5)
                    for i, L in enumerate(Ls):
                        for j, al in enumerate(alphas):
                            q = [x for x in out if (x["robot"], x["orientation"], x["controller"], x["alpha"], x["L_m"]) == (rb, o, c, float(al), L)]
                            if not q:
                                continue
                            q = q[0]
                            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color=CLASS_COLOR.get(q["class"], "#fff"), ec="k", lw=0.5))
                            if q["first_saturation_joint"]:
                                t = f"{q['class']}\nsat j{q['first_saturation_joint']}\n@{q['first_saturation_t_s']:.2f}s"
                            else:
                                t = f"{q['class']}\nno sat."
                            ax.text(j, i, t, ha="center", va="center", fontsize=6)
                    ax.set_title(f"{rb} {c} orientation={o}", fontsize=9)
                    ax.set_xlabel("alpha")
                    ax.set_ylabel("L [m]")
        fig.suptitle("Documented torque limit: first saturated joint and time, alpha x L (friction x0, paper gains, T_max 10 s; single start)", fontsize=10)
        fig.savefig(os.path.join(a.out, "speed_L_torque_binding.png"), dpi=110)
        plt.close(fig)

    # ---- direct FR3 vs UR5e comparison plots ------------------------------
    def load(r):
        return np.load(os.path.join(r["dir"], "traj.npz"))

    def compare(name, tag_levels, ori, title):
        # tag_levels: list of (tag, level)
        nl = len(tag_levels)
        fig, axs = plt.subplots(4, nl, figsize=(3.3 * nl + 1, 10), sharex="col", squeeze=False, layout="constrained")
        col = {"FR3": "tab:blue", "UR5e": "tab:orange"}
        ls = {"pic": "-", "servo": "--"}
        for j, (tag, lv) in enumerate(tag_levels):
            for rb in ("FR3", "UR5e"):
                for c in ("pic", "servo"):
                    r = idx.get((tag, rb, ori, lv, c))
                    if r is None:
                        continue
                    z = load(r)
                    t = z["t"]
                    lab = f"{rb} {'PIC-DS' if c == 'pic' else 'servo'}"
                    axs[0, j].semilogy(t, np.maximum(1e3 * z["dist_goal"], 1e-6), color=col[rb], ls=ls[c], label=lab)
                    axs[1, j].plot(t, np.linalg.norm(z["tcp_vel"], axis=1), color=col[rb], ls=ls[c])
                    ratio = np.abs(z["tau_app"]) / z["tau_limit"]
                    ok = ~np.isnan(ratio).all(axis=1)
                    rr = np.full(len(t), np.nan)
                    rr[ok] = ratio[ok].max(axis=1)
                    axs[2, j].plot(t, rr, color=col[rb], ls=ls[c])
                    axs[3, j].plot(t, z["sigma_min_Jp"], color=col[rb], ls=ls[c])
            axs[0, j].set_title(lv, fontsize=9)
            axs[0, j].axhline(1.0, color="gray", ls=":", lw=0.8)
        axs[0, 0].set_ylabel("distance to goal [mm]\n(dotted: 1 mm threshold)")
        axs[1, 0].set_ylabel("TCP speed [m/s]")
        axs[2, 0].set_ylabel("max_j |applied tau| / limit [-]")
        axs[3, 0].set_ylabel("sigma_min(J_p) [m/rad]")
        for ax in axs[-1]:
            ax.set_xlabel("time [s]")
        for ax in axs.flat:
            ax.grid(alpha=0.3)
        axs[0, 0].legend(fontsize=7)
        fig.suptitle(f"{title}  (orientation={ori}; friction x0 diagnostic plant; single start)", fontsize=10)
        fig.savefig(os.path.join(a.out, name), dpi=110)
        plt.close(fig)

    for ori in ("hold", "none"):
        compare(f"cmp_alpha_{ori}.png", [("var_alpha", l) for l in ("alpha0.5", "alpha1", "alpha2", "alpha4")], ori,
                "FR3 vs UR5e: DS speed factor")
        compare(f"cmp_placement_{ori}.png", [("var_placement", l) for l in ("r0.3_nearbase", "r0.5_nominal", "r0.8_extended")],
                ori, "FR3 vs UR5e: attractor placement")
        compare(f"cmp_com_inertia_{ori}.png", [("var_com", l) for l in ("comx_plus5cm", "comx_minus5cm", "comz_plus5cm")] +
                [("var_inertia", l) for l in ("inertia0.5", "inertia2")], ori, "FR3 vs UR5e: COM shift / inertia")
    print("wrote", a.out, "; runs scanned:", len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
