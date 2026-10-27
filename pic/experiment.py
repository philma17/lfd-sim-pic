"""Config-driven experiment runner: YAML in -> unique run directories with data, plots, report.

    python experiments/run_experiment.py configs/experiments/nominal_fr3.yaml
    python experiments/run_experiment.py configs/experiments/speed_scaling.yaml
    python experiments/run_experiment.py configs/experiments/nominal_fr3.yaml --set plant.dof_frictionloss_scale=0

Result layout (never overwritten; a new batch id is created for every call):

    results/runs/<YYYYmmdd-HHMMSS>_<experiment>/
        batch_report.md, batch_summary.csv, batch.json
        <condition>__<controller>/
            manifest.json  provenance: git, hashes, versions, resolved parameters, applied plant
            config_resolved.yaml
            metrics.json   curated metrics + status
            traj.npz       all arrays (measured + exact + derived), see pic/analysis.py
            trajectory.csv same data, one row per sample
            report.md, plots/*.png
        compare/...        overlay plots across the runs of the batch

Exit codes: 0 all runs finished and were saved; 1 at least one run raised an error or failed to
save/plot (details in <run>/error.txt); 3 only with --strict: all runs finished, but at least one
did not converge. "Not converged" is a legitimate scientific outcome and is NOT an error.
"""
from __future__ import annotations

import copy
import csv
import datetime as dt_mod
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True

import mujoco
import numpy as np
import yaml

from common import sim_dynamic
from common.scenario import ROOT
from common.sim_dynamic import TAU_MAX_DOC

from . import analysis, plots
from .controller import PICParams
from .sim_pic import build_case, run as run_pic

CONFIG_DIR = os.path.join(ROOT, "configs")
RESULTS_ROOT = os.path.join(ROOT, "results", "runs")
CONTROLLERS = ("servo", "pic", "pic_pulse")
# top-level keys of a run config (picds.yaml keys + additions); anything else is a typo
CONFIG_KEYS = {"shape", "dataset", "model_source", "time_rate", "robot", "sim", "task", "plant",
               "pic", "disturbance", "validation", "ds_speed_factor", "analysis"}
META_KEYS = {"extends", "experiment", "conditions"}
# keys that may be ADDED by a condition/--set although the base config does not contain them
ADDABLE = {("ds_speed_factor",), ("task", "q_start_rad")}
ADDABLE_PARENTS = {("plant",), ("analysis",)}


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------
def _resolve_yaml(name: str, relative_to: str | None = None) -> str:
    cands = [name]
    if relative_to:
        cands.append(os.path.join(relative_to, name))
    cands += [os.path.join(ROOT, name), os.path.join(CONFIG_DIR, name),
              os.path.join(CONFIG_DIR, "experiments", name)]
    for c in list(cands):
        if not c.endswith((".yaml", ".yml")):
            cands.append(c + ".yaml")
    for c in cands:
        if os.path.isfile(c):
            return os.path.abspath(c)
    raise FileNotFoundError(f"config '{name}' not found; tried: {cands}")


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) \
            else copy.deepcopy(v)
    return out


def load_experiment_file(name: str, _seen=()) -> tuple[dict, list]:
    """Load a YAML file and resolve `extends` (child values win, dicts merge, lists replace).
    Returns (merged dict, list of absolute file paths that contributed)."""
    path = _resolve_yaml(name)
    if path in _seen:
        raise ValueError(f"circular 'extends': {path}")
    with open(path, encoding="utf-8") as fh:
        d = yaml.safe_load(fh) or {}
    files = [path]
    if "extends" in d:
        base, bfiles = load_experiment_file(_resolve_yaml(d["extends"], os.path.dirname(path)),
                                            _seen + (path,))
        files = bfiles + files
        d = {**deep_merge(base, {k: v for k, v in d.items() if k != "extends"}),
             "extends": d["extends"]}
    return d, files


def set_path(d: dict, dotted: str, value) -> None:
    """d[a][b][c] = value for 'a.b.c'. A typo cannot silently create a new key: the key must
    already exist, or be whitelisted (plant.*, analysis.*, ds_speed_factor, task.q_start_rad)."""
    parts = tuple(dotted.split("."))
    if parts[:-1] in ADDABLE_PARENTS:
        d.setdefault(parts[0], {})                          # plant / analysis may be created
    cur = d
    for p in parts[:-1]:
        if not isinstance(cur.get(p), dict):
            raise KeyError(f"cannot set '{dotted}': '{p}' is not a section of the config")
        cur = cur[p]
    if parts[-1] not in cur and parts not in ADDABLE and parts[:-1] not in ADDABLE_PARENTS:
        raise KeyError(f"unknown config key '{dotted}' (typo?). Existing keys at this level: "
                       f"{sorted(cur)}")
    cur[parts[-1]] = value


def parse_set(items: list) -> dict:
    out = {}
    for it in items or []:
        if "=" not in it:
            raise ValueError(f"--set expects key=value, got '{it}'")
        k, v = it.split("=", 1)
        out[k.strip()] = yaml.safe_load(v)
    return out


def sanitize(s: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(s))


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_obj(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------
def git_info() -> dict:
    def g(*a):
        return subprocess.run(["git", "-C", ROOT, *a], capture_output=True, text=True, timeout=30)
    try:
        head = g("rev-parse", "HEAD")
        if head.returncode != 0:
            return {"available": False, "reason": head.stderr.strip()}
        st = g("status", "--porcelain")
        return {"available": True, "commit": head.stdout.strip(),
                "branch": g("rev-parse", "--abbrev-ref", "HEAD").stdout.strip(),
                "dirty": bool(st.stdout.strip()),
                "dirty_files": [ln[3:] for ln in st.stdout.splitlines()][:50]}
    except (OSError, subprocess.SubprocessError) as e:
        return {"available": False, "reason": repr(e)}


def environment_info() -> dict:
    info = {"python": sys.version.split()[0], "platform": platform.platform(),
            "numpy": np.__version__, "mujoco": mujoco.__version__}
    for mod in ("scipy", "torch", "matplotlib", "yaml"):
        try:
            info[mod] = __import__(mod).__version__
        except ImportError:                                  # recorded, not hidden
            info[mod] = "not installed"
    return info


def model_hashes(case) -> dict:
    from common import ds as dsmod
    from common.robots import MENAGERIE_DIR
    out = {"fr3_mjcf_sha256": sha256_file(os.path.join(MENAGERIE_DIR, "franka_fr3", "fr3.xml")),
           "ds_model_pth_sha256": sha256_file(dsmod.pth_path(case.cfg["shape"],
                                                             case.cfg.get("dataset", "LASA")))}
    for f in sorted(os.listdir(os.path.join(ROOT, "common"))):
        if f.endswith(".py"):
            out[f"common/{f}"] = sha256_file(os.path.join(ROOT, "common", f))
    return out


# --------------------------------------------------------------------------
# running
# --------------------------------------------------------------------------
def execute_controller(controller: str, case, cfg: dict):
    """Run one controller on a prepared Case. Returns the raw result dict."""
    if controller == "servo":
        return sim_dynamic.run(case.robot, case.ds, case.emb, case.x0_n, case.rate, case.tgt,
                               case.sp, q_start=case.q_start, n_substeps=1,
                               tau_limit=case.tau_limit)
    pc = cfg["pic"]
    params = PICParams.from_cfg(pc)
    ns = pc["null_space"]
    null_d_q = float(ns["d_q"]) if ns["enabled"] else 0.0
    clip = bool(pc["torque_limit"]["clip"])
    pulses = []
    if controller == "pic_pulse":
        pulses = cfg["disturbance"]["pulses"]
        if not pulses:
            raise ValueError("controller 'pic_pulse' needs disturbance.pulses in the config")
    return run_pic(case, params, null_d_q=null_d_q, clip=clip, pulses=pulses)


def write_csv(path: str, arr: dict, stride: int = 1) -> None:
    """CSV copy of the data, every `stride`-th sample (traj.npz always holds all samples)."""
    stride = int(stride)
    if stride < 1:
        raise ValueError("analysis.csv_stride must be >= 1")
    arr = {k: (v[::stride] if np.ndim(v) >= 1 and len(v) == len(arr["t"]) else v)
           for k, v in arr.items()}
    t = arr["t"]
    N1 = len(t)
    cols, names = [t], ["t_s"]

    def add(prefix, mat, unit, labels=None):
        mat = np.asarray(mat, float)
        if mat.ndim == 1:
            mat = mat[:, None]
        for j in range(mat.shape[1]):
            lab = labels[j] if labels else (str(j + 1) if mat.shape[1] > 1 else "")
            names.append(f"{prefix}{lab}_{unit}")
            cols.append(mat[:, j])

    xyz = ["_x", "_y", "_z"]
    add("q", arr["q"], "rad")
    add("qd", arr["qd"], "rad_s")
    add("tcp", arr["tcp"], "m", [s[1:] for s in xyz])
    add("tcp_ref", arr["tcp_ref"], "m", [s[1:] for s in xyz])
    add("goal", np.tile(arr["goal"], (N1, 1)), "m", [s[1:] for s in xyz])
    add("err_sync", arr["err_sync"], "m")
    add("err_geom", arr["err_geom"], "m")
    add("dist_goal", arr["dist_goal"], "m")
    add("tcp_vel", arr["tcp_vel"], "m_s", [s[1:] for s in xyz])
    add("d_tcp_acc", arr["d_tcp_acc"], "m_s2", [s[1:] for s in xyz])
    add("d_tcp_jerk", arr["d_tcp_jerk"], "m_s3", [s[1:] for s in xyz])
    add("d_qdd", arr["d_qdd"], "rad_s2")
    add("d_qddd", arr["d_qddd"], "rad_s3")
    add("tau_cmd", arr["tau_cmd"], "Nm")
    add("tau_app", arr["tau_app"], "Nm")
    add("sat_joint", arr["sat_joint"], "flag")
    add("sat_any", arr["sat_any"], "flag")
    add("sigma_min_Jp", arr["sigma_min_Jp"], "m_rad")
    add("manipulability_Jp", arr["manipulability_Jp"], "-")
    for k in ("q_des", "q_gap"):
        if k in arr:
            add(k, arr[k], "rad")
    for k in sorted(x for x in arr if x.startswith("pic_")):
        add(k, arr[k], "-")
    for k in ("F_task", "F_ext"):
        if k in arr:
            add(k, arr[k], "N", [s[1:] for s in xyz])
    if "tau_grav" in arr:
        add("tau_grav", arr["tau_grav"], "Nm")
    header = ",".join(names)
    np.savetxt(path, np.column_stack(cols), delimiter=",", header=header, comments="",
               fmt="%.7g")


def run_one(controller: str, label: str, case, cfg: dict, batch_dir: str, batch_id: str,
            exp_name: str, config_files: list, start_label: str, argv: list,
            make_plots: bool) -> dict:
    run_id = f"{sanitize(label)}__{controller}"
    rdir = os.path.join(batch_dir, run_id)
    os.makedirs(rdir, exist_ok=False)                        # never reuse a run directory
    an = analysis.analysis_params(cfg)
    lim = {"q_min_rad": case.robot.q_min.tolist(), "q_max_rad": case.robot.q_max.tolist(),
           "qd_max_rad_s": case.robot.qd_max.tolist(),
           "tau_doc_nominal_Nm": TAU_MAX_DOC["FR3"][0].tolist()}
    manifest = {
        "status": "incomplete", "run_id": run_id, "batch_id": batch_id, "label": label,
        "experiment": exp_name, "controller": controller,
        "created_utc": dt_mod.datetime.now(dt_mod.timezone.utc).isoformat(timespec="seconds"),
        "command_line": argv, "config_files": config_files,
        "config_sha256": sha256_obj(cfg), "git": git_info(), "environment": environment_info(),
        "robot": "FR3", "execution": "dynamic simulation (MuJoCo, not kinematic, not ideal)",
        "interface": ("joint position servo, ctrl = q_des" if controller == "servo" else
                      "direct joint torque via qfrc_applied (actuation disabled)"),
        "scenario": {"shape": cfg["shape"], "dataset": cfg.get("dataset"), "task": cfg["task"],
                     "sim": cfg["sim"], "seed": cfg["sim"].get("seed")},
        "ds_speed_factor": case.speed_factor, "time_rate_effective": case.rate,
        "plant": case.plant_report, "torque_limit_Nm": case.tau_limit.tolist(),
        "controller_params": (None if controller == "servo" else cfg["pic"]),
        "pulses": (cfg["disturbance"]["pulses"] if controller == "pic_pulse" else []),
        "q_start_rad": case.q_start.tolist(), "start_label": start_label,
        "start_ik": {"pos_err_m": case.ik.pos_err, "sigma_min": case.ik.sigma_min,
                     "margin_pct": case.ik.margin_pct},
        "x0_n": case.x0_n.tolist(), "robot_limits": lim, "analysis": an,
        "model_hashes": model_hashes(case),
    }
    with open(os.path.join(rdir, "config_resolved.yaml"), "w", encoding="utf-8") as fh:
        yaml.safe_dump(jsonable(cfg), fh, sort_keys=False)
    _dump(os.path.join(rdir, "manifest.json"), manifest)    # marks the run as incomplete

    out = {"run_id": run_id, "dir": rdir, "controller": controller, "label": label}
    t0 = time.perf_counter()
    try:
        res = execute_controller(controller, case, cfg)
        wall = time.perf_counter() - t0
        arr = analysis.standardize(controller, res, case, case.tau_limit, an)
        metrics = analysis.summarize(controller, res, arr, case, an)
        metrics["wall_time_s"] = wall
        metrics["status"] = "complete"
        np.savez_compressed(os.path.join(rdir, "traj.npz"), **arr)
        write_csv(os.path.join(rdir, "trajectory.csv"), arr, an["csv_stride"])
        _dump(os.path.join(rdir, "metrics.json"), metrics)
        manifest["status"] = "complete"
        manifest["wall_time_s"] = wall
        _dump(os.path.join(rdir, "manifest.json"), manifest)
        if make_plots:
            plots.make_run_plots(plots.RunData(rdir))
        write_run_report(rdir, manifest, metrics, make_plots)
        out.update(metrics=metrics, ok=True)
    except Exception:                          # run boundary: record, continue, exit code != 0
        manifest["status"] = "error"
        _dump(os.path.join(rdir, "manifest.json"), manifest)
        with open(os.path.join(rdir, "error.txt"), "w", encoding="utf-8") as fh:
            traceback.print_exc(file=fh)
        traceback.print_exc()
        out.update(ok=False, metrics=None)
    return out


def _dump(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(jsonable(obj), fh, indent=1)


def write_run_report(rdir: str, man: dict, m: dict, with_plots: bool) -> None:
    pl = man["plant"]["requested"]
    L = [f"# Run `{man['run_id']}`", "",
         f"- **Status:** {m['failure_class']} (data complete)",
         f"- **Controller:** {man['controller']} — {man['interface']}",
         f"- **Execution:** {man['execution']}",
         f"- **Commit:** `{man['git'].get('commit', 'unknown')}`"
         + (" (**working tree dirty**)" if man["git"].get("dirty") else ""),
         f"- **DS speed factor α:** {man['ds_speed_factor']:g}",
         f"- **Plant:** friction×{pl['dof_frictionloss_scale']:g}, damping×{pl['dof_damping_scale']:g}, "
         f"gravity×{pl['gravity_scale']:g}, mass×{pl['mass_scale']:g}, inertia×{pl['inertia_scale']:g}, "
         f"torque limit×{pl['torque_limit_scale']:g}",
         f"- **Start:** `{man['start_label']}`, q0 = {np.round(man['q_start_rad'], 4).tolist()} rad",
         "", "## Key metrics", "", "| metric | value |", "|---|---|"]
    keys = [("converged", "converged (final error < threshold)"),
            ("final_dist_m", "final distance to goal [m]"),
            ("success_threshold_m", "success threshold [m]"), ("t_reach_s", "first time below threshold [s]"),
            ("sim_duration_s", "simulated duration [s]"), ("wall_time_s", "wall-clock time [s]"),
            ("rmse_sync_m", "RMSE time-synchronous error [m]"),
            ("max_err_sync_m", "max time-synchronous error [m]"),
            ("geom_dev_max_m", "max geometric deviation [m]"),
            ("tcp_speed_peak_m_s", "peak TCP speed [m/s]"),
            ("d_tcp_acc_peak_m_s2", "peak TCP acceleration [m/s²] (derived)"),
            ("d_tcp_jerk_peak_m_s3", "peak TCP jerk [m/s³] (derived)"),
            ("d_tcp_acc_peak_after_startup_m_s2", "peak TCP acceleration after start-up window [m/s²]"),
            ("d_tcp_jerk_peak_after_startup_m_s3", "peak TCP jerk after start-up window [m/s³]"),
            ("peak_cmd_torque_ratio", "peak |commanded τ| / limit"),
            ("peak_applied_torque_ratio", "peak |applied τ| / limit"),
            ("n_saturated_steps", "saturated steps"),
            ("saturated_steps_per_joint", "saturated steps per joint"),
            ("n_pos_limit_viol_steps", "joint position limit violations (steps)"),
            ("n_vel_limit_viol_steps", "joint velocity limit violations (steps)"),
            ("sigma_min_Jp_min", "min σ_min(J_p) along path [m/rad]"),
            ("contact_detected", "contact detected")]
    keys += [(k, k) for k in ("tank_s_min_J", "tank_s_max_J", "tank_clamp_steps", "min_beta_R",
                              "max_abs_servo_gap_rad") if k in m]
    for k, lab in keys:
        v = m[k]
        L.append(f"| {lab} | {v:.6g} |" if isinstance(v, float) else f"| {lab} | {v} |")
    L += ["", "Acceleration/jerk are derived by Savitzky-Golay filtering (see `pic/analysis.py`),",
          "all other time series are measured simulator states or exact functions of them.", ""]
    if with_plots:
        L += ["## Plots", ""] + [f"![{f}](plots/{f})" for f in sorted(os.listdir(os.path.join(rdir, "plots")))]
    with open(os.path.join(rdir, "report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


SUMMARY_FIELDS = ["run_id", "controller", "failure_class", "converged", "final_dist_m", "t_reach_s",
                  "rmse_sync_m", "geom_dev_max_m", "tcp_speed_peak_m_s", "d_tcp_acc_peak_m_s2",
                  "d_tcp_jerk_peak_m_s3", "peak_cmd_torque_ratio", "peak_applied_torque_ratio",
                  "saturated_fraction", "sigma_min_Jp_min", "wall_time_s"]


def run_experiment(config: str, sets: dict | None = None, controllers: list | None = None,
                   out_root: str | None = None, make_plots: bool = True, tag: str | None = None,
                   argv: list | None = None) -> dict:
    raw, files = load_experiment_file(config)
    exp = raw.get("experiment") or {}
    name = sanitize(tag or exp.get("name") or os.path.splitext(os.path.basename(files[-1]))[0])
    ctrls = list(controllers or exp.get("controllers") or ["servo", "pic"])
    bad = [c for c in ctrls if c not in CONTROLLERS]
    if bad:
        raise ValueError(f"unknown controllers {bad}; allowed {CONTROLLERS}")
    conditions = raw.get("conditions") or [{"label": "nominal"}]
    base_cfg = {k: v for k, v in raw.items() if k not in META_KEYS}
    unknown = sorted(set(base_cfg) - CONFIG_KEYS)
    if unknown:
        raise ValueError(f"unknown top-level config keys {unknown}")
    labels = [sanitize(c.get("label", f"cond{i}")) for i, c in enumerate(conditions)]
    if len(set(labels)) != len(labels):
        raise ValueError(f"condition labels must be unique: {labels}")

    # resolve every condition BEFORE running anything: configuration errors surface immediately
    resolved = []
    for cond, lab in zip(conditions, labels):
        cfg = copy.deepcopy(base_cfg)
        for k, v in {**(cond.get("set") or {}), **(sets or {})}.items():
            set_path(cfg, k, v)
        resolved.append((lab, cond, cfg))

    os.makedirs(out_root or RESULTS_ROOT, exist_ok=True)
    stamp = dt_mod.datetime.now().strftime("%Y%m%d-%H%M%S")
    batch_id, i = f"{stamp}_{name}", 1
    while os.path.exists(os.path.join(out_root or RESULTS_ROOT, batch_id)):
        i += 1
        batch_id = f"{stamp}_{name}-{i}"
    batch_dir = os.path.join(out_root or RESULTS_ROOT, batch_id)
    os.makedirs(batch_dir)
    print(f"batch {batch_id}  ->  {batch_dir}")

    results = []
    for lab, cond, cfg in resolved:
        try:
            case = build_case(cfg)
        except Exception:                                    # configuration/case boundary
            traceback.print_exc()
            with open(os.path.join(batch_dir, f"error_{lab}.txt"), "w", encoding="utf-8") as fh:
                traceback.print_exc(file=fh)
            results.append({"run_id": f"{lab}__*", "ok": False, "metrics": None,
                            "controller": "-", "label": lab})
            continue
        for c in ctrls:
            print(f"  running {lab} / {c} ...", flush=True)
            r = run_one(c, lab, case, cfg, batch_dir, batch_id, name, files,
                        str(cond.get("start_label", "default")),
                        argv or sys.argv, make_plots)
            results.append(r)
            if r["ok"]:
                m = r["metrics"]
                print(f"    {m['failure_class']:15s} final {1e3 * m['final_dist_m']:9.4f} mm"
                      f"  t_reach {m['t_reach_s']}  peak applied tau/lim {m['peak_applied_torque_ratio']:.3f}"
                      f"  sat steps {m['n_saturated_steps']}  wall {m['wall_time_s']:.1f}s")
    ok_runs = [r for r in results if r["ok"]]
    summary = write_batch_summary(batch_dir, batch_id, name, results, files)
    if make_plots and ok_runs:
        try:
            make_batch_compare(batch_dir, ok_runs, name)
        except Exception:
            traceback.print_exc()
            with open(os.path.join(batch_dir, "error_compare.txt"), "w", encoding="utf-8") as fh:
                traceback.print_exc(file=fh)
            summary["compare_failed"] = True
    summary["batch_dir"] = batch_dir
    return summary


def make_batch_compare(batch_dir: str, ok_runs: list, name: str) -> None:
    runs = [plots.RunData(r["dir"]) for r in ok_runs]
    labels = sorted({r["label"] for r in ok_runs})
    ctrls = sorted({r["controller"] for r in ok_runs})
    if len(labels) > 1:
        for c in ctrls:
            sel = [rr for rr, r in zip(runs, ok_runs) if r["controller"] == c]
            if len(sel) > 1:
                plots.make_compare_plots(sel, os.path.join(batch_dir, "compare", c),
                                         title=f"{name}: {plots.CTRL_LABEL.get(c, c)}")
    elif len(ctrls) > 1:
        plots.make_compare_plots(runs, os.path.join(batch_dir, "compare", "controllers"),
                                 title=f"{name}: controllers")


def write_batch_summary(batch_dir, batch_id, name, results, files) -> dict:
    rows = []
    for r in results:
        m = r["metrics"] or {}
        rows.append({"run_id": r["run_id"], "controller": r["controller"],
                     **{k: m.get(k) for k in SUMMARY_FIELDS[2:]}})
        if not r["ok"]:
            rows[-1]["failure_class"] = "ERROR"
    with open(os.path.join(batch_dir, "batch_summary.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(rows)
    n_err = sum(not r["ok"] for r in results)
    n_nc = sum(1 for r in results if r["ok"] and not r["metrics"]["converged"])
    L = [f"# Batch `{batch_id}`", "", f"- configs: {', '.join(os.path.relpath(f, ROOT) for f in files)}",
         f"- runs: {len(results)}, errors: {n_err}, finished but not converged: {n_nc}", "",
         "| run | class | final error [mm] | t_reach [s] | RMSE sync [mm] | peak τ cmd/lim | "
         "peak τ app/lim | saturated [%] |", "|---|---|---|---|---|---|---|---|"]
    for r, row in zip(results, rows):
        if not r["ok"]:
            L.append(f"| {r['run_id']} | **ERROR** (see error.txt) | | | | | | |")
            continue
        m = r["metrics"]
        tr = "-" if m["t_reach_s"] is None else f"{m['t_reach_s']:.3f}"
        L.append(f"| {r['run_id']} | {m['failure_class']} | {1e3 * m['final_dist_m']:.4g} | {tr} | "
                 f"{1e3 * m['rmse_sync_m']:.3g} | {m['peak_cmd_torque_ratio']:.3f} | "
                 f"{m['peak_applied_torque_ratio']:.3f} | {100 * m['saturated_fraction']:.2f} |")
    L += ["", "`stalled` = not converged and the TCP is (almost) at rest at the end; "
          "`timeout_moving` = not converged but still moving when T_max was reached.", ""]
    with open(os.path.join(batch_dir, "batch_report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    summ = {"batch_id": batch_id, "n_runs": len(results), "n_errors": n_err,
            "n_not_converged": n_nc, "runs": [r["run_id"] for r in results]}
    _dump(os.path.join(batch_dir, "batch.json"), summ)
    return summ
