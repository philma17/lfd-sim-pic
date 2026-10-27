"""Config-driven scenario setup: dataset / shape / start / embedding / unsafe set.

This follows the logic of LfD_baseline `experiments/compare_traj.py::setup` (revision a06ed55,
lines 238-318) and uses the same baseline modules (vendored byte-identical in common/:
datasets.shape_scaling, embedding.make_embedding, unsafe.UnsafeSet). It is a re-implementation as a
function of the config because compare_traj.py is an argparse script of another repository and cannot
be imported; it is NOT a copy of the script, and the baseline files were not touched.

Config keys (all optional; the defaults reproduce the historical LASA/Sine cell exactly):
    dataset: LASA | 2D_Shapes | 3D_Shapes      shape: Sine | Five_Obstacle_DS | Cshape_bottom ...
    task.start: demo_mean                       mean demonstration start (default)
                {index: i}                      i-th start of the verified init_range set, drawn with
                                                multistart.{seed, n_starts} (as compare_traj.py)
                {x0: [..]}                      explicit start in normalized DS coordinates
    task.{center, L_m, plane, anchor}           2D embedding (PlanarEmbedding)
    spatial.{center, L_m, axes, anchor}         3D embedding (SpatialEmbedding); baseline defaults
    multistart.{seed, n_starts}                 default seed 0, n_starts 20 (baseline dynamic_baseline.yaml)

Always the verified network model_f.pth (source "pth"). Models other than LASA/Sine need a full
S2NNDS directory (see experiments/prepare_s2nnds.py) in the environment variable S2NNDS_DIR.
"""
from __future__ import annotations

import json
import os

import numpy as np

from common import ds as dsmod
from common.datasets import shape_scaling
from common.ds import LearnedDS, demo_time_rate
from common.embedding import make_embedding
from common.unsafe import UnsafeSet

SPATIAL_DEFAULT = {"center": [0.50, 0.0, 0.50], "L_m": 0.10,
                   "axes": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], "anchor": [0.0, 0.0, 0.0]}
MULTISTART_DEFAULT = {"seed": 0, "n_starts": 20}


def s2nnds_config(dataset: str, shape: str):
    p = os.path.join(dsmod.S2NNDS_DIR, "config_files", dataset, f"{shape}_config.json")
    if not os.path.exists(p):
        return None, p
    with open(p, encoding="utf-8") as fh:
        return json.load(fh), p


def load_scenario(cfg: dict) -> dict:
    dataset, shape = cfg.get("dataset", "LASA"), cfg["shape"]
    if cfg.get("model_source", "pth") != "pth":
        raise ValueError("only the verified model_f.pth is supported (model_source: pth)")
    ds = LearnedDS(shape, dataset, source="pth")
    dim = int(ds.f.dim)
    sc = shape_scaling(dataset, shape)
    if dim != len(sc.x0_n):
        raise ValueError(f"{dataset}/{shape}: model dim {dim} != demonstration dim {len(sc.x0_n)}")
    if dim not in (2, 3):
        raise ValueError(f"{dim}D models are not supported")
    rate = demo_time_rate(sc) if cfg.get("time_rate", "demo") == "demo" else float(cfg["time_rate"])
    s2, s2_path = s2nnds_config(dataset, shape)
    warnings = []

    # ---- start ---------------------------------------------------------
    t = cfg["task"]
    start = t.get("start", "demo_mean")
    box = None
    if s2 is not None and s2.get("plotting", {}).get("init_range"):
        box = np.asarray(s2["plotting"]["init_range"], float)
    if start == "demo_mean":
        x0, label = np.asarray(sc.x0_n, float), "demo_mean"
    elif isinstance(start, dict) and "x0" in start:
        x0, label = np.asarray(start["x0"], float), "x0"
        if len(x0) != dim:
            raise ValueError(f"task.start.x0 needs {dim} values")
    elif isinstance(start, dict) and "index" in start:
        if box is None:
            raise ValueError(f"no init_range for {dataset}/{shape} (S2NNDS config missing: {s2_path}); "
                             "cannot draw start set (set S2NNDS_DIR, see experiments/prepare_s2nnds.py)")
        ms = {**MULTISTART_DEFAULT, **(cfg.get("multistart") or {})}
        X0 = np.random.default_rng(int(ms["seed"])).uniform(box[:, 0], box[:, 1],
                                                           size=(int(ms["n_starts"]), dim))
        i = int(start["index"])
        if not 0 <= i < len(X0):
            raise ValueError(f"task.start.index must be in 0..{len(X0) - 1}")
        x0, label = X0[i], f"start{i}"
    else:
        raise ValueError(f"unknown task.start {start!r}")
    dom = None
    if s2 is not None:
        dom = np.asarray(s2["domain"]["range"], float)
        if np.any(x0 < dom[:, 0]) or np.any(x0 > dom[:, 1]):
            warnings.append("x0 is outside the S2NNDS training domain")
    if box is not None and (np.any(x0 < box[:, 0]) or np.any(x0 > box[:, 1])):
        warnings.append("x0 is outside the verified initial set (no convergence guarantee)")

    # ---- embedding -----------------------------------------------------
    if dim == 2:
        emb = make_embedding(2, t["center"], float(t["L_m"]), plane=t["plane"], anchor=t["anchor"])
    else:
        sp3 = {**SPATIAL_DEFAULT, **(cfg.get("spatial") or {})}
        emb = make_embedding(3, sp3["center"], float(sp3["L_m"]), axes=sp3["axes"], anchor=sp3["anchor"])

    # ---- unsafe set (optional clearance measure) -------------------------
    unsafe = UnsafeSet(s2.get("unsafe") if s2 else None, dim)
    if not unsafe.supported:
        warnings.append(f"unsafe-set clearance disabled: {unsafe.reason}")
    return {"ds": ds, "sc": sc, "rate": rate, "dim": dim, "x0": x0, "start_label": label, "emb": emb,
            "unsafe": unsafe if unsafe.supported else None, "unsafe_info": unsafe.describe(),
            "s2nnds_config": s2_path if s2 is not None else None, "domain": dom, "init_box": box,
            "warnings": warnings, "dataset": dataset, "shape": shape}
