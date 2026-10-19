"""Baut DS, Einbettung, Ziel und Simulationsparameter aus einer YAML-Config."""
from __future__ import annotations

import os

import numpy as np
import yaml

from .ds import LearnedDS, PlanarEmbedding, demo_time_rate, lasa_scaling
from .ik import IKParams, Target
from .sim import SimParams

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(name: str) -> dict:
    if os.path.isabs(name) or os.path.exists(name):
        path = name
    elif os.path.exists(os.path.join(ROOT, name)):
        path = os.path.join(ROOT, name)
    else:
        path = os.path.join(ROOT, "configs", name)
    with open(path) as fh:
        return yaml.safe_load(fh)


def sim_params(cfg: dict) -> SimParams:
    s = dict(cfg["sim"])
    ik = IKParams(**s.pop("ik", {}))
    return SimParams(ik=ik, **s)


def build_ds(cfg: dict):
    """Model source: cfg["model_source"], default "pth" = the verified network
    S2NNDS/models_verified/<dataset>/<shape>/model_f.pth. "onnx" = models_onnx/<dataset>/<shape>.onnx
    (the default until 2026-10-18; identical weights for LASA/Sine, different networks for
    CShape, PShape_benchmark, Sshape_benchmark, 2D_Shapes/Five_Obstacle_DS, see tests/test_ds.py)."""
    ds = LearnedDS(cfg["shape"], cfg.get("dataset", "LASA"),
                   source=cfg.get("model_source", "pth"))
    sc = lasa_scaling(cfg["shape"])
    if cfg.get("time_rate", "demo") == "demo":
        rate = demo_time_rate(sc)
    else:
        rate = float(cfg["time_rate"])
    return ds, sc, rate


def anchor_point(ds, sc, rate, sp: SimParams, mode) -> np.ndarray:
    if isinstance(mode, (list, tuple)):
        return np.asarray(mode, float)
    if mode == "bbox_center":
        X = ds.rollout_ideal(sc.x0_n, sp.dt, int(round(sp.T_max / sp.dt)), rate)
        return 0.5 * (X.min(0) + X.max(0))
    if mode == "goal":
        return np.zeros(2)
    raise ValueError(mode)


def midpose_setup(cfg: dict, robot, ds, sc, rate, sp: SimParams):
    """Einbettung durch den Home-TCP und Orientierungsziele (nur Phase-0-Test)."""
    m = cfg["midpose"]
    p_home, R_home = robot.fk(robot.q_home)
    anchor = anchor_point(ds, sc, rate, sp, m["anchor"])
    emb = PlanarEmbedding(p_home, m["e1"], m["e2"], m["L"], anchor)
    targets = {
        "pos": Target("pos"),
        "axis5": Target("axis5", a_des=R_home[:, 2].copy()),
        "full6": Target("full6", R_des=R_home.copy()),
    }
    return emb, targets
