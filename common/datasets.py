"""Normierung fuer 2D_Shapes / 3D_Shapes (wie S2NNDS/main.py) und Roboteraufnahmen.

main.py, generate_demo_data:
  2D_Shapes: CSVs in Datasets/2D_Shapes/<name>/, Unterabtastung um
             config.hyperparameters.subsample, Verschiebung auf den Endpunkt
             (pos - pos[:, -1]), ps = globales max |x| aller Demos.
  3D_Shapes: Datasets/3D_Shapes/3D_<name>.mat, Feld "data", Zeilen 0-2 = Position,
             Verschiebung auf den Endpunkt, ps wie oben.
  x0_n = Mittel der Demo-Starts / ps (entspricht initial_set_center).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from .ds import S2NNDS_DIR, LasaScaling


def _config(dataset: str, name: str) -> dict:
    with open(os.path.join(S2NNDS_DIR, "config_files", dataset, name + "_config.json")) as fh:
        return json.load(fh)


def shape_scaling(dataset: str, name: str) -> LasaScaling:
    if dataset == "LASA":
        from .ds import lasa_scaling
        return lasa_scaling(name)
    cfg = _config(dataset, name)
    pos, vel = [], []
    if dataset == "2D_Shapes":
        folder = os.path.join(S2NNDS_DIR, "Datasets", "2D_Shapes", name)
        sub = cfg["hyperparameters"].get("subsample", 10)
        for f in sorted(os.listdir(folder)):
            if not f.endswith(".csv"):
                continue
            df = pd.read_csv(os.path.join(folder, f))
            p = np.vstack([df.x, df.y])[:, ::sub]
            v = np.vstack([df.dx, df.dy])[:, ::sub]
            pos.append(p - p[:, -1:])
            vel.append(v)
        dt = 0.01
    elif dataset == "3D_Shapes":
        import scipy.io
        mat = scipy.io.loadmat(os.path.join(S2NNDS_DIR, "Datasets", "3D_Shapes", f"3D_{name}.mat"))
        for traj in np.squeeze(mat["data"]):
            p, v = traj[:3, :], traj[3:, :]
            pos.append(p - p[:, -1:])
            vel.append(v)
        dt = 0.01
    else:
        raise ValueError(dataset)
    P = np.concatenate(pos, axis=1)
    V = np.concatenate(vel, axis=1)
    ps = float(np.max(np.abs(P)))
    vs = float(np.max(np.abs(V)))
    x0 = np.mean([p[:, 0] for p in pos], axis=0) / ps
    return LasaScaling(name, ps, vs, x0, [p.T / ps for p in pos], dt)


def unsafe_set(dataset: str, name: str):
    """Unsafe-Region aus der Original-Config: Liste von (Zentrum_n, Radius_n)."""
    u = _config(dataset, name)["unsafe"]
    if u["shape"] != "Circle":
        raise NotImplementedError(u["shape"])
    c = np.atleast_2d(np.asarray(u["center"], float))
    return [(ci, float(u["radius"])) for ci in c]


# --------------------------------------------------------------------------
# Roboteraufnahmen (libfranka-Log): Zeile 1 JSON-Parameter, Zeile 2 Kopf, Rest Daten
# --------------------------------------------------------------------------
# Schluessel nach dem Befund aus five_obstacle_identify.py: Ebene in beiden Faellen
# horizontal; im ersten Versuch ist das Werkzeug ~54 deg gegen die Ebenennormale gekippt.
RECORDINGS = {
    "tool_tilted": "Robot_Data/2D_Shapes/Five_Obstacle_DS.csv",
    "tool_tilted_perturbed": "Robot_Data/2D_Shapes/Five_Obstacle_DS_perturbed.csv",
    "hand_down": "robot_demonstrations/3D_Shapes/Recording_Five_Obstacle_DS.csv",
}


def load_recording(key: str):
    path = os.path.join(S2NNDS_DIR, RECORDINGS[key])
    with open(path) as fh:
        params = json.loads(fh.readline())
    df = pd.read_csv(path, header=1)
    t = (df["time"].to_numpy() - df["time"].iloc[0]) * 1e-3          # ms -> s
    q = df[[f"q[{i}]" for i in range(7)]].to_numpy()
    # O_T_EE ist spaltenweise (column-major) abgelegt
    T = df[[f"O_T_EE[{i}]" for i in range(16)]].to_numpy().reshape(-1, 4, 4).transpose(0, 2, 1)
    x = df[["x", "y"]].to_numpy()
    v = df[["dx", "dy"]].to_numpy()
    return {"params": params, "t": t, "q": q, "T": T, "x": x, "v": v, "path": path}
