"""Gelerntes S2-NNDS als reine Positions-DS.

Fakten aus dem Original-Repo (S2NNDS/main.py, generate_demo_data):
  * x_n = x / ps, ps = globales Maximum |x| ueber ALLE Demos eines Shapes.
  * Zielpunkt der LASA-Demos liegt bereits im Ursprung (kein Shift noetig).
  * f_theta(x_n) ~ v / vs, vs = globales Maximum |v| ueber alle Demos.
  * Rollout im Original (Plotter.py): explizites Euler x_{k+1} = x_k + dt f(x_k).

Hier: v_world = v_peak * (f1 e1 + f2 e2), d.h. x_n_dot = (v_peak / L) f(x_n),
L = Meter pro Normierungseinheit (physikalische Groesse von ps).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import onnx
from onnx import numpy_helper

S2NNDS_DIR = os.environ.get("S2NNDS_DIR", "/home/philipp/S2NNDS")

# LASA-Shape -> Attributname in pyLasaDataset (wie in S2NNDS/main.py)
LASA_DATASET = {
    "Angle": "Angle", "Worm": "Worm", "CShape": "CShape", "GShape": "GShape",
    "Sshape": "Sshape", "Sine": "Sine", "Sine_new": "Sine", "PShape": "PShape",
    "NShape": "NShape",
}


# --------------------------------------------------------------------------
# ONNX -> numpy MLP (float64, deterministisch, schnell fuer batch=1)
# --------------------------------------------------------------------------
_ACT = {
    "Tanh": np.tanh,
    "Relu": lambda x: np.maximum(x, 0.0),
    "Sin": np.sin,
    "Softplus": lambda x: np.logaddexp(0.0, x),
}


class OnnxMLP:
    """Wertet eine lineare Kette aus MatMul/Gemm/Add/Aktivierung aus.

    Unbekannte Knoten fuehren zu einem Fehler (kein stilles Raten).
    """

    def __init__(self, path: str):
        self.path = path
        model = onnx.load(path)
        g = model.graph
        self.input_name = g.input[0].name
        self.output_name = g.output[0].name
        self.dim = int(g.input[0].type.tensor_type.shape.dim[1].dim_value)
        inits = {t.name: numpy_helper.to_array(t).astype(np.float64) for t in g.initializer}
        self.ops = []  # Liste von (typ, parameter)
        cur = self.input_name
        for n in g.node:
            if n.input[0] != cur:
                raise ValueError(f"Nicht-lineare Graphstruktur in {path}: {n.op_type}")
            attrs = {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}
            if n.op_type == "MatMul":
                self.ops.append(("matmul", inits[n.input[1]]))
            elif n.op_type == "Gemm":
                W = inits[n.input[1]]
                if attrs.get("transB", 0):
                    W = W.T
                b = inits[n.input[2]] if len(n.input) > 2 else np.zeros(W.shape[1])
                self.ops.append(("affine", (attrs.get("alpha", 1.0) * W, attrs.get("beta", 1.0) * b)))
            elif n.op_type == "Add":
                self.ops.append(("add", inits[n.input[1]]))
            elif n.op_type == "Elu":
                self.ops.append(("elu", attrs.get("alpha", 1.0)))
            elif n.op_type in _ACT:
                self.ops.append(("act", n.op_type))
            else:
                raise ValueError(f"Nicht unterstuetzter ONNX-Knoten {n.op_type} in {path}")
            cur = n.output[0]
        if cur != self.output_name:
            raise ValueError("Graph endet nicht im deklarierten Output")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        y = np.asarray(x, dtype=np.float64)
        for typ, p in self.ops:
            if typ == "matmul":
                y = y @ p
            elif typ == "affine":
                y = y @ p[0] + p[1]
            elif typ == "add":
                y = y + p
            elif typ == "elu":
                y = np.where(y > 0, y, p * (np.exp(np.minimum(y, 0.0)) - 1.0))
            else:
                y = _ACT[p](y)
        return y


def onnx_path(shape: str, dataset: str = "LASA") -> str:
    return os.path.join(S2NNDS_DIR, "models_onnx", dataset, shape + ".onnx")


def pth_path(shape: str, dataset: str = "LASA") -> str:
    return os.path.join(S2NNDS_DIR, "models_verified", dataset, shape, "model_f.pth")


def load_torch_reference(path: str):
    """Laedt model_f.pth (gepickeltes Modul) read-only aus dem Original-Repo.

    Braucht NNModels aus S2NNDS. sys.dont_write_bytecode verhindert, dass
    __pycache__ im Original-Repo geschrieben wird.
    """
    import sys
    import torch
    sys.dont_write_bytecode = True
    if S2NNDS_DIR not in sys.path:
        sys.path.insert(0, S2NNDS_DIR)
    import NNModels  # noqa: F401  (fuer das Unpickling noetig)
    model = torch.load(path, map_location="cpu", weights_only=False)
    model.eval()
    return model


# --------------------------------------------------------------------------
# Normierung (aus den LASA-Daten berechnet, nicht hart kodiert)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class LasaScaling:
    shape: str
    ps_mm: float           # globales max |x| aller Demos [mm]
    vs_mm_s: float         # globales max |v| aller Demos [mm/s]
    x0_n: np.ndarray       # mittlerer Demo-Start, normiert (wie initial_set_center)
    demos_n: list = field(repr=False)  # normierte Demo-Positionen, je (T,2)
    dt_demo: float = 0.0


def lasa_scaling(shape: str) -> LasaScaling:
    import pyLasaDataset as lasa
    ds = getattr(lasa.DataSet, LASA_DATASET[shape])
    P = np.concatenate([d.pos for d in ds.demos], axis=1)
    V = np.concatenate([d.vel for d in ds.demos], axis=1)
    ends = np.array([d.pos[:, -1] for d in ds.demos])
    if np.max(np.abs(ends)) > 1e-9:
        raise ValueError(f"{shape}: Demo-Enden nicht im Ursprung, Normierung pruefen")
    ps = float(np.max(np.abs(P)))
    vs = float(np.max(np.abs(V)))
    x0 = np.mean([d.pos[:, 0] for d in ds.demos], axis=0) / ps
    return LasaScaling(shape, ps, vs, x0, [d.pos.T / ps for d in ds.demos], float(ds.dt))


# --------------------------------------------------------------------------
# Einbettung der 2D-Normebene in den Arbeitsraum
# --------------------------------------------------------------------------
@dataclass
class PlanarEmbedding:
    """p = origin + L * [e1 e2] (x_n - anchor_n)."""

    origin: np.ndarray
    e1: np.ndarray
    e2: np.ndarray
    L: float
    anchor_n: np.ndarray

    def __post_init__(self):
        self.origin = np.asarray(self.origin, float)
        self.e1 = np.asarray(self.e1, float) / np.linalg.norm(self.e1)
        e2 = np.asarray(self.e2, float)
        e2 = e2 - (e2 @ self.e1) * self.e1
        self.e2 = e2 / np.linalg.norm(e2)
        self.normal = np.cross(self.e1, self.e2)
        self.anchor_n = np.asarray(self.anchor_n, float)
        self.E = np.stack([self.e1, self.e2], axis=1)  # (3,2)

    def to_world(self, x_n: np.ndarray) -> np.ndarray:
        x_n = np.asarray(x_n, float)
        return self.origin + self.L * ((x_n - self.anchor_n) @ self.E.T)

    def to_plane(self, p: np.ndarray):
        d = np.asarray(p, float) - self.origin
        x_n = d @ self.E / self.L + self.anchor_n
        return x_n, d @ self.normal

    def vel_to_world(self, v2: np.ndarray) -> np.ndarray:
        return np.asarray(v2, float) @ self.E.T


class PthMLP(OnnxMLP):
    """model_f.pth (NNModels.DyanmicsNet) als numpy-MLP in float64.

    forward (NNModels.py): y = act(W_i y) fuer alle Schichten ausser der letzten,
    danach Tanh(W_last y). Keine Biases (bias=False im Original).
    """

    def __init__(self, path: str):
        import torch.nn as nn
        m = load_torch_reference(path)
        acts = {nn.Tanh: "Tanh", nn.ReLU: "Relu"}
        act = acts.get(type(m.sigmoid_f))
        if act is None and type(m.sigmoid_f).__name__ != "ELU":
            raise ValueError(f"Aktivierung {type(m.sigmoid_f).__name__} nicht unterstuetzt")
        self.path = path
        self.dim = m.layers_f[0].in_features
        self.ops = []
        for i, layer in enumerate(m.layers_f):
            if layer.bias is not None:
                raise ValueError("Bias unerwartet")
            self.ops.append(("matmul", layer.weight.detach().double().numpy().T))
            if i < len(m.layers_f) - 1:
                self.ops.append(("elu", 1.0) if act is None else ("act", act))
            else:
                self.ops.append(("act", "Tanh"))


class LearnedDS:
    """x_n_dot = (v_peak / L) f(x_n); v_world = v_peak * E f(x_n).

    source: "onnx" (models_onnx/, Standard) oder "pth" (models_verified/.../model_f.pth).
    """

    def __init__(self, shape: str, dataset: str = "LASA", source: str = "onnx"):
        self.shape = shape
        self.source = source
        self.f = OnnxMLP(onnx_path(shape, dataset)) if source == "onnx" else PthMLP(
            pth_path(shape, dataset))

    def f_n(self, x_n: np.ndarray) -> np.ndarray:
        return self.f(np.asarray(x_n, float).reshape(1, -1))[0]

    def rollout_ideal(self, x0_n, dt: float, n_steps: int, rate: float) -> np.ndarray:
        """Explizites Euler wie im Original (Plotter.py), rate = v_peak / L [1/s].

        Rueckgabe: (n_steps+1, 2), Zeile k gehoert zu t = k*dt.
        """
        X = np.empty((n_steps + 1, 2))
        X[0] = x0_n
        for k in range(n_steps):
            X[k + 1] = X[k] + dt * rate * self.f_n(X[k])
        return X


def demo_time_rate(sc: LasaScaling) -> float:
    """Demo-treue Zeitbasis: v_peak / L = vs / ps [1/s]."""
    return sc.vs_mm_s / sc.ps_mm
