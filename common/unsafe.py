"""Unsafe sets of the S2NNDS configs as signed distance functions in normalized coordinates.

Semantics follow S2NNDS/main.py (generate_domain_data), where a point is unsafe if
  Circle    : ||x - c|| <= r for any center c (centers reshaped to (-1, dim)); 2D or 3D
  Rectangle : unsafe_min <= x <= unsafe_max per axis; for "unbounded" (2D only) the open side
              is replaced by +-100 exactly as in main.py
  Custom    : eval(function)(x, y) <= 0
Signed distance (normalized units, negative = inside):
  Circle    : min_c ||x - c|| - r                       (exact)
  Rectangle : exact Euclidean distance to the box, -(depth) inside
  Custom    : NOT implemented (no distance for an arbitrary level set); the metric is disabled
              and reported as such. No obstacle safety is claimed for these shapes.
World clearance = L * signed distance (the embeddings are isotropic: orthonormal axes, one L).
"""
from __future__ import annotations

import numpy as np


class UnsafeSet:
    def __init__(self, spec: dict | None, dim: int):
        self.spec = spec or {}
        self.dim = dim
        self.shape = self.spec.get("shape")
        self.supported = False
        self.reason = None
        self.centers = self.radius = self.lo = self.hi = None
        if not spec:
            self.reason = "no 'unsafe' entry in the S2NNDS config"
        elif self.shape == "Circle":
            self.centers = np.asarray(spec["center"], float).reshape(-1, dim)
            self.radius = float(spec["radius"])
            self.supported = True
        elif self.shape == "Rectangle":
            rng = spec["range"]
            if "unbounded" not in spec:
                self.lo = np.array([rng[i][0] for i in range(dim)], float)
                self.hi = np.array([rng[i][1] for i in range(dim)], float)
                self.supported = True
            elif dim == 2:
                u, mm = spec["unbounded"], spec["max_min"]
                if u == "x" and mm == "min":
                    self.lo, self.hi = [rng[0][0], rng[1][0]], [100.0, rng[1][1]]
                elif u == "x" and mm == "max":
                    self.lo, self.hi = [-100.0, rng[1][0]], [rng[0][0], rng[1][1]]
                elif u == "y" and mm == "min":
                    self.lo, self.hi = [rng[0][0], rng[1][0]], [rng[0][1], 100.0]
                elif u == "y" and mm == "max":
                    self.lo, self.hi = [rng[0][0], -100.0], [rng[0][1], rng[1][0]]
                if self.lo is not None:
                    self.lo, self.hi = np.asarray(self.lo, float), np.asarray(self.hi, float)
                    self.supported = True
                else:
                    self.reason = f"Rectangle variant unbounded={u}, max_min={mm} not known"
            else:
                self.reason = "unbounded Rectangle is only defined for 2D in S2NNDS"
        elif self.shape == "Custom":
            self.reason = ("unsafe shape 'Custom' (level set of a config expression): clearance "
                           "metric not implemented, obstacle safety NOT evaluated")
        else:
            self.reason = f"unsafe shape '{self.shape}' not supported"

    def signed_distance(self, X):
        """X (N, dim) normalized -> (N,) signed distance in normalized units."""
        X = np.atleast_2d(np.asarray(X, float))
        if not self.supported:
            raise RuntimeError(self.reason)
        if self.shape == "Circle":
            d = np.linalg.norm(X[:, None, :] - self.centers[None], axis=2) - self.radius
            return d.min(axis=1)
        q = np.maximum(self.lo - X, X - self.hi)           # > 0 outside along that axis
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
        inside = np.minimum(q.max(axis=1), 0.0)
        return outside + inside

    def describe(self) -> dict:
        d = {"shape": self.shape, "supported": self.supported, "reason": self.reason}
        if self.shape == "Circle" and self.supported:
            d.update(centers_n=self.centers.tolist(), radius_n=self.radius,
                     geometry="sphere" if self.dim == 3 else "circle")
        if self.shape == "Rectangle" and self.supported:
            d.update(lo_n=self.lo.tolist(), hi_n=self.hi.tolist())
        if self.shape == "Custom":
            d["function"] = self.spec.get("function")
        return d
