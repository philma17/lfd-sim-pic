"""Embeddings of normalized DS coordinates into the robot world frame, for any model dimension.

2D models: common.ds.PlanarEmbedding (unchanged), p = origin + L [e1 e2] (x_n - anchor_n).
3D models: SpatialEmbedding below, p = origin + L A (x_n - anchor_n) with an orthonormal 3x3
           matrix A (columns = world directions of the DS axes). No plane, no projection: the
           inverse is exact, the out-of-plane offset is identically 0 and normal = 0, so the
           plane correction term k_perp * d_perp * normal in sim_dynamic.run vanishes and all
           three Cartesian velocity components come from the DS.

Both classes expose the same interface used by common.sim_dynamic.run:
  to_world(x_n), to_plane(p) -> (x_n, d_perp), vel_to_world(v_n), E (3 x dim), normal, L,
  origin, anchor_n.  `embedding_dim(emb)` returns dim = E.shape[1].
"""
from __future__ import annotations

import numpy as np

from .ds import PlanarEmbedding


class SpatialEmbedding:
    """p = origin + L * A (x_n - anchor_n), A orthonormal (3x3)."""

    def __init__(self, origin, axes, L: float, anchor_n):
        A = np.asarray(axes, float)
        if A.shape != (3, 3):
            raise ValueError(f"spatial axes must be 3x3, got {A.shape}")
        if not np.allclose(A.T @ A, np.eye(3), atol=1e-9):
            raise ValueError("spatial axes must be orthonormal (distances are scaled by L only)")
        if L <= 0:
            raise ValueError("L must be > 0")
        self.origin = np.asarray(origin, float).reshape(3)
        self.E = A
        self.L = float(L)
        self.anchor_n = np.asarray(anchor_n, float).reshape(3)
        self.normal = np.zeros(3)

    def to_world(self, x_n):
        return self.origin + self.L * ((np.asarray(x_n, float) - self.anchor_n) @ self.E.T)

    def to_plane(self, p):
        """Name kept for interface compatibility with PlanarEmbedding: exact inverse, no
        projection. Returns (x_n, 0.0)."""
        d = np.asarray(p, float) - self.origin
        return d @ self.E / self.L + self.anchor_n, 0.0

    def vel_to_world(self, v_n):
        return np.asarray(v_n, float) @ self.E.T


def embedding_dim(emb) -> int:
    return int(np.asarray(emb.E).shape[1])


def make_embedding(dim: int, center, L: float, plane=None, axes=None, anchor=None):
    """dim 2 -> PlanarEmbedding(center, e1, e2, L, anchor); dim 3 -> SpatialEmbedding."""
    anchor = np.zeros(dim) if anchor is None else np.asarray(anchor, float)
    if len(anchor) != dim:
        raise ValueError(f"anchor has {len(anchor)} entries, model dimension is {dim}")
    if dim == 2:
        return PlanarEmbedding(np.asarray(center, float), plane["e1"], plane["e2"], float(L),
                               anchor)
    if dim == 3:
        return SpatialEmbedding(center, np.eye(3) if axes is None else axes, L, anchor)
    raise ValueError(f"no world embedding for {dim}D models")
