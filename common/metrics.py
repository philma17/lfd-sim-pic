"""Bahnmetriken.

Geometrische Metriken (Chamfer, Frechet, DTW) arbeiten auf Bogenlaengen-
Resampling und sind damit invariant gegen Zeitumparametrisierung.
Der zeitsynchrone RMSE vergleicht dagegen bewusst gleiche Zeitpunkte gegen
das ideale DS (gleicher Start, gleiche Zeitbasis).
"""
from __future__ import annotations

import numpy as np


def arclength(P: np.ndarray) -> np.ndarray:
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def resample_arclength(P: np.ndarray, n: int) -> np.ndarray:
    """n Punkte gleichverteilt in der Bogenlaenge. Stillstand (Nullsegmente) wird entfernt."""
    P = np.asarray(P, float)
    s = arclength(P)
    keep = np.concatenate([[True], np.diff(s) > 0])
    P, s = P[keep], s[keep]
    if len(P) < 2:
        return np.repeat(P[:1], n, axis=0)
    t = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(t, s, P[:, d]) for d in range(P.shape[1])], axis=1)


def point_to_polyline(X: np.ndarray, P: np.ndarray, chunk: int = 512) -> np.ndarray:
    """Exakter Abstand jedes Punkts in X zum Polygonzug P."""
    A, B = P[:-1], P[1:]
    AB = B - A
    L2 = np.maximum(np.einsum("ij,ij->i", AB, AB), 1e-300)
    out = np.empty(len(X))
    for i in range(0, len(X), chunk):
        x = X[i:i + chunk, None, :]
        t = np.clip(np.einsum("kij,ij->ki", x - A[None], AB) / L2, 0.0, 1.0)
        proj = A[None] + t[..., None] * AB[None]
        out[i:i + chunk] = np.sqrt(np.min(np.sum((x - proj) ** 2, axis=2), axis=1))
    return out


def chamfer(P: np.ndarray, Q: np.ndarray, n: int = 1000):
    """Symmetrischer Chamfer-Abstand (mean) und Hausdorff (max), Punkt-zu-Polygonzug."""
    Pr, Qr = resample_arclength(P, n), resample_arclength(Q, n)
    d1 = point_to_polyline(Pr, Qr)
    d2 = point_to_polyline(Qr, Pr)
    return 0.5 * (d1.mean() + d2.mean()), max(d1.max(), d2.max())


def frechet_discrete(P: np.ndarray, Q: np.ndarray, n: int = 400) -> float:
    Pr, Qr = resample_arclength(P, n), resample_arclength(Q, n)
    D = np.linalg.norm(Pr[:, None] - Qr[None], axis=2).tolist()
    prev = np.maximum.accumulate(np.asarray(D[0])).tolist()
    for i in range(1, n):
        Di = D[i]
        row = [max(prev[0], Di[0])] + [0.0] * (n - 1)
        for j in range(1, n):
            a = prev[j] if prev[j] < prev[j - 1] else prev[j - 1]
            if row[j - 1] < a:
                a = row[j - 1]
            row[j] = a if a > Di[j] else Di[j]
        prev = row
    return float(prev[-1])


def dtw_mean(P: np.ndarray, Q: np.ndarray, n: int = 400) -> float:
    """DTW-Kosten geteilt durch die Pfadlaenge (mittlerer Abstand entlang der Zuordnung)."""
    Pr, Qr = resample_arclength(P, n), resample_arclength(Q, n)
    D = np.linalg.norm(Pr[:, None] - Qr[None], axis=2).tolist()
    inf = float("inf")
    Cp = [0.0] + [inf] * n          # Zeile i-1 (Kosten)
    Kp = [0] * (n + 1)              # Zeile i-1 (Schrittzahl)
    for i in range(1, n + 1):
        Di = D[i - 1]
        Cr = [inf] * (n + 1)
        Kr = [0] * (n + 1)
        for j in range(1, n + 1):
            c, k = Cp[j - 1], Kp[j - 1]          # diagonal
            if Cp[j] < c:
                c, k = Cp[j], Kp[j]              # oben
            if Cr[j - 1] < c:
                c, k = Cr[j - 1], Kr[j - 1]      # links
            Cr[j] = Di[j - 1] + c
            Kr[j] = k + 1
        Cp, Kp = Cr, Kr
    return float(Cp[n] / Kp[n])


def rmse_time_sync(P: np.ndarray, Q: np.ndarray) -> float:
    if P.shape != Q.shape:
        raise ValueError("zeitsynchroner RMSE braucht gleiche Zeitbasis")
    return float(np.sqrt(np.mean(np.sum((P - Q) ** 2, axis=1))))


def max_dev_time_sync(P: np.ndarray, Q: np.ndarray) -> float:
    return float(np.max(np.linalg.norm(P - Q, axis=1)))
