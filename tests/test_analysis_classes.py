"""Unit tests of the convergence classes and first-violation logic (synthetic arrays, no simulation)."""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.dont_write_bytecode = True

import numpy as np

from pic import analysis as A

AN = dict(A.ANALYSIS_DEFAULTS)
DT = 0.002


def arrays(N=1000, n=3, dist_end=0.0, dist0=0.1, speed_end=0.0, sat_frac=0.0):
    t = np.arange(N + 1) * DT
    dist = np.linspace(dist0, dist_end, N + 1)
    v = np.zeros((N + 1, 3))
    v[-200:, 0] = speed_end
    sat = np.zeros((N + 1, n), np.int8)
    sat[: int(sat_frac * N), 1] = 1
    return {"q": np.zeros((N + 1, n)), "qd": np.zeros((N + 1, n)), "t": t, "dist_goal": dist, "tcp_vel": v,
            "sat_joint": sat, "sat_any": sat.max(axis=1), "q_min": -np.ones(n), "q_max": np.ones(n),
            "qd_max": np.ones(n)}


class TestClasses(unittest.TestCase):
    def cls(self, arr, success):
        return A.failure_class({"finite": True, "success": success}, arr, AN, DT)

    def test_converged(self):
        self.assertEqual(self.cls(arrays(dist_end=1e-4), True), "converged")

    def test_stalled_vs_slow(self):
        self.assertEqual(self.cls(arrays(dist_end=0.05, speed_end=0.0), False), "stalled")
        self.assertEqual(self.cls(arrays(dist_end=0.05, speed_end=0.01), False), "slow")

    def test_diverged_threshold(self):
        self.assertEqual(self.cls(arrays(dist0=0.1, dist_end=0.16, speed_end=0.01), False), "diverged")
        self.assertEqual(self.cls(arrays(dist0=0.1, dist_end=0.14, speed_end=0.01), False), "slow")

    def test_saturated_has_priority_over_diverged(self):
        self.assertEqual(self.cls(arrays(dist_end=0.5, speed_end=0.01, sat_frac=0.2), False), "saturated")
        self.assertEqual(self.cls(arrays(dist_end=0.5, speed_end=0.01, sat_frac=0.01), False), "diverged")

    def test_nonfinite(self):
        a = arrays()
        a["q"][5, 0] = np.nan
        self.assertEqual(self.cls(a, False), "nonfinite")


class TestFirstViolation(unittest.TestCase):
    def test_none(self):
        self.assertEqual(A.first_violation(arrays(), DT)["limit"], "none")

    def test_earliest_wins_and_joint_reported(self):
        a = arrays()
        a["qd"][100, 2] = 2.0                       # velocity limit at step 100, joint 3
        a["q"][50, 0] = 1.5                         # position limit at step 50, joint 1
        a["sat_joint"][300, 1] = 1                  # torque saturation later
        fv = A.first_violation(a, DT)
        self.assertEqual((fv["limit"], fv["joint"]), ("joint_position_limit", 1))
        self.assertAlmostEqual(fv["t_s"], 50 * DT)
        a["q"][50, 0] = 0.0
        self.assertEqual(A.first_violation(a, DT)["limit"], "joint_velocity_limit")

    def test_obstacle(self):
        a = arrays()
        a["clearance_m"] = np.ones(len(a["t"]))
        a["clearance_m"][10] = -0.01
        self.assertEqual(A.first_violation(a, DT)["limit"], "obstacle_penetration")


if __name__ == "__main__":
    unittest.main()
