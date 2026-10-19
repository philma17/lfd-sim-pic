"""Targeted validations of the PIC-DS prototype.  Run:  ~/lfd_env/bin/python -B -m unittest discover -s tests -v

Closed-loop tests (convergence, push, energy) use the DIAGNOSTIC plant with dof_frictionloss = 0,
because with the unmodified Menagerie friction and the paper gains the arm does not leave the
start (test_known_limitation_unmodified_plant_stalls documents this). Nothing here claims a
passivity guarantee: the energy tests check numerical behaviour of this implementation only.
"""
import hashlib
import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.dont_write_bytecode = True

import mujoco
import numpy as np

from common.scenario import load_config
from pic.controller import (Coupling, DampingBasis, PICController, PICParams, damping_matrix,
                            smooth_step)
from pic.sim_pic import apply_tcp_force, build_case, null_projector, run

CFG = load_config(os.path.join(ROOT, "configs", "picds.yaml"))
PC = CFG["pic"]
VAL = CFG["validation"]
DQ = PC["null_space"]["d_q"]
PULSES = CFG["disturbance"]["pulses"]


class TestCoupling(unittest.TestCase):
    def test_conditions_eq_11_12_14(self):
        for kind in ("quintic", "cosine"):
            c = Coupling(10.0, 1.0, 0.01, kind)
            s = np.linspace(-1.0, 11.0, 241)
            z = np.linspace(-0.05, 0.05, 201)
            S, Z = np.meshgrid(s, z)
            a, bs, bR = c.alpha(S), c.beta_s(Z, S), c.beta_R(Z, S)
            tol = 1e-12
            self.assertTrue(np.all((a >= -tol) & (a <= 1 + tol)))
            self.assertTrue(np.all(a[S >= 10.0] <= tol))                       # Eq. 11
            self.assertTrue(np.all(np.abs(bs[(S <= 0) & (Z >= 0)]) <= tol))    # Eq. 12
            self.assertTrue(np.all(np.abs(bs[(S >= 10.0) & (Z <= 0)]) <= tol))
            self.assertTrue(np.all((bs >= -tol) & (bs <= 1 + tol)))
            self.assertTrue(np.all((bR >= -tol) & (bR <= 1 + tol)))
            pos = Z >= 0
            self.assertTrue(np.allclose(bR[pos], bs[pos], atol=1e-12))         # Eq. 14, z >= 0
            neg = Z < 0
            self.assertTrue(np.all(bR[neg] >= bs[neg] - tol))                  # Eq. 14, z < 0

    def test_smooth_step_properties(self):
        x = np.array([-1.0, 0.0, 0.5, 1.0, 2.0])
        h = smooth_step(x, 0.0, 1.0)
        self.assertTrue(np.allclose(h, [0, 0, 0.5, 1, 1]))

    def test_full_tank_at_rest_blocks_f_R(self):
        """Documented consequence of Eq. 14 + Eq. 12: beta_R(0, s_bar) = beta_s(0, s_bar) = 0.
        With decomposition 'none' the arm cannot start from a full tank (hence s0 < s_bar)."""
        c = Coupling(10.0, 1.0, 0.01)
        self.assertEqual(float(c.beta_R(0.0, 10.0)), 0.0)
        self.assertEqual(float(c.beta_R(0.0, 5.0)), 1.0)

    def test_param_validation(self):
        base = PICParams.from_cfg(PC)
        with self.assertRaises(ValueError):
            PICParams(s0=11.0).validate()
        with self.assertRaises(ValueError):
            PICParams(s0=0.0).validate()
        with self.assertRaises(NotImplementedError):
            PICParams(decomposition="linear").validate()
        with self.assertRaises(ValueError):
            Coupling(10.0, 6.0, 0.01)
        self.assertEqual(base.decomposition, "none")


class TestDamping(unittest.TestCase):
    def test_basis_and_eigenstructure(self):
        b = DampingBasis(3, 1e-6)
        lam = (20.0, 200.0, 150.0)
        for f in (np.array([0.03, 0.01, 0.0]), np.array([0.0, -0.02, 0.02]),
                  np.array([0.0, 0.0, 0.05])):
            Q = b.update(f)
            self.assertTrue(np.allclose(Q.T @ Q, np.eye(3), atol=1e-12))
            self.assertTrue(np.allclose(Q[:, 0], f / np.linalg.norm(f)))
            D = damping_matrix(Q, lam)
            self.assertTrue(np.allclose(D @ f, lam[0] * f))                    # f is eigenvector
            self.assertTrue(np.all(np.linalg.eigvalsh(D) >= 0))

    def test_basis_kept_below_eta(self):
        b = DampingBasis(3, 1e-6)
        Q1 = b.update(np.array([0.0, 0.1, 0.0])).copy()
        Q2 = b.update(np.array([1e-9, 0.0, 0.0]))
        self.assertTrue(np.array_equal(Q1, Q2))
        self.assertEqual(b.n_kept, 1)

    def test_antiparallel_switch_stays_orthonormal(self):
        b = DampingBasis(3, 1e-6)
        b.update(np.array([1.0, 0.0, 0.0]))
        Q = b.update(np.array([0.0, 1.0, 0.0]))
        self.assertTrue(np.allclose(Q.T @ Q, np.eye(3), atol=1e-12))

    def test_tank_never_leaves_interval_pure_controller(self):
        c = PICController(PICParams.from_cfg(PC))
        rng = np.random.default_rng(0)                                         # fixed seed
        for _ in range(20000):
            c.step(rng.normal(size=3) * 0.5, rng.normal(size=3) * 0.05, 0.002)
        self.assertEqual(c.n_clamp, 0)
        self.assertTrue(0.0 <= c.s <= c.p.s_max)


class TestNullProjector(unittest.TestCase):
    def test_projector(self):
        rng = np.random.default_rng(1)
        J = rng.normal(size=(3, 7))
        N, nd = null_projector(J)
        self.assertEqual(nd, 4)
        self.assertTrue(np.allclose(J @ N, 0, atol=1e-12))
        self.assertTrue(np.allclose(N @ N, N, atol=1e-12))
        N0, nd0 = null_projector(rng.normal(size=(3, 3)))                      # empty null space
        self.assertEqual(nd0, 0)
        self.assertTrue(np.all(N0 == 0))


class TestProvenance(unittest.TestCase):
    def test_vendored_files_unchanged(self):
        d = os.path.join(ROOT, "common")
        lines = [ln.split() for ln in open(os.path.join(d, "VENDORED.txt")) if ln.strip().endswith(".py")]
        self.assertEqual(len(lines), 8)
        for h, name in lines:
            with open(os.path.join(d, name), "rb") as fh:
                self.assertEqual(hashlib.sha256(fh.read()).hexdigest(), h, name)

    def test_start_and_servo_match_baseline_results(self):
        p = "/home/philipp/lfd_baseline/results/dynamic_baseline/metrics.json"
        if not os.path.exists(p):
            self.skipTest("baseline results not available")
        found = []

        def walk(o):
            if isinstance(o, dict):
                if o.get("robot") == "FR3" and o.get("mode") == "pos" and "q_start_rad" in o:
                    found.append(o)
                for v in o.values():
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)

        walk(json.load(open(p)))
        self.assertEqual(len(found), 1)
        case = build_case(CFG)
        self.assertTrue(np.allclose(case.q_start, found[0]["q_start_rad"], atol=1e-12))
        # servo comparison run reproduces the stored baseline final error (same code, same start)
        from common import sim_dynamic
        res = sim_dynamic.run(case.robot, case.ds, case.emb, case.x0_n, case.rate, case.tgt,
                              case.sp, q_start=case.q_start, n_substeps=1)
        self.assertAlmostEqual(res["final_dist_m"],
                               found[0]["dynamic"]["tracking"]["final_goal_err_m"], places=9)


class TestSimPIC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case = build_case(CFG, friction_scale=0.0)          # DIAGNOSTIC plant
        cls.params = PICParams.from_cfg(PC)
        cls.free = run(cls.case, cls.params, null_d_q=DQ)
        cls.push = run(cls.case, cls.params, null_d_q=DQ, pulses=PULSES)

    def test_1_tank_stays_in_interval(self):
        for r in (self.free, self.push):
            s = r["traj"]["s"]
            self.assertGreaterEqual(s.min(), -VAL["tank_tol_J"])
            self.assertLessEqual(s.max(), self.params.s_max + VAL["tank_tol_J"])
            self.assertEqual(r["tank"]["n_clamp_steps"], 0)   # the clamp never had to act
            self.assertLessEqual(r["tank"]["max_excursion_J"], VAL["tank_tol_J"])

    def test_2_converges_to_goal(self):
        r = self.free
        self.assertTrue(r["finite"])
        self.assertTrue(r["success"], f"final {r['final_dist_m']:.3e} m")
        self.assertLess(r["final_dist_m"], r["success_threshold_m"])
        self.assertEqual(r["torque"]["n_clip_steps"], 0)
        self.assertEqual(r["n_pos_limit_viol_steps"], 0)

    def test_3_push_recovers(self):
        r, r0 = self.push, self.free
        dev = np.linalg.norm(r["traj"]["P"] - r0["traj"]["P"], axis=1)
        self.assertGreater(dev.max(), 1e-3, "the pulse had no visible effect")
        self.assertTrue(r["success"], f"final {r['final_dist_m']:.3e} m")
        self.assertEqual(r["torque"]["n_clip_steps"], 0)
        # pulse really acted at t = 1 s and nowhere else
        t, F = r["traj"]["t"], np.linalg.norm(r["traj"]["F_ext"], axis=1)
        self.assertTrue(np.all(F[t < 1.0] == 0) and np.all(F[t >= 1.05] == 0))
        self.assertAlmostEqual(F.max(), 5.0, places=9)

    def test_4_no_f_kinetic_energy_decays_monotonically(self):
        v0 = np.array([0.0, CFG["validation"]["no_f_v0_m_s"], 0.0])
        r = run(self.case, self.params, null_d_q=DQ, use_f=False, v0=v0, T_max=3.0)
        KE = r["traj"]["KE"]
        self.assertGreater(KE[0], 1e-3)
        inc = np.diff(KE)
        self.assertLessEqual(inc.max(), VAL["ke_increase_tol_J"])
        self.assertLess(KE[-1], 1e-6 * KE[0])
        self.assertTrue(0 <= r["traj"]["s"].min() and r["traj"]["s"].max() <= self.params.s_max)

    def test_clipped_and_unclipped_torques_are_logged_separately(self):
        lim = np.full(7, 0.5)                                   # absurdly small, forces clipping
        r = run(self.case, self.params, null_d_q=DQ, tau_limit=lim, T_max=0.2)
        tr = r["traj"]
        self.assertGreater(r["torque"]["n_clip_steps"], 0)
        n = len(tr["t"]) - 1
        self.assertTrue(np.all(np.abs(tr["tau_applied"][:n]) <= lim + 1e-12))
        self.assertGreater(np.abs(tr["tau_cmd"][:n]).max(), 0.5)
        self.assertTrue(np.allclose(tr["tau_applied"][:n], np.clip(tr["tau_cmd"][:n], -lim, lim)))

    def test_model_flags_restored_and_mjcf_untouched(self):
        m = self.case.robot.model
        self.assertEqual(int(m.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_ACTUATION), 0)

    def test_xfrc_pulse_equals_jacobian_transpose_force(self):
        robot = self.case.robot
        m = robot.model
        F = np.array([0.0, 3.0, 4.0])
        p, _ = robot.fk(self.case.q_start)
        Jp = robot.jac()[:3]
        qa = []
        for mode in ("xfrc", "qfrc"):
            d = mujoco.MjData(m)
            d.qpos[robot.qadr] = self.case.q_start
            mujoco.mj_forward(m, d)
            if mode == "xfrc":
                apply_tcp_force(m, d, robot.site_body, p, F)
            else:
                d.qfrc_applied[robot.dadr] = Jp.T @ F
            mujoco.mj_forward(m, d)
            qa.append(d.qacc.copy())
        self.assertTrue(np.allclose(qa[0], qa[1], atol=1e-9))


class TestKnownLimitation(unittest.TestCase):
    def test_known_limitation_unmodified_plant_stalls(self):
        """Menagerie FR3 Coulomb friction (up to 1.137 Nm) vs. the paper gains (lambda1 = 20 N s/m,
        |f| <= 0.05 m/s -> TCP force <= 1 N): the arm does not converge. Documented in README."""
        case = build_case(CFG, friction_scale=1.0)
        r = run(case, PICParams.from_cfg(PC), null_d_q=DQ, T_max=4.0)
        self.assertFalse(r["success"])
        self.assertGreater(r["final_dist_m"], 0.05)
        self.assertLess(r["energy"]["KE_max_J"], 1e-4)


if __name__ == "__main__":
    unittest.main()
