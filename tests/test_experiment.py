"""Tests for the configurable experiment framework (plant variations, speed factor, runner,
analysis). Run:  python -B -m unittest tests.test_experiment -v   (or the whole discover command)

These check that a requested parameter really changes the simulated physics, not only the config.
"""
import copy
import hashlib
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.dont_write_bytecode = True

import mujoco
import numpy as np

from common.robots import MENAGERIE_DIR, ROBOTS, Robot
from common.scenario import load_config
from common.sim_dynamic import TAU_MAX_DOC, gravity_torque
from pic import analysis
from pic.experiment import (deep_merge, load_experiment_file, parse_set, run_experiment,
                            sanitize, set_path)
from pic.plant import apply_plant, resolve_plant
from pic.sim_pic import build_case

CFG = load_config("configs/picds.yaml")
Q0 = np.array([0.003253751299604534, -0.24427262864244112, 0.003369135969120155,
               -2.0812376788143045, 0.0007605664156411323, 1.2226910587064925, -0.7853])


class TestPlant(unittest.TestCase):
    def test_unknown_key_and_bad_values_rejected(self):
        with self.assertRaises(ValueError):
            resolve_plant({"frcition_scale": 1.0})
        with self.assertRaises(ValueError):
            resolve_plant({"mass_scale": 0.0})
        with self.assertRaises(ValueError):
            resolve_plant({"dof_frictionloss_scale": -1.0})

    def test_mjcf_untouched_and_nominal_robot_unmodified(self):
        path = os.path.join(MENAGERIE_DIR, "franka_fr3", "fr3.xml")
        h0 = hashlib.sha256(open(path, "rb").read()).hexdigest()
        cfg = copy.deepcopy(CFG)
        cfg["plant"] = {"dof_frictionloss_scale": 0.0, "mass_scale": 2.0, "gravity_scale": 0.5}
        case = build_case(cfg)
        self.assertEqual(hashlib.sha256(open(path, "rb").read()).hexdigest(), h0)
        nom = Robot(ROBOTS["FR3"]).model
        self.assertTrue(np.allclose(case.robot_nominal.model.body_mass, nom.body_mass))
        self.assertTrue(np.allclose(case.robot_nominal.model.dof_frictionloss, nom.dof_frictionloss))
        self.assertTrue(np.allclose(case.robot_nominal.model.opt.gravity, nom.opt.gravity))

    def test_friction_changes_dynamics(self):
        """A 0.6 Nm joint-1 torque is below stiction (1.137 Nm): no motion with nominal friction,
        motion with friction switched off."""
        out = {}
        for fs in (1.0, 0.0):
            r = Robot(ROBOTS["FR3"])
            apply_plant(r, {"dof_frictionloss_scale": fs})
            m = r.model
            d = mujoco.MjData(m)
            d.qpos[r.qadr] = Q0
            mujoco.mj_forward(m, d)
            m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
            for _ in range(150):
                d.qfrc_applied[:] = gravity_torque(r, d.qpos[r.qadr])
                d.qfrc_applied[r.dadr[0]] += 0.6
                mujoco.mj_step(m, d)
            out[fs] = abs(d.qpos[r.qadr[0]] - Q0[0])
        self.assertLess(out[1.0], 1e-3)
        self.assertGreater(out[0.0], 1e-2)

    def test_gravity_and_mass_scale_gravity_torque(self):
        g0 = gravity_torque(Robot(ROBOTS["FR3"]), Q0)
        for key, factor in (("gravity_scale", 0.5), ("mass_scale", 2.0)):
            r = Robot(ROBOTS["FR3"])
            apply_plant(r, {key: factor})
            self.assertTrue(np.allclose(gravity_torque(r, Q0), factor * g0, rtol=1e-9, atol=1e-9),
                            key)
        r = Robot(ROBOTS["FR3"])
        apply_plant(r, {"gravity_scale": 0.0})
        self.assertLess(np.abs(gravity_torque(r, Q0)).max(), 1e-12)

    def test_inertia_scale_changes_mass_matrix_not_gravity(self):
        def M(r):
            d = mujoco.MjData(r.model)
            d.qpos[r.qadr] = Q0
            mujoco.mj_forward(r.model, d)
            out = np.zeros((r.model.nv, r.model.nv))
            mujoco.mj_fullM(r.model, d, out)       # MuJoCo >= 3.x signature (m, d, dst)
            return out
        r0, r1 = Robot(ROBOTS["FR3"]), Robot(ROBOTS["FR3"])
        apply_plant(r1, {"inertia_scale": 3.0})
        self.assertTrue(np.allclose(gravity_torque(r0, Q0), gravity_torque(r1, Q0)))
        self.assertGreater(np.trace(M(r1)), np.trace(M(r0)))

    def test_servo_torque_limit_is_enforced_by_mujoco(self):
        r = Robot(ROBOTS["FR3"])
        apply_plant(r, {"torque_limit_scale": 0.25})
        m = r.model
        lim = 0.25 * TAU_MAX_DOC["FR3"][0]
        self.assertTrue(np.all(m.actuator_forcelimited == 1))
        self.assertTrue(np.allclose(m.actuator_forcerange[:, 1], lim))
        d = mujoco.MjData(m)
        d.qpos[r.qadr] = Q0
        d.ctrl[:] = Q0 + 1.0                     # huge position error -> huge unclamped command
        mujoco.mj_forward(m, d)
        self.assertTrue(np.allclose(np.abs(d.actuator_force), lim))


class TestSpeedFactor(unittest.TestCase):
    def test_alpha_scales_velocity_not_geometry(self):
        c1, c2 = build_case(CFG), build_case(CFG, speed_factor=2.0)
        self.assertAlmostEqual(c2.rate / c1.rate, 2.0, places=12)
        self.assertEqual(c2.speed_factor, 2.0)
        from common.sim_dynamic import ideal_rollout
        X1 = ideal_rollout(c1.ds, c1.x0_n, c1.sp.dt, 3, c1.rate)
        X2 = ideal_rollout(c2.ds, c2.x0_n, c2.sp.dt, 3, c2.rate)
        # x_dot = alpha f(x): the first Euler step is exactly twice as long
        self.assertTrue(np.allclose(X2[1] - X2[0], 2.0 * (X1[1] - X1[0]), rtol=1e-12, atol=1e-15))
        self.assertAlmostEqual(c2.emb.L * c2.rate / (c1.emb.L * c1.rate), 2.0, places=12)

    def test_invalid_alpha(self):
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                build_case(CFG, speed_factor=bad)

    def test_config_key(self):
        cfg = copy.deepcopy(CFG)
        cfg["ds_speed_factor"] = 0.5
        self.assertEqual(build_case(cfg).speed_factor, 0.5)


class TestStart(unittest.TestCase):
    def test_explicit_start_used_and_validated(self):
        cfg = copy.deepcopy(CFG)
        cfg["task"]["q_start_rad"] = Q0.tolist()
        case = build_case(cfg)
        self.assertTrue(np.allclose(case.q_start, Q0))
        cfg["task"]["q_start_rad"] = (Q0 + 10.0).tolist()
        with self.assertRaises(ValueError):
            build_case(cfg)


class TestConfigHandling(unittest.TestCase):
    def test_extends_and_merge(self):
        raw, files = load_experiment_file("configs/experiments/speed_scaling.yaml")
        self.assertEqual(raw["sim"]["T_max"], 20.0)
        self.assertEqual(raw["sim"]["dt"], 0.002)          # inherited from picds.yaml
        self.assertEqual(len(files), 2)
        self.assertEqual(deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 3}}), {"a": {"b": 3, "c": 2}})

    def test_set_path_rejects_typos(self):
        cfg = {"pic": {"lambda": [1, 2, 3]}, "plant": {}, "task": {}}   # no "analysis" section yet
        set_path(cfg, "pic.lambda", [4, 5, 6])
        set_path(cfg, "plant.gravity_scale", 0.5)           # whitelisted section
        set_path(cfg, "analysis.csv_stride", 1)             # section is created on demand
        self.assertEqual(cfg["analysis"], {"csv_stride": 1})
        set_path(cfg, "ds_speed_factor", 2.0)
        set_path(cfg, "task.q_start_rad", [0.0] * 7)
        with self.assertRaises(KeyError):
            set_path(cfg, "pic.lamda", [1, 2, 3])
        with self.assertRaises(KeyError):
            set_path(cfg, "nosuch.key", 1)
        self.assertEqual(parse_set(["a.b=0.5", "x=[1, 2]"]), {"a.b": 0.5, "x": [1, 2]})
        self.assertEqual(sanitize("a b/c"), "a_b_c")


class TestAnalysis(unittest.TestCase):
    def test_savgol_derivatives_of_known_signal(self):
        dt = 0.002
        t = np.arange(0, 2, dt)
        v = np.sin(2 * np.pi * t)[:, None]
        acc = analysis.sg_derivative(v, dt, 1, 0.05, 3)[:, 0]
        jerk = analysis.sg_derivative(v, dt, 2, 0.05, 3)[:, 0]
        w = 2 * np.pi
        sl = slice(50, -50)                                  # away from the edges
        self.assertLess(np.abs(acc[sl] - w * np.cos(w * t[sl])).max(), 1e-3 * w)
        self.assertLess(np.abs(jerk[sl] + w**2 * np.sin(w * t[sl])).max(), 2e-2 * w**2)

    def test_missing_torque_is_nan_not_zero(self):
        out = analysis._pad(np.ones((4, 7)), 4)
        self.assertEqual(out.shape, (5, 7))
        self.assertTrue(np.all(np.isnan(out[-1])))
        r = analysis.max_ratio(out, np.ones(7))
        self.assertTrue(np.isnan(r[-1]) and np.all(r[:-1] == 1.0))


class TestRunner(unittest.TestCase):
    def test_short_batch_is_complete_unique_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            kw = dict(sets={"sim.T_max": 1.0, "plant.dof_frictionloss_scale": 0.0,
                            "analysis.csv_stride": 1},
                      out_root=tmp, make_plots=True, argv=["test"])
            s1 = run_experiment("nominal_fr3", **kw)
            s2 = run_experiment("nominal_fr3", **kw)
            self.assertNotEqual(s1["batch_dir"], s2["batch_dir"])
            self.assertEqual(s1["n_errors"], 0)
            run = os.path.join(s1["batch_dir"], "nominal__pic")
            for f in ("manifest.json", "metrics.json", "traj.npz", "trajectory.csv",
                      "config_resolved.yaml", "report.md", "plots/01_tcp_3d.png",
                      "plots/05_torques.png"):
                self.assertTrue(os.path.isfile(os.path.join(run, f)), f)
            man = json.load(open(os.path.join(run, "manifest.json")))
            self.assertEqual(man["status"], "complete")
            self.assertEqual(man["plant"]["applied"]["dof_frictionloss_Nm"], [0.0] * 7)
            self.assertIn("commit", man["git"]) if man["git"]["available"] else None
            z = np.load(os.path.join(run, "traj.npz"))
            self.assertEqual(z["tau_cmd"].shape, (501, 7))
            self.assertTrue(np.all(np.isnan(z["tau_cmd"][-1])))
            self.assertTrue(np.all(np.isfinite(z["tau_cmd"][:-1])))
            # the stored distance matches the stored TCP positions and goal
            self.assertTrue(np.allclose(z["dist_goal"], np.linalg.norm(z["tcp"] - z["goal"], axis=1)))
            with open(os.path.join(run, "trajectory.csv")) as fh:
                self.assertEqual(len(fh.readlines()), 502)           # header + 501 rows

    def test_bad_config_fails_before_running(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(KeyError):
                run_experiment("nominal_fr3", sets={"pic.lamda": [1, 2, 3]}, out_root=tmp,
                               make_plots=False)
            self.assertEqual(os.listdir(tmp), [])


if __name__ == "__main__":
    unittest.main()


class TestQaccRecorder(unittest.TestCase):
    def test_servo_qacc_recorded_without_changing_the_simulation(self):
        from common import sim_dynamic
        from pic.experiment import execute_controller
        cfg = copy.deepcopy(CFG)
        cfg["plant"] = {"dof_frictionloss_scale": 0.0}
        cfg["sim"]["T_max"] = 1.0
        import mujoco
        step_before = mujoco.mj_step
        case = build_case(cfg)
        plain = sim_dynamic.run(case.robot, case.ds, case.emb, case.x0_n, case.rate, case.tgt, case.sp,
                                q_start=case.q_start, n_substeps=1, tau_limit=case.tau_limit)
        rec = execute_controller("servo", case, cfg)
        np.testing.assert_array_equal(rec["traj"]["Q"], plain["traj"]["Q"])       # bit-identical states
        self.assertEqual(rec["traj"]["qacc"].shape, (500, 7))
        self.assertTrue(np.all(np.isfinite(rec["traj"]["qacc"])))
        # qacc is consistent with the stored velocities: dqd/dt over one step ~ qacc (implicit integrator: loose check)
        qd = rec["traj"]["QD"]
        fd = np.diff(qd, axis=0) / 0.002
        self.assertLess(np.median(np.abs(fd[10:] - rec["traj"]["qacc"][10:])) , 1e-2 * max(1.0, np.abs(fd).max()))
        self.assertIs(mujoco.mj_step, step_before)                                 # patch removed after the run
