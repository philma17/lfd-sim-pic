"""Step 2 / A / B tests: regression of the historical numbers, UR5e in the runner, actuator mapping,
gravity hold, torque limits, orientation modes, datasets. Run:
    python -B -m unittest tests.test_robots_orientation -v
Extra-dataset tests are skipped unless S2NNDS_DIR points to a directory prepared with
experiments/prepare_s2nnds.py (the skip reason is printed by unittest)."""
import copy
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.dont_write_bytecode = True

import mujoco
import numpy as np

from common import ds as dsmod
from common import sim_dynamic
from common.ik import rot_error
from common.robots import ROBOTS, Robot
from common.scenario import load_config
from common.sim_dynamic import TAU_MAX_DOC, position_servo_map
from pic.controller import Coupling, PICParams
from pic.plant import apply_plant
from pic.sim_pic import build_case, run

BASE = load_config("configs/picds.yaml")
PARAMS = PICParams.from_cfg(BASE["pic"])
DQ = BASE["pic"]["null_space"]["d_q"]
EXTRA = os.path.isdir(os.path.join(dsmod.S2NNDS_DIR, "models_verified", "2D_Shapes", "Five_Obstacle_DS"))


def cfg_for(robot="FR3", orientation="none", friction=0.0):
    c = copy.deepcopy(BASE)
    c["robot"] = robot
    c["task"]["orientation"] = orientation
    c["plant"] = {"dof_frictionloss_scale": friction}
    return c


class TestRegressionHistoric(unittest.TestCase):
    """LASA/Sine FR3 defaults must give the historical numbers (EXPERIMENT_LOG.md) after the refactor."""

    def test_fr3_historic_numbers(self):
        hist = {0.0: dict(servo=(1.006959e-11, 6.128), pic=(1.053198237e-06, 6.924)),
                1.0: dict(servo=(2.00456e-07, 6.21), pic=(9.41185020e-02, None))}
        for fs, h in hist.items():
            case = build_case(BASE, friction_scale=fs)
            sv = sim_dynamic.run(case.robot, case.ds, case.emb, case.x0_n, case.rate, case.tgt, case.sp,
                                 q_start=case.q_start, n_substeps=1, tau_limit=case.tau_limit)
            pi = run(case, PARAMS, null_d_q=DQ)
            self.assertAlmostEqual(sv["final_dist_m"], h["servo"][0], delta=1e-9 + 1e-5 * h["servo"][0],
                                   msg=f"servo {fs}")
            self.assertEqual(sv["t_reach_s"], h["servo"][1])
            self.assertAlmostEqual(pi["final_dist_m"], h["pic"][0], delta=2e-9 + 1e-6 * h["pic"][0],
                                   msg=f"pic {fs}")
            self.assertEqual(pi["t_reach_s"], h["pic"][1])
            self.assertEqual(pi["null_dim_min_max"], [4, 4])

    def test_start_configuration_matches_baseline(self):
        case = build_case(BASE)
        np.testing.assert_allclose(case.q_start, [0.00325375129960457, -0.2442726286424409,
                                                  0.0033691359691201943, -2.0812376788143045,
                                                  0.0007605664156411551, 1.2226910587064925, -0.7853],
                                   atol=1e-12)
        case = build_case(cfg_for("UR5e"))
        np.testing.assert_allclose(case.q_start, [0.12270310518854471, -1.3860191609835344,
                                                  -1.7932420290232634, 4.57140371012737,
                                                  -3.6538239140983904, 1.54801647123979], atol=1e-9)


class TestRobotsInRunner(unittest.TestCase):
    def test_robot_comes_from_config_and_unknown_is_rejected(self):
        self.assertEqual(build_case(cfg_for("UR5e")).robot_name, "UR5e")
        self.assertEqual(build_case(cfg_for("UR5e")).robot.n, 6)
        with self.assertRaises(ValueError):
            build_case(cfg_for("Panda"))

    def test_joint_order_and_actuator_mapping(self):
        """Actuator i drives joint i of the robot spec, 1:1, with the documented kp."""
        for name, kp0 in (("FR3", 4500.0), ("UR5e", 2000.0)):
            r = Robot(ROBOTS[name])
            sv = position_servo_map(r)
            self.assertEqual(sv.act.tolist(), list(range(r.n)))
            for i, j in enumerate(r.spec.joints):
                jid = mujoco.mj_name2id(r.model, mujoco.mjtObj.mjOBJ_JOINT, j)
                self.assertEqual(int(r.model.actuator_trnid[sv.act[i], 0]), jid)
                self.assertEqual(int(r.dadr[i]), int(r.model.jnt_dofadr[jid]))
            self.assertEqual(sv.kp[0], kp0)

    def test_gravity_hold_3s_tcp_drift(self):
        """Hold ctrl = q0 + g(q0)/kp for 3 s (sim_dynamic.static_sag with gravity_ff): TCP drift < 0.05 mm."""
        for name in ("FR3", "UR5e"):
            case = build_case(cfg_for(name))
            sag = sim_dynamic.static_sag(case.robot, case.q_start, hold_s=3.0, gravity_ff=True)
            self.assertLess(sag["tcp_disp_mm"], 0.05, f"{name}: {sag['tcp_disp_mm']} mm")
            self.assertLess(sag["residual_qvel_max"], 1e-3)

    def test_torque_limits_enforced_alike(self):
        """87/12 Nm (FR3) and 150/28 Nm (UR5e), also scaled, are enforced in the servo (forcerange) and
        PIC-DS (clipping) through the same code path."""
        for name, lim in (("FR3", [87, 87, 87, 87, 12, 12, 12]), ("UR5e", [150, 150, 150, 28, 28, 28])):
            np.testing.assert_allclose(TAU_MAX_DOC[name][0], lim)
            for scale in (1.0, 0.5):
                c = cfg_for(name)
                c["plant"]["torque_limit_scale"] = scale
                case = build_case(c)
                np.testing.assert_allclose(case.tau_limit, scale * np.asarray(lim, float))
                m = case.robot.model
                self.assertTrue(np.all(m.actuator_forcelimited == 1))
                np.testing.assert_allclose(m.actuator_forcerange[:, 1], case.tau_limit)
                d = mujoco.MjData(m)
                d.qpos[case.robot.qadr] = case.q_start
                d.ctrl[:] = case.q_start + 2.0                       # absurd position error
                mujoco.mj_forward(m, d)
                np.testing.assert_allclose(np.abs(d.actuator_force), case.tau_limit)
                r = run(case, PARAMS, null_d_q=DQ, T_max=1.0)
                self.assertTrue(np.all(r["traj"]["tau_limit"] == case.tau_limit))
                self.assertTrue(np.all(np.abs(r["traj"]["tau_applied"][:-1]) <= case.tau_limit + 1e-9))

    def test_synthetic_friction_ur5e_relative_and_absolute(self):
        r = Robot(ROBOTS["UR5e"])
        rep = apply_plant(r, {})
        fr3 = np.array([1.137, 1.137, 1.137, 1.137, 0.763, 0.44])
        lim_fr3 = np.array([87, 87, 87, 87, 12, 12.0])
        np.testing.assert_allclose(r.model.dof_frictionloss,
                                   fr3 / lim_fr3 * np.array([150, 150, 150, 28, 28, 28]))
        self.assertEqual(rep["applied"]["synthetic_friction"], "relative")
        r2 = Robot(ROBOTS["UR5e"])
        apply_plant(r2, {"synthetic_friction": "absolute"})
        np.testing.assert_allclose(r2.model.dof_frictionloss, fr3)
        r3 = Robot(ROBOTS["UR5e"])
        apply_plant(r3, {"synthetic_friction": "none"})
        self.assertTrue(np.all(r3.model.dof_frictionloss == 0))
        with self.assertRaises(ValueError):                     # must not overwrite real FR3 friction
            apply_plant(Robot(ROBOTS["FR3"]), {"synthetic_friction": "relative"})

    def test_ur5e_pic_and_servo_execute(self):
        case = build_case(cfg_for("UR5e"))
        sv = sim_dynamic.run(case.robot, case.ds, case.emb, case.x0_n, case.rate, case.tgt, case.sp,
                             q_start=case.q_start, n_substeps=1, tau_limit=case.tau_limit)
        pi = run(case, PARAMS, null_d_q=DQ)
        self.assertTrue(sv["success"] and pi["success"])
        self.assertEqual(pi["null_dim_min_max"], [3, 3])


class TestOrientation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = {}
        for robot in ("FR3", "UR5e"):
            for mode in ("none", "hold"):
                case = build_case(cfg_for(robot, mode))
                cls.res[(robot, mode)] = (case, run(case, PARAMS, null_d_q=DQ, T_max=3.0))

    def test_none_is_unchanged(self):
        case, r = self.res[("FR3", "none")]
        self.assertIsNone(case.orient)
        self.assertEqual(r["orientation"]["mode"], "none")
        self.assertEqual(r["null_dim_min_max"], [4, 4])
        self.assertTrue(np.all(r["traj"]["tau_rot"] == 0))
        self.assertTrue(np.all(r["traj"]["E_rot"] == 0))

    def test_null_dims(self):
        self.assertEqual(self.res[("FR3", "hold")][1]["null_dim_min_max"], [1, 1])
        self.assertEqual(self.res[("UR5e", "hold")][1]["null_dim_min_max"], [0, 0])

    def test_hold_orientation_error_converges_at_standstill(self):
        """Start with a tilt, f = 0 (no DS motion): the orientation error must decay to ~0."""
        for robot in ("FR3", "UR5e"):
            case = build_case(cfg_for(robot, "hold"))
            q0 = case.q_start.copy()
            q0[-1] += 0.15
            e0 = np.linalg.norm(rot_error(case.robot.fk(q0)[1], case.orient.R_des))
            self.assertGreater(e0, 0.05)
            r = run(case, PARAMS, null_d_q=DQ, use_f=False, q_start=q0, T_max=6.0)
            e = r["traj"]["ori_err"]
            self.assertLess(e[-1], 0.05 * e0, f"{robot}: {e0} -> {e[-1]}")
            self.assertTrue(r["finite"])

    def test_hold_spring_energy_in_bookkeeping(self):
        """With E_rot in W the per-step energy increase stays at the numerical level in a tilt-release run;
        without E_rot the released spring energy would appear as large positive dW."""
        case = build_case(cfg_for("FR3", "hold"))
        q0 = case.q_start.copy()
        q0[-1] += 0.15
        r = run(case, PARAMS, null_d_q=DQ, use_f=False, q_start=q0, T_max=3.0)
        self.assertGreater(r["traj"]["E_rot"][0], 1e-2)               # 1/2 * 50 * 0.15^2 ~ 0.56 J
        self.assertLess(r["energy"]["sum_positive_dW_J"], 1e-3)
        dW_without = np.diff(r["traj"]["KE"] + r["traj"]["s"])
        self.assertGreater(np.sum(dW_without[dW_without > 0]), 10 * max(r["energy"]["sum_positive_dW_J"], 1e-9))

    def test_gain_derivation(self):
        case = build_case(cfg_for("FR3", "hold"))
        o = case.orient
        self.assertEqual((o.K_r, o.zeta), (50.0, 1.414))
        w = np.linalg.eigvalsh(o.Lambda_r)
        np.testing.assert_allclose(np.linalg.eigvalsh(o.D_r), 2 * 1.414 * np.sqrt(50.0) * np.sqrt(w), rtol=1e-9)
        from pic.sim_pic import verify_orientation_gain_source
        chk = verify_orientation_gain_source()
        if not chk["checked"]:
            self.skipTest(chk["reason"])
        self.assertTrue(chk["match"])


class TestCouplingGrid(unittest.TestCase):
    def test_eq_14_and_ranges_on_grid(self):
        """Numerical check of coupling-function properties on a (z, s) grid for both step kinds
        (the full Eq. 11/12/14 test is tests/test_pic.py::test_conditions_eq_11_12_14)."""
        for kind in ("quintic", "cosine"):
            cp = Coupling(10.0, 1.0, 0.01, kind)
            z = np.linspace(-0.05, 0.05, 101)
            s = np.linspace(0.0, 10.0, 201)
            Z, S = np.meshgrid(z, s)
            br, bs, al = cp.beta_R(Z, S), cp.beta_s(Z, S), cp.alpha(S)
            for a in (br, bs, al):
                self.assertTrue(np.all((a >= -1e-12) & (a <= 1 + 1e-12)))
            self.assertTrue(np.allclose(br[Z >= 0], bs[Z >= 0]))        # Eq. 14: beta_R = beta_s for z >= 0


@unittest.skipUnless(EXTRA, "extra S2NNDS models not prepared (set S2NNDS_DIR, experiments/prepare_s2nnds.py)")
class TestDatasets(unittest.TestCase):
    def test_other_datasets_build_and_run_short(self):
        for robot in ("FR3", "UR5e"):
            for dataset, shape, dim in (("2D_Shapes", "Five_Obstacle_DS", 2), ("3D_Shapes", "Cshape_bottom", 3)):
                c = cfg_for(robot)
                c["dataset"], c["shape"] = dataset, shape
                case = build_case(c)
                self.assertEqual(len(case.x0_n), dim)
                self.assertIsNotNone(case.unsafe)
                r = run(case, PARAMS, null_d_q=DQ, T_max=1.0)
                self.assertTrue(r["finite"])
                self.assertIn("clearance_min_m", r)

    def test_start_index_from_init_range(self):
        c = cfg_for("FR3")
        c["task"]["start"] = {"index": 3}
        case = build_case(c)
        box = case.scenario["init_box"]
        self.assertTrue(np.all(case.x0_n >= box[:, 0]) and np.all(case.x0_n <= box[:, 1]))
        c["task"]["start"] = {"index": 999}
        with self.assertRaises(ValueError):
            build_case(c)


if __name__ == "__main__":
    unittest.main()
