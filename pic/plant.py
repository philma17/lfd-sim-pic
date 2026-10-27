"""Physical plant variations for the MuJoCo FR3 model, applied IN MEMORY only (MJCF untouched).

Every variation is a multiplicative factor on the nominal Menagerie value, 1.0 = nominal.
`apply_plant` returns a report with the factors AND the resulting model arrays, so a run record
shows what actually reached the physics engine (not only what was requested).

| key (cfg `plant:`)        | MuJoCo parameter changed                          | status   |
|---------------------------|---------------------------------------------------|----------|
| dof_frictionloss_scale    | model.dof_frictionloss (Coulomb joint friction)   | verified |
| dof_damping_scale         | model.dof_damping (viscous joint friction)        | verified |
| gravity_scale             | model.opt.gravity (z component, scaled)           | verified |
| mass_scale                | model.body_mass AND body_inertia (together)       | verified |
| inertia_scale             | model.body_inertia only (extra factor)            | verified |
| torque_limit_scale        | actuator limit = scale * TAU_MAX_DOC (see below)  | verified |

Torque limits: the documented FR3 limits (common.sim_dynamic.TAU_MAX_DOC, 87/12 Nm) are NOT
part of the Menagerie MJCF. They are enforced here in two places, both with the scaled limit:
  * PIC-DS: manual clipping of the commanded torque in pic.sim_pic.run (commanded and applied
    torque are logged separately),
  * position servo: model.actuator_forcelimited/forcerange are set, so MuJoCo clamps
    actuator_force; the unclamped servo law is logged as commanded torque.

Not implemented (extension points, see HOWTO_EXPERIMENTS.md): per-link COM offset
(model.body_ipos), per-link mass, armature, arbitrary gravity direction.
"""
from __future__ import annotations

import mujoco
import numpy as np

from common.sim_dynamic import TAU_MAX_DOC, position_servo_map

PLANT_DEFAULTS = {
    "dof_frictionloss_scale": 1.0,
    "dof_damping_scale": 1.0,
    "gravity_scale": 1.0,
    "mass_scale": 1.0,
    "inertia_scale": 1.0,
    "torque_limit_scale": 1.0,
    "servo_enforce_torque_limit": True,
}


def resolve_plant(plant: dict | None) -> dict:
    """Defaults + unknown-key check (a typo in a YAML file must not be silently ignored)."""
    plant = dict(plant or {})
    unknown = sorted(set(plant) - set(PLANT_DEFAULTS))
    if unknown:
        raise ValueError(f"unknown plant keys {unknown}; allowed: {sorted(PLANT_DEFAULTS)}")
    out = {**PLANT_DEFAULTS, **plant}
    for k, v in out.items():
        if k == "servo_enforce_torque_limit":
            out[k] = bool(v)
            continue
        v = float(v)
        if not np.isfinite(v) or v < 0:
            raise ValueError(f"plant.{k} must be finite and >= 0, got {v}")
        if k in ("mass_scale", "inertia_scale", "torque_limit_scale") and v <= 0:
            raise ValueError(f"plant.{k} must be > 0, got {v}")
        out[k] = v
    return out


def torque_limit(robot_name: str, scale: float) -> np.ndarray:
    return TAU_MAX_DOC[robot_name][0] * float(scale)


def apply_plant(robot, plant: dict | None) -> dict:
    """Modify robot.model in memory; return the report. Call once per freshly built Robot."""
    p = resolve_plant(plant)
    m = robot.model
    nominal = {"gravity": m.opt.gravity.copy()}
    m.dof_frictionloss[:] *= p["dof_frictionloss_scale"]
    m.dof_damping[:] *= p["dof_damping_scale"]
    m.opt.gravity[:] = m.opt.gravity * p["gravity_scale"]
    m.body_mass[:] *= p["mass_scale"]
    m.body_inertia[:] *= p["mass_scale"] * p["inertia_scale"]
    # derived constants (subtree mass, invweights) depend on mass/inertia
    mujoco.mj_setConst(m, mujoco.MjData(m))
    lim = torque_limit(robot.spec.name, p["torque_limit_scale"])
    if p["servo_enforce_torque_limit"]:
        sv = position_servo_map(robot)
        m.actuator_forcelimited[sv.act] = 1
        m.actuator_forcerange[sv.act, 0] = -lim
        m.actuator_forcerange[sv.act, 1] = lim
    robot.friction_scale = p["dof_frictionloss_scale"]    # kept for older callers
    return {
        "requested": p,
        "nominal_gravity_m_s2": nominal["gravity"].tolist(),
        "applied": {
            "dof_frictionloss_Nm": m.dof_frictionloss[robot.dadr].tolist(),
            "dof_damping_Nms_rad": m.dof_damping[robot.dadr].tolist(),
            "gravity_m_s2": m.opt.gravity.tolist(),
            "total_mass_kg": float(m.body_mass.sum()),
            "torque_limit_Nm": lim.tolist(),
            "servo_forcelimited": bool(p["servo_enforce_torque_limit"]),
        },
    }
