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
| synthetic_friction        | UR5e only: model.dof_frictionloss from the FR3     | SYNTHETIC|
|                           | values (relative to the torque limit or absolute)  |          |
| com_shift                 | model.body_ipos of one link (local frame)          | verified |

Torque limits: the documented FR3 limits (common.sim_dynamic.TAU_MAX_DOC, 87/12 Nm) are NOT
part of the Menagerie MJCF. They are enforced here in two places, both with the scaled limit:
  * PIC-DS: manual clipping of the commanded torque in pic.sim_pic.run (commanded and applied
    torque are logged separately),
  * position servo: model.actuator_forcelimited/forcerange are set, so MuJoCo clamps
    actuator_force; the unclamped servo law is logged as commanded torque.

SYNTHETIC UR5e friction (ASSUMPTION, open question for the supervisor): the Menagerie UR5e has
dof_frictionloss = 0 and dof_damping = 0. With synthetic_friction = "relative" (default for the UR5e)
frictionloss_i = FR3_frictionloss_i / FR3_torque_limit_i * UR5e_torque_limit_i  for i = 1..6
(the same fraction of the torque limit; joint i of the UR5e is paired with joint i of the FR3 by INDEX,
which is an assumption). "absolute": frictionloss_i = FR3_frictionloss_i (same Nm). "none": 0.
dof_frictionloss_scale multiplies the result afterwards.

COM shift: model.body_ipos[body] += dxyz (body-local frame, metres). MuJoCo stores the inertia about the
COM, so the inertia about the joints changes by the parallel-axis term automatically; the mass is unchanged.

Not implemented (extension points, see HOWTO_EXPERIMENTS.md): per-link mass, armature, arbitrary
gravity direction.
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
    "synthetic_friction": None,          # None -> "relative" for robots without friction in the MJCF
    "com_shift": None,                   # {link: "last" | <body name>, dxyz_m: [dx, dy, dz]}
}
FR3_REF = ("FR3", np.array([1.137, 1.137, 1.137, 1.137, 0.763, 0.44, 0.248]))   # Menagerie, = model values


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
        if k == "synthetic_friction":
            if v not in (None, "none", "relative", "absolute"):
                raise ValueError("plant.synthetic_friction must be none | relative | absolute")
            out[k] = v
            continue
        if k == "com_shift":
            if v is not None:
                if set(v) - {"link", "dxyz_m"} or len(v.get("dxyz_m", [])) != 3:
                    raise ValueError("plant.com_shift needs {link: ..., dxyz_m: [dx, dy, dz]}")
                v = {"link": str(v.get("link", "last")), "dxyz_m": [float(x) for x in v["dxyz_m"]]}
            out[k] = v
            continue
        v = float(v)
        if not np.isfinite(v) or v < 0:
            raise ValueError(f"plant.{k} must be finite and >= 0, got {v}")
        if k in ("mass_scale", "inertia_scale", "torque_limit_scale") and v <= 0:
            raise ValueError(f"plant.{k} must be > 0, got {v}")
        out[k] = v
    return out


def synthetic_frictionloss(robot_name: str, mode: str) -> np.ndarray:
    """Synthetic joint Coulomb friction for a robot whose MJCF has none (see module docstring)."""
    ref = FR3_REF[1]
    n = len(TAU_MAX_DOC[robot_name][0])
    if n > len(ref):
        raise ValueError("more joints than the FR3 reference")
    if mode == "absolute":
        return ref[:n].copy()
    return ref[:n] / TAU_MAX_DOC["FR3"][0][:n] * TAU_MAX_DOC[robot_name][0]


def torque_limit(robot_name: str, scale: float) -> np.ndarray:
    return TAU_MAX_DOC[robot_name][0] * float(scale)


def apply_plant(robot, plant: dict | None) -> dict:
    """Modify robot.model in memory; return the report. Call once per freshly built Robot."""
    p = resolve_plant(plant)
    m = robot.model
    nominal = {"gravity": m.opt.gravity.copy()}
    syn = None
    if np.all(m.dof_frictionloss[robot.dadr] == 0.0):
        syn = p["synthetic_friction"] or "relative"
        if syn != "none" and robot.spec.name != "FR3":
            m.dof_frictionloss[robot.dadr] = synthetic_frictionloss(robot.spec.name, syn)
        else:
            syn = "none" if robot.spec.name != "FR3" else None
    elif p["synthetic_friction"] not in (None, "none"):
        raise ValueError(f"{robot.spec.name} MJCF already has joint friction; synthetic_friction "
                         "would overwrite a real model value")
    m.dof_frictionloss[:] *= p["dof_frictionloss_scale"]
    m.dof_damping[:] *= p["dof_damping_scale"]
    m.opt.gravity[:] = m.opt.gravity * p["gravity_scale"]
    m.body_mass[:] *= p["mass_scale"]
    m.body_inertia[:] *= p["mass_scale"] * p["inertia_scale"]
    com = None
    if p["com_shift"] is not None:
        link = p["com_shift"]["link"]
        b = robot.site_body if link == "last" else mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, link)
        if b < 0:
            raise ValueError(f"plant.com_shift: body '{link}' not found")
        m.body_ipos[b] += np.asarray(p["com_shift"]["dxyz_m"], float)
        com = {"body": m.body(b).name, "dxyz_m": p["com_shift"]["dxyz_m"],
               "ipos_after_m": m.body_ipos[b].tolist()}
    # derived constants (subtree mass, invweights) depend on mass/inertia/COM
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
            "synthetic_friction": syn,
            "com_shift": com,
            "armature_kg_m2": m.dof_armature[robot.dadr].tolist(),
        },
    }
