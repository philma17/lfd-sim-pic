"""Robotermodelle (MuJoCo Menagerie) mit dokumentierten Limits.

Gelenkpositionslimits werden aus dem MJCF gelesen und gegen die dokumentierten
Herstellerwerte geprueft (tests/test_robots.py). Geschwindigkeitslimits stehen
NICHT im MJCF; sie kommen ausschliesslich aus den unten zitierten Quellen.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import mujoco
import numpy as np

MENAGERIE_DIR = os.environ.get(
    "MENAGERIE_DIR",
    os.path.expanduser("~/.cache/robot_descriptions/mujoco_menagerie"),
)
# Commit des lokalen Menagerie-Checkouts zum Zeitpunkt von Phase 0
MENAGERIE_COMMIT = "d7b481bac63d3d600ab404ee74f7474c5efe8a67"


@dataclass(frozen=True)
class RobotSpec:
    name: str
    mjcf: str                     # relativ zu MENAGERIE_DIR
    joints: tuple
    site: str                     # TCP-Referenz (Flansch, kein Werkzeug)
    q_min_doc: tuple              # dokumentierte Positionslimits [rad]
    q_max_doc: tuple
    qd_max: tuple                 # dokumentierte Geschwindigkeitslimits [rad/s]
    sources: dict = field(default_factory=dict)
    caveats: tuple = ()


FR3 = RobotSpec(
    name="FR3",
    mjcf="franka_fr3/fr3.xml",
    joints=tuple(f"fr3_joint{i}" for i in range(1, 8)),
    site="attachment_site",
    q_min_doc=(-2.9007, -1.8361, -2.9007, -3.0770, -2.8763, 0.4398, -3.0508),
    q_max_doc=(2.9007, 1.8361, 2.9007, -0.1169, 2.8763, 4.6216, 3.0508),
    qd_max=(2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26),
    sources={
        "position+velocity": "frankaemika/franka_description, robots/fr3/joint_limits.yaml "
                             "(main, abgerufen 2026-10-16)",
        "position (Datenblatt)": "FR3 Datasheet: A1,A3 +-166 deg, A2 +-105 deg, A4 -176/-7 deg, "
                                 "A5 +-165 deg, A6 25/265 deg, A7 +-175 deg",
        "velocity (Datenblatt)": "FR3 Datasheet: A1-A4 150 deg/s (=2.618 rad/s)",
        "mjcf": "mujoco_menagerie/franka_fr3/fr3.xml (aus franka_description abgeleitet)",
    },
    caveats=(
        "joint_limits.yaml enthaelt zusaetzlich positionsabhaengige Geschwindigkeitslimits "
        "(velocity_offset, deceleration_limit). Diese sind NICHT modelliert; die konstanten "
        "Limits sind daher eine optimistische Obergrenze fuer FR3.",
        "Datenblatt-Positionslimits weichen um bis zu 0.3 deg von joint_limits.yaml ab "
        "(Rundung auf ganze Grad).",
        "A5-A7 Geschwindigkeit nur aus joint_limits.yaml belegt, nicht im Datenblatt geprueft.",
    ),
)

UR5E = RobotSpec(
    name="UR5e",
    mjcf="universal_robots_ur5e/ur5e.xml",
    joints=("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"),
    site="attachment_site",
    q_min_doc=(-2 * np.pi, -2 * np.pi, -np.pi, -2 * np.pi, -2 * np.pi, -2 * np.pi),
    q_max_doc=(2 * np.pi, 2 * np.pi, np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi),
    qd_max=(np.pi,) * 6,
    sources={
        "position+velocity": "UniversalRobots/Universal_Robots_ROS2_Description, "
                             "config/ur5e/joint_limits.yaml (rolling, abgerufen 2026-10-16), "
                             "zitiert UR5e User Manual v5.8",
        "datasheet": "UR5e Tech Spec: alle Gelenke +-360 deg, max. 180 deg/s",
        "mjcf": "mujoco_menagerie/universal_robots_ur5e/ur5e.xml",
    },
    caveats=(
        "Ellbogen: Datenblatt +-360 deg, praktisch durch die Bauform auf ca. +-180 deg begrenzt; "
        "ROS2-Description und Menagerie setzen +-pi. Wir verwenden +-pi.",
        "MJCF rundet: 6.28319 statt 2*pi, 3.1415 statt pi (Abweichung < 1e-4 rad).",
        "Beschleunigungslimits sind laut Quelle nicht oeffentlich; nicht modelliert.",
    ),
)

ROBOTS = {"FR3": FR3, "UR5e": UR5E}


# Franka-Hand-EE-Rahmen (libfranka F_T_EE-Standard bei montierter Hand): 0.1034 m entlang
# Flansch-z, -45 deg um z. In den Five_Obstacle-Aufnahmen per FK bestaetigt (Streuung < 1e-6).
FRANKA_HAND_TCP = (np.array([0.0, 0.0, 0.1034]),
                   np.array([[np.cos(-np.pi / 4), -np.sin(-np.pi / 4), 0.0],
                             [np.sin(-np.pi / 4), np.cos(-np.pi / 4), 0.0],
                             [0.0, 0.0, 1.0]]))


def _quat_z_to(a: np.ndarray) -> np.ndarray:
    """Quaternion (w, x, y, z), das die z-Achse auf a dreht."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    z = np.array([0.0, 0.0, 1.0])
    c = np.cross(z, a)
    s, d = np.linalg.norm(c), float(z @ a)
    if s < 1e-12:
        return np.array([1.0, 0, 0, 0]) if d > 0 else np.array([0.0, 1.0, 0, 0])
    ang = np.arctan2(s, d)
    ax = c / s
    return np.concatenate([[np.cos(ang / 2)], np.sin(ang / 2) * ax])


class Robot:
    """Kinematik-Wrapper: FK, geometrischer Jacobian (Welt), Limits.

    tcp_offset = (p_off, R_off) im Site-Rahmen: TCP = Site * Offset. Ohne Offset ist der
    TCP die Site selbst (Phase 0/1, bitgleich zum frueheren Verhalten).
    tool_radius > 0 haengt eine Kapsel Site->TCP als Werkzeug-Kollisionsgeometrie an.
    obstacles: Liste von dicts {name, center, axis, radius, height} (Zylinder, Welt).
    """

    def __init__(self, spec: RobotSpec, tcp_offset=None, tool_radius: float = 0.0,
                 obstacles=None):
        self.spec = spec
        path = os.path.join(MENAGERIE_DIR, spec.mjcf)
        if tool_radius > 0 or obstacles:
            self.model = self._build(path, spec, tcp_offset, tool_radius, obstacles or [])
        else:
            self.model = mujoco.MjModel.from_xml_path(path)
        self.data = mujoco.MjData(self.model)
        self.p_off = None if tcp_offset is None else np.asarray(tcp_offset[0], float)
        self.R_off = None if tcp_offset is None else np.asarray(tcp_offset[1], float)
        self._p_tcp = np.zeros(3)
        jids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in spec.joints]
        if min(jids) < 0:
            raise ValueError(f"{spec.name}: Gelenk nicht im MJCF gefunden")
        self.qadr = np.array([self.model.jnt_qposadr[j] for j in jids])
        self.dadr = np.array([self.model.jnt_dofadr[j] for j in jids])
        self.n = len(jids)
        rng = np.array([self.model.jnt_range[j] for j in jids])
        self.q_min, self.q_max = rng[:, 0].copy(), rng[:, 1].copy()
        self.qd_max = np.asarray(spec.qd_max, float)
        self.site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, spec.site)
        if self.site < 0:
            raise ValueError(f"{spec.name}: Site {spec.site} fehlt")
        self.site_body = int(self.model.site_bodyid[self.site])
        g = self.model
        self.obstacle_geoms = [i for i in range(g.ngeom)
                               if (g.geom(i).name or "").startswith("obstacle")]
        # Hindernisse koennen in Gruppen (tag) vorliegen, z. B. gleiche Grundflaeche, andere Hoehe
        self.obstacle_tags = {}
        for i in self.obstacle_geoms:
            tag = g.geom(i).name.split("|")[1] if "|" in g.geom(i).name else ""
            self.obstacle_tags.setdefault(tag, []).append(i)
        self.tool_geoms = [i for i in range(g.ngeom) if g.geom(i).name == "tool"]
        self.link_geoms = [i for i in range(g.ngeom)
                           if g.geom_group[i] == 3 and g.geom_bodyid[i] != 0
                           and i not in self.tool_geoms]
        key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        self.q_home = self.model.key_qpos[key][self.qadr].copy() if key >= 0 else np.zeros(self.n)
        self._jp = np.zeros((3, self.model.nv))
        self._jr = np.zeros((3, self.model.nv))

    def set_q(self, q: np.ndarray) -> None:
        self.data.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    @staticmethod
    def _build(path, spec, tcp_offset, tool_radius, obstacles):
        """Modell per MjSpec um Werkzeugkapsel und Hindernis-Zylinder erweitern."""
        base = mujoco.MjModel.from_xml_path(path)
        sid = mujoco.mj_name2id(base, mujoco.mjtObj.mjOBJ_SITE, spec.site)
        bname = base.body(int(base.site_bodyid[sid])).name
        s_pos = base.site_pos[sid].copy()
        s_R = np.zeros(9)
        mujoco.mju_quat2Mat(s_R, base.site_quat[sid])
        s_R = s_R.reshape(3, 3)
        ms = mujoco.MjSpec.from_file(path)
        if tool_radius > 0:
            p_off = np.zeros(3) if tcp_offset is None else np.asarray(tcp_offset[0], float)
            a, b = s_pos, s_pos + s_R @ p_off
            ms.body(bname).add_geom(
                name="tool", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                pos=0.5 * (a + b), quat=_quat_z_to(b - a),
                size=[tool_radius, 0.5 * np.linalg.norm(b - a), 0.0],
                group=3, contype=0, conaffinity=0, rgba=[0.2, 0.2, 0.8, 1.0])
        for i, ob in enumerate(obstacles):
            ax = np.asarray(ob["axis"], float) / np.linalg.norm(ob["axis"])
            c = np.asarray(ob["center"], float) + 0.5 * ob["height"] * ax
            tag = f"|{ob['tag']}" if ob.get("tag") else ""
            ms.worldbody.add_geom(
                name=f"obstacle{i}{tag}", type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=c,
                quat=_quat_z_to(ax), size=[ob["radius"], 0.5 * ob["height"], 0.0],
                group=4, contype=0, conaffinity=0, rgba=[0.9, 0.1, 0.1, 0.6])
        return ms.compile()

    def fk(self, q: np.ndarray):
        self.set_q(q)
        p = self.data.site_xpos[self.site].copy()
        R = self.data.site_xmat[self.site].reshape(3, 3).copy()
        if self.p_off is None:
            return p, R
        self._p_tcp = p + R @ self.p_off
        return self._p_tcp.copy(), R @ self.R_off

    def jac(self) -> np.ndarray:
        """6xn Jacobian [Jp; Jr] im Weltsystem am TCP (nach set_q/fk)."""
        if self.p_off is None:
            mujoco.mj_jacSite(self.model, self.data, self._jp, self._jr, self.site)
        else:
            mujoco.mj_jac(self.model, self.data, self._jp, self._jr, self._p_tcp, self.site_body)
        return np.vstack([self._jp[:, self.dadr], self._jr[:, self.dadr]])

    def clearance(self, geoms, distmax: float = 1.0, tag=None) -> np.ndarray:
        """Abstand (m) jeder Geometrie in geoms zum naechsten Hindernis (optional nur Gruppe
        tag), nach set_q/fk. Negativ = Durchdringung."""
        out = np.full(len(geoms), distmax)
        ft = np.zeros(6)
        obs = self.obstacle_geoms if tag is None else self.obstacle_tags[tag]
        for k, g in enumerate(geoms):
            for o in obs:
                out[k] = min(out[k], mujoco.mj_geomDistance(self.model, self.data, g, o,
                                                            distmax, ft))
        return out

    def limit_margin_pct(self, q: np.ndarray) -> np.ndarray:
        """Abstand zum naechsten Positionslimit in % des Gelenkbereichs (50 % = Mitte)."""
        rng = self.q_max - self.q_min
        return 100.0 * np.minimum(q - self.q_min, self.q_max - q) / rng

    def link_summary(self) -> dict:
        """Gelenkabstaende in der Home-Pose (fuer den Bericht: Linklaengen getrennt von Reichweite)."""
        self.set_q(self.q_home)
        jids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in self.spec.joints]
        anchors = [self.data.xanchor[j].copy() for j in jids] + [self.data.site_xpos[self.site].copy()]
        seg = [float(np.linalg.norm(b - a)) for a, b in zip(anchors[:-1], anchors[1:])]
        return {"anchor_distances_m": seg, "sum_m": float(sum(seg))}
