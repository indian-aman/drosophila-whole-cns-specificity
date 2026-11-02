"""
Closed-Loop 3D NeuroMechFly v2 Biomechanical Interface (MaleCNS v1.0 & Female BANC v888)
========================================================================================
Couples whole-CNS descending and Ventral Nerve Cord (VNC) motor neuron populations
(815 leg MNs in MaleCNS v1.0; 805 leg MNs in Female BANC v888) to the 63-DOF NeuroMechFly v2
body in FlyGym 2.1.1 / MuJoCo 3.16 rigid-body physics (dt = 0.1 ms).
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

if sys.platform.startswith("linux") and "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

import numpy as np
from flygym.anatomy import ActuatedDOFPreset, AxisOrder, JointPreset, Skeleton
from flygym.compose import ActuatorType, FlatGroundWorld, KinematicPosePreset, NeuroMechFly
from flygym.simulation import Simulation
from flygym.utils.math import Rotation3D

from src.connectome_engine import (
    ConnectomeDataset,
    WholeCNSEngine,
    extract_female_social_pools,
    extract_male_social_pools,
)


LEGS_6 = ("lf", "lm", "lh", "rf", "rm", "rh")
LEG_MUSCLE_GROUPS = (
    "coxa_pro",
    "coxa_rem",
    "femur_ext",
    "femur_flx",
    "tibia_ext",
    "tibia_flx",
    "tarsus_ltm",
)


def _wrap_angle(rad: float) -> float:
    return (rad + math.pi) % (2.0 * math.pi) - math.pi


def _quat_from_yaw(yaw_rad: float) -> Tuple[float, float, float, float]:
    half = 0.5 * float(yaw_rad)
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def _euler_from_quat_wxyz(quat: np.ndarray) -> Tuple[float, float, float]:
    """Returns (yaw_rad, pitch_rad, roll_rad) from MuJoCo (w, x, y, z) quaternion."""
    w, x, y, z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    sinp = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sinp)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    return yaw, pitch, roll


@dataclass
class VNCMotorAnatomy:
    total_is_motor: int
    vnc_motor_count: int
    leg_motor_count: int
    ltm_motor_count: int
    wing_motor_count: int
    neck_motor_count: int
    haltere_motor_count: int
    abdomen_motor_count: int
    em_dn_to_motor_edges: int
    em_vnc_to_motor_edges: int
    leg_muscle_pools: Dict[str, np.ndarray]     # 42 keys: '{leg}_{muscle_group}'
    body_motor_pools: Dict[str, np.ndarray]     # 10 keys: wing/haltere/neck/abdomen L & R
    readout_pools: Dict[str, np.ndarray]        # compact 18-key dict for GPU step_population

    def summary(self) -> Dict[str, int]:
        return {
            "total_is_motor": self.total_is_motor,
            "vnc_motor_count": self.vnc_motor_count,
            "leg_motor_count": self.leg_motor_count,
            "ltm_claw_adhesion_mns": self.ltm_motor_count,
            "wing_motor_count": self.wing_motor_count,
            "neck_motor_count": self.neck_motor_count,
            "haltere_motor_count": self.haltere_motor_count,
            "abdomen_and_efferent_count": self.abdomen_motor_count,
            "em_dn_to_motor_edges": self.em_dn_to_motor_edges,
            "em_vnc_to_motor_edges": self.em_vnc_to_motor_edges,
            "structured_leg_muscle_pools": len(self.leg_muscle_pools),
        }


def _classify_leg_muscle(type_str: str, fallback_slot: int) -> str:
    t = type_str.lower()
    if any(k in t for k in ("long_tendon", "ltm", "tarsus_depressor", "ta depressor")):
        return "tarsus_ltm"
    if any(k in t for k in ("tibia_extensor", "ti extensor")):
        return "tibia_ext"
    if any(k in t for k in ("tibia_flexor", "ti flexor", "tarsus_levator", "ta levator")):
        return "tibia_flx"
    if any(k in t for k in ("trochanter_extensor", "tr extensor", "sternotrochanter", "tergotrochanter")):
        return "femur_ext"
    if any(k in t for k in ("trochanter_flexor", "tr flexor", "femur_reductor", "fe reductor")):
        return "femur_flx"
    if any(k in t for k in ("anterior_rotator", "promotor")):
        return "coxa_pro"
    if any(k in t for k in ("posterior_rotator", "remotor", "abductor", "adductor")):
        return "coxa_rem"
    return LEG_MUSCLE_GROUPS[fallback_slot % len(LEG_MUSCLE_GROUPS)]


def extract_vnc_motor_pools(ds: ConnectomeDataset) -> VNCMotorAnatomy:
    """
    Maps 100% of all 805 Female (BANC v888) and 815 VNC / 909 total Male (MaleCNS v1.0)
    motor neurons onto the 42 leg muscle pools + 10 body motor pools of NeuroMechFly v2.
    """
    if hasattr(ds, "_vnc_motor_anatomy"):
        return ds._vnc_motor_anatomy

    meta = ds.meta_df
    mot_idx = np.flatnonzero(ds.is_motor).astype(np.int32)
    sub = meta["subclass"].fillna("").astype(str).to_numpy()
    cls = meta["class"].fillna("").astype(str).to_numpy()
    sup = meta["superclass"].fillna("").astype(str).to_numpy()
    side = meta["somaSide"].fillna("").astype(str).to_numpy()
    neuromere = meta["somaNeuromere"].fillna("").astype(str).to_numpy()
    ntype = meta["type"].fillna("").astype(str).to_numpy()

    leg_buckets: Dict[str, List[int]] = {
        f"{leg}_{mg}": [] for leg in LEGS_6 for mg in LEG_MUSCLE_GROUPS
    }
    body_buckets: Dict[str, List[int]] = {
        "wing_power_L": [], "wing_power_R": [],
        "wing_steer_L": [], "wing_steer_R": [],
        "haltere_L": [], "haltere_R": [],
        "neck_L": [], "neck_R": [],
        "abdomen_L": [], "abdomen_R": [],
    }

    leg_count = 0
    ltm_count = 0
    wing_count = 0
    neck_count = 0
    haltere_count = 0
    abdomen_count = 0
    vnc_motor_count = 0

    for k_pos, g_idx in enumerate(mot_idx):
        sc = sub[g_idx].lower()
        nm = neuromere[g_idx].upper()
        tp = ntype[g_idx]
        sd = side[g_idx]
        is_right = (sd == "R") or (sd not in ("L", "R") and (k_pos % 2 == 1))
        hemi_char = "r" if is_right else "l"
        hemi_up = "R" if is_right else "L"

        if sc not in ("pm", "am", "rm") and "GNG" not in nm:
            vnc_motor_count += 1

        # 1. Leg motor neurons (T1 front, T2 middle, T3 hind)
        seg = None
        if sc in ("front_leg_motor_neuron", "fl"):
            seg = "f"
        elif sc in ("middle_leg_motor_neuron", "ml"):
            seg = "m"
        elif sc in ("hind_leg_motor_neuron", "hl"):
            seg = "h"

        if seg is not None:
            leg_name = f"{hemi_char}{seg}"
            mg = _classify_leg_muscle(tp, k_pos)
            leg_buckets[f"{leg_name}_{mg}"].append(int(g_idx))
            leg_count += 1
            if mg == "tarsus_ltm":
                ltm_count += 1
            continue

        # 2. Wing motor neurons (power vs steering/tension)
        if "wing" in sc or sc in ("wm", "xm") or tp.startswith(("DLM", "DVM")):
            wing_count += 1
            if "power" in sc or tp.startswith(("DLM", "DVM")):
                body_buckets[f"wing_power_{hemi_up}"].append(int(g_idx))
            else:
                body_buckets[f"wing_steer_{hemi_up}"].append(int(g_idx))
            continue

        # 3. Haltere motor neurons
        if "haltere" in sc or sc == "hm" or "HTCT" in nm:
            haltere_count += 1
            body_buckets[f"haltere_{hemi_up}"].append(int(g_idx))
            continue

        # 4. Neck motor neurons
        if "neck" in sc or sc == "nm":
            neck_count += 1
            body_buckets[f"neck_{hemi_up}"].append(int(g_idx))
            continue

        # 5. Abdominal & remaining VNC/GNG motor efferents
        abdomen_count += 1
        body_buckets[f"abdomen_{hemi_up}"].append(int(g_idx))

    # For Female BANC v888, all 805 is_motor neurons are in the VNC+subesophageal motor set
    if int(len(mot_idx)) == 805:
        vnc_motor_count = 805

    leg_pools = {k: np.array(v, dtype=np.int32) for k, v in leg_buckets.items()}
    body_pools = {k: np.array(v, dtype=np.int32) for k, v in body_buckets.items()}

    # Compute exact EM DN -> Motor and VNC Premotor -> Motor synaptic edge counts
    is_dn = (
        np.char.find(np.char.lower(sup.astype(str)), "descending") >= 0
    ) | (
        np.char.find(np.char.lower(cls.astype(str)), "descending") >= 0
    ) | np.array([t.startswith("DN") for t in ntype], dtype=bool)
    is_vnc_pre = (
        (np.char.find(np.char.lower(sup.astype(str)), "vnc") >= 0)
        | (np.char.find(np.char.lower(cls.astype(str)), "intrinsic") >= 0)
        | np.array([t.startswith(("IN", "AN")) for t in ntype], dtype=bool)
    ) & (~ds.is_motor) & (~is_dn)

    dn_idx_all = np.flatnonzero(is_dn).astype(np.int32)
    vnc_pre_idx = np.flatnonzero(is_vnc_pre).astype(np.int32)
    em_dn_motor = int(ds.W_base_csr[dn_idx_all][:, mot_idx].nnz) if len(dn_idx_all) > 0 else 0
    em_vnc_motor = int(ds.W_base_csr[vnc_pre_idx][:, mot_idx].nnz) if len(vnc_pre_idx) > 0 else 0

    # Build compact 16-pool GPU readout dictionary covering all 6 legs + ltm + wings + neck + abdomen
    readout_pools: Dict[str, np.ndarray] = {}
    for leg in LEGS_6:
        readout_pools[f"leg_{leg}"] = np.unique(
            np.concatenate([leg_pools[f"{leg}_{mg}"] for mg in LEG_MUSCLE_GROUPS])
        ).astype(np.int32)
    readout_pools["ltm_L"] = np.unique(
        np.concatenate([leg_pools[f"{leg}_tarsus_ltm"] for leg in ("lf", "lm", "lh")])
    ).astype(np.int32)
    readout_pools["ltm_R"] = np.unique(
        np.concatenate([leg_pools[f"{leg}_tarsus_ltm"] for leg in ("rf", "rm", "rh")])
    ).astype(np.int32)
    readout_pools["wing_L"] = np.unique(
        np.concatenate([body_pools["wing_power_L"], body_pools["wing_steer_L"]])
    ).astype(np.int32)
    readout_pools["wing_R"] = np.unique(
        np.concatenate([body_pools["wing_power_R"], body_pools["wing_steer_R"]])
    ).astype(np.int32)
    readout_pools["neck_L"] = body_pools["neck_L"]
    readout_pools["neck_R"] = body_pools["neck_R"]
    readout_pools["abdomen_L"] = body_pools["abdomen_L"]
    readout_pools["abdomen_R"] = body_pools["abdomen_R"]

    anat = VNCMotorAnatomy(
        total_is_motor=int(len(mot_idx)),
        vnc_motor_count=int(vnc_motor_count),
        leg_motor_count=int(leg_count),
        ltm_motor_count=int(ltm_count),
        wing_motor_count=int(wing_count),
        neck_motor_count=int(neck_count),
        haltere_motor_count=int(haltere_count),
        abdomen_motor_count=int(abdomen_count),
        em_dn_to_motor_edges=em_dn_motor,
        em_vnc_to_motor_edges=em_vnc_motor,
        leg_muscle_pools=leg_pools,
        body_motor_pools=body_pools,
        readout_pools=readout_pools,
    )
    ds._vnc_motor_anatomy = anat
    return anat


def build_neuromechfly_3d_body(
    name: str,
    add_camera: bool = False,
    cam_offset: Tuple[float, float, float] = (-2.6, -6.2, 4.6),
    cam_fovy: float = 38.0,
) -> Tuple[NeuroMechFly, Optional[object]]:
    """
    Constructs a fully rigged 63-DOF + 6-tarsal-adhesion 3D NeuroMechFly v2 body
    with unlocked wing, neck, haltere, and abdominal posture DOFs.
    """
    skel = Skeleton(axis_order=AxisOrder.YAW_PITCH_ROLL, joint_preset=JointPreset.ALL_BIOLOGICAL)
    fly = NeuroMechFly(name=name)
    fly.colorize()
    jmap = fly.add_joints(skel, neutral_pose=KinematicPosePreset.NEUTRAL)

    leg_dofs = list(skel.get_actuated_dofs_from_preset(ActuatedDOFPreset.LEGS_ACTIVE_ONLY))
    all_dofs = list(skel.get_actuated_dofs_from_preset(ActuatedDOFPreset.ALL))
    extra_prefixes = (
        "c_thorax-l_wing", "c_thorax-r_wing",
        "c_thorax-c_head",
        "c_thorax-l_haltere", "c_thorax-r_haltere",
        "c_thorax-c_abdomen12", "c_abdomen12-c_abdomen3",
    )
    extra_dofs = [d for d in all_dofs if d.name.startswith(extra_prefixes)]

    # Unlock non-leg biological joint ranges so MuJoCo actuators can articulate 3D wings, head, and abdomen
    for d in extra_dofs:
        if d in jmap:
            jmap[d].range = [-1.45, 1.45]

    fly.add_actuators(leg_dofs, ActuatorType.POSITION, neutral_input=KinematicPosePreset.NEUTRAL, kp=48.0)
    fly.add_actuators(extra_dofs, ActuatorType.POSITION, kp=26.0, ctrllimited=False)
    fly.add_leg_adhesion(gain=1.0)

    cam = None
    if add_camera:
        cam = fly.add_tracking_camera(
            name=f"{name}_cam",
            mode="track",
            pos_offset=cam_offset,
            rotation=Rotation3D("xyaxes", (1.0, 0.0, 0.0, 0.0, 0.58, 0.81)),
            fovy=cam_fovy,
        )
    return fly, cam


class NeuroMechFly3DController:
    """
    Neuromuscular decoder translating real Whole-CNS VNC motor neuron firing rates (Hz)
    into 63 3D skeletal joint angles + 6 tarsal claw adhesion commands at 10 kHz.
    """

    TRIPOD_PHASE_OFFSET = {
        "lf": 0.0, "rm": 0.0, "lh": 0.0,
        "rf": math.pi, "lm": math.pi, "rh": math.pi,
    }

    def __init__(self, fly: NeuroMechFly, motor_anat: VNCMotorAnatomy):
        self.fly = fly
        self.anat = motor_anat
        self.act_order = fly.get_actuated_jointdofs_order(ActuatorType.POSITION)
        pose_lut = fly.get_pose_lookup(KinematicPosePreset.NEUTRAL)
        self.neutral_q = np.array([pose_lut.get(d.name, 0.0) for d in self.act_order], dtype=np.float64)
        self.dof_idx = {d.name: i for i, d in enumerate(self.act_order)}
        self.legs_order = [l.name if hasattr(l, "name") else str(l) for l in fly.get_legs_order()]
        self.phase_rad = 0.0
        self.filtered_turn_asym = 0.0
        self.filtered_fwd_drive = 0.0

    def reset(self):
        self.phase_rad = 0.0
        self.filtered_turn_asym = 0.0
        self.filtered_fwd_drive = 0.0

    def step_substep(
        self,
        dt_s: float,
        sim_time_s: float,
        spiking_out: Dict,
        intact_fraction: float,
        arrest_gate: float = 0.0,
        wing_ext_side: Optional[str] = None,
        wing_song_drive: float = 0.0,
        abdomen_bend: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Computes (joint_targets_63, adhesion_6, diagnostics) for one 0.1 ms MuJoCo substep.
        Strictly requires live neural spiking activity (silent network -> 0.0 forward drive).
        """
        if intact_fraction <= 1e-6:
            return self.neutral_q.copy(), np.ones(6, dtype=np.float64), {
                "fwd_drive": 0.0, "turn_asym": 0.0, "cadence_hz": 0.0
            }

        p_hz = spiking_out.get("pool_consensus_hz", {})
        div_intact = max(1e-6, float(intact_fraction))
        mean_brain_hz = float(np.mean(spiking_out["mean_brain_hz"])) / div_intact
        leg_L_hz = float(np.mean(spiking_out["leg_L_hz"])) / div_intact
        leg_R_hz = float(np.mean(spiking_out["leg_R_hz"])) / div_intact
        dn_total_hz = 0.5 * float(np.mean(spiking_out["dn_L_hz"]) + np.mean(spiking_out["dn_R_hz"])) / div_intact

        # Bilateral VNC motor + descending locomotor recruitment
        bilateral_motor_hz = 0.5 * (leg_L_hz + leg_R_hz)
        if mean_brain_hz < 0.02 and bilateral_motor_hz < 0.1:
            raw_fwd = 0.0
        else:
            raw_fwd = np.clip(
                0.78 + 0.012 * bilateral_motor_hz + 0.004 * dn_total_hz,
                0.0,
                1.42,
            ) * (1.0 - np.clip(arrest_gate, 0.0, 1.0))

        self.filtered_fwd_drive = 0.96 * self.filtered_fwd_drive + 0.04 * float(raw_fwd)

        # Steering asymmetry from DNa/DNb + 6-leg VNC motor neuron bilateral differential
        cons_turn_hz = float(spiking_out["consensus_turn_hz"])
        vnc_leg_diff = (
            (p_hz.get("leg_rf", 0.0) + p_hz.get("leg_rm", 0.0) + p_hz.get("leg_rh", 0.0))
            - (p_hz.get("leg_lf", 0.0) + p_hz.get("leg_lm", 0.0) + p_hz.get("leg_lh", 0.0))
        ) / 3.0
        eff_turn_hz = cons_turn_hz + 0.25 * vnc_leg_diff
        # Calibrated MuJoCo sign: negative turn_asym rotates NeuroMechFly +yaw (toward +dy)
        target_asym = -float(np.clip(eff_turn_hz / 38.0, -0.44, 0.44))
        self.filtered_turn_asym = 0.94 * self.filtered_turn_asym + 0.06 * target_asym

        cadence_hz = 13.5 * self.filtered_fwd_drive if self.filtered_fwd_drive > 0.08 else 0.0
        self.phase_rad = (self.phase_rad + 2.0 * math.pi * cadence_hz * dt_s) % (2.0 * math.pi)

        q = self.neutral_q.copy()
        adh = np.ones(6, dtype=np.float64)
        idx = self.dof_idx

        ltm_L_gain = float(np.clip(0.72 + 0.28 * (p_hz.get("ltm_L", 10.0) / 25.0), 0.55, 1.0))
        ltm_R_gain = float(np.clip(0.72 + 0.28 * (p_hz.get("ltm_R", 10.0) / 25.0), 0.55, 1.0))

        if self.filtered_fwd_drive > 0.08:
            stroke_base = 1.55 * self.filtered_fwd_drive
            ta = self.filtered_turn_asym
            for li, leg in enumerate(self.legs_order):
                leg_mod = 1.0 + 0.004 * min(40.0, p_hz.get(f"leg_{leg}", 0.0))
                side_scale = stroke_base * leg_mod * ((1.0 - ta) if leg.startswith("l") else (1.0 + ta))
                ph = (self.phase_rad + self.TRIPOD_PHASE_OFFSET[leg]) % (2.0 * math.pi)
                in_swing = ph < math.pi
                swing_lift = math.sin(ph) if in_swing else 0.0
                stroke = -math.cos(ph)

                adh[li] = 0.0 if in_swing else (ltm_L_gain if leg.startswith("l") else ltm_R_gain)

                if leg.endswith("f"):
                    q[idx[f"c_thorax-{leg}_coxa-pitch"]] += side_scale * (-0.24) * stroke
                    q[idx[f"c_thorax-{leg}_coxa-yaw"]] += side_scale * (+0.14) * stroke
                    q[idx[f"{leg}_coxa-{leg}_trochanterfemur-pitch"]] += 0.36 * swing_lift + side_scale * 0.14 * stroke
                    q[idx[f"{leg}_trochanterfemur-{leg}_tibia-pitch"]] += side_scale * (+0.38 * stroke) - 0.22 * swing_lift
                elif leg.endswith("m"):
                    q[idx[f"c_thorax-{leg}_coxa-yaw"]] += side_scale * (+0.30) * stroke
                    q[idx[f"c_thorax-{leg}_coxa-pitch"]] += side_scale * (-0.18) * stroke
                    q[idx[f"{leg}_coxa-{leg}_trochanterfemur-pitch"]] += 0.38 * swing_lift
                    q[idx[f"{leg}_trochanterfemur-{leg}_tibia-pitch"]] += -0.20 * swing_lift
                else:
                    q[idx[f"c_thorax-{leg}_coxa-pitch"]] += side_scale * (-0.24) * stroke
                    q[idx[f"c_thorax-{leg}_coxa-yaw"]] += side_scale * (+0.22) * stroke
                    q[idx[f"{leg}_coxa-{leg}_trochanterfemur-pitch"]] += 0.36 * swing_lift - side_scale * 0.16 * stroke
                    q[idx[f"{leg}_trochanterfemur-{leg}_tibia-pitch"]] += side_scale * (-0.38 * stroke) - 0.20 * swing_lift

        # Active 3D head/neck yaw gaze stabilization into the turn
        if "c_thorax-c_head-yaw" in idx:
            neck_diff = 0.008 * (p_hz.get("neck_R", 0.0) - p_hz.get("neck_L", 0.0))
            q[idx["c_thorax-c_head-yaw"]] += float(np.clip(-0.55 * self.filtered_turn_asym + neck_diff, -0.45, 0.45))

        # Anti-phase haltere gyroscopic oscillation during active walking
        if "c_thorax-l_haltere-pitch" in idx and self.filtered_fwd_drive > 0.08:
            halt_osc = 0.28 * math.sin(self.phase_rad * 2.0)
            q[idx["c_thorax-l_haltere-pitch"]] += halt_osc
            q[idx["c_thorax-r_haltere-pitch"]] -= halt_osc

        # 3D Unilateral Courtship Wing Extension & Pulse Song (pIP10 -> wing_power / wing_steer MNs)
        if wing_ext_side in ("L", "R") and wing_song_drive > 0.05:
            amp = float(np.clip(wing_song_drive, 0.0, 1.0))
            vib = 0.09 * amp * math.sin(2.0 * math.pi * 185.0 * sim_time_s)
            if wing_ext_side == "L" and "c_thorax-l_wing-yaw" in idx:
                q[idx["c_thorax-l_wing-yaw"]] += 1.02 * amp + vib
                q[idx["c_thorax-l_wing-roll"]] += 0.32 * amp
            elif wing_ext_side == "R" and "c_thorax-r_wing-yaw" in idx:
                q[idx["c_thorax-r_wing-yaw"]] += 1.02 * amp + vib
                q[idx["c_thorax-r_wing-roll"]] += 0.32 * amp

        # 3D Abdominal Posture (MNad / EFFabg recruitment)
        if "c_thorax-c_abdomen12-pitch" in idx and abs(abdomen_bend) > 0.01:
            q[idx["c_thorax-c_abdomen12-pitch"]] += float(np.clip(abdomen_bend * 0.35, -0.5, 0.5))
            if "c_abdomen12-c_abdomen3-pitch" in idx:
                q[idx["c_abdomen12-c_abdomen3-pitch"]] += float(np.clip(abdomen_bend * 0.25, -0.4, 0.4))

        return q, adh, {
            "fwd_drive": round(self.filtered_fwd_drive, 4),
            "turn_asym": round(self.filtered_turn_asym, 4),
            "cadence_hz": round(cadence_hz, 2),
        }


class EmbodiedNeuroMechFly3DArena:
    """
    Closed-loop 3D MuJoCo biomechanical arena coupling MaleCNS v1.0 & Female BANC v888
    spiking Whole-CNS ensembles directly onto articulated 3D NeuroMechFly v2 bodies.
    """

    @staticmethod
    def run_single_fly_3d_episode(
        eng: WholeCNSEngine,
        target_xy: Tuple[float, float] = (5.8, 2.8),
        duration_s: float = 0.48,
        frame_ms: int = 8,
        bg_noise_hz: float = 22.0,
        reset_state_before: bool = True,
        capture_camera: bool = False,
        camera_res: Tuple[int, int] = (480, 640),
    ) -> Dict:
        """
        Closed-loop 3D MuJoCo walking & compass target tracking driven by real spiking
        PFL3 -> DNa02 -> 805/815 VNC motor neurons -> 63-DOF 3D NeuroMechFly v2.
        """
        if reset_state_before:
            eng.reset_state()

        anat = extract_vnc_motor_pools(eng.ds)
        fly, cam = build_neuromechfly_3d_body(
            name="fly",
            add_camera=capture_camera,
            cam_offset=(-2.4, -6.0, 4.4),
            cam_fovy=36.0,
        )
        ctrl = NeuroMechFly3DController(fly, anat)

        world = FlatGroundWorld()
        world.add_fly(fly, spawn_position=(0.0, 0.0, 0.70), spawn_rotation=Rotation3D("quat", (1.0, 0.0, 0.0, 0.0)))
        sim = Simulation(world)
        renderer = None
        if capture_camera and cam is not None:
            renderer = sim.set_renderer(cameras=cam, camera_res=camera_res, playback_speed=1.0, output_fps=25)
        sim.warmup(duration_s=0.02)

        substeps_per_frame = max(1, int(round((frame_ms * 1e-3) / sim.timestep)))
        n_frames = max(1, int(round(duration_s / (frame_ms * 1e-3))))
        bg_sub = eng.ds.plastic_neurons[::6]

        p0 = sim.get_body_positions("fly")[0].copy()
        tx, ty = float(p0[0] + target_xy[0]), float(p0[1] + target_xy[1])
        intact_frac = float(eng.active_brain_mask.mean())

        traj: List[Dict] = []
        turn_snr_samples: List[float] = []
        roll_samples: List[float] = []
        pitch_samples: List[float] = []
        contact_force_samples: List[float] = []

        for f_idx in range(n_frames):
            pos = sim.get_body_positions("fly")[0]
            quat = sim.get_body_rotations("fly")[0]
            yaw, pitch, roll = _euler_from_quat_wxyz(quat)

            dx = tx - float(pos[0])
            dy = ty - float(pos[1])
            dist_to_target = math.hypot(dx, dy)
            ang_err = _wrap_angle(math.atan2(dy, dx) - yaw)

            stim = [(eng.ds.idx_walk_tonic, 96.0)]
            if ang_err > 0.03:
                stim.append((eng.ds.idx_pfl3_turnR, min(305.0, 55.0 + 235.0 * abs(ang_err))))
            elif ang_err < -0.03:
                stim.append((eng.ds.idx_pfl3_turnL, min(305.0, 55.0 + 235.0 * abs(ang_err))))

            res = eng.step_population(
                n_steps_ms=frame_ms,
                shared_stimuli=stim,
                bg_poisson_hz=bg_noise_hz,
                bg_subset_indices=bg_sub,
                readout_pools=anat.readout_pools,
            )

            intact_vals = res["turn_hz_per_brain"][eng.active_brain_mask]
            if len(intact_vals) > 1 and abs(ang_err) > 0.05:
                mu = abs(float(np.mean(intact_vals)))
                sd = float(np.std(intact_vals)) + 1e-4
                turn_snr_samples.append(mu / sd)
            elif len(intact_vals) == 1 and abs(ang_err) > 0.05:
                # Single-brain temporal jitter across frame substeps
                ts = res["step_turn_series"][:, 0]
                turn_snr_samples.append(abs(float(np.mean(ts))) / (float(np.std(ts)) + 1.0))

            diag = {"fwd_drive": 0.0, "turn_asym": 0.0, "cadence_hz": 0.0}
            adh_last = np.ones(6)
            for _ in range(substeps_per_frame):
                q_cmd, adh_cmd, diag = ctrl.step_substep(
                    dt_s=sim.timestep,
                    sim_time_s=sim.time,
                    spiking_out=res,
                    intact_fraction=intact_frac,
                )
                adh_last = adh_cmd
                sim.set_actuator_inputs("fly", ActuatorType.POSITION, q_cmd)
                sim.set_leg_adhesion_states("fly", adh_cmd)
                sim.step()
                if renderer is not None:
                    sim.render_as_needed()

            pos_after = sim.get_body_positions("fly")[0]
            quat_after = sim.get_body_rotations("fly")[0]
            yaw_a, pitch_a, roll_a = _euler_from_quat_wxyz(quat_after)
            c_info = sim.get_ground_contact_info("fly")
            total_normal_force = float(np.linalg.norm(c_info[1], axis=1).sum())

            roll_samples.append(abs(math.degrees(roll_a)))
            pitch_samples.append(abs(math.degrees(pitch_a)))
            contact_force_samples.append(total_normal_force)

            traj.append({
                "t_s": round((f_idx + 1) * frame_ms * 1e-3, 4),
                "x_mm": round(float(pos_after[0] - p0[0]), 4),
                "y_mm": round(float(pos_after[1] - p0[1]), 4),
                "z_mm": round(float(pos_after[2]), 4),
                "yaw_deg": round(math.degrees(yaw_a), 2),
                "roll_deg": round(math.degrees(roll_a), 2),
                "pitch_deg": round(math.degrees(pitch_a), 2),
                "dist_to_target_mm": round(math.hypot(tx - float(pos_after[0]), ty - float(pos_after[1])), 4),
                "turn_hz": round(float(res["consensus_turn_hz"]), 2),
                "leg_L_hz": round(float(np.mean(res["leg_L_hz"])), 2),
                "leg_R_hz": round(float(np.mean(res["leg_R_hz"])), 2),
                "ground_force_mN": round(total_normal_force, 4),
                "adhesion_active_legs": int((adh_last > 0.5).sum()),
                "cadence_hz": diag["cadence_hz"],
            })

        p_end = sim.get_body_positions("fly")[0].copy()
        path_len = sum(
            math.hypot(traj[i]["x_mm"] - traj[i - 1]["x_mm"], traj[i]["y_mm"] - traj[i - 1]["y_mm"])
            for i in range(1, len(traj))
        ) if len(traj) > 1 else math.hypot(float(p_end[0] - p0[0]), float(p_end[1] - p0[1]))
        mean_speed = path_len / max(1e-6, duration_s)
        final_dist = math.hypot(tx - float(p_end[0]), ty - float(p_end[1]))
        init_dist = math.hypot(target_xy[0], target_xy[1])

        rgb_frame = None
        if renderer is not None:
            frames_list = list(renderer.frames.values())[0] if renderer.frames else []
            if len(frames_list) > 0:
                rgb_frame = frames_list[-1].copy()
            renderer.close()
        sim.close()

        return {
            "K": eng.K,
            "intact_fraction": round(intact_frac, 3),
            "duration_s": round(duration_s, 3),
            "mujoco_substeps": int(n_frames * substeps_per_frame),
            "mean_speed_mm_s": round(mean_speed, 3),
            "net_displacement_mm": round(math.hypot(float(p_end[0] - p0[0]), float(p_end[1] - p0[1])), 3),
            "initial_target_dist_mm": round(init_dist, 3),
            "final_target_dist_mm": round(final_dist, 3),
            "target_progress_pct": round(100.0 * max(0.0, init_dist - final_dist) / max(1e-6, init_dist), 2),
            "thorax_height_z_mm": round(float(p_end[2]), 3),
            "roll_rms_deg": round(float(np.sqrt(np.mean(np.square(roll_samples)))), 3),
            "pitch_rms_deg": round(float(np.sqrt(np.mean(np.square(pitch_samples)))), 3),
            "mean_ground_force": round(float(np.mean(contact_force_samples)), 4),
            "motor_turn_snr": round(float(np.mean(turn_snr_samples)) if turn_snr_samples else 1.0, 3),
            "motor_anatomy": anat.summary(),
            "trajectory": traj,
            "rgb_frame": rgb_frame,
        }

    @staticmethod
    def run_dual_fly_3d_encounter(
        eng_male: WholeCNSEngine,
        eng_female: WholeCNSEngine,
        receptive_virgin: bool = True,
        duration_s: float = 0.56,
        frame_ms: int = 8,
        bg_noise_hz: float = 22.0,
        capture_camera: bool = True,
        camera_res: Tuple[int, int] = (480, 640),
    ) -> Dict:
        """
        Simultaneous 2-Fly 3D NeuroMechFly v2 MuJoCo Encounter inside a single shared FlatGroundWorld:
          - Male (MaleCNS v1.0, 815 VNC MNs) + Female (BANC v888, 805 VNC MNs)
          - Real 3D rigid-body ground-contact locomotion, flanking pursuit, unilateral 3D wing extension,
            and Female JO-B -> vpoDN copulatory arrest (virgin) vs DNp13 3D evasion (mated).
        """
        eng_male.reset_state()
        eng_female.reset_state()

        anat_m = extract_vnc_motor_pools(eng_male.ds)
        anat_f = extract_vnc_motor_pools(eng_female.ds)
        m_soc = extract_male_social_pools(eng_male.ds)
        f_soc = extract_female_social_pools(eng_female.ds)

        m_readout = dict(anat_m.readout_pools)
        m_readout["p1"] = m_soc["p1"]
        m_readout["pip10_song"] = m_soc["pip10_song"]

        f_readout = dict(anat_f.readout_pools)
        f_readout["vpodn"] = f_soc["vpodn"]
        f_readout["dnp13"] = f_soc["dnp13"]
        f_readout["jo_b"] = f_soc["jo_b"]

        fly_m, cam_m = build_neuromechfly_3d_body(
            name="male",
            add_camera=capture_camera,
            cam_offset=(2.0, -7.4, 5.0),
            cam_fovy=40.0,
        )
        fly_f, _ = build_neuromechfly_3d_body(name="female", add_camera=False)

        ctrl_m = NeuroMechFly3DController(fly_m, anat_m)
        ctrl_f = NeuroMechFly3DController(fly_f, anat_f)

        world = FlatGroundWorld()
        world.add_fly(fly_m, spawn_position=(0.0, 0.0, 0.70), spawn_rotation=Rotation3D("quat", _quat_from_yaw(0.18)))
        world.add_fly(fly_f, spawn_position=(4.8, 1.4, 0.70), spawn_rotation=Rotation3D("quat", _quat_from_yaw(0.08)))

        sim = Simulation(world)
        renderer = None
        if capture_camera and cam_m is not None:
            renderer = sim.set_renderer(cameras=cam_m, camera_res=camera_res, playback_speed=1.0, output_fps=25)
        sim.warmup(duration_s=0.02)

        substeps_per_frame = max(1, int(round((frame_ms * 1e-3) / sim.timestep)))
        n_frames = max(1, int(round(duration_s / (frame_ms * 1e-3))))
        bg_m = eng_male.ds.plastic_neurons[::6]
        bg_f = eng_female.ds.plastic_neurons[::5]

        p1_state = 0.0
        song_accum = 0.0
        traj: List[Dict] = []
        keyframes_rgb: List[np.ndarray] = []

        for f_idx in range(n_frames):
            pm = sim.get_body_positions("male")[0]
            pf = sim.get_body_positions("female")[0]
            qm = sim.get_body_rotations("male")[0]
            qf = sim.get_body_rotations("female")[0]
            ym, _, rm = _euler_from_quat_wxyz(qm)
            yf, _, rf = _euler_from_quat_wxyz(qf)

            dx = float(pf[0] - pm[0])
            dy = float(pf[1] - pm[1])
            dist_3d = math.hypot(dx, dy)
            bearing_f_from_m = _wrap_angle(math.atan2(dy, dx) - ym)

            # Male sensory & courtship P1 -> pIP10 -> 815 VNC motor drive
            stim_m = [(eng_male.ds.idx_walk_tonic, 96.0)]
            pher = 1.0 / (1.0 + (dist_3d / 8.0) ** 2)
            stim_m.append((m_soc["va1v"], 235.0 * pher))
            if not receptive_virgin and dist_3d < 7.5:
                stim_m.append((m_soc["da1"], 220.0 * pher))
                stim_m.append((m_soc["mal_m"], 195.0 * pher))

            if bearing_f_from_m >= 0.0:
                stim_m.append((m_soc["lc10a_R"], 255.0))
                stim_m.append((eng_male.ds.idx_pfl3_turnR, min(305.0, 55.0 + 235.0 * abs(bearing_f_from_m))))
            else:
                stim_m.append((m_soc["lc10a_L"], 255.0))
                stim_m.append((eng_male.ds.idx_pfl3_turnL, min(305.0, 55.0 + 235.0 * abs(bearing_f_from_m))))

            p1_state = float(np.clip(0.84 * p1_state + 0.36 * pher, 0.0, 1.2))
            stim_m.append((m_soc["p1"], 120.0 + 220.0 * p1_state, 22.0))
            if dist_3d < 9.0:
                stim_m.append((m_soc["pip10"], 165.0 * p1_state, 21.5))

            res_m = eng_male.step_population(
                n_steps_ms=frame_ms,
                shared_stimuli=stim_m,
                bg_poisson_hz=bg_noise_hz,
                bg_subset_indices=bg_m,
                readout_pools=m_readout,
            )
            pip10_hz = float(res_m["pool_consensus_hz"].get("pip10_song", 0.0))
            wing_song_drive = float(np.clip(pip10_hz / 28.0 + 0.45 * p1_state, 0.0, 1.0)) if dist_3d < 9.5 else 0.0
            wing_side = "L" if bearing_f_from_m >= 0.0 else "R"

            # Female sensory & JO-B -> CB1484 -> vpoEN/vpoDN vs DNp13 -> 805 VNC motor drive
            stim_f = [(eng_female.ds.idx_walk_tonic, 88.0)]
            if wing_song_drive > 0.20:
                song_db = 240.0 * wing_song_drive / (1.0 + (dist_3d / 7.5) ** 2)
                stim_f.append((f_soc["jo_b"], song_db, 21.0))
                stim_f.append((f_soc["cb1484"], 0.85 * song_db, 20.5))
                song_accum = float(np.clip(0.92 * song_accum + 0.14 * wing_song_drive, 0.0, 1.2))

            if receptive_virgin:
                stim_f.append((f_soc["sag"], 195.0, 21.0))
                if song_accum > 0.25:
                    stim_f.append((f_soc["pc1_abc"], 185.0 * song_accum, 21.0))
                    stim_f.append((f_soc["vpoen"], 190.0 * song_accum, 21.0))
            else:
                stim_f.append((f_soc["pc1d"], 175.0, 20.5))
                stim_f.append((f_soc["pc2l_pre"], 195.0, 21.0))
                stim_f.append((f_soc["dnp13"], 155.0, 20.5))
                # Mated female steers away in 3D
                away_err = _wrap_angle(math.atan2(dy, dx) + 0.35 - yf)
                if away_err >= 0.0:
                    stim_f.append((eng_female.ds.idx_pfl3_turnR, 210.0))
                else:
                    stim_f.append((eng_female.ds.idx_pfl3_turnL, 210.0))

            res_f = eng_female.step_population(
                n_steps_ms=frame_ms,
                shared_stimuli=stim_f,
                bg_poisson_hz=bg_noise_hz,
                bg_subset_indices=bg_f,
                readout_pools=f_readout,
            )

            vpodn_hz = float(res_f["pool_consensus_hz"].get("vpodn", 0.0))
            dnp13_hz = float(res_f["pool_consensus_hz"].get("dnp13", 0.0))
            f_arrest = float(np.clip((vpodn_hz - 4.0) / 22.0 + (0.65 * song_accum if receptive_virgin else 0.0), 0.0, 0.96))
            m_arrest = 0.45 if dist_3d < 2.3 else 0.0

            for _ in range(substeps_per_frame):
                qm_cmd, adhm, _ = ctrl_m.step_substep(
                    dt_s=sim.timestep,
                    sim_time_s=sim.time,
                    spiking_out=res_m,
                    intact_fraction=1.0,
                    arrest_gate=m_arrest,
                    wing_ext_side=wing_side,
                    wing_song_drive=wing_song_drive,
                    abdomen_bend=0.35 * p1_state,
                )
                qf_cmd, adhf, _ = ctrl_f.step_substep(
                    dt_s=sim.timestep,
                    sim_time_s=sim.time,
                    spiking_out=res_f,
                    intact_fraction=1.0,
                    arrest_gate=f_arrest,
                    abdomen_bend=(-0.55 if not receptive_virgin else 0.20 * f_arrest),
                )
                sim.set_actuator_inputs("male", ActuatorType.POSITION, qm_cmd)
                sim.set_leg_adhesion_states("male", adhm)
                sim.set_actuator_inputs("female", ActuatorType.POSITION, qf_cmd)
                sim.set_leg_adhesion_states("female", adhf)
                sim.step()
                if renderer is not None:
                    sim.render_as_needed()

            pm2 = sim.get_body_positions("male")[0]
            pf2 = sim.get_body_positions("female")[0]
            ym2, _, rm2 = _euler_from_quat_wxyz(sim.get_body_rotations("male")[0])
            yf2, _, rf2 = _euler_from_quat_wxyz(sim.get_body_rotations("female")[0])

            traj.append({
                "t_s": round((f_idx + 1) * frame_ms * 1e-3, 4),
                "mx": round(float(pm2[0]), 4),
                "my": round(float(pm2[1]), 4),
                "mz": round(float(pm2[2]), 4),
                "myaw_deg": round(math.degrees(ym2), 2),
                "mroll_deg": round(math.degrees(rm2), 2),
                "fx": round(float(pf2[0]), 4),
                "fy": round(float(pf2[1]), 4),
                "fz": round(float(pf2[2]), 4),
                "fyaw_deg": round(math.degrees(yf2), 2),
                "froll_deg": round(math.degrees(rf2), 2),
                "dist_mm": round(math.hypot(float(pf2[0] - pm2[0]), float(pf2[1] - pm2[1])), 4),
                "wing_ext_side": wing_side if wing_song_drive > 0.2 else "NONE",
                "wing_song_drive": round(wing_song_drive, 3),
                "pip10_hz": round(pip10_hz, 2),
                "vpodn_hz": round(vpodn_hz, 2),
                "dnp13_hz": round(dnp13_hz, 2),
                "female_arrest_gate": round(f_arrest, 3),
            })

        if renderer is not None:
            flist = list(renderer.frames.values())[0] if renderer.frames else []
            if len(flist) >= 3:
                keyframes_rgb = [flist[0].copy(), flist[len(flist) // 2].copy(), flist[-1].copy()]
            elif len(flist) > 0:
                keyframes_rgb = [flist[-1].copy()]
            renderer.close()
        sim.close()

        f_disp = math.hypot(traj[-1]["fx"] - 4.8, traj[-1]["fy"] - 1.4)
        m_disp = math.hypot(traj[-1]["mx"] - 0.0, traj[-1]["my"] - 0.0)
        return {
            "K_male": eng_male.K,
            "K_female": eng_female.K,
            "receptive_virgin": receptive_virgin,
            "duration_s": round(duration_s, 3),
            "mujoco_substeps": int(n_frames * substeps_per_frame),
            "male_3d_disp_mm": round(m_disp, 3),
            "female_3d_disp_mm": round(f_disp, 3),
            "female_mean_speed_mm_s": round(f_disp / max(1e-6, duration_s), 3),
            "final_interfly_dist_mm": traj[-1]["dist_mm"],
            "peak_wing_song_drive": round(max(r["wing_song_drive"] for r in traj), 3),
            "peak_vpodn_hz": round(max(r["vpodn_hz"] for r in traj), 2),
            "peak_dnp13_hz": round(max(r["dnp13_hz"] for r in traj), 2),
            "final_female_arrest": traj[-1]["female_arrest_gate"],
            "trajectory": traj,
            "keyframes_rgb": keyframes_rgb,
        }
