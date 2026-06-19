#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reference-only FastDiagonalTrot + light VMC v4 bring-up node for Fanfan.

This node is intentionally not an RL policy runner.  It generates a conservative
CPG/VMC position target, converts policy/URDF joint order to the existing real
motor order through JointSemanticMapper, and sends position targets through the
same HTTP batch endpoint used by the existing Fanfan nodes.
"""

import csv
import math
import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from .imu_serial_interface import ImuSerialInterface
from .motor_state_interface import MotorSnapshot, MotorStateHttpInterface
from .semantic_mapper import JointSemanticMapper


LEG_ORDER = ("FR", "FL", "RR", "RL")
REAR_LEGS = ("RR", "RL")
LEG_START = {"FR": 0, "FL": 3, "RR": 6, "RL": 9}
JOINT_SUFFIX = ("hip", "thigh", "calf")
JOINT_NAMES = tuple(f"{leg}_{joint}" for leg in LEG_ORDER for joint in JOINT_SUFFIX)
HIP_OUTWARD_SIGNS = {"FR": -1.0, "FL": 1.0, "RR": -1.0, "RL": 1.0}
PAIR_A = ("FR", "RL")
PAIR_B = ("FL", "RR")
PAIR_OFFSETS = {"FR": 0.0, "RL": 0.0, "FL": 0.5, "RR": 0.5}
REAL_TEST_MODES = ("air", "touch", "assist", "short_free")


# Fallback only.  The node prefers JointSemanticMapper.default_joint_angle so it
# stays aligned with the existing real-machine mapper and motor order.
FALLBACK_DEFAULT_STAND_POLICY = np.array(
    [
        -0.1047, 0.4363, -0.8727,  # FR
        0.1047, 0.7000, 1.1500,  # FL
        0.1571, -0.5934, 1.7628,  # RR
        -0.1571, 0.5934, -1.7628,  # RL
    ],
    dtype=np.float32,
)


def clamp(value: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, value))


def smootherstep(s: float) -> float:
    s = clamp(float(s), 0.0, 1.0)
    return s * s * s * (s * (s * 6.0 - 15.0) + 10.0)


def wrap_phase(x: float) -> float:
    return float(x - math.floor(x))


def wrap_to_pi(x: float) -> float:
    return float((x + math.pi) % (2.0 * math.pi) - math.pi)


def pctl(values: List[float], percentile: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def mode_scale(mode: str, air: float, touch: float, assist: float, short_free: float) -> float:
    return {
        "air": air,
        "touch": touch,
        "assist": assist,
        "short_free": short_free,
    }[mode]


@dataclass
class Feedback:
    q_policy: np.ndarray
    dq_policy: np.ndarray
    torque_policy: np.ndarray
    current_policy: np.ndarray
    temp_policy: np.ndarray
    online: np.ndarray
    max_age_ms: float
    valid: bool
    source: str


class FanfanCpgVmcV4Node(Node):
    def __init__(self):
        super().__init__("fanfan_cpg_vmc_v4_node")

        self._declare_parameters()
        self.mapper = JointSemanticMapper()
        self.motor_ids = self.mapper.get_real_motor_ids()
        self.real_joint_names = list(self.mapper.real_joint_names)
        self.policy_joint_names = self.mapper.get_policy_joint_names()
        self.default_policy = self._select_default_stand()
        self.default_real = self.mapper.policy_target_to_real_target(self.default_policy, clamp=True)

        self.http_session = requests.Session()
        self.motor = MotorStateHttpInterface(
            base_url=self.motor_base_url,
            timeout=self.http_timeout,
            stale_recheck_ms=self.max_motor_age_ms,
            enable_stale_recheck=True,
        )
        self.imu: Optional[ImuSerialInterface] = None
        self.imu_valid = False
        self.motor_feedback_valid = False
        self._last_feedback_warn = 0.0
        self._last_send_info = 0.0
        self._stop_requested = False
        self._soft_stop_active = False
        self._soft_stop_start = 0.0
        self._soft_stop_from = self.default_policy.copy()
        self.node_state = "WARMUP"
        self.stop_reason = "none"
        self.safety_stop = False
        self.safety_stop_reason = "none"
        self.target_yaw = 0.0

        self.q_last_cmd = self.default_policy.copy()
        self.q_cmd_final = self.default_policy.copy()
        self.yaw_hip_offset = np.zeros(4, dtype=np.float32)
        self.rear_late_clearance = np.zeros(4, dtype=np.float32)

        self.phase = 0.0
        self.start_time = time.time()
        self._last_update_time = self.start_time
        self._run_started = False

        self.stats: Dict[str, List[float]] = {
            "roll_abs": [],
            "pitch_abs": [],
            "yaw_abs": [],
            "yaw_error_abs": [],
            "q_error_max": [],
            "tau_max": [],
            "current_max": [],
            "temp_max": [],
            "rate_clip": [],
            "torque_clip": [],
            "rr_early_guard": [],
            "rl_early_guard": [],
            "rr_late_guard": [],
            "rl_late_guard": [],
            "rear_late_qerr": [],
            "rear_late_tau": [],
        }

        self.pub_target = self.create_publisher(Float32MultiArray, "/mydog/fanfan_cpg_vmc_v4_target_real", 10)
        self.pub_debug = self.create_publisher(Float32MultiArray, "/mydog/fanfan_cpg_vmc_v4_debug", 10)

        self._setup_csv()
        self._start_imu_if_needed()
        self._validate_send_preconditions()
        self._print_startup_summary()
        if self.enable_send:
            self._countdown()
            self._send_default_stand()
        else:
            self.get_logger().warn("enable_send=False: dry-run only, no motor commands will be sent.")

        period = 1.0 / max(self.gait_hz, 1.0)
        self.timer = self.create_timer(period, self.update)

    def _declare_parameters(self):
        self.declare_parameter("motor_base_url", "http://127.0.0.1:8000")
        self.declare_parameter("enable_send", False)
        self.declare_parameter("test_mode", "air")
        self.declare_parameter("duration_s", 5.0)
        self.declare_parameter("warmup_s", 2.0)
        self.declare_parameter("soft_start_s", 2.0)
        self.declare_parameter("soft_stop_s", 1.5)
        self.declare_parameter("auto_stop_after_duration", True)
        self.declare_parameter("allow_long_free_test", False)
        self.declare_parameter("gait_hz", 60.0)
        self.declare_parameter("http_timeout", 0.08)
        self.declare_parameter("max_motor_age_ms", 150.0)
        self.declare_parameter("csv_path", "")
        self.declare_parameter("stand_source", "mapper")
        self.declare_parameter("dry_run_virtual_feedback", True)
        self.declare_parameter("require_stand_ready", True)
        self.declare_parameter("stand_ready_q_error_threshold", 0.35)
        self.declare_parameter("allow_start_from_any_pose", False)

        self.declare_parameter("step_hz", 1.15)
        self.declare_parameter("duty_factor", 0.61)
        self.declare_parameter("stride_length", 0.022)
        self.declare_parameter("front_swing_height", 0.048)
        self.declare_parameter("rear_swing_height", 0.067)
        self.declare_parameter("swing_lift_peak_phase", 0.45)
        self.declare_parameter("touchdown_phase", 0.82)
        self.declare_parameter("early_stance_blend", 0.12)
        self.declare_parameter("thigh_length", 0.1560608)
        self.declare_parameter("calf_length", 0.1489418)

        self.declare_parameter("real_stride_scale_air", 0.70)
        self.declare_parameter("real_stride_scale_touch", 0.55)
        self.declare_parameter("real_stride_scale_assist", 0.65)
        self.declare_parameter("real_stride_scale_short_free", 0.70)
        self.declare_parameter("real_swing_height_scale_air", 0.85)
        self.declare_parameter("real_swing_height_scale_touch", 0.75)
        self.declare_parameter("real_swing_height_scale_assist", 0.80)
        self.declare_parameter("real_swing_height_scale_short_free", 0.85)
        self.declare_parameter("real_vmc_scale_air", 0.0)
        self.declare_parameter("real_vmc_scale_touch", 0.35)
        self.declare_parameter("real_vmc_scale_assist", 0.50)
        self.declare_parameter("real_vmc_scale_short_free", 0.60)

        self.declare_parameter("target_base_height", 0.288)
        self.declare_parameter("target_pitch", -0.04)
        self.declare_parameter("height_sign", -1.0)
        self.declare_parameter("height_kp_z", 0.30)
        self.declare_parameter("height_kd_z", 0.04)
        self.declare_parameter("height_corr_limit_m", 0.004)
        self.declare_parameter("roll_sign", 1.0)
        self.declare_parameter("roll_kp_z", 0.025)
        self.declare_parameter("roll_kd_z", 0.006)
        self.declare_parameter("roll_corr_limit_m", 0.0035)
        self.declare_parameter("pitch_sign", -1.0)
        self.declare_parameter("pitch_kp_z", 0.025)
        self.declare_parameter("pitch_kd_z", 0.005)
        self.declare_parameter("pitch_corr_limit_m", 0.003)
        self.declare_parameter("enable_light_yaw_damping", True)
        self.declare_parameter("yaw_sign", 1.0)
        self.declare_parameter("yaw_kp_hip", 0.0025)
        self.declare_parameter("yaw_kd_hip", 0.006)
        self.declare_parameter("yaw_hip_limit_rad", 0.007)
        self.declare_parameter("yaw_hip_rate_limit_rad", 0.001)

        self.declare_parameter("rear_late_swing_guard_enable", True)
        self.declare_parameter("rear_late_swing_phase_start", 0.28)
        self.declare_parameter("rear_late_swing_phase_end", 0.38)
        self.declare_parameter("rear_late_swing_progress_start", 0.70)
        self.declare_parameter("rear_late_swing_progress_end", 0.95)
        self.declare_parameter("rear_touchdown_progress_start", 0.85)
        self.declare_parameter("rear_late_swing_clearance_margin_m", 0.003)
        self.declare_parameter("rear_late_swing_min_height_m", 0.003)
        self.declare_parameter("rear_late_swing_guard_rate_limit_m", 0.001)
        self.declare_parameter("rear_late_swing_clearance_sign", 1.0)
        self.declare_parameter("rear_late_swing_descent_soft_enable", True)
        self.declare_parameter("rear_late_swing_descent_scale", 0.50)
        self.declare_parameter("rear_early_contact_guard_enable", True)
        self.declare_parameter("rear_early_contact_force_threshold", 10.0)
        self.declare_parameter("rear_early_contact_qerr_threshold", 0.18)
        self.declare_parameter("rear_early_contact_tau_threshold_nm", 8.0)
        self.declare_parameter("rear_early_contact_current_threshold_a", 999.0)
        self.declare_parameter("rear_early_contact_phase_start", 0.28)
        self.declare_parameter("rear_early_contact_phase_end", 0.40)
        self.declare_parameter("rear_early_contact_lift_relief_m", 0.002)
        self.declare_parameter("rear_early_contact_relief_sign", 1.0)
        self.declare_parameter("rear_touchdown_vmc_ramp", 0.20)
        self.declare_parameter("rear_touchdown_kp_ramp", 0.24)
        self.declare_parameter("rear_touchdown_kp_scale", 0.75)

        self.declare_parameter("hip_kp", 40.0)
        self.declare_parameter("thigh_kp", 70.0)
        self.declare_parameter("calf_kp", 70.0)
        self.declare_parameter("kd", 5.0)
        self.declare_parameter("support_hip_kp", 45.0)
        self.declare_parameter("support_thigh_kp", 80.0)
        self.declare_parameter("support_calf_kp", 85.0)
        self.declare_parameter("support_kd", 5.5)
        self.declare_parameter("touchdown_hip_kp", 35.0)
        self.declare_parameter("touchdown_thigh_kp", 60.0)
        self.declare_parameter("touchdown_calf_kp", 60.0)
        self.declare_parameter("touchdown_kd", 6.0)
        self.declare_parameter("rear_early_contact_hip_kp_limit", 35.0)
        self.declare_parameter("rear_early_contact_thigh_kp_limit", 55.0)
        self.declare_parameter("rear_early_contact_calf_kp_limit", 55.0)
        self.declare_parameter("rear_early_contact_kd", 6.5)

        self.declare_parameter("max_target_rate_rad_s_hip", 1.2)
        self.declare_parameter("max_target_rate_rad_s_thigh", 1.5)
        self.declare_parameter("max_target_rate_rad_s_calf", 1.5)
        self.declare_parameter("max_q_error_warn", 0.25)
        self.declare_parameter("max_q_error_stop", 0.45)
        self.declare_parameter("torque_soft_warn_nm", 8.0)
        self.declare_parameter("torque_soft_limit_nm", 10.0)
        self.declare_parameter("torque_stop_nm", 17.0)
        self.declare_parameter("current_warn_a", 999.0)
        self.declare_parameter("current_stop_a", 999.0)
        self.declare_parameter("max_roll_deg_warn", 8.0)
        self.declare_parameter("max_roll_deg_stop", 12.0)
        self.declare_parameter("max_pitch_deg_warn", 8.0)
        self.declare_parameter("max_pitch_deg_stop", 12.0)
        self.declare_parameter("max_motor_temp_warn_c", 65.0)
        self.declare_parameter("max_motor_temp_stop_c", 75.0)

        self.declare_parameter("send_speed", 0.0)
        self.declare_parameter("send_torque", 0.0)
        self.declare_parameter("imu_port", "/dev/myimu")
        self.declare_parameter("imu_read_hz", 100.0)
        self.declare_parameter("require_imu_for_air_send", False)

        # Cache parameters as attributes.
        for name in self._parameter_names():
            setattr(self, name, self.get_parameter(name).value)
        self.motor_base_url = str(self.motor_base_url).rstrip("/")
        self.test_mode = str(self.test_mode)
        if self.test_mode not in REAL_TEST_MODES:
            raise RuntimeError(f"Invalid test_mode={self.test_mode!r}; choose from {REAL_TEST_MODES}")
        self.enable_send = bool(self.enable_send)
        self.allow_long_free_test = bool(self.allow_long_free_test)
        self.dry_run_virtual_feedback = bool(self.dry_run_virtual_feedback)
        self.require_stand_ready = bool(self.require_stand_ready)
        self.allow_start_from_any_pose = bool(self.allow_start_from_any_pose)
        self.auto_stop_after_duration = bool(self.auto_stop_after_duration)
        if self.test_mode == "short_free" and float(self.duration_s) > 3.0 and not self.allow_long_free_test:
            self.get_logger().warn("short_free duration_s > 3.0; limiting to 3.0 unless allow_long_free_test=true.")
            self.duration_s = 3.0

    @staticmethod
    def _parameter_names() -> Tuple[str, ...]:
        return (
            "motor_base_url", "enable_send", "test_mode", "duration_s", "warmup_s", "soft_start_s",
            "soft_stop_s", "auto_stop_after_duration", "allow_long_free_test", "gait_hz", "http_timeout",
            "max_motor_age_ms", "csv_path", "stand_source", "step_hz", "duty_factor", "stride_length",
            "dry_run_virtual_feedback", "require_stand_ready", "stand_ready_q_error_threshold",
            "allow_start_from_any_pose",
            "front_swing_height", "rear_swing_height", "swing_lift_peak_phase", "touchdown_phase",
            "early_stance_blend", "thigh_length", "calf_length", "real_stride_scale_air",
            "real_stride_scale_touch", "real_stride_scale_assist", "real_stride_scale_short_free",
            "real_swing_height_scale_air", "real_swing_height_scale_touch", "real_swing_height_scale_assist",
            "real_swing_height_scale_short_free", "real_vmc_scale_air", "real_vmc_scale_touch",
            "real_vmc_scale_assist", "real_vmc_scale_short_free", "target_base_height", "target_pitch",
            "height_sign", "height_kp_z", "height_kd_z", "height_corr_limit_m", "roll_sign", "roll_kp_z",
            "roll_kd_z", "roll_corr_limit_m", "pitch_sign", "pitch_kp_z", "pitch_kd_z", "pitch_corr_limit_m",
            "enable_light_yaw_damping", "yaw_sign", "yaw_kp_hip", "yaw_kd_hip", "yaw_hip_limit_rad",
            "yaw_hip_rate_limit_rad", "rear_late_swing_guard_enable", "rear_late_swing_phase_start",
            "rear_late_swing_phase_end", "rear_late_swing_clearance_margin_m", "rear_late_swing_min_height_m",
            "rear_late_swing_progress_start", "rear_late_swing_progress_end", "rear_touchdown_progress_start",
            "rear_late_swing_guard_rate_limit_m", "rear_late_swing_clearance_sign",
            "rear_late_swing_descent_soft_enable", "rear_late_swing_descent_scale",
            "rear_early_contact_guard_enable", "rear_early_contact_force_threshold",
            "rear_early_contact_qerr_threshold", "rear_early_contact_tau_threshold_nm",
            "rear_early_contact_current_threshold_a", "rear_early_contact_phase_start",
            "rear_early_contact_phase_end", "rear_early_contact_lift_relief_m",
            "rear_early_contact_relief_sign", "rear_touchdown_vmc_ramp", "rear_touchdown_kp_ramp",
            "rear_touchdown_kp_scale", "hip_kp", "thigh_kp", "calf_kp", "kd", "support_hip_kp",
            "support_thigh_kp", "support_calf_kp", "support_kd", "touchdown_hip_kp", "touchdown_thigh_kp",
            "touchdown_calf_kp", "touchdown_kd", "rear_early_contact_hip_kp_limit",
            "rear_early_contact_thigh_kp_limit", "rear_early_contact_calf_kp_limit", "rear_early_contact_kd",
            "max_target_rate_rad_s_hip", "max_target_rate_rad_s_thigh", "max_target_rate_rad_s_calf",
            "max_q_error_warn", "max_q_error_stop", "torque_soft_warn_nm", "torque_soft_limit_nm",
            "torque_stop_nm", "current_warn_a", "current_stop_a", "max_roll_deg_warn", "max_roll_deg_stop",
            "max_pitch_deg_warn", "max_pitch_deg_stop", "max_motor_temp_warn_c", "max_motor_temp_stop_c",
            "send_speed", "send_torque", "imu_port", "imu_read_hz", "require_imu_for_air_send",
        )

    def _select_default_stand(self) -> np.ndarray:
        source = str(self.stand_source).strip().lower()
        if source == "fallback":
            self.get_logger().warn("Using fallback DEFAULT_STAND from the bring-up plan.")
            return FALLBACK_DEFAULT_STAND_POLICY.copy()
        self.get_logger().info("Using JointSemanticMapper.default_joint_angle as real-machine stand pose.")
        return self.mapper.default_joint_angle.astype(np.float32).copy()

    def _start_imu_if_needed(self):
        try:
            self.imu = ImuSerialInterface(port=str(self.imu_port), read_hz=float(self.imu_read_hz))
            self.imu.start()
            self.imu_valid = self.imu.wait_until_ready(timeout=2.0)
            if not self.imu_valid:
                self.get_logger().warn("WARNING: IMU not ready.")
        except Exception as exc:
            self.imu = None
            self.imu_valid = False
            self.get_logger().warn(f"WARNING: IMU unavailable: {exc}")

    def _validate_send_preconditions(self):
        if not self.enable_send:
            return
        if self.test_mode in ("touch", "assist", "short_free") and not self.imu_valid:
            raise RuntimeError("IMU is required before sending in touch/assist/short_free mode.")
        if self.test_mode == "air" and bool(self.require_imu_for_air_send) and not self.imu_valid:
            raise RuntimeError("IMU is required for air send because require_imu_for_air_send=true.")
        if self.test_mode in ("assist", "short_free"):
            feedback = self._read_feedback()
            if not feedback.valid:
                raise RuntimeError("Motor feedback is required before sending in assist/short_free mode.")
        if bool(self.require_stand_ready) and not bool(self.allow_start_from_any_pose):
            feedback = self._read_feedback()
            if not feedback.valid:
                raise RuntimeError("Motor feedback is required for stand-ready check before enable_send=true.")
            stand_error = float(np.max(np.abs(feedback.q_policy - self.default_policy)))
            if stand_error > float(self.stand_ready_q_error_threshold):
                raise RuntimeError(
                    "Stand-ready check failed: "
                    f"max(|q_actual - START_STAND|)={stand_error:.3f} rad "
                    f"> {float(self.stand_ready_q_error_threshold):.3f}. "
                    "Place robot in stand pose or pass allow_start_from_any_pose:=true for a conservative warmup."
                )
        if bool(self.allow_start_from_any_pose):
            self.warmup_s = max(float(self.warmup_s), 3.0)
            self.max_target_rate_rad_s_hip = min(float(self.max_target_rate_rad_s_hip), 0.6)
            self.max_target_rate_rad_s_thigh = min(float(self.max_target_rate_rad_s_thigh), 0.8)
            self.max_target_rate_rad_s_calf = min(float(self.max_target_rate_rad_s_calf), 0.8)
            self.real_vmc_scale_air = 0.0
            self.real_vmc_scale_touch = 0.0
            self.real_vmc_scale_assist = 0.0
            self.real_vmc_scale_short_free = 0.0
            self.get_logger().warn(
                "allow_start_from_any_pose=true: forcing warmup_s>=3, reducing target rate, and disabling VMC."
            )

    def _print_startup_summary(self):
        self.get_logger().warn(
            "fanfan_cpg_vmc_v4_node | "
            f"enable_send={self.enable_send} mode={self.test_mode} duration={float(self.duration_s):.2f}s "
            f"step_hz={float(self.step_hz):.2f} duty={float(self.duty_factor):.2f} "
            f"stride={float(self.stride_length):.3f} front_h={float(self.front_swing_height):.3f} "
            f"rear_h={float(self.rear_swing_height):.3f} vmc_scale={self.real_vmc_scale():.2f}"
        )
        self.get_logger().warn(
            "URDF hip outward signs: "
            f"FR={HIP_OUTWARD_SIGNS['FR']:+.0f} FL={HIP_OUTWARD_SIGNS['FL']:+.0f} "
            f"RR={HIP_OUTWARD_SIGNS['RR']:+.0f} RL={HIP_OUTWARD_SIGNS['RL']:+.0f}"
        )
        if not self.imu_valid:
            self.get_logger().warn("WARNING: IMU feedback unavailable. VMC uses zero roll/pitch/yaw.")

    def _countdown(self):
        for i in (3, 2, 1):
            self.get_logger().warn(f"ENABLE_SEND TRUE: starting motor commands in {i}...")
            time.sleep(1.0)

    def _setup_csv(self):
        path = str(self.csv_path).strip()
        if not path:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.expanduser(f"~/mydog_ros2_ws/src/mydog_policy/mydog_policy/docs/cpg_vmc_v4_{stamp}.csv")
        path = os.path.expanduser(path)
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self.csv_path = path
        self.csv_file = open(path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(self._csv_header())
        self.csv_file.flush()
        self.get_logger().warn(f"[CSV] writing CPG/VMC v4 data to {path}")

    def _csv_header(self) -> List[str]:
        header = [
            "time", "dt", "test_mode", "enable_send", "node_state", "stop_reason", "safety_stop_reason",
            "dry_run_virtual_feedback", "phase", "active_swing_pair", "support_pair",
            "leg_phase_FR", "leg_phase_FL", "leg_phase_RR", "leg_phase_RL",
            "swing_progress_FR", "swing_progress_FL", "swing_progress_RR", "swing_progress_RL",
            "swing_mask_FR", "swing_mask_FL", "swing_mask_RR", "swing_mask_RL",
            "support_mask_FR", "support_mask_FL", "support_mask_RR", "support_mask_RL",
            "base_roll", "base_pitch", "base_yaw", "roll_abs", "pitch_abs", "yaw_abs",
            "target_yaw", "yaw_error", "yaw_error_abs",
            "base_ang_vel_x", "base_ang_vel_y", "base_ang_vel_z",
            "base_lin_acc_x", "base_lin_acc_y", "base_lin_acc_z",
            "real_vmc_scale", "vmc_height_corr_z", "vmc_roll_corr_z", "vmc_pitch_corr_z", "yaw_corr_hip",
            "yaw_hip_offset_FR", "yaw_hip_offset_FL", "yaw_hip_offset_RR", "yaw_hip_offset_RL",
            "vmc_weight_FR", "vmc_weight_FL", "vmc_weight_RR", "vmc_weight_RL",
            "rear_late_swing_guard_active_RR", "rear_late_swing_guard_active_RL",
            "rear_late_swing_clearance_offset_RR", "rear_late_swing_clearance_offset_RL",
            "rear_late_swing_descent_scale_applied_RR", "rear_late_swing_descent_scale_applied_RL",
            "rear_late_swing_window_active_RR", "rear_late_swing_window_active_RL",
            "rear_late_swing_is_descending_RR", "rear_late_swing_is_descending_RL",
            "rear_late_swing_guard_reason_RR", "rear_late_swing_guard_reason_RL",
            "rear_early_contact_guard_active_RR", "rear_early_contact_guard_active_RL",
            "rear_early_contact_score_RR", "rear_early_contact_score_RL", "early_contact_source",
            "rear_touchdown_kp_ramp_weight_RR", "rear_touchdown_kp_ramp_weight_RL",
        ]
        for prefix in (
            "q_ref",
            "q_cmd_raw",
            "q_cmd_final",
            "q_actual",
            "q_actual_for_safety",
            "q_error",
            "q_error_for_safety",
            "q_error_real_feedback",
            "dq_actual",
            "tau_est",
            "tau_est_for_safety",
            "current",
            "temp",
            "rate_limited_delta",
            "q_ref_cmd_diff",
        ):
            header += [f"{prefix}_{name}" for name in JOINT_NAMES]
        header += [f"raw_motor_target_0x{mid:02X}" for mid in (0x11, 0x12, 0x13, 0x21, 0x22, 0x23, 0x31, 0x32, 0x33, 0x41, 0x42, 0x43)]
        header += [
            "max_q_error", "max_tau_est", "max_current", "max_temp", "safety_warn", "safety_stop",
            "rate_clip_ratio", "torque_clip_ratio", "communication_ok", "semantic_to_motor_mapping_ok",
            "hip_outward_sign_FR", "hip_outward_sign_FL", "hip_outward_sign_RR", "hip_outward_sign_RL",
            "use_urdf_hip_outward_signs",
        ]
        return header

    def real_stride_scale(self) -> float:
        return mode_scale(
            self.test_mode,
            float(self.real_stride_scale_air),
            float(self.real_stride_scale_touch),
            float(self.real_stride_scale_assist),
            float(self.real_stride_scale_short_free),
        )

    def real_swing_height_scale(self) -> float:
        return mode_scale(
            self.test_mode,
            float(self.real_swing_height_scale_air),
            float(self.real_swing_height_scale_touch),
            float(self.real_swing_height_scale_assist),
            float(self.real_swing_height_scale_short_free),
        )

    def real_vmc_scale(self) -> float:
        return mode_scale(
            self.test_mode,
            float(self.real_vmc_scale_air),
            float(self.real_vmc_scale_touch),
            float(self.real_vmc_scale_assist),
            float(self.real_vmc_scale_short_free),
        )

    def update(self):
        now = time.time()
        dt = clamp(now - self._last_update_time, 1.0 / 500.0, 0.10)
        self._last_update_time = now
        elapsed = now - self.start_time

        feedback = self._read_feedback()
        imu = self._read_imu()
        if not self._run_started:
            self.target_yaw = float(imu["rpy"][2])
            self._run_started = True

        if self.auto_stop_after_duration and elapsed >= float(self.duration_s) and not self._soft_stop_active:
            self._request_soft_stop("duration_reached")
        if self.safety_stop and not self._soft_stop_active:
            self._request_soft_stop(self.safety_stop_reason)

        if self._soft_stop_active:
            self.node_state = "SAFETY_STOP" if self.safety_stop else "SOFT_STOP"
            q_cmd_raw, debug = self._soft_stop_target(now)
        else:
            self.node_state = "WARMUP" if elapsed < float(self.warmup_s) else "GAIT"
            self.phase = wrap_phase(self.phase + float(self.step_hz) * dt)
            run_warm = clamp(elapsed / max(float(self.warmup_s), 1.0e-3), 0.0, 1.0)
            soft_start = smootherstep(clamp(elapsed / max(float(self.soft_start_s), 1.0e-3), 0.0, 1.0))
            q_cmd_raw, debug = self._build_target(self.phase, soft_start * run_warm, imu, feedback, dt)

        q_cmd_final, safety = self._apply_safety_output(q_cmd_raw, feedback, dt, debug)
        target_real = self.mapper.policy_target_to_real_target(q_cmd_final, clamp=True)
        self.publish_array(self.pub_target, target_real)
        self.publish_array(
            self.pub_debug,
            np.array(
                [
                    self.phase,
                    float(debug["leg_phase"]["FR"]),
                    float(debug["leg_phase"]["FL"]),
                    float(debug["leg_phase"]["RR"]),
                    float(debug["leg_phase"]["RL"]),
                    float(safety["max_q_error"]),
                    float(safety["max_tau_est"]),
                    float(safety["rate_clip_ratio"]),
                    float(safety["torque_clip_ratio"]),
                ],
                dtype=np.float32,
            ),
        )

        sent = False
        if self.enable_send:
            sent = self._send_motion_batch(target_real, debug["kp"], debug["kd"])
        self._write_csv(now, dt, debug, safety, feedback, imu, sent, q_cmd_raw, target_real)

        if self._soft_stop_active and now - self._soft_stop_start >= float(self.soft_stop_s):
            self.node_state = "DONE"
            self.get_logger().warn(f"soft stop complete: {self.stop_reason}")
            rclpy.shutdown()

    def _read_imu(self) -> Dict[str, np.ndarray]:
        if self.imu is None:
            return self._zero_imu(False)
        snap = self.imu.get_latest()
        if not snap.valid:
            return self._zero_imu(False)
        self.imu_valid = True
        rpy = np.asarray(snap.rpy_deg, dtype=np.float32) * (math.pi / 180.0)
        return {
            "valid": True,
            "rpy": rpy,
            "gyro": np.asarray(snap.gyro_rad_s, dtype=np.float32).reshape(3),
            "acc": np.asarray(snap.acc_g, dtype=np.float32).reshape(3),
        }

    @staticmethod
    def _zero_imu(valid: bool) -> Dict[str, np.ndarray]:
        return {
            "valid": valid,
            "rpy": np.zeros(3, dtype=np.float32),
            "gyro": np.zeros(3, dtype=np.float32),
            "acc": np.zeros(3, dtype=np.float32),
        }

    def _read_feedback(self) -> Feedback:
        try:
            snap: MotorSnapshot = self.motor.get_latest()
            q_policy, dq_policy = self.mapper.real_to_policy_abs_q_dq(snap.q_real, snap.dq_real)
            torque_real = np.asarray(snap.torque, dtype=np.float32).reshape(12)
            temp_real = np.asarray(snap.temp, dtype=np.float32).reshape(12)
            torque_policy = torque_real[self.mapper.policy_to_real_index] * self.mapper.joint_sign
            temp_policy = temp_real[self.mapper.policy_to_real_index]
            current_policy = np.full(12, np.nan, dtype=np.float32)
            self.motor_feedback_valid = bool(snap.valid and np.all(np.isfinite(q_policy)))
            return Feedback(
                q_policy=q_policy,
                dq_policy=dq_policy,
                torque_policy=torque_policy.astype(np.float32),
                current_policy=current_policy,
                temp_policy=temp_policy.astype(np.float32),
                online=np.asarray(snap.online, dtype=bool).reshape(12),
                max_age_ms=float(np.max(snap.age_ms)),
                valid=self.motor_feedback_valid,
                source="http",
            )
        except Exception as exc:
            now = time.time()
            if now - self._last_feedback_warn > 1.0:
                self._last_feedback_warn = now
                self.get_logger().warn(f"WARNING: motor feedback unavailable: {exc}")
            return Feedback(
                q_policy=self.q_cmd_final.copy(),
                dq_policy=np.zeros(12, dtype=np.float32),
                torque_policy=np.full(12, np.nan, dtype=np.float32),
                current_policy=np.full(12, np.nan, dtype=np.float32),
                temp_policy=np.full(12, np.nan, dtype=np.float32),
                online=np.zeros(12, dtype=bool),
                max_age_ms=float("inf"),
                valid=False,
                source="none",
            )

    def _build_target(self, phase: float, warm: float, imu: Dict[str, np.ndarray], feedback: Feedback, dt: float):
        stride = float(self.stride_length) * self.real_stride_scale()
        front_height = float(self.front_swing_height) * self.real_swing_height_scale()
        rear_height = float(self.rear_swing_height) * self.real_swing_height_scale()
        vmc_scale = self.real_vmc_scale()

        q = self.default_policy.copy()
        leg_phase = {leg: wrap_phase(phase - PAIR_OFFSETS[leg]) for leg in LEG_ORDER}
        swing_fraction = max(0.05, 1.0 - float(self.duty_factor))
        swing_mask = {leg: float(leg_phase[leg] < swing_fraction) for leg in LEG_ORDER}
        swing_progress = {
            leg: clamp(leg_phase[leg] / swing_fraction, 0.0, 1.0) if swing_mask[leg] > 0.5 else 1.0
            for leg in LEG_ORDER
        }
        support_mask = {leg: 1.0 - swing_mask[leg] for leg in LEG_ORDER}
        active_pair = "FR+RL" if swing_mask["FR"] > 0.5 else "FL+RR"
        support_pair = "FL+RR" if active_pair == "FR+RL" else "FR+RL"

        rpy = imu["rpy"]
        gyro = imu["gyro"]
        height_error = 0.0  # no real base-height measurement on this node
        roll_error = float(rpy[0])
        pitch_error = float(rpy[1] - float(self.target_pitch))
        vmc_height = clamp(
            float(self.height_sign) * (float(self.height_kp_z) * height_error - float(self.height_kd_z) * 0.0),
            -float(self.height_corr_limit_m),
            float(self.height_corr_limit_m),
        ) * vmc_scale
        vmc_roll = clamp(
            float(self.roll_sign) * (float(self.roll_kp_z) * roll_error + float(self.roll_kd_z) * float(gyro[0])),
            -float(self.roll_corr_limit_m),
            float(self.roll_corr_limit_m),
        ) * vmc_scale
        vmc_pitch = clamp(
            float(self.pitch_sign) * (float(self.pitch_kp_z) * pitch_error + float(self.pitch_kd_z) * float(gyro[1])),
            -float(self.pitch_corr_limit_m),
            float(self.pitch_corr_limit_m),
        ) * vmc_scale
        yaw_corr = 0.0
        yaw_error = wrap_to_pi(float(rpy[2]) - float(self.target_yaw))
        if bool(self.enable_light_yaw_damping):
            yaw_corr = clamp(
                float(self.yaw_sign)
                * (float(self.yaw_kp_hip) * yaw_error + float(self.yaw_kd_hip) * float(gyro[2])),
                -float(self.yaw_hip_limit_rad),
                float(self.yaw_hip_limit_rad),
            ) * vmc_scale
        last_yaw = self.yaw_hip_offset.copy()
        desired_yaw = np.array(
            [HIP_OUTWARD_SIGNS[leg] * yaw_corr for leg in LEG_ORDER],
            dtype=np.float32,
        )
        yaw_step = np.clip(
            desired_yaw - last_yaw,
            -float(self.yaw_hip_rate_limit_rad),
            float(self.yaw_hip_rate_limit_rad),
        )
        self.yaw_hip_offset = last_yaw + yaw_step

        vmc_weight = np.zeros(4, dtype=np.float32)
        rear_late_guard = {"RR": 0.0, "RL": 0.0}
        rear_late_window = {"RR": 0.0, "RL": 0.0}
        rear_late_descending = {"RR": 0.0, "RL": 0.0}
        rear_late_reason = {"RR": "", "RL": ""}
        rear_descent_scale = {"RR": 1.0, "RL": 1.0}
        rear_early_guard = {"RR": 0.0, "RL": 0.0}
        rear_early_score = {"RR": 0.0, "RL": 0.0}
        rear_touchdown_weight = {"RR": 0.0, "RL": 0.0}
        early_source = "q_error_only"

        for leg_i, leg in enumerate(LEG_ORDER):
            idx = LEG_START[leg]
            phase_local = leg_phase[leg]
            is_swing = swing_mask[leg] > 0.5
            leg_swing_phase = swing_progress[leg]
            stance_phase = clamp((phase_local - swing_fraction) / max(1.0 - swing_fraction, 1e-6), 0.0, 1.0)

            vmc_w = support_mask[leg] * smootherstep(min(1.0, stance_phase / max(float(self.early_stance_blend), 1e-6)))
            if leg in REAR_LEGS and is_swing:
                vmc_w = 0.0
            vmc_weight[leg_i] = float(vmc_w)

            base_x = 0.5 * stride if is_swing else -0.5 * stride
            swing_advance = smootherstep(leg_swing_phase)
            x_delta = (-0.5 * stride + stride * swing_advance) if is_swing else base_x
            height = rear_height if leg in REAR_LEGS else front_height
            if is_swing:
                peak = max(float(self.swing_lift_peak_phase), 1e-6)
                td = max(float(self.touchdown_phase), peak + 1e-6)
                if leg_swing_phase <= peak:
                    lift = smootherstep(leg_swing_phase / peak)
                elif leg_swing_phase <= td:
                    down = smootherstep((leg_swing_phase - peak) / max(td - peak, 1e-6))
                    lift = 1.0 - down
                else:
                    lift = 0.0
                z_delta = height * lift
            else:
                z_delta = 0.0

            rear_late_active = (
                leg in REAR_LEGS
                and is_swing
                and float(self.rear_late_swing_progress_start)
                <= leg_swing_phase
                <= float(self.rear_late_swing_progress_end)
            )
            rear_is_descending = (
                rear_late_active and leg_swing_phase >= max(float(self.swing_lift_peak_phase), 1.0e-6)
            )
            if leg in REAR_LEGS:
                rear_late_window[leg] = float(rear_late_active)
                rear_late_descending[leg] = float(rear_is_descending)

            if rear_is_descending and bool(self.rear_late_swing_descent_soft_enable):
                # Keep a portion of the current lift so late swing descends more softly.
                z_delta += max(0.0, z_delta) * (1.0 - float(self.rear_late_swing_descent_scale))
                rear_descent_scale[leg] = float(self.rear_late_swing_descent_scale)

            if leg in REAR_LEGS and is_swing and bool(self.rear_late_swing_guard_enable):
                if rear_late_active:
                    min_clearance = float(self.rear_late_swing_min_height_m) + float(self.rear_late_swing_clearance_margin_m)
                    height_error = max(0.0, min_clearance - z_delta)
                    desired = float(self.rear_late_swing_clearance_sign) * height_error
                    last = self.rear_late_clearance[leg_i]
                    step = clamp(desired - float(last), -float(self.rear_late_swing_guard_rate_limit_m), float(self.rear_late_swing_guard_rate_limit_m))
                    self.rear_late_clearance[leg_i] = float(last + step)
                    if abs(self.rear_late_clearance[leg_i]) > 1.0e-6:
                        rear_late_guard[leg] = 1.0
                        rear_late_reason[leg] = "clearance_low"
                    else:
                        rear_late_reason[leg] = "height_ok"
                else:
                    self.rear_late_clearance[leg_i] *= 0.5
                    if leg in REAR_LEGS:
                        rear_late_reason[leg] = "outside_window"
                z_delta += float(self.rear_late_clearance[leg_i])

            q_feedback = feedback.q_policy if feedback.valid else self.q_cmd_final
            qerr_leg = float(np.max(np.abs(self.q_cmd_final[idx : idx + 3] - q_feedback[idx : idx + 3])))
            tau_leg = float(np.nanmax(np.abs(feedback.torque_policy[idx : idx + 3]))) if np.any(np.isfinite(feedback.torque_policy[idx : idx + 3])) else float("nan")
            current_leg = float(np.nanmax(np.abs(feedback.current_policy[idx : idx + 3]))) if np.any(np.isfinite(feedback.current_policy[idx : idx + 3])) else float("nan")
            early_score = qerr_leg / max(float(self.rear_early_contact_qerr_threshold), 1e-6)
            if math.isfinite(tau_leg):
                early_score = max(early_score, tau_leg / max(float(self.rear_early_contact_tau_threshold_nm), 1e-6))
                early_source = "q_error_or_torque"
            if math.isfinite(current_leg):
                early_score = max(early_score, current_leg / max(float(self.rear_early_contact_current_threshold_a), 1e-6))
                early_source = "q_error_or_current"
            if (
                leg in REAR_LEGS
                and is_swing
                and bool(self.rear_early_contact_guard_enable)
                and float(self.rear_early_contact_phase_start) <= leg_swing_phase <= float(self.rear_early_contact_phase_end)
                and early_score > 1.0
            ):
                rear_early_guard[leg] = 1.0
                z_delta += float(self.rear_early_contact_relief_sign) * float(self.rear_early_contact_lift_relief_m)
            if leg in REAR_LEGS:
                rear_early_score[leg] = float(early_score)
                if not is_swing and stance_phase < float(self.rear_touchdown_kp_ramp):
                    rear_touchdown_weight[leg] = smootherstep(stance_phase / max(float(self.rear_touchdown_kp_ramp), 1e-6))

            z_delta += vmc_w * (vmc_height + vmc_pitch + (vmc_roll if leg in ("FL", "RL") else -vmc_roll))
            hip = self.default_policy[idx] + HIP_OUTWARD_SIGNS[leg] * 0.02 * support_mask[leg] + self.yaw_hip_offset[leg_i]
            thigh, calf = self._ik_delta_to_q(leg, x_delta, z_delta)
            q[idx] = hip
            q[idx + 1] = thigh
            q[idx + 2] = calf

        q = self.default_policy + warm * (q - self.default_policy)
        q = np.clip(q, self.mapper.policy_lower_limit, self.mapper.policy_upper_limit)
        kp, kd = self._compute_gains(swing_mask, support_mask, rear_early_guard, rear_touchdown_weight)
        return q.astype(np.float32), {
            "leg_phase": leg_phase,
            "swing_progress": swing_progress,
            "swing_mask": swing_mask,
            "support_mask": support_mask,
            "active_pair": active_pair,
            "support_pair": support_pair,
            "vmc_height": vmc_height,
            "vmc_roll": vmc_roll,
            "vmc_pitch": vmc_pitch,
            "yaw_corr": yaw_corr,
            "target_yaw": float(self.target_yaw),
            "yaw_error": yaw_error,
            "yaw_offsets": self.yaw_hip_offset.copy(),
            "vmc_weight": vmc_weight,
            "rear_late_guard": rear_late_guard,
            "rear_late_window": rear_late_window,
            "rear_late_descending": rear_late_descending,
            "rear_late_reason": rear_late_reason,
            "rear_late_clearance": {leg: float(self.rear_late_clearance[LEG_ORDER.index(leg)]) for leg in REAR_LEGS},
            "rear_descent_scale": rear_descent_scale,
            "rear_early_guard": rear_early_guard,
            "rear_early_score": rear_early_score,
            "rear_touchdown_weight": rear_touchdown_weight,
            "early_source": early_source,
            "kp": kp,
            "kd": kd,
        }

    def _ik_delta_to_q(self, leg: str, x_delta: float, z_delta: float) -> Tuple[float, float]:
        idx = LEG_START[leg]
        thigh0 = float(self.default_policy[idx + 1])
        calf0 = float(self.default_policy[idx + 2])
        # Conservative local linearization, not a full foot-frame solver.  It
        # keeps the node deployment-safe and directionally aligned with the
        # existing real-machine IK gait nodes.
        front_sign = 1.0 if leg in ("FR", "FL") else -1.0
        thigh = thigh0 + front_sign * 2.2 * float(x_delta) + 1.8 * float(z_delta)
        calf = calf0 - 3.0 * float(z_delta) - front_sign * 1.2 * float(x_delta)
        return float(thigh), float(calf)

    def _compute_gains(self, swing_mask, support_mask, rear_early_guard, rear_touchdown_weight):
        kp = np.zeros(12, dtype=np.float32)
        kd = np.zeros(12, dtype=np.float32)
        for leg in LEG_ORDER:
            idx = LEG_START[leg]
            if swing_mask[leg] > 0.5:
                leg_kp = [float(self.hip_kp), float(self.thigh_kp), float(self.calf_kp)]
                leg_kd = float(self.kd)
            else:
                leg_kp = [float(self.support_hip_kp), float(self.support_thigh_kp), float(self.support_calf_kp)]
                leg_kd = float(self.support_kd)
            if leg in REAR_LEGS:
                td = float(rear_touchdown_weight[leg])
                if td > 0.0:
                    touch_kp = np.array([float(self.touchdown_hip_kp), float(self.touchdown_thigh_kp), float(self.touchdown_calf_kp)], dtype=np.float32)
                    support_kp = np.array(leg_kp, dtype=np.float32)
                    leg_kp = (touch_kp * (1.0 - td) + support_kp * td).tolist()
                    leg_kd = float(self.touchdown_kd) * (1.0 - td) + leg_kd * td
                if rear_early_guard[leg] > 0.5:
                    leg_kp = [
                        min(leg_kp[0], float(self.rear_early_contact_hip_kp_limit)),
                        min(leg_kp[1], float(self.rear_early_contact_thigh_kp_limit)),
                        min(leg_kp[2], float(self.rear_early_contact_calf_kp_limit)),
                    ]
                    leg_kd = float(self.rear_early_contact_kd)
            kp[idx : idx + 3] = np.asarray(leg_kp, dtype=np.float32)
            kd[idx : idx + 3] = leg_kd
        return kp, kd

    def _apply_safety_output(self, q_raw: np.ndarray, feedback: Feedback, dt: float, debug: dict):
        q_actual_real = feedback.q_policy if feedback.valid else self.q_cmd_final
        dq_actual_real = feedback.dq_policy if feedback.valid else np.zeros(12, dtype=np.float32)
        use_virtual = (not self.enable_send) and bool(self.dry_run_virtual_feedback)
        if use_virtual:
            q_actual_for_safety = self.q_last_cmd.copy()
        else:
            q_actual_for_safety = q_actual_real.copy()
        rate_limits = np.array(
            [float(self.max_target_rate_rad_s_hip), float(self.max_target_rate_rad_s_thigh), float(self.max_target_rate_rad_s_calf)] * 4,
            dtype=np.float32,
        )
        delta = q_raw - self.q_last_cmd
        max_step = rate_limits * max(dt, 1.0e-4)
        clipped_delta = np.clip(delta, -max_step, max_step)
        q_rate = self.q_last_cmd + clipped_delta
        if use_virtual:
            dq_actual_for_safety = clipped_delta / max(dt, 1.0e-4)
        else:
            dq_actual_for_safety = dq_actual_real.copy()
        rate_clip_mask = np.abs(delta - clipped_delta) > 1.0e-6

        kp = np.asarray(debug["kp"], dtype=np.float32).reshape(12)
        kd = np.asarray(debug["kd"], dtype=np.float32).reshape(12)
        tau_est = kp * (q_rate - q_actual_for_safety) - kd * dq_actual_for_safety
        abs_tau = np.abs(tau_est)
        scale = np.ones(12, dtype=np.float32)
        soft = float(self.torque_soft_limit_nm)
        hard = float(self.torque_stop_nm)
        mask_soft = abs_tau > soft
        scale[mask_soft] = np.clip((hard - abs_tau[mask_soft]) / max(hard - soft, 1.0e-6), 0.0, 1.0)
        q_torque = q_actual_for_safety + scale * (q_rate - q_actual_for_safety)
        torque_clip_mask = mask_soft
        tau_final = kp * (q_torque - q_actual_for_safety) - kd * dq_actual_for_safety

        q_cmd = np.clip(q_torque, self.mapper.policy_lower_limit, self.mapper.policy_upper_limit)
        self.q_last_cmd = q_cmd.astype(np.float32).copy()
        self.q_cmd_final = q_cmd.astype(np.float32).copy()

        q_error_for_safety = q_cmd - q_actual_for_safety
        q_error_real_feedback = q_cmd - q_actual_real
        current = feedback.current_policy
        temp = feedback.temp_policy
        max_q_error = float(np.max(np.abs(q_error_for_safety)))
        max_tau = float(np.nanmax(np.abs(tau_final))) if np.any(np.isfinite(tau_final)) else float("nan")
        max_current = float(np.nanmax(np.abs(current))) if np.any(np.isfinite(current)) else float("nan")
        max_temp = float(np.nanmax(temp)) if np.any(np.isfinite(temp)) else float("nan")

        warn = ""
        if max_q_error > float(self.max_q_error_warn):
            warn = "q_error_warn"
        if math.isfinite(max_tau) and max_tau > float(self.torque_soft_warn_nm):
            warn = (warn + "|tau_warn").strip("|")
        if math.isfinite(max_current) and max_current > float(self.current_warn_a):
            warn = (warn + "|current_warn").strip("|")
        if math.isfinite(max_temp) and max_temp > float(self.max_motor_temp_warn_c):
            warn = (warn + "|temp_warn").strip("|")

        self._check_stop_conditions(max_q_error, max_tau, max_current, max_temp, feedback)
        return q_cmd, {
            "dry_run_virtual_feedback": use_virtual,
            "q_actual": q_actual_real,
            "q_actual_for_safety": q_actual_for_safety,
            "dq_actual": dq_actual_real,
            "dq_actual_for_safety": dq_actual_for_safety,
            "q_error": q_error_real_feedback,
            "q_error_for_safety": q_error_for_safety,
            "q_error_real_feedback": q_error_real_feedback,
            "tau_est": tau_final,
            "tau_est_for_safety": tau_final,
            "current": current,
            "temp": temp,
            "max_q_error": max_q_error,
            "max_tau_est": max_tau,
            "max_current": max_current,
            "max_temp": max_temp,
            "rate_delta": clipped_delta,
            "rate_clip_ratio": float(np.mean(rate_clip_mask.astype(np.float32))),
            "torque_clip_ratio": float(np.mean(torque_clip_mask.astype(np.float32))),
            "safety_warn": warn,
        }

    def _check_stop_conditions(self, max_q_error, max_tau, max_current, max_temp, feedback):
        if self.safety_stop:
            return
        imu = self._read_imu()
        roll_deg = abs(float(imu["rpy"][0]) * 180.0 / math.pi)
        pitch_deg = abs(float(imu["rpy"][1]) * 180.0 / math.pi)
        reason = ""
        if roll_deg > float(self.max_roll_deg_stop):
            reason = "roll_stop"
        elif pitch_deg > float(self.max_pitch_deg_stop):
            reason = "pitch_stop"
        elif max_q_error > float(self.max_q_error_stop):
            reason = "q_error_stop"
        elif math.isfinite(max_tau) and max_tau > float(self.torque_stop_nm):
            reason = "torque_stop"
        elif math.isfinite(max_current) and max_current > float(self.current_stop_a):
            reason = "current_stop"
        elif math.isfinite(max_temp) and max_temp > float(self.max_motor_temp_stop_c):
            reason = "temp_stop"
        elif self.enable_send and feedback.valid and feedback.max_age_ms > float(self.max_motor_age_ms):
            reason = "communication_timeout"
        if reason:
            self.safety_stop = True
            self.safety_stop_reason = reason
            self.stop_reason = reason
            self.get_logger().error(f"SAFETY STOP: {reason}")

    def _soft_stop_target(self, now: float):
        alpha = smootherstep(clamp((now - self._soft_stop_start) / max(float(self.soft_stop_s), 1.0e-3), 0.0, 1.0))
        q = (1.0 - alpha) * self._soft_stop_from + alpha * self.default_policy
        kp = np.array([float(self.support_hip_kp), float(self.support_thigh_kp), float(self.support_calf_kp)] * 4, dtype=np.float32)
        kd = np.full(12, float(self.support_kd), dtype=np.float32)
        debug = self._empty_debug()
        debug["kp"] = kp
        debug["kd"] = kd
        return q.astype(np.float32), debug

    def _empty_debug(self):
        return {
            "leg_phase": {leg: 0.0 for leg in LEG_ORDER},
            "swing_progress": {leg: 1.0 for leg in LEG_ORDER},
            "swing_mask": {leg: 0.0 for leg in LEG_ORDER},
            "support_mask": {leg: 1.0 for leg in LEG_ORDER},
            "active_pair": "SOFT_STOP",
            "support_pair": "FR+FL+RR+RL",
            "vmc_height": 0.0,
            "vmc_roll": 0.0,
            "vmc_pitch": 0.0,
            "yaw_corr": 0.0,
            "target_yaw": float(self.target_yaw),
            "yaw_error": 0.0,
            "yaw_offsets": np.zeros(4, dtype=np.float32),
            "vmc_weight": np.zeros(4, dtype=np.float32),
            "rear_late_guard": {"RR": 0.0, "RL": 0.0},
            "rear_late_window": {"RR": 0.0, "RL": 0.0},
            "rear_late_descending": {"RR": 0.0, "RL": 0.0},
            "rear_late_reason": {"RR": "soft_stop", "RL": "soft_stop"},
            "rear_late_clearance": {"RR": 0.0, "RL": 0.0},
            "rear_descent_scale": {"RR": 1.0, "RL": 1.0},
            "rear_early_guard": {"RR": 0.0, "RL": 0.0},
            "rear_early_score": {"RR": 0.0, "RL": 0.0},
            "rear_touchdown_weight": {"RR": 0.0, "RL": 0.0},
            "early_source": "none",
            "kp": np.array([float(self.hip_kp), float(self.thigh_kp), float(self.calf_kp)] * 4, dtype=np.float32),
            "kd": np.full(12, float(self.kd), dtype=np.float32),
        }

    def _request_soft_stop(self, reason: str):
        if self._soft_stop_active:
            return
        self._soft_stop_active = True
        self._soft_stop_start = time.time()
        self._soft_stop_from = self.q_cmd_final.copy()
        self.stop_reason = reason
        self.get_logger().warn(f"soft stop requested: {reason}")

    def _send_default_stand(self):
        items = []
        kp = [float(self.support_hip_kp), float(self.support_thigh_kp), float(self.support_calf_kp)] * 4
        kd = [float(self.support_kd)] * 12
        for i, mid in enumerate(self.motor_ids):
            items.append(
                {
                    "motor_id": int(mid),
                    "position": float(self.default_real[i]),
                    "speed": 0.0,
                    "torque": 0.0,
                    "kp": float(kp[i]),
                    "kd": float(kd[i]),
                }
            )
        r = self.http_session.post(
            f"{self.motor_base_url}/api/rs04/motion_mode_run_batch",
            json={"items": items, "enable_first": True, "stop_first": False},
            timeout=max(float(self.http_timeout), 0.5),
        )
        if r.status_code != 200:
            raise RuntimeError(f"default stand failed HTTP {r.status_code}: {r.text}")

    def _send_motion_batch(self, target_real: np.ndarray, kp_policy: np.ndarray, kd_policy: np.ndarray) -> bool:
        kp_real = np.zeros(12, dtype=np.float32)
        kd_real = np.zeros(12, dtype=np.float32)
        kp_real[self.mapper.policy_to_real_index] = kp_policy
        kd_real[self.mapper.policy_to_real_index] = kd_policy
        items = []
        for i, mid in enumerate(self.motor_ids):
            items.append(
                {
                    "motor_id": int(mid),
                    "position": float(target_real[i]),
                    "speed": float(self.send_speed),
                    "torque": float(self.send_torque),
                    "kp": float(kp_real[i]),
                    "kd": float(kd_real[i]),
                }
            )
        try:
            r = self.http_session.post(
                f"{self.motor_base_url}/api/rs04/motion_batch_fast",
                json={"items": items, "enable_first": False, "stop_first": False},
                timeout=float(self.http_timeout),
            )
            if r.status_code != 200:
                self.get_logger().warn(f"[SEND] HTTP {r.status_code}: {r.text}")
                return False
            now = time.time()
            if now - self._last_send_info > 1.0:
                self._last_send_info = now
                self.get_logger().info(f"[SEND] cpg_vmc_v4 ok mode={self.test_mode}")
            return True
        except Exception as exc:
            self.get_logger().warn(f"[SEND] request failed: {exc}")
            return False

    def _write_csv(self, now, dt, debug, safety, feedback: Feedback, imu, sent, q_cmd_raw, target_real):
        rpy = imu["rpy"]
        gyro = imu["gyro"]
        acc = imu["acc"]
        q_ref = np.asarray(q_cmd_raw, dtype=np.float32).reshape(12)
        q_cmd = self.q_cmd_final.copy()
        q_actual = safety["q_actual"]
        q_error = safety["q_error"]
        q_ref_cmd_diff = q_ref - q_cmd
        raw_motor_target = np.asarray(target_real, dtype=np.float32).reshape(12)
        remapped_real = self.mapper.policy_target_to_real_target(q_cmd, clamp=True)
        semantic_to_motor_mapping_ok = bool(np.allclose(raw_motor_target, remapped_real, atol=1.0e-6))
        yaw_error = wrap_to_pi(float(rpy[2]) - float(self.target_yaw))
        row = [
            f"{now:.6f}", f"{dt:.6f}", self.test_mode, int(self.enable_send),
            self.node_state, self.stop_reason, self.safety_stop_reason,
            int(bool(safety["dry_run_virtual_feedback"])), f"{self.phase:.6f}",
            debug["active_pair"], debug["support_pair"],
        ]
        row += [f"{debug['leg_phase'][leg]:.6f}" for leg in LEG_ORDER]
        row += [f"{debug['swing_progress'][leg]:.6f}" for leg in LEG_ORDER]
        row += [int(debug["swing_mask"][leg] > 0.5) for leg in LEG_ORDER]
        row += [int(debug["support_mask"][leg] > 0.5) for leg in LEG_ORDER]
        row += [
            f"{float(rpy[0]):.6f}", f"{float(rpy[1]):.6f}", f"{float(rpy[2]):.6f}",
            f"{abs(float(rpy[0])):.6f}", f"{abs(float(rpy[1])):.6f}", f"{abs(float(rpy[2])):.6f}",
            f"{float(self.target_yaw):.6f}", f"{yaw_error:.6f}", f"{abs(yaw_error):.6f}",
            f"{float(gyro[0]):.6f}", f"{float(gyro[1]):.6f}", f"{float(gyro[2]):.6f}",
            f"{float(acc[0]):.6f}", f"{float(acc[1]):.6f}", f"{float(acc[2]):.6f}",
            f"{self.real_vmc_scale():.6f}", f"{debug['vmc_height']:.6f}", f"{debug['vmc_roll']:.6f}",
            f"{debug['vmc_pitch']:.6f}", f"{debug['yaw_corr']:.6f}",
        ]
        row += [f"{float(x):.6f}" for x in debug["yaw_offsets"]]
        row += [f"{float(x):.6f}" for x in debug["vmc_weight"]]
        row += [
            int(debug["rear_late_guard"]["RR"] > 0.5), int(debug["rear_late_guard"]["RL"] > 0.5),
            f"{debug['rear_late_clearance']['RR']:.6f}", f"{debug['rear_late_clearance']['RL']:.6f}",
            f"{debug['rear_descent_scale']['RR']:.6f}", f"{debug['rear_descent_scale']['RL']:.6f}",
            int(debug["rear_late_window"]["RR"] > 0.5), int(debug["rear_late_window"]["RL"] > 0.5),
            int(debug["rear_late_descending"]["RR"] > 0.5), int(debug["rear_late_descending"]["RL"] > 0.5),
            debug["rear_late_reason"]["RR"], debug["rear_late_reason"]["RL"],
            int(debug["rear_early_guard"]["RR"] > 0.5), int(debug["rear_early_guard"]["RL"] > 0.5),
            f"{debug['rear_early_score']['RR']:.6f}", f"{debug['rear_early_score']['RL']:.6f}",
            debug["early_source"],
            f"{debug['rear_touchdown_weight']['RR']:.6f}", f"{debug['rear_touchdown_weight']['RL']:.6f}",
        ]
        for arr in (
            q_ref,
            q_ref,
            q_cmd,
            q_actual,
            safety["q_actual_for_safety"],
            q_error,
            safety["q_error_for_safety"],
            safety["q_error_real_feedback"],
            safety["dq_actual"],
            safety["tau_est"],
            safety["tau_est_for_safety"],
            safety["current"],
            safety["temp"],
            safety["rate_delta"],
            q_ref_cmd_diff,
        ):
            row += [f"{float(x):.6f}" if math.isfinite(float(x)) else "nan" for x in np.asarray(arr).reshape(12)]
        row += [f"{float(x):.6f}" for x in raw_motor_target]
        row += [
            f"{safety['max_q_error']:.6f}",
            f"{safety['max_tau_est']:.6f}" if math.isfinite(safety["max_tau_est"]) else "nan",
            f"{safety['max_current']:.6f}" if math.isfinite(safety["max_current"]) else "nan",
            f"{safety['max_temp']:.6f}" if math.isfinite(safety["max_temp"]) else "nan",
            safety["safety_warn"],
            int(self.safety_stop),
            f"{safety['rate_clip_ratio']:.6f}",
            f"{safety['torque_clip_ratio']:.6f}",
            int(feedback.valid),
            int(semantic_to_motor_mapping_ok),
            HIP_OUTWARD_SIGNS["FR"], HIP_OUTWARD_SIGNS["FL"], HIP_OUTWARD_SIGNS["RR"], HIP_OUTWARD_SIGNS["RL"], 1,
        ]
        self.csv_writer.writerow(row)
        self.csv_file.flush()

        self.stats["roll_abs"].append(abs(float(rpy[0])) * 180.0 / math.pi)
        self.stats["pitch_abs"].append(abs(float(rpy[1])) * 180.0 / math.pi)
        self.stats["yaw_abs"].append(abs(float(rpy[2])) * 180.0 / math.pi)
        self.stats["yaw_error_abs"].append(abs(yaw_error) * 180.0 / math.pi)
        self.stats["q_error_max"].append(float(safety["max_q_error"]))
        if math.isfinite(safety["max_tau_est"]):
            self.stats["tau_max"].append(float(safety["max_tau_est"]))
        if math.isfinite(safety["max_current"]):
            self.stats["current_max"].append(float(safety["max_current"]))
        if math.isfinite(safety["max_temp"]):
            self.stats["temp_max"].append(float(safety["max_temp"]))
        self.stats["rate_clip"].append(float(safety["rate_clip_ratio"]))
        self.stats["torque_clip"].append(float(safety["torque_clip_ratio"]))
        self.stats["rr_early_guard"].append(float(debug["rear_early_guard"]["RR"]))
        self.stats["rl_early_guard"].append(float(debug["rear_early_guard"]["RL"]))
        self.stats["rr_late_guard"].append(float(debug["rear_late_guard"]["RR"]))
        self.stats["rl_late_guard"].append(float(debug["rear_late_guard"]["RL"]))

    @staticmethod
    def publish_array(pub, arr):
        msg = Float32MultiArray()
        msg.data = np.asarray(arr, dtype=np.float32).reshape(-1).tolist()
        pub.publish(msg)

    def destroy_node(self):
        try:
            self._request_soft_stop("shutdown")
        except Exception:
            pass
        try:
            self._print_summary()
        except Exception as exc:
            self.get_logger().warn(f"summary failed: {exc}")
        try:
            if self.csv_file is not None:
                self.csv_file.flush()
                self.csv_file.close()
        except Exception:
            pass
        try:
            if self.imu is not None:
                self.imu.stop()
        except Exception:
            pass
        try:
            self.motor.close()
            self.http_session.close()
        except Exception:
            pass
        super().destroy_node()

    def _print_summary(self):
        result = "PASS"
        if self.safety_stop:
            result = "FAIL"
        elif pctl(self.stats["q_error_max"], 95) >= float(self.max_q_error_warn):
            result = "CAUTION"
        elif pctl(self.stats["roll_abs"], 95) >= float(self.max_roll_deg_warn) or pctl(self.stats["pitch_abs"], 95) >= float(self.max_pitch_deg_warn):
            result = "CAUTION"
        self.get_logger().warn(
            "[SUMMARY] "
            f"duration={time.time() - self.start_time:.2f}s mode={self.test_mode} send={self.enable_send} csv={self.csv_path} "
            f"roll/pitch/yaw_error max={max(self.stats['roll_abs'] or [0]):.2f}/"
            f"{max(self.stats['pitch_abs'] or [0]):.2f}/{max(self.stats['yaw_error_abs'] or [0]):.2f}deg "
            f"q_error p95/max={pctl(self.stats['q_error_max'],95):.3f}/{max(self.stats['q_error_max'] or [0]):.3f} "
            f"tau p95/p99/max={pctl(self.stats['tau_max'],95):.2f}/{pctl(self.stats['tau_max'],99):.2f}/{max(self.stats['tau_max'] or [float('nan')]):.2f} "
            f"current max={max(self.stats['current_max'] or [float('nan')]):.2f} temp max={max(self.stats['temp_max'] or [float('nan')]):.1f} "
            f"rate_clip={np.mean(self.stats['rate_clip'] or [0]):.3f} torque_clip={np.mean(self.stats['torque_clip'] or [0]):.3f} "
            f"RR/RL early={np.mean(self.stats['rr_early_guard'] or [0]):.3f}/{np.mean(self.stats['rl_early_guard'] or [0]):.3f} "
            f"RR/RL late={np.mean(self.stats['rr_late_guard'] or [0]):.3f}/{np.mean(self.stats['rl_late_guard'] or [0]):.3f} "
            f"stop_reason={self.stop_reason} safety_stop_reason={self.safety_stop_reason} "
            f"REAL_MACHINE_TEST_RESULT={result}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = FanfanCpgVmcV4Node()

    def _handle_sigint(signum, frame):
        node._request_soft_stop("ctrl_c")

    try:
        signal.signal(signal.SIGINT, _handle_sigint)
    except Exception:
        pass

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node._request_soft_stop("ctrl_c")
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
