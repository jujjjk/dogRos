#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fanfan_cpg_vmc_v4_migration_node.py

IsaacLab V4 FastDiagonalTrot + Light VMC + safety_profile
``performance_soft_output_v2_light_vmc_balance_v4`` 的 ROS2 真机迁移外壳。

三个 joint 空间 (见 sim_real_semantic_bridge.py):
    sim_semantic  : V4 core 内部空间 = IsaacLab / golden CSV 空间 (q_ref_sim / q_cmd_final_sim ...)
    real_policy   : JointSemanticMapper.real_to_policy_abs_q_dq() 返回的空间
    real_motor    : 真正发给电机 0x11~0x43 的空间

链路:
    电机反馈 → mapper.real_to_policy_abs_q_dq → real_policy
            → bridge.real_policy_to_sim       → sim_semantic → core
    core.step → q_cmd_final_sim
            → bridge.sim_to_real_policy        → real_policy
            → mapper.policy_target_to_real_target → real_motor → HTTP

sim_compare 永远在 sim_semantic 空间和 golden CSV 比较。
本轮只做空间转换桥，不改 gait / IK / VMC / Kp/Kd / swing height。
"""

from __future__ import annotations

import csv
import math
import os
import time
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node

from .fanfan_v4_migration_core import (
    FanfanV4MigrationCore,
    V4Config,
    CoreInputs,
    SIM_V4_DEFAULT_JOINT_POS_SIM,
    POLICY_JOINT_NAMES,
    POLICY_LEG_ORDER,
    URDF_HIP_OUTWARD_SIGNS,
    MIGRATION_CORE_VERSION,
)
from .semantic_mapper import JointSemanticMapper
from .sim_real_semantic_bridge import SimRealSemanticBridge

try:
    from .motor_state_interface import MotorStateHttpInterface
except Exception:  # pragma: no cover
    MotorStateHttpInterface = None

try:
    from .imu_serial_interface import ImuSerialInterface
except Exception:  # pragma: no cover
    ImuSerialInterface = None

import requests

MOTOR_ID_LABELS = ("0x11", "0x12", "0x13", "0x21", "0x22", "0x23",
                   "0x31", "0x32", "0x33", "0x41", "0x42", "0x43")

VALID_TEST_MODES = ("sim_compare", "air", "touch", "assist", "short_free", "stand_only")

TEST_MODE_VMC_SCALE = {
    "sim_compare": 1.0,
    "air": 0.0,
    "touch": 0.25,
    "assist": 0.5,
    "short_free": 1.0,
    "stand_only": 0.0,
}


class FanfanV4MigrationNode(Node):
    def __init__(self):
        super().__init__("fanfan_cpg_vmc_v4_migration_node")

        self._declare_params()
        self._read_params()

        self.mapper = JointSemanticMapper()
        self.bridge = SimRealSemanticBridge(self.mapper, verbose=False)
        self.bridge.print_bridge(self.get_logger())

        self.core = FanfanV4MigrationCore(
            cfg=V4Config(dt=float(self.dt), use_base_height_estimate=self.use_base_height_estimate,
                         base_height_estimate_m=self.base_height_estimate_m),
            trot_preset=self.trot_preset,
            support_kp_level=self.support_kp_level,
            safety_profile=self.safety_profile,
            default_joint_pos_policy=SIM_V4_DEFAULT_JOINT_POS_SIM.copy(),
        )

        # 默认站姿对比 (在 real_policy 空间比较，才有意义)
        self.sim_v4_default_sim = SIM_V4_DEFAULT_JOINT_POS_SIM.copy()
        self.sim_v4_default_real_policy = self.bridge.sim_to_real_policy(self.sim_v4_default_sim)
        self.mapper_default_real_policy = np.asarray(self.mapper.default_joint_angle, dtype=np.float64).reshape(12)
        self.default_real_policy_diff = self.sim_v4_default_real_policy - self.mapper_default_real_policy
        self.default_real_policy_diff_max = float(np.max(np.abs(self.default_real_policy_diff)))
        self._report_default_pose()

        self.http_session = requests.Session()
        self.motor = None
        self.imu = None
        self.imu_valid = False
        self.motor_feedback_valid = False
        self._init_interfaces()

        self.node_state = "init"
        self.stop_reason = ""
        self.safety_stop_reason = ""
        self.start_time = time.time()
        self.last_time = self.start_time
        self.soft_stop_active = False
        self._soft_stop_from_sim = self.sim_v4_default_sim.copy()
        self._soft_stop_t0 = 0.0
        self.q_cmd_final_sim = self.core.default_joint_pos.copy()
        self._target_yaw = 0.0

        self.golden = None
        if self.test_mode == "sim_compare" or self.sim_compare_csv_path:
            self._load_golden_csv()

        self._csv_file = None
        self._csv_writer = None
        self._csv_header = None
        self._open_csv()

        self._cmp_q_ref_max = []
        self._cmp_q_cmd_max = []

        self._preflight_checks()

        if self.enable_send:
            self._countdown(3)
            if self.test_mode == "stand_only":
                self._send_default_stand()

        self.node_state = "running"
        self.get_logger().info(
            f"[MIGRATION] start mode={self.test_mode} enable_send={self.enable_send} "
            f"dry_run_virtual_feedback={self.dry_run_virtual_feedback} stand_source={self.stand_source} "
            f"bridge_identity={self.bridge.is_identity} core={MIGRATION_CORE_VERSION}"
        )
        self.timer = self.create_timer(self.dt, self._on_timer)

    # ------------------------------------------------------------------
    def _declare_params(self):
        p = self.declare_parameter
        p("enable_send", False)
        p("test_mode", "air")
        p("duration_s", 5.0)
        p("dry_run_virtual_feedback", True)
        p("stand_source", "sim_v4")
        p("require_stand_ready", True)
        p("allow_start_from_any_pose", False)
        p("auto_stop_after_duration", True)
        p("allow_long_free_test", False)
        p("sim_compare_csv_path", "")
        p("control_hz", 50.0)
        p("trot_preset", "balanced")
        p("support_kp_level", "mid_soft")
        p("safety_profile", "performance_soft_output_v2_light_vmc_balance_v4")
        p("motor_base_url", "http://127.0.0.1:8000")
        p("http_timeout", 0.1)
        p("imu_port", "/dev/myimu")
        p("imu_read_hz", 100.0)
        p("require_imu_for_air_send", False)
        p("send_speed", 0.0)
        p("send_torque", 0.0)
        p("max_motor_age_ms", 200.0)
        p("csv_path", "")
        p("stop_roll_deg", 12.0)
        p("stop_pitch_deg", 12.0)
        p("stop_q_error_rad", 0.45)
        p("stop_tau_nm", 17.0)
        p("stop_current_a", 40.0)
        p("stop_temp_c", 75.0)
        p("stand_ready_tol_rad", 0.20)
        p("soft_stop_sec", 3.0)
        p("base_height_estimate_m", 0.288)
        p("use_base_height_estimate", False)

    def _read_params(self):
        g = lambda n: self.get_parameter(n).value
        self.enable_send = bool(g("enable_send"))
        self.test_mode = str(g("test_mode"))
        if self.test_mode not in VALID_TEST_MODES:
            raise RuntimeError(f"test_mode={self.test_mode!r} 不支持，应为 {VALID_TEST_MODES}")
        self.duration_s = float(g("duration_s"))
        self.dry_run_virtual_feedback = bool(g("dry_run_virtual_feedback"))
        self.stand_source = str(g("stand_source"))
        self.require_stand_ready = bool(g("require_stand_ready"))
        self.allow_start_from_any_pose = bool(g("allow_start_from_any_pose"))
        self.auto_stop_after_duration = bool(g("auto_stop_after_duration"))
        self.allow_long_free_test = bool(g("allow_long_free_test"))
        self.sim_compare_csv_path = str(g("sim_compare_csv_path"))
        self.control_hz = float(g("control_hz"))
        self.dt = 1.0 / max(self.control_hz, 1.0)
        self.trot_preset = str(g("trot_preset"))
        self.support_kp_level = str(g("support_kp_level"))
        self.safety_profile = str(g("safety_profile"))
        self.motor_base_url = str(g("motor_base_url")).rstrip("/")
        self.http_timeout = float(g("http_timeout"))
        self.imu_port = str(g("imu_port"))
        self.imu_read_hz = float(g("imu_read_hz"))
        self.require_imu_for_air_send = bool(g("require_imu_for_air_send"))
        self.send_speed = float(g("send_speed"))
        self.send_torque = float(g("send_torque"))
        self.max_motor_age_ms = float(g("max_motor_age_ms"))
        self.csv_path = str(g("csv_path"))
        self.stop_roll_deg = float(g("stop_roll_deg"))
        self.stop_pitch_deg = float(g("stop_pitch_deg"))
        self.stop_q_error_rad = float(g("stop_q_error_rad"))
        self.stop_tau_nm = float(g("stop_tau_nm"))
        self.stop_current_a = float(g("stop_current_a"))
        self.stop_temp_c = float(g("stop_temp_c"))
        self.stand_ready_tol_rad = float(g("stand_ready_tol_rad"))
        self.soft_stop_sec = float(g("soft_stop_sec"))
        self.base_height_estimate_m = float(g("base_height_estimate_m"))
        self.use_base_height_estimate = bool(g("use_base_height_estimate"))

        if self.test_mode == "short_free" and not self.allow_long_free_test and self.duration_s > 3.0:
            self.get_logger().warn(f"short_free duration {self.duration_s:.1f}s > 3.0s，自动限制为 3.0s")
            self.duration_s = 3.0
        if self.enable_send and self.dry_run_virtual_feedback:
            self.get_logger().warn("enable_send=true: 自动禁用 dry_run_virtual_feedback，使用真实反馈做 safety。")
            self.dry_run_virtual_feedback = False

    def _report_default_pose(self):
        self.get_logger().info(
            f"[STAND] stand_source={self.stand_source} "
            f"default_real_policy_diff_max (sim_v4@real_policy vs mapper_default) = "
            f"{self.default_real_policy_diff_max:.4f} rad"
        )
        if self.default_real_policy_diff_max > 0.05 and self.stand_source == "sim_v4":
            self.get_logger().warn(
                "WARNING: bridge.sim_to_real_policy(sim_v4_default) 与 mapper_default 差异 > 0.05 rad。"
                "stand_source=sim_v4 时真机必须先 stand_only 进入 sim_v4 default，不能直接从 mapper default 进入 gait。"
            )

    # ------------------------------------------------------------------
    def _init_interfaces(self):
        if MotorStateHttpInterface is not None:
            try:
                self.motor = MotorStateHttpInterface(base_url=self.motor_base_url, timeout=self.http_timeout)
            except Exception as exc:
                self.get_logger().warn(f"motor interface init failed: {exc}")
                self.motor = None
        if ImuSerialInterface is not None and self._imu_needed():
            try:
                self.imu = ImuSerialInterface(port=self.imu_port, read_hz=self.imu_read_hz)
                self.imu.start()
                self.imu_valid = bool(self.imu.wait_until_ready(timeout=2.0))
                if not self.imu_valid:
                    self.get_logger().warn("IMU 未就绪 (2s 超时)。")
            except Exception as exc:
                self.get_logger().warn(f"IMU init failed: {exc}")
                self.imu = None
                self.imu_valid = False

    def _imu_needed(self) -> bool:
        return self.test_mode in ("touch", "assist", "short_free") or self.enable_send

    # ------------------------------------------------------------------
    def _preflight_checks(self):
        if not self.enable_send:
            return
        if self.test_mode in ("touch", "assist", "short_free") and not self.imu_valid:
            raise RuntimeError(f"enable_send=true 且 test_mode={self.test_mode}: IMU 不可用，拒绝启动。")
        if self.test_mode == "air" and self.require_imu_for_air_send and not self.imu_valid:
            raise RuntimeError("air 发送要求 IMU (require_imu_for_air_send=true)，但 IMU 不可用。")
        feedback = self._read_feedback()
        if self.test_mode in ("assist", "short_free") and not feedback["valid"]:
            raise RuntimeError(f"enable_send=true 且 test_mode={self.test_mode}: 电机反馈不可用，拒绝启动。")
        if self.require_stand_ready and self.test_mode != "stand_only":
            if not feedback["valid"]:
                raise RuntimeError("require_stand_ready=true: 电机反馈不可用，无法确认是否处于 sim_v4 default。")
            # 在 sim_semantic 空间比较 q_actual 与 sim_v4 default
            q_actual_sim = self.bridge.real_policy_to_sim(feedback["q_policy"])
            err = float(np.max(np.abs(q_actual_sim - self.sim_v4_default_sim)))
            if err > self.stand_ready_tol_rad and not self.allow_start_from_any_pose:
                raise RuntimeError(
                    f"require_stand_ready=true: q_actual(sim) 距 sim_v4 default {err:.3f} rad > "
                    f"{self.stand_ready_tol_rad:.3f}。请先 stand_only。"
                )

    def _countdown(self, seconds: int):
        for i in range(seconds, 0, -1):
            self.get_logger().warn(f"[MIGRATION] enable_send=true，{i} 秒后开始发送电机命令...")
            time.sleep(1.0)

    # ------------------------------------------------------------------
    def _read_imu(self) -> dict:
        if self.imu is None:
            return {"valid": False, "rpy": np.zeros(3), "gyro": np.zeros(3), "acc": np.zeros(3)}
        try:
            snap = self.imu.get_latest()
            if not getattr(snap, "valid", False):
                return {"valid": False, "rpy": np.zeros(3), "gyro": np.zeros(3), "acc": np.zeros(3)}
            self.imu_valid = True
            rpy = np.asarray(snap.rpy_deg, dtype=np.float64).reshape(3) * (math.pi / 180.0)
            return {
                "valid": True,
                "rpy": rpy,
                "gyro": np.asarray(snap.gyro_rad_s, dtype=np.float64).reshape(3),
                "acc": np.asarray(snap.acc_g, dtype=np.float64).reshape(3),
            }
        except Exception:
            return {"valid": False, "rpy": np.zeros(3), "gyro": np.zeros(3), "acc": np.zeros(3)}

    def _read_feedback(self) -> dict:
        """返回 real_policy 空间的反馈 (q_policy/dq_policy/torque_policy)。"""
        fallback_real_policy = self.bridge.sim_to_real_policy(self.q_cmd_final_sim)
        empty = {
            "valid": False,
            "q_policy": fallback_real_policy,
            "dq_policy": np.zeros(12),
            "torque_policy": np.full(12, np.nan),
            "current_policy": np.full(12, np.nan),
            "temp_policy": np.full(12, np.nan),
            "max_age_ms": float("inf"),
            "communication_ok": False,
        }
        if self.motor is None:
            return empty
        try:
            snap = self.motor.get_latest()
            q_policy, dq_policy = self.mapper.real_to_policy_abs_q_dq(snap.q_real, snap.dq_real)
            torque_real = np.asarray(snap.torque, dtype=np.float64).reshape(12)
            temp_real = np.asarray(snap.temp, dtype=np.float64).reshape(12)
            torque_policy = torque_real[self.mapper.policy_to_real_index] * self.mapper.joint_sign
            temp_policy = temp_real[self.mapper.policy_to_real_index]
            valid = bool(snap.valid and np.all(np.isfinite(q_policy)))
            self.motor_feedback_valid = valid
            return {
                "valid": valid,
                "q_policy": np.asarray(q_policy, dtype=np.float64),
                "dq_policy": np.asarray(dq_policy, dtype=np.float64),
                "torque_policy": torque_policy,
                "current_policy": np.full(12, np.nan),
                "temp_policy": temp_policy,
                "max_age_ms": float(np.max(snap.age_ms)),
                "communication_ok": valid,
            }
        except Exception as exc:
            now = time.time()
            if now - getattr(self, "_last_fb_warn", 0.0) > 1.0:
                self._last_fb_warn = now
                self.get_logger().warn(f"motor feedback unavailable: {exc}")
            return empty

    def _feedback_to_sim(self, feedback: dict) -> dict:
        """把 real_policy 反馈转成 sim_semantic 空间。"""
        if feedback["valid"]:
            q_actual_sim = self.bridge.real_policy_to_sim(feedback["q_policy"])
            dq_actual_sim = self.bridge.real_policy_dq_to_sim(feedback["dq_policy"])
            tau_sim = self.bridge.real_policy_tau_to_sim(np.nan_to_num(feedback["torque_policy"]))
        else:
            q_actual_sim = self.bridge.real_policy_to_sim(feedback["q_policy"])
            dq_actual_sim = np.zeros(12)
            tau_sim = np.full(12, np.nan)
        return {"q_actual_sim": q_actual_sim, "dq_actual_sim": dq_actual_sim, "tau_sim": tau_sim}

    # ------------------------------------------------------------------
    def _on_timer(self):
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        rel_wall = now - self.start_time

        imu = self._read_imu()
        feedback = self._read_feedback()
        fb_sim = self._feedback_to_sim(feedback)
        if imu["valid"]:
            self._target_yaw = float(imu["rpy"][2])

        if self.auto_stop_after_duration and rel_wall >= self.duration_s and not self.soft_stop_active:
            self._request_soft_stop("duration_reached")

        if self.test_mode == "stand_only":
            self._run_stand_only(now, rel_wall, dt, imu, feedback, fb_sim)
            return

        if self.soft_stop_active:
            core_out = self._soft_stop_step(now)
        else:
            core_out = self._core_step(imu, feedback, fb_sim)
        debug = core_out["debug_info"]
        q_cmd_final_sim = np.asarray(core_out["q_cmd_final_sim"], dtype=np.float64)
        self.q_cmd_final_sim = q_cmd_final_sim.copy()

        # sim_semantic -> real_policy -> real_motor
        q_cmd_real_policy = self.bridge.sim_to_real_policy(q_cmd_final_sim)
        target_real = self.mapper.policy_target_to_real_target(q_cmd_real_policy, clamp=True)
        kp_sim = np.asarray(core_out["kp_sim"], dtype=np.float64)
        kd_sim = np.asarray(core_out["kd_sim"], dtype=np.float64)

        safety = self._safety_check(imu, feedback, fb_sim)

        sent = False
        if self.enable_send:
            sent = self._send_motion(target_real, kp_sim, kd_sim)

        # sim_compare 用核心确定性步时对齐
        rel_core = float(debug.get("relative_time", rel_wall))
        cmp = self._sim_compare(rel_core, debug) if self.golden is not None else None

        self._write_csv_row(rel_core, dt, debug, feedback, fb_sim, imu, safety,
                            q_cmd_real_policy, target_real, sent, cmp)

        if self.soft_stop_active and (now - self._soft_stop_t0) >= self.soft_stop_sec:
            self.get_logger().info(f"[MIGRATION] soft stop done ({self.stop_reason}). shutting down.")
            self._shutdown()

    def _core_step(self, imu: dict, feedback: dict, fb_sim: dict) -> dict:
        vmc_scale = TEST_MODE_VMC_SCALE.get(self.test_mode, 1.0)
        inp = CoreInputs(
            roll=float(imu["rpy"][0]),
            pitch=float(imu["rpy"][1]),
            yaw=float(imu["rpy"][2]),
            gyro=tuple(float(v) for v in imu["gyro"]),
            lin_vel=(0.0, 0.0, 0.0),
            imu_valid=bool(imu["valid"]),
            q_actual_sim=fb_sim["q_actual_sim"] if feedback["valid"] else None,
            dq_actual_sim=fb_sim["dq_actual_sim"] if feedback["valid"] else None,
            tau_sim=fb_sim["tau_sim"] if feedback["valid"] else None,
            feedback_valid=bool(feedback["valid"]),
            base_height_m=None,
            foot_force=None,
            test_mode=self.test_mode,
            dry_run_virtual_feedback=self.dry_run_virtual_feedback,
            vmc_scale=vmc_scale,
        )
        return self.core.step(inp)

    # ------------------------------------------------------------------
    def _run_stand_only(self, now, rel, dt, imu, feedback, fb_sim):
        if not hasattr(self, "_stand_from_sim"):
            self._stand_from_sim = (fb_sim["q_actual_sim"].copy() if feedback["valid"]
                                    else self.sim_v4_default_sim.copy())
            self._stand_t0 = now
        ramp = min(1.0, (now - self._stand_t0) / max(self.soft_stop_sec, 1.0e-3))
        s = ramp * ramp * (3.0 - 2.0 * ramp)
        q_target_sim = self._stand_from_sim + s * (self.sim_v4_default_sim - self._stand_from_sim)
        self.q_cmd_final_sim = q_target_sim.copy()
        q_cmd_real_policy = self.bridge.sim_to_real_policy(q_target_sim)
        target_real = self.mapper.policy_target_to_real_target(q_cmd_real_policy, clamp=True)
        kp = np.array([self.core.cfg.fast_trot_support_hip_kp, self.core.cfg.fast_trot_support_thigh_kp,
                       self.core.cfg.fast_trot_support_calf_kp] * 4)
        kd = np.full(12, self.core.cfg.fast_trot_support_kd)
        sent = False
        if self.enable_send:
            sent = self._send_motion(target_real, kp, kd)
        safety = self._safety_check(imu, feedback, fb_sim)
        debug = self._stand_debug(q_target_sim, kp, kd, "stand_only")
        self._write_csv_row(rel, dt, debug, feedback, fb_sim, imu, safety,
                            q_cmd_real_policy, target_real, sent, None)
        if rel >= self.duration_s:
            self.get_logger().info("[MIGRATION] stand_only done.")
            self._shutdown()

    def _stand_debug(self, q_sim, kp, kd, node_state):
        z = self.core._forward_sagittal(q_sim[1::3], q_sim[2::3])[1]
        clearance = z - self.core.default_foot_z
        z4 = np.zeros(4)
        z12 = np.zeros(12)
        return {
            "relative_time": 0.0, "phase": 0.0, "warmup": 0.0, "duty_factor": self.core.cfg.fast_trot_duty_factor,
            "active_swing_pair": 0, "support_pair": 0,
            "leg_phase": z4.copy(), "swing_progress": z4.copy(),
            "swing_mask": np.zeros(4, dtype=bool), "support_mask": np.ones(4, dtype=bool),
            "phase_to_switch": 0.0, "phase_switch_guard_active": False, "phase_switch_guard_strength": 0.0,
            "q_cpg_sim": q_sim.copy(), "q_ref_sim": q_sim.copy(),
            "q_vmc_delta_sim": z12.copy(), "q_cmd_raw_sim": q_sim.copy(),
            "q_cmd_final_sim": q_sim.copy(), "q_actual_sim": q_sim.copy(),
            "q_error_sim": z12.copy(), "q_ref_cmd_diff_sim": z12.copy(),
            "kp_sim": kp.copy(), "kd_sim": kd.copy(),
            "fk_clearance_ref": clearance.copy(), "fk_clearance_cmd": clearance.copy(),
            "fk_clearance_actual": clearance.copy(), "predicted_foot_height": clearance.copy(),
            "height_source": "unavailable", "early_contact_source": "unavailable", "real_vmc_scale": 0.0,
            "vmc_weight": z4.copy(), "vmc_height_corr_z": 0.0, "vmc_roll_corr_z": 0.0, "vmc_pitch_corr_z": 0.0,
            "vmc_foot_z_offset": z4.copy(), "vmc_foot_x_offset": z4.copy(), "vmc_foot_y_offset": z4.copy(),
            "vmc_foot_x_corr": 0.0, "vmc_foot_y_corr": 0.0,
            "yaw_target": 0.0, "yaw_error": 0.0, "yaw_corr_hip_raw": 0.0, "yaw_corr_hip": 0.0,
            "yaw_hip_offset": z4.copy(), "yaw_hip_rate_limited": z4.copy(),
            "rear_preswing_unload_gate": z4.copy(), "rear_preswing_vmc_fade": np.ones(4),
            "rear_touchdown_vmc_ramp_weight": z4.copy(),
            "phase_switch_vmc_weight_scale_applied": 1.0, "phase_switch_yaw_weight_scale_applied": 1.0,
            "phase_switch_kp_scale_applied": 1.0,
            "rear_late_swing_window_active": np.zeros(4, dtype=bool),
            "rear_late_swing_guard_active": np.zeros(4, dtype=bool),
            "rear_late_swing_clearance_offset": z4.copy(), "rear_late_swing_height": z4.copy(),
            "rear_late_swing_height_error": z4.copy(), "rear_late_swing_descent_scale_applied": np.ones(4),
            "rear_early_contact_guard_active": np.zeros(4, dtype=bool), "rear_early_contact_score": z4.copy(),
            "rear_early_contact_relief_offset": z4.copy(),
            "rear_touchdown_kp_scale": np.ones(4), "rear_early_contact_kp_scale": np.ones(4),
            "rear_touchdown_kp_ramp_weight": z4.copy(), "guard_kp_scale": z12.copy(),
            "tau_est": z12.copy(), "rate_limited_delta": z12.copy(),
            "rate_clip_ratio": 0.0, "torque_clip_ratio": 0.0, "support_preload_delta_z": z4.copy(),
            "support_preload_gate": z4.copy(), "preload_gate": z4.copy(), "early_stance_gate": z4.copy(),
            "frequency": 0.0, "stride": 0.0, "swing_height": 0.0, "node_state": node_state,
        }

    # ------------------------------------------------------------------
    def _request_soft_stop(self, reason: str):
        if self.soft_stop_active:
            return
        self.soft_stop_active = True
        self._soft_stop_from_sim = self.q_cmd_final_sim.copy()
        self._soft_stop_t0 = time.time()
        if not self.stop_reason:
            self.stop_reason = reason
        self.get_logger().warn(f"[MIGRATION] soft stop requested: {reason}")

    def _soft_stop_step(self, now):
        ramp = min(1.0, (now - self._soft_stop_t0) / max(self.soft_stop_sec, 1.0e-3))
        s = ramp * ramp * (3.0 - 2.0 * ramp)
        q_target_sim = self._soft_stop_from_sim + s * (self.sim_v4_default_sim - self._soft_stop_from_sim)
        kp = np.array([self.core.cfg.fast_trot_support_hip_kp, self.core.cfg.fast_trot_support_thigh_kp,
                       self.core.cfg.fast_trot_support_calf_kp] * 4)
        kd = np.full(12, self.core.cfg.fast_trot_support_kd)
        debug = self._stand_debug(q_target_sim, kp, kd, "soft_stop")
        return {"q_cmd_final_sim": q_target_sim, "kp_sim": kp, "kd_sim": kd, "debug_info": debug}

    # ------------------------------------------------------------------
    def _safety_check(self, imu: dict, feedback: dict, fb_sim: dict) -> dict:
        roll_deg = abs(float(imu["rpy"][0]) * 180.0 / math.pi)
        pitch_deg = abs(float(imu["rpy"][1]) * 180.0 / math.pi)

        use_virtual = (not self.enable_send) and self.dry_run_virtual_feedback
        if feedback["valid"] and not use_virtual:
            q_actual_sim = fb_sim["q_actual_sim"]
            tau = fb_sim["tau_sim"]
            current = feedback["current_policy"]
            temp = feedback["temp_policy"]
            comm_ok = feedback["communication_ok"]
        else:
            q_actual_sim = self.q_cmd_final_sim.copy()
            tau = np.zeros(12)
            current = np.zeros(12)
            temp = np.zeros(12)
            comm_ok = True

        q_error_sim = self.q_cmd_final_sim - q_actual_sim
        max_q_error = float(np.max(np.abs(q_error_sim)))
        max_tau = float(np.nanmax(np.abs(tau))) if np.any(np.isfinite(tau)) else 0.0
        max_current = float(np.nanmax(np.abs(current))) if np.any(np.isfinite(current)) else 0.0
        max_temp = float(np.nanmax(temp)) if np.any(np.isfinite(temp)) else 0.0

        stop = False
        reason = ""
        if imu["valid"]:
            if roll_deg > self.stop_roll_deg:
                stop, reason = True, f"roll {roll_deg:.1f}deg"
            elif pitch_deg > self.stop_pitch_deg:
                stop, reason = True, f"pitch {pitch_deg:.1f}deg"
        if not stop and not use_virtual and feedback["valid"]:
            if max_q_error > self.stop_q_error_rad:
                stop, reason = True, f"q_error {max_q_error:.3f}rad"
            elif max_tau > self.stop_tau_nm:
                stop, reason = True, f"tau {max_tau:.1f}Nm"
            elif np.isfinite(max_current) and max_current > self.stop_current_a:
                stop, reason = True, f"current {max_current:.1f}A"
            elif np.isfinite(max_temp) and max_temp > self.stop_temp_c:
                stop, reason = True, f"temp {max_temp:.1f}C"
            elif feedback["max_age_ms"] > self.max_motor_age_ms:
                stop, reason = True, f"comm_timeout age {feedback['max_age_ms']:.0f}ms"

        if stop and not self.soft_stop_active:
            self.safety_stop_reason = reason
            self.stop_reason = reason
            self._request_soft_stop(reason)

        return {
            "q_actual_sim": q_actual_sim,
            "q_error_sim": q_error_sim,
            "roll_deg": roll_deg,
            "pitch_deg": pitch_deg,
            "max_q_error": max_q_error,
            "max_tau_est": max_tau,
            "max_current": max_current,
            "max_temp": max_temp,
            "communication_ok": comm_ok,
            "safety_warn": False,
            "safety_stop": stop or self.soft_stop_active,
        }

    # ------------------------------------------------------------------
    def _send_motion(self, target_real, kp_policy, kd_policy) -> bool:
        kp_real = np.zeros(12)
        kd_real = np.zeros(12)
        kp_real[self.mapper.policy_to_real_index] = kp_policy
        kd_real[self.mapper.policy_to_real_index] = kd_policy
        items = []
        for i, mid in enumerate(self.mapper.get_real_motor_ids()):
            items.append({
                "motor_id": int(mid),
                "position": float(target_real[i]),
                "speed": float(self.send_speed),
                "torque": float(self.send_torque),
                "kp": float(kp_real[i]),
                "kd": float(kd_real[i]),
            })
        try:
            r = self.http_session.post(
                f"{self.motor_base_url}/api/rs04/motion_batch_fast",
                json={"items": items, "enable_first": False, "stop_first": False},
                timeout=self.http_timeout,
            )
            if r.status_code != 200:
                self.get_logger().warn(f"[SEND] HTTP {r.status_code}: {r.text}")
                return False
            return True
        except Exception as exc:
            self.get_logger().warn(f"[SEND] failed: {exc}")
            return False

    def _send_default_stand(self):
        # 用 bridge 把 sim_v4 default 转到 real_policy 再发，不要直接当 mapper policy 发
        q_real_policy = self.bridge.sim_to_real_policy(self.sim_v4_default_sim)
        default_real = self.mapper.policy_target_to_real_target(q_real_policy, clamp=True)
        kp = [float(self.core.cfg.fast_trot_support_hip_kp), float(self.core.cfg.fast_trot_support_thigh_kp),
              float(self.core.cfg.fast_trot_support_calf_kp)] * 4
        kd = [float(self.core.cfg.fast_trot_support_kd)] * 12
        items = []
        for i, mid in enumerate(self.mapper.get_real_motor_ids()):
            items.append({"motor_id": int(mid), "position": float(default_real[i]),
                          "speed": 0.0, "torque": 0.0, "kp": float(kp[i]), "kd": float(kd[i])})
        try:
            r = self.http_session.post(
                f"{self.motor_base_url}/api/rs04/motion_mode_run_batch",
                json={"items": items, "enable_first": True, "stop_first": False},
                timeout=max(self.http_timeout, 0.5),
            )
            if r.status_code != 200:
                raise RuntimeError(f"default stand HTTP {r.status_code}: {r.text}")
        except Exception as exc:
            self.get_logger().warn(f"[SEND] default stand failed: {exc}")

    # ------------------------------------------------------------------
    def _load_golden_csv(self):
        path = self.sim_compare_csv_path
        if not path:
            self.get_logger().warn("test_mode=sim_compare 但 sim_compare_csv_path 为空，跳过对齐。")
            return
        path = os.path.expanduser(path)
        if not os.path.isfile(path):
            self.get_logger().error(f"golden CSV 不存在: {path}")
            return
        times, rows = [], []
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    times.append(float(row.get("time", "nan")))
                except (TypeError, ValueError):
                    times.append(float("nan"))
                rows.append(row)
        self.golden = {"times": np.asarray(times, dtype=np.float64), "rows": rows}
        self.get_logger().info(f"[SIM_COMPARE] loaded golden CSV {path} rows={len(rows)}")

    def _golden_nearest(self, rel: float):
        times = self.golden["times"]
        idx = int(np.nanargmin(np.abs(times - rel)))
        return idx, self.golden["rows"][idx], float(times[idx])

    @staticmethod
    def _golden_vec(row: dict, prefix: str):
        out = []
        for i in range(12):
            v = row.get(f"{prefix}_{i}")
            out.append(float(v) if v not in (None, "") else float("nan"))
        return np.asarray(out, dtype=np.float64)

    def _sim_compare(self, rel: float, debug: dict) -> dict:
        idx, grow, gtime = self._golden_nearest(rel)
        # golden q_ref_* (= simulator_q_ref) / q_cmd_final_* 都是 sim_semantic 空间
        g_q_ref = self._golden_vec(grow, "q_ref")
        g_q_cmd = self._golden_vec(grow, "q_cmd_final")
        try:
            g_phase = float(grow.get("base_phase", "nan"))
        except (TypeError, ValueError):
            g_phase = float("nan")
        q_ref_sim = debug["q_ref_sim"]
        q_cmd_sim = debug["q_cmd_final_sim"]
        q_ref_abs_diff = np.abs(q_ref_sim - g_q_ref)
        q_cmd_abs_diff = np.abs(q_cmd_sim - g_q_cmd)
        q_ref_abs_diff_max = float(np.nanmax(q_ref_abs_diff))
        q_cmd_abs_diff_max = float(np.nanmax(q_cmd_abs_diff))
        self._cmp_q_ref_max.append(q_ref_abs_diff_max)
        self._cmp_q_cmd_max.append(q_cmd_abs_diff_max)
        return {
            "q_ref_sim_compare": g_q_ref,
            "q_cmd_sim_compare": g_q_cmd,
            "q_ref_abs_diff": q_ref_abs_diff,
            "q_cmd_abs_diff": q_cmd_abs_diff,
            "q_ref_abs_diff_max": q_ref_abs_diff_max,
            "q_cmd_abs_diff_max": q_cmd_abs_diff_max,
            "phase_sim_compare": g_phase,
            "phase_diff": float(debug["phase"] - g_phase) if math.isfinite(g_phase) else float("nan"),
        }

    # ------------------------------------------------------------------
    def _open_csv(self):
        path = self.csv_path
        if not path:
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = f"fanfan_v4_migration_{self.test_mode}_{ts}.csv"
        path = os.path.expanduser(path)
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        self._csv_path_out = path
        self._csv_file = open(path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_file)
        self.get_logger().info(f"[CSV] writing -> {path}")

    def _build_row(self, rel, dt, debug, feedback, fb_sim, imu, safety, q_cmd_real_policy, target_real, sent, cmp):
        cols = []

        def s(name, value):
            cols.append((name, value))

        def vec_joint(name, arr):
            arr = np.asarray(arr, dtype=np.float64).reshape(12)
            for i, jn in enumerate(POLICY_JOINT_NAMES):
                cols.append((f"{name}_{jn}", float(arr[i])))

        def vec_leg(name, arr):
            arr = np.asarray(arr, dtype=np.float64).reshape(4)
            for i, ln in enumerate(POLICY_LEG_ORDER):
                cols.append((f"{name}_{ln}", float(arr[i])))

        q_ref_sim = np.asarray(debug["q_ref_sim"], dtype=np.float64)
        q_cmd_final_sim = np.asarray(debug["q_cmd_final_sim"], dtype=np.float64)
        q_actual_sim = np.asarray(safety["q_actual_sim"], dtype=np.float64)
        q_error_sim = np.asarray(safety["q_error_sim"], dtype=np.float64)

        # real_policy 空间 (经 bridge 转换)
        q_ref_real_policy = self.bridge.sim_to_real_policy(q_ref_sim)
        q_cmd_final_real_policy = np.asarray(q_cmd_real_policy, dtype=np.float64)
        q_actual_real_policy = (feedback["q_policy"] if feedback["valid"]
                                else self.bridge.sim_to_real_policy(q_actual_sim))
        q_error_real_policy = q_cmd_final_real_policy - q_actual_real_policy

        # --- 基础 ---
        s("time", time.time())
        s("relative_time", rel)
        s("dt", dt)
        s("test_mode", self.test_mode)
        s("enable_send", int(self.enable_send))
        s("node_state", debug.get("node_state", self.node_state if not self.soft_stop_active else "soft_stop"))
        s("phase", debug["phase"])
        s("phase_sim_compare", cmp["phase_sim_compare"] if cmp else float("nan"))
        s("phase_diff", cmp["phase_diff"] if cmp else float("nan"))
        s("active_swing_pair", debug["active_swing_pair"])
        s("support_pair", debug["support_pair"])
        vec_leg("leg_phase", debug["leg_phase"])
        vec_leg("swing_progress", debug["swing_progress"])
        vec_leg("swing_mask", np.asarray(debug["swing_mask"], dtype=np.float64))
        vec_leg("support_mask", np.asarray(debug["support_mask"], dtype=np.float64))

        # --- bridge ---
        s("bridge_enabled", int(self.bridge.enabled))
        s("bridge_is_identity", int(self.bridge.is_identity))
        s("bridge_roundtrip_error_max", self.bridge.roundtrip_error_max)
        for i, jn in enumerate(POLICY_JOINT_NAMES):
            s(f"sim_to_real_index_{jn}", int(self.bridge.sim_to_real_index[i]))
        for i, jn in enumerate(POLICY_JOINT_NAMES):
            s(f"sim_to_real_sign_{jn}", float(self.bridge.sim_to_real_sign[i]))
        for i, jn in enumerate(POLICY_JOINT_NAMES):
            s(f"sim_to_real_offset_{jn}", float(self.bridge.sim_to_real_offset[i]))

        # --- 迁移状态 ---
        s("direct_migration_enabled", 1)
        s("migration_core_version", MIGRATION_CORE_VERSION)
        s("stand_source", self.stand_source)

        # --- default stand (sim 与 real_policy 两个版本) ---
        vec_joint("sim_v4_default_sim", self.sim_v4_default_sim)
        vec_joint("sim_v4_default_real_policy", self.sim_v4_default_real_policy)
        vec_joint("mapper_default_real_policy", self.mapper_default_real_policy)
        vec_joint("default_real_policy_diff", self.default_real_policy_diff)
        s("default_real_policy_diff_max", self.default_real_policy_diff_max)

        # --- reference / output (sim_semantic) ---
        vec_joint("q_cpg_sim", debug["q_cpg_sim"])
        vec_joint("q_ref_sim", q_ref_sim)
        vec_joint("q_vmc_delta_sim", debug["q_vmc_delta_sim"])
        vec_joint("q_cmd_raw_sim", debug["q_cmd_raw_sim"])
        vec_joint("q_cmd_final_sim", q_cmd_final_sim)
        vec_joint("q_actual_sim", q_actual_sim)
        vec_joint("q_error_sim", q_error_sim)
        vec_joint("q_ref_cmd_diff_sim", debug["q_ref_cmd_diff_sim"])

        # --- reference / output (real_policy) ---
        vec_joint("q_ref_real_policy", q_ref_real_policy)
        vec_joint("q_cmd_final_real_policy", q_cmd_final_real_policy)
        vec_joint("q_actual_real_policy", q_actual_real_policy)
        vec_joint("q_error_real_policy", q_error_real_policy)

        # --- sim compare (sim_semantic 空间对 golden) ---
        vec_joint("q_ref_sim_compare", cmp["q_ref_sim_compare"] if cmp else np.full(12, np.nan))
        vec_joint("q_cmd_sim_compare", cmp["q_cmd_sim_compare"] if cmp else np.full(12, np.nan))
        vec_joint("q_ref_abs_diff", cmp["q_ref_abs_diff"] if cmp else np.full(12, np.nan))
        vec_joint("q_cmd_abs_diff", cmp["q_cmd_abs_diff"] if cmp else np.full(12, np.nan))
        s("q_ref_abs_diff_max", cmp["q_ref_abs_diff_max"] if cmp else float("nan"))
        s("q_cmd_abs_diff_max", cmp["q_cmd_abs_diff_max"] if cmp else float("nan"))

        # --- FK / clearance ---
        vec_leg("fk_clearance_ref", debug["fk_clearance_ref"])
        vec_leg("fk_clearance_cmd", debug["fk_clearance_cmd"])
        vec_leg("fk_clearance_actual", debug["fk_clearance_actual"])
        vec_leg("predicted_foot_height", debug["predicted_foot_height"])

        # --- VMC ---
        s("height_source", debug["height_source"])
        s("early_contact_source", debug["early_contact_source"])
        s("real_vmc_scale", debug["real_vmc_scale"])
        s("vmc_height_corr_z", debug["vmc_height_corr_z"])
        s("vmc_roll_corr_z", debug["vmc_roll_corr_z"])
        s("vmc_pitch_corr_z", debug["vmc_pitch_corr_z"])
        s("yaw_corr_hip", debug["yaw_corr_hip"])
        vec_leg("yaw_hip_offset", debug["yaw_hip_offset"])
        vec_leg("vmc_weight", debug["vmc_weight"])

        # --- rear guard (RR=2, RL=3) ---
        rr, rl = 2, 3
        late_win = np.asarray(debug["rear_late_swing_window_active"], dtype=np.float64)
        late_act = np.asarray(debug["rear_late_swing_guard_active"], dtype=np.float64)
        late_off = np.asarray(debug["rear_late_swing_clearance_offset"], dtype=np.float64)
        descent = np.asarray(debug["rear_late_swing_descent_scale_applied"], dtype=np.float64)
        early_act = np.asarray(debug["rear_early_contact_guard_active"], dtype=np.float64)
        early_score = np.asarray(debug["rear_early_contact_score"], dtype=np.float64)
        td_ramp = np.asarray(debug["rear_touchdown_kp_ramp_weight"], dtype=np.float64)
        s("rear_late_swing_window_active_RR", float(late_win[rr]))
        s("rear_late_swing_window_active_RL", float(late_win[rl]))
        s("rear_late_swing_guard_active_RR", float(late_act[rr]))
        s("rear_late_swing_guard_active_RL", float(late_act[rl]))
        s("rear_late_swing_clearance_offset_RR", float(late_off[rr]))
        s("rear_late_swing_clearance_offset_RL", float(late_off[rl]))
        s("rear_late_swing_descent_scale_applied_RR", float(descent[rr]))
        s("rear_late_swing_descent_scale_applied_RL", float(descent[rl]))
        s("rear_early_contact_guard_active_RR", float(early_act[rr]))
        s("rear_early_contact_guard_active_RL", float(early_act[rl]))
        s("rear_early_contact_score_RR", float(early_score[rr]))
        s("rear_early_contact_score_RL", float(early_score[rl]))
        s("rear_touchdown_kp_ramp_weight_RR", float(td_ramp[rr]))
        s("rear_touchdown_kp_ramp_weight_RL", float(td_ramp[rl]))

        # --- gains / safety (kp/kd 是增益幅值, sim==real_policy) ---
        vec_joint("kp", debug["kp_sim"])
        vec_joint("kd", debug["kd_sim"])
        vec_joint("rate_limited_delta", debug["rate_limited_delta"])
        vec_joint("tau_est", debug["tau_est"])
        vec_joint("current", feedback["current_policy"])
        vec_joint("temp", feedback["temp_policy"])
        s("max_q_error", safety["max_q_error"])
        s("max_tau_est", safety["max_tau_est"])
        s("max_current", safety["max_current"])
        s("max_temp", safety["max_temp"])
        s("rate_clip_ratio", debug["rate_clip_ratio"])
        s("torque_clip_ratio", debug["torque_clip_ratio"])
        s("safety_warn", int(safety["safety_warn"]))
        s("safety_stop", int(safety["safety_stop"]))
        s("stop_reason", self.stop_reason)
        s("safety_stop_reason", self.safety_stop_reason)
        s("communication_ok", int(safety["communication_ok"]))

        # --- mapping (real_motor) ---
        target_real = np.asarray(target_real, dtype=np.float64).reshape(12)
        for i, lab in enumerate(MOTOR_ID_LABELS):
            s(f"raw_motor_target_{lab}", float(target_real[i]))
        remap = self.mapper.policy_target_to_real_target(q_cmd_final_real_policy, clamp=True)
        s("semantic_to_motor_mapping_ok", int(bool(np.allclose(target_real, remap, atol=1.0e-6))))
        s("sent", int(bool(sent)))
        return cols

    def _write_csv_row(self, rel, dt, debug, feedback, fb_sim, imu, safety, q_cmd_real_policy, target_real, sent, cmp):
        cols = self._build_row(rel, dt, debug, feedback, fb_sim, imu, safety, q_cmd_real_policy, target_real, sent, cmp)
        if self._csv_header is None:
            self._csv_header = [c[0] for c in cols]
            self._csv_writer.writerow(self._csv_header)
        self._csv_writer.writerow([c[1] for c in cols])
        self._csv_file.flush()

    # ------------------------------------------------------------------
    def _shutdown(self):
        if self.node_state == "stopped":
            return
        self.node_state = "stopped"
        try:
            self.timer.cancel()
        except Exception:
            pass
        self._print_summary()
        try:
            if self._csv_file:
                self._csv_file.close()
        except Exception:
            pass
        rclpy.shutdown()

    def _print_summary(self):
        self.get_logger().info("=" * 70)
        self.get_logger().info(f"[SUMMARY] mode={self.test_mode} enable_send={self.enable_send} csv={self._csv_path_out}")
        self.get_logger().info(
            f"[BRIDGE] is_identity={self.bridge.is_identity} roundtrip_error_max={self.bridge.roundtrip_error_max:.3e} "
            f"sim_to_real_sign={self.bridge.sim_to_real_sign.tolist()}"
        )
        self.get_logger().info(
            f"[SUMMARY] stand_source={self.stand_source} default_real_policy_diff_max="
            f"{self.default_real_policy_diff_max:.4f} rad"
        )
        self.get_logger().info(
            f"[SUMMARY] hip_outward_signs FR/FL/RR/RL = {URDF_HIP_OUTWARD_SIGNS.tolist()} (use_urdf=True)"
        )
        if self.stop_reason:
            self.get_logger().info(f"[SUMMARY] stop_reason={self.stop_reason}")
        if self._cmp_q_ref_max:
            qref = float(np.max(self._cmp_q_ref_max))
            qcmd = float(np.max(self._cmp_q_cmd_max))
            ref_grade = "PASS" if qref < 0.03 else ("CAUTION" if qref <= 0.08 else "FAIL")
            cmd_grade = "PASS" if qcmd < 0.10 else ("CAUTION" if qcmd <= 0.25 else "FAIL")
            self.get_logger().info(f"[SIM_COMPARE] (sim_semantic 空间) q_ref_abs_diff_max={qref:.4f} rad -> {ref_grade}")
            self.get_logger().info(f"[SIM_COMPARE] (sim_semantic 空间) q_cmd_abs_diff_max={qcmd:.4f} rad -> {cmd_grade}")
            if ref_grade != "PASS" or cmd_grade == "FAIL":
                self.get_logger().warn(
                    "[SIM_COMPARE] 差异来源: default stand / phase·warmup / support preload / VMC input / "
                    "contact force / rate limiter / torque backoff / kp·kd / joint sign·order"
                )
            self.get_logger().warn("[SIM_COMPARE] 在 sim_compare 通过前，不允许上 enable_send=true。")
        self.get_logger().info("=" * 70)


def main(args=None):
    rclpy.init(args=args)
    node = FanfanV4MigrationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().warn("[MIGRATION] Ctrl+C -> soft stop")
        node._request_soft_stop("ctrl_c")
        t0 = time.time()
        while rclpy.ok() and (time.time() - t0) < node.soft_stop_sec + 0.5:
            rclpy.spin_once(node, timeout_sec=node.dt)
    finally:
        if rclpy.ok():
            try:
                node._shutdown()
            except Exception:
                pass


if __name__ == "__main__":
    main()
