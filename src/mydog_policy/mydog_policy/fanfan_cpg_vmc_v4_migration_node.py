#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fanfan_cpg_vmc_v4_migration_node.py

IsaacLab V4 FastDiagonalTrot + Light VMC + safety_profile
``performance_soft_output_v2_light_vmc_balance_v4`` 的 ROS2 真机迁移外壳。

这个节点只做外壳，不重新写 gait：
    读取 ROS2 参数
    读取 IMU
    读取电机反馈
    调用 fanfan_v4_migration_core.FanfanV4MigrationCore.step()
    用 JointSemanticMapper 把 policy q_cmd_final 映射到真机电机目标
    通过已有 HTTP 接口 (/api/rs04/motion_batch_fast) 发送电机
    写 CSV
    执行 safety stop / soft stop
    打印 summary / sim_compare PASS/CAUTION/FAIL

验收标准：ROS2 dry-run CSV 与 IsaacLab V4 golden CSV 对齐 (q_ref / q_cmd_final /
phase / swing_mask / support_mask / rear guard / kp/kd / rate filter)。
sim_compare 没通过前不允许上 enable_send=true。
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
    SIM_V4_DEFAULT_JOINT_POS_POLICY,
    POLICY_JOINT_NAMES,
    POLICY_LEG_ORDER,
    URDF_HIP_OUTWARD_SIGNS,
    MIGRATION_CORE_VERSION,
)
from .semantic_mapper import JointSemanticMapper

try:
    from .motor_state_interface import MotorStateHttpInterface, MotorSnapshot
except Exception:  # pragma: no cover
    MotorStateHttpInterface = None
    MotorSnapshot = None

try:
    from .imu_serial_interface import ImuSerialInterface
except Exception:  # pragma: no cover
    ImuSerialInterface = None

import requests

# 真机电机 ID 顺序 (real_motor_ids)，与 semantic_mapper / motor_state_interface 一致
MOTOR_ID_LABELS = ("0x11", "0x12", "0x13", "0x21", "0x22", "0x23",
                   "0x31", "0x32", "0x33", "0x41", "0x42", "0x43")

VALID_TEST_MODES = ("sim_compare", "air", "touch", "assist", "short_free", "stand_only")

# 每个 test_mode 的 VMC 比例 (CPG reference 不变, 只缩放 VMC correction)
TEST_MODE_VMC_SCALE = {
    "sim_compare": 1.0,   # sim_compare 完整复现仿真
    "air": 0.0,           # 架空: VMC 默认 0
    "touch": 0.25,        # 脚尖轻触: 小比例
    "assist": 0.5,        # 半承重手扶: 中等
    "short_free": 1.0,    # 极短放手: 满
    "stand_only": 0.0,
}


def _wrap_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class FanfanV4MigrationNode(Node):
    def __init__(self):
        super().__init__("fanfan_cpg_vmc_v4_migration_node")

        self._declare_params()
        self._read_params()

        self.mapper = JointSemanticMapper()
        self.core = FanfanV4MigrationCore(
            cfg=V4Config(dt=float(self.dt)),
            trot_preset=self.trot_preset,
            support_kp_level=self.support_kp_level,
            safety_profile=self.safety_profile,
            default_joint_pos_policy=self._resolve_stand_default(),
        )

        # 默认站姿对比 (sim_v4 vs mapper)
        self.default_cmp = FanfanV4MigrationCore.default_pose_comparison(self.mapper.default_joint_angle)
        self._report_default_pose()

        # 接口
        self.http_session = requests.Session()
        self.motor: Optional[object] = None
        self.imu: Optional[object] = None
        self.imu_valid = False
        self.motor_feedback_valid = False
        self._init_interfaces()

        # 状态
        self.node_state = "init"
        self.stop_reason = ""
        self.safety_stop_reason = ""
        self.start_time = time.time()
        self.last_time = self.start_time
        self.soft_stop_active = False
        self._soft_stop_from = SIM_V4_DEFAULT_JOINT_POS_POLICY.copy()
        self._soft_stop_t0 = 0.0
        self.q_cmd_final = self.core.default_joint_pos.copy()
        self._last_send_log = 0.0
        self._target_yaw = 0.0

        # sim_compare golden
        self.golden = None
        if self.test_mode == "sim_compare" or self.sim_compare_csv_path:
            self._load_golden_csv()

        # CSV
        self._csv_file = None
        self._csv_writer = None
        self._csv_header = None
        self._open_csv()

        # sim_compare 统计
        self._cmp_q_ref_max = []
        self._cmp_q_cmd_max = []

        # 安全启动检查
        self._preflight_checks()

        # 倒计时 (enable_send)
        if self.enable_send:
            self._countdown(3)
            if self.test_mode == "stand_only":
                self._send_default_stand()

        self.node_state = "running"
        self.get_logger().info(
            f"[MIGRATION] start mode={self.test_mode} enable_send={self.enable_send} "
            f"dry_run_virtual_feedback={self.dry_run_virtual_feedback} stand_source={self.stand_source} "
            f"duration={self.duration_s:.1f}s core={MIGRATION_CORE_VERSION}"
        )

        self.timer = self.create_timer(self.dt, self._on_timer)

    # ------------------------------------------------------------------
    # 参数
    # ------------------------------------------------------------------
    def _declare_params(self):
        p = self.declare_parameter
        p("enable_send", False)
        p("test_mode", "air")
        p("duration_s", 5.0)
        p("dry_run_virtual_feedback", True)
        p("stand_source", "sim_v4")           # sim_v4 | mapper | fallback
        p("require_stand_ready", True)
        p("allow_start_from_any_pose", False)
        p("auto_stop_after_duration", True)
        p("allow_long_free_test", False)
        p("sim_compare_csv_path", "")
        p("control_hz", 50.0)
        p("trot_preset", "balanced")
        p("support_kp_level", "mid_soft")
        p("safety_profile", "performance_soft_output_v2_light_vmc_balance_v4")
        # 接口
        p("motor_base_url", "http://127.0.0.1:8000")
        p("http_timeout", 0.1)
        p("imu_port", "/dev/myimu")
        p("imu_read_hz", 100.0)
        p("require_imu_for_air_send", False)
        p("send_speed", 0.0)
        p("send_torque", 0.0)
        p("max_motor_age_ms", 200.0)
        p("csv_path", "")
        # 安全阈值
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

        # short_free 安全：默认限制 duration<=3s
        if self.test_mode == "short_free" and not self.allow_long_free_test:
            if self.duration_s > 3.0:
                self.get_logger().warn(
                    f"short_free duration {self.duration_s:.1f}s > 3.0s，自动限制为 3.0s "
                    f"(设置 allow_long_free_test=true 可放开)"
                )
                self.duration_s = 3.0

        # enable_send=true 自动禁用虚拟反馈
        if self.enable_send and self.dry_run_virtual_feedback:
            self.get_logger().warn("enable_send=true: 自动禁用 dry_run_virtual_feedback，使用真实反馈做 safety。")
            self.dry_run_virtual_feedback = False

    def _resolve_stand_default(self) -> np.ndarray:
        if self.stand_source == "sim_v4":
            return SIM_V4_DEFAULT_JOINT_POS_POLICY.copy()
        if self.stand_source == "mapper":
            return np.asarray(self.mapper.default_joint_angle, dtype=np.float64).reshape(12).copy()
        # fallback: sim_v4
        return SIM_V4_DEFAULT_JOINT_POS_POLICY.copy()

    def _report_default_pose(self):
        d = self.default_cmp
        diff_max = d["default_policy_diff_max"]
        self.get_logger().info(
            f"[STAND] stand_source={self.stand_source} "
            f"max(abs(mapper_default - sim_v4_default))={diff_max:.4f} rad"
        )
        if diff_max > 0.05 and self.stand_source == "sim_v4":
            self.get_logger().warn(
                "WARNING: mapper default 与 sim_v4 default 差异 > 0.05 rad。"
                "stand_source=sim_v4 时真机必须先 stand_only 进入 sim_v4 default，"
                "不能直接从 mapper default 进入 gait。"
            )

    # ------------------------------------------------------------------
    # 接口初始化
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
        # 真实发送 + 需要姿态的模式才必须 IMU；dry-run 也可读 IMU 但不强制。
        return self.test_mode in ("touch", "assist", "short_free") or self.enable_send

    # ------------------------------------------------------------------
    # 启动前安全检查
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
            q_actual = feedback["q_policy"]
            err = float(np.max(np.abs(q_actual - SIM_V4_DEFAULT_JOINT_POS_POLICY)))
            if err > self.stand_ready_tol_rad and not self.allow_start_from_any_pose:
                raise RuntimeError(
                    f"require_stand_ready=true: q_actual 距 sim_v4 default {err:.3f} rad > {self.stand_ready_tol_rad:.3f}。"
                    "请先 stand_only 进入 sim_v4 default，再进入 gait。"
                )

    def _countdown(self, seconds: int):
        for i in range(seconds, 0, -1):
            self.get_logger().warn(f"[MIGRATION] enable_send=true，{i} 秒后开始发送电机命令...")
            time.sleep(1.0)

    # ------------------------------------------------------------------
    # IMU / 反馈
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
        empty = {
            "valid": False,
            "q_policy": self.q_cmd_final.copy(),
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

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
    def _on_timer(self):
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        rel = now - self.start_time

        imu = self._read_imu()
        feedback = self._read_feedback()
        if imu["valid"]:
            self._target_yaw = float(imu["rpy"][2])

        # duration / soft stop
        if self.auto_stop_after_duration and rel >= self.duration_s and not self.soft_stop_active:
            self._request_soft_stop("duration_reached")

        if self.test_mode == "stand_only":
            self._run_stand_only(now, rel, dt, imu, feedback)
            return

        if self.soft_stop_active:
            q_cmd_final, debug, core_out = self._soft_stop_step(now)
        else:
            core_out = self._core_step(imu, feedback)
            q_cmd_final = core_out["q_cmd_final_policy"]
            debug = core_out["debug_info"]
        self.q_cmd_final = np.asarray(q_cmd_final, dtype=np.float64).copy()

        # 安全检查 (用真实反馈; dry-run 用虚拟)
        safety = self._safety_check(imu, feedback)

        # 映射到真机
        target_real = self.mapper.policy_target_to_real_target(self.q_cmd_final, clamp=True)
        kp_policy = np.asarray(core_out["kp_policy"], dtype=np.float64)
        kd_policy = np.asarray(core_out["kd_policy"], dtype=np.float64)

        sent = False
        if self.enable_send and not self.soft_stop_active:
            sent = self._send_motion(target_real, kp_policy, kd_policy)
        elif self.enable_send and self.soft_stop_active:
            sent = self._send_motion(target_real, kp_policy, kd_policy)

        # sim_compare
        cmp = self._sim_compare(rel, debug) if self.golden is not None else None

        self._write_csv_row(rel, dt, debug, core_out, feedback, imu, safety, target_real, sent, cmp)

        # soft stop 完成 -> 关闭
        if self.soft_stop_active and (now - self._soft_stop_t0) >= self.soft_stop_sec:
            self.get_logger().info(f"[MIGRATION] soft stop done ({self.stop_reason}). shutting down.")
            self._shutdown()

    def _core_step(self, imu: dict, feedback: dict) -> dict:
        vmc_scale = TEST_MODE_VMC_SCALE.get(self.test_mode, 1.0)
        base_height = None
        if feedback.get("valid") and not self.dry_run_virtual_feedback:
            base_height = None  # 真机一般没有 base height
        inp = CoreInputs(
            roll=float(imu["rpy"][0]),
            pitch=float(imu["rpy"][1]),
            yaw=float(imu["rpy"][2]),
            gyro=tuple(float(v) for v in imu["gyro"]),
            lin_vel=(0.0, 0.0, 0.0),
            imu_valid=bool(imu["valid"]),
            q_actual_policy=feedback["q_policy"] if feedback["valid"] else None,
            dq_actual_policy=feedback["dq_policy"] if feedback["valid"] else None,
            tau_actual_policy=feedback["torque_policy"] if feedback["valid"] else None,
            feedback_valid=bool(feedback["valid"]),
            base_height_m=base_height if self.use_base_height_estimate else None,
            foot_force=None,  # 真机无 contact force
            test_mode=self.test_mode,
            dry_run_virtual_feedback=self.dry_run_virtual_feedback,
            vmc_scale=vmc_scale,
        )
        return self.core.step(inp)

    # ------------------------------------------------------------------
    # stand_only: 当前 q_actual -> sim_v4 default 软插值
    # ------------------------------------------------------------------
    def _run_stand_only(self, now, rel, dt, imu, feedback):
        if not hasattr(self, "_stand_from"):
            if feedback["valid"]:
                self._stand_from = feedback["q_policy"].copy()
            else:
                self._stand_from = SIM_V4_DEFAULT_JOINT_POS_POLICY.copy()
            self._stand_t0 = now
        ramp = min(1.0, (now - self._stand_t0) / max(self.soft_stop_sec, 1.0e-3))
        s = ramp * ramp * (3.0 - 2.0 * ramp)
        q_target = self._stand_from + s * (SIM_V4_DEFAULT_JOINT_POS_POLICY - self._stand_from)
        self.q_cmd_final = q_target.copy()
        target_real = self.mapper.policy_target_to_real_target(q_target, clamp=True)
        kp = np.array([self.core.cfg.fast_trot_support_hip_kp, self.core.cfg.fast_trot_support_thigh_kp,
                       self.core.cfg.fast_trot_support_calf_kp] * 4)
        kd = np.full(12, self.core.cfg.fast_trot_support_kd)
        sent = False
        if self.enable_send:
            sent = self._send_motion(target_real, kp, kd)
        safety = self._safety_check(imu, feedback)
        debug = self._stand_only_debug(q_target, kp, kd)
        self._write_csv_row(rel, dt, debug, {"kp_policy": kp, "kd_policy": kd}, feedback, imu, safety,
                            target_real, sent, None)
        if rel >= self.duration_s:
            self.get_logger().info("[MIGRATION] stand_only done.")
            self._shutdown()

    def _stand_only_debug(self, q_target, kp, kd):
        z = self.core._forward_sagittal(q_target[1::3], q_target[2::3])[1]
        clearance = z - self.core.default_foot_z
        zeros4 = np.zeros(4)
        zeros12 = np.zeros(12)
        return {
            "relative_time": 0.0, "phase": 0.0, "warmup": 0.0, "duty_factor": self.core.cfg.fast_trot_duty_factor,
            "active_swing_pair": 0, "support_pair": 0,
            "leg_phase": zeros4.copy(), "swing_progress": zeros4.copy(),
            "swing_mask": np.zeros(4, dtype=bool), "support_mask": np.ones(4, dtype=bool),
            "phase_to_switch": 0.0, "phase_switch_guard_active": False, "phase_switch_guard_strength": 0.0,
            "q_cpg_policy": q_target.copy(), "q_ref_policy": q_target.copy(),
            "q_vmc_delta_policy": zeros12.copy(), "q_cmd_raw_policy": q_target.copy(),
            "q_cmd_final_policy": q_target.copy(), "q_actual_policy": q_target.copy(),
            "q_error_policy": zeros12.copy(), "q_ref_cmd_diff": zeros12.copy(),
            "kp_policy": kp.copy(), "kd_policy": kd.copy(),
            "fk_clearance_ref": clearance.copy(), "fk_clearance_cmd": clearance.copy(),
            "fk_clearance_actual": clearance.copy(), "predicted_foot_height": clearance.copy(),
            "height_source": "unavailable", "early_contact_source": "unavailable", "real_vmc_scale": 0.0,
            "vmc_weight": zeros4.copy(), "vmc_height_corr_z": 0.0, "vmc_roll_corr_z": 0.0, "vmc_pitch_corr_z": 0.0,
            "vmc_foot_z_offset": zeros4.copy(), "vmc_foot_x_offset": zeros4.copy(), "vmc_foot_y_offset": zeros4.copy(),
            "vmc_foot_x_corr": 0.0, "vmc_foot_y_corr": 0.0,
            "yaw_target": 0.0, "yaw_error": 0.0, "yaw_corr_hip_raw": 0.0, "yaw_corr_hip": 0.0,
            "yaw_hip_offset": zeros4.copy(), "yaw_hip_rate_limited": zeros4.copy(),
            "rear_preswing_unload_gate": zeros4.copy(), "rear_preswing_vmc_fade": np.ones(4),
            "rear_touchdown_vmc_ramp_weight": zeros4.copy(),
            "phase_switch_vmc_weight_scale_applied": 1.0, "phase_switch_yaw_weight_scale_applied": 1.0,
            "phase_switch_kp_scale_applied": 1.0,
            "rear_late_swing_window_active": np.zeros(4, dtype=bool),
            "rear_late_swing_guard_active": np.zeros(4, dtype=bool),
            "rear_late_swing_clearance_offset": zeros4.copy(), "rear_late_swing_height": zeros4.copy(),
            "rear_late_swing_height_error": zeros4.copy(), "rear_late_swing_descent_scale_applied": np.ones(4),
            "rear_early_contact_guard_active": np.zeros(4, dtype=bool), "rear_early_contact_score": zeros4.copy(),
            "rear_early_contact_relief_offset": zeros4.copy(),
            "rear_touchdown_kp_scale": np.ones(4), "rear_early_contact_kp_scale": np.ones(4),
            "rear_touchdown_kp_ramp_weight": zeros4.copy(), "guard_kp_scale": zeros12.copy(),
            "tau_est": zeros12.copy(), "rate_limited_delta": zeros12.copy(),
            "rate_clip_ratio": 0.0, "torque_clip_ratio": 0.0, "support_preload_delta_z": zeros4.copy(),
            "support_preload_gate": zeros4.copy(), "preload_gate": zeros4.copy(), "early_stance_gate": zeros4.copy(),
            "frequency": 0.0, "stride": 0.0, "swing_height": 0.0,
        }

    # ------------------------------------------------------------------
    # soft stop
    # ------------------------------------------------------------------
    def _request_soft_stop(self, reason: str):
        if self.soft_stop_active:
            return
        self.soft_stop_active = True
        self._soft_stop_from = self.q_cmd_final.copy()
        self._soft_stop_t0 = time.time()
        if not self.stop_reason:
            self.stop_reason = reason
        self.get_logger().warn(f"[MIGRATION] soft stop requested: {reason}")

    def _soft_stop_step(self, now):
        ramp = min(1.0, (now - self._soft_stop_t0) / max(self.soft_stop_sec, 1.0e-3))
        s = ramp * ramp * (3.0 - 2.0 * ramp)
        q_target = self._soft_stop_from + s * (SIM_V4_DEFAULT_JOINT_POS_POLICY - self._soft_stop_from)
        kp = np.array([self.core.cfg.fast_trot_support_hip_kp, self.core.cfg.fast_trot_support_thigh_kp,
                       self.core.cfg.fast_trot_support_calf_kp] * 4)
        kd = np.full(12, self.core.cfg.fast_trot_support_kd)
        debug = self._stand_only_debug(q_target, kp, kd)
        debug["node_state"] = "soft_stop"
        return q_target, debug, {"q_cmd_final_policy": q_target, "kp_policy": kp, "kd_policy": kd, "debug_info": debug}

    # ------------------------------------------------------------------
    # 安全检查
    # ------------------------------------------------------------------
    def _safety_check(self, imu: dict, feedback: dict) -> dict:
        roll_deg = abs(float(imu["rpy"][0]) * 180.0 / math.pi)
        pitch_deg = abs(float(imu["rpy"][1]) * 180.0 / math.pi)

        use_virtual = (not self.enable_send) and self.dry_run_virtual_feedback
        if feedback["valid"] and not use_virtual:
            q_actual = feedback["q_policy"]
            tau = feedback["torque_policy"]
            current = feedback["current_policy"]
            temp = feedback["temp_policy"]
            comm_ok = feedback["communication_ok"]
        else:
            q_actual = self.q_cmd_final.copy()
            tau = np.zeros(12)
            current = np.zeros(12)
            temp = np.zeros(12)
            comm_ok = True

        q_error = self.q_cmd_final - q_actual
        max_q_error = float(np.max(np.abs(q_error)))
        max_tau = float(np.nanmax(np.abs(tau))) if np.any(np.isfinite(tau)) else 0.0
        max_current = float(np.nanmax(np.abs(current))) if np.any(np.isfinite(current)) else 0.0
        max_temp = float(np.nanmax(temp)) if np.any(np.isfinite(temp)) else 0.0

        warn = False
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
            "q_actual": q_actual,
            "q_error": q_error,
            "roll_deg": roll_deg,
            "pitch_deg": pitch_deg,
            "max_q_error": max_q_error,
            "max_tau_est": max_tau,
            "max_current": max_current,
            "max_temp": max_temp,
            "communication_ok": comm_ok,
            "safety_warn": warn,
            "safety_stop": stop or self.soft_stop_active,
        }

    # ------------------------------------------------------------------
    # 发送
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
        default_real = self.mapper.real_default_pose_for_motor_order()
        kp = [float(self.core.cfg.fast_trot_support_hip_kp), float(self.core.cfg.fast_trot_support_thigh_kp),
              float(self.core.cfg.fast_trot_support_calf_kp)] * 4
        kd = [float(self.core.cfg.fast_trot_support_kd)] * 12
        items = []
        for i, mid in enumerate(self.mapper.get_real_motor_ids()):
            items.append({
                "motor_id": int(mid), "position": float(default_real[i]),
                "speed": 0.0, "torque": 0.0, "kp": float(kp[i]), "kd": float(kd[i]),
            })
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
    # golden CSV / sim_compare
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
        times = []
        rows = []
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self._golden_fields = reader.fieldnames or []
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
        # golden 每关节列为 prefix_0 .. prefix_11
        vals = []
        for i in range(12):
            v = row.get(f"{prefix}_{i}")
            vals.append(float(v) if v not in (None, "") else float("nan"))
        return np.asarray(vals, dtype=np.float64)

    def _sim_compare(self, rel: float, debug: dict) -> dict:
        idx, grow, gtime = self._golden_nearest(rel)
        # golden q_ref_* = simulator_q_ref (policy order, sign=1) ; q_cmd_final_* = final_q_cmd
        g_q_ref = self._golden_vec(grow, "q_ref")
        g_q_cmd = self._golden_vec(grow, "q_cmd_final")
        try:
            g_phase = float(grow.get("base_phase", "nan"))
        except (TypeError, ValueError):
            g_phase = float("nan")
        q_ref = debug["q_ref_policy"]
        q_cmd = debug["q_cmd_final_policy"]
        q_ref_abs_diff = np.abs(q_ref - g_q_ref)
        q_cmd_abs_diff = np.abs(q_cmd - g_q_cmd)
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
    # CSV
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

    def _build_row(self, rel, dt, debug, core_out, feedback, imu, safety, target_real, sent, cmp):
        cfg = self.core.cfg
        cols = []  # list of (name, value)

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

        # 基础
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

        # 迁移状态
        s("direct_migration_enabled", 1)
        s("migration_core_version", MIGRATION_CORE_VERSION)
        s("stand_source", self.stand_source)
        vec_joint("sim_v4_default_policy", self.default_cmp["sim_v4_default_policy"])
        vec_joint("mapper_default_policy", self.default_cmp["mapper_default_policy"])
        vec_joint("default_policy_diff", self.default_cmp["default_policy_diff"])
        s("default_policy_diff_max", self.default_cmp["default_policy_diff_max"])

        # reference / output
        vec_joint("q_cpg_policy", debug["q_cpg_policy"])
        vec_joint("q_ref_policy", debug["q_ref_policy"])
        vec_joint("q_vmc_delta_policy", debug["q_vmc_delta_policy"])
        vec_joint("q_cmd_raw_policy", debug["q_cmd_raw_policy"])
        vec_joint("q_cmd_final_policy", debug["q_cmd_final_policy"])
        vec_joint("q_actual_policy", safety["q_actual"])
        vec_joint("q_error_policy", debug["q_error_policy"])
        vec_joint("q_ref_cmd_diff", debug["q_ref_cmd_diff"])

        # sim compare
        vec_joint("q_ref_sim_compare", cmp["q_ref_sim_compare"] if cmp else np.full(12, np.nan))
        vec_joint("q_cmd_sim_compare", cmp["q_cmd_sim_compare"] if cmp else np.full(12, np.nan))
        vec_joint("q_ref_abs_diff", cmp["q_ref_abs_diff"] if cmp else np.full(12, np.nan))
        vec_joint("q_cmd_abs_diff", cmp["q_cmd_abs_diff"] if cmp else np.full(12, np.nan))
        s("q_ref_abs_diff_max", cmp["q_ref_abs_diff_max"] if cmp else float("nan"))
        s("q_cmd_abs_diff_max", cmp["q_cmd_abs_diff_max"] if cmp else float("nan"))

        # FK / clearance
        vec_leg("fk_clearance_ref", debug["fk_clearance_ref"])
        vec_leg("fk_clearance_cmd", debug["fk_clearance_cmd"])
        vec_leg("fk_clearance_actual", debug["fk_clearance_actual"])
        vec_leg("predicted_foot_height", debug["predicted_foot_height"])

        # VMC
        s("height_source", debug["height_source"])
        s("early_contact_source", debug["early_contact_source"])
        s("real_vmc_scale", debug["real_vmc_scale"])
        s("vmc_height_corr_z", debug["vmc_height_corr_z"])
        s("vmc_roll_corr_z", debug["vmc_roll_corr_z"])
        s("vmc_pitch_corr_z", debug["vmc_pitch_corr_z"])
        s("yaw_corr_hip", debug["yaw_corr_hip"])
        vec_leg("yaw_hip_offset", debug["yaw_hip_offset"])
        vec_leg("vmc_weight", debug["vmc_weight"])

        # rear guard (RR=2, RL=3)
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

        # gains / safety
        vec_joint("kp", debug["kp_policy"])
        vec_joint("kd", debug["kd_policy"])
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

        # mapping
        target_real = np.asarray(target_real, dtype=np.float64).reshape(12)
        for i, lab in enumerate(MOTOR_ID_LABELS):
            s(f"raw_motor_target_{lab}", float(target_real[i]))
        remap = self.mapper.policy_target_to_real_target(self.q_cmd_final, clamp=True)
        s("semantic_to_motor_mapping_ok", int(bool(np.allclose(target_real, remap, atol=1.0e-6))))
        s("sent", int(bool(sent)))
        return cols

    def _write_csv_row(self, rel, dt, debug, core_out, feedback, imu, safety, target_real, sent, cmp):
        cols = self._build_row(rel, dt, debug, core_out, feedback, imu, safety, target_real, sent, cmp)
        if self._csv_header is None:
            self._csv_header = [c[0] for c in cols]
            self._csv_writer.writerow(self._csv_header)
        self._csv_writer.writerow([c[1] for c in cols])
        self._csv_file.flush()

    # ------------------------------------------------------------------
    # 关闭 / summary
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
            f"[SUMMARY] stand_source={self.stand_source} default_policy_diff_max="
            f"{self.default_cmp['default_policy_diff_max']:.4f} rad"
        )
        self.get_logger().info(
            f"[SUMMARY] hip_outward_signs FR/FL/RR/RL = {URDF_HIP_OUTWARD_SIGNS.tolist()} (use_urdf=True)"
        )
        if self.stop_reason:
            self.get_logger().info(f"[SUMMARY] stop_reason={self.stop_reason}")
        if self._cmp_q_ref_max:
            qref = float(np.max(self._cmp_q_ref_max))
            qcmd = float(np.max(self._cmp_q_cmd_max))
            ref_grade = "PASS" if qref < 0.03 else ("CAUTION" if qref < 0.08 else "FAIL")
            cmd_grade = "PASS" if qcmd < 0.10 else ("CAUTION" if qcmd < 0.15 else
                        ("CAUTION" if qcmd < 0.25 else "FAIL"))
            self.get_logger().info(f"[SIM_COMPARE] q_ref_abs_diff_max={qref:.4f} rad -> {ref_grade}")
            self.get_logger().info(f"[SIM_COMPARE] q_cmd_abs_diff_max={qcmd:.4f} rad -> {cmd_grade}")
            if ref_grade != "PASS" or cmd_grade in ("FAIL",):
                self.get_logger().warn(
                    "[SIM_COMPARE] 差异来源排查清单: default stand 不一致 / phase/warmup 不一致 / "
                    "support preload 不一致 / VMC input 缺失 / contact force 缺失 / rate limiter 不一致 / "
                    "torque backoff 不一致 / kp/kd 不一致 / joint sign/order 不一致"
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
        # 给 soft stop 一点时间执行
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
