#!/usr/bin/env python3
import csv
import math
import os
import threading
import time
import numpy as np
import requests
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from .motor_state_interface import MotorStateHttpInterface
from .semantic_mapper import JointSemanticMapper
LEG_ORDER = ('FR', 'FL', 'RR', 'RL')
LEG_START = {'FR': 0, 'FL': 3, 'RR': 6, 'RL': 9}
STABLE_SWING_ORDER = ('RR', 'FL', 'RL', 'FR')
LOOP_OFFSETS = {'RR': 0.0, 'FL': 0.80, 'RL': 0.50, 'FR': 0.30}
HIP_OUTWARD_SIGNS = {'FR': 1.0, 'FL': 1.0, 'RR': -1.0, 'RL': 1.0}
LEG_SIDE = {'FR': -1.0, 'RR': -1.0, 'FL': 1.0, 'RL': 1.0}
DIAGONAL_PARTNER = {'FR': 'RL', 'RL': 'FR', 'FL': 'RR', 'RR': 'FL'}

class FanfanIkGaitNode(Node):

    def __init__(self):
        super().__init__('fanfan_ik_gait_node')
        self.declare_parameter('motor_base_url', 'http://127.0.0.1:8000')
        self.declare_parameter('enable_send', False)
        self.declare_parameter('motion_mode', 'stable_calf_walk')
        self.declare_parameter('gait_hz', 80.0)
        self.declare_parameter('step_hz', 0.72)
        self.declare_parameter('stand_sec', 3.0)
        self.declare_parameter('warmup_sec', 4.0)
        self.declare_parameter('stand_kp', 40.0)
        self.declare_parameter('stand_kd', 4.2)
        self.declare_parameter('send_kp', 32.0)
        self.declare_parameter('send_kd', 4.0)
        self.declare_parameter('send_speed', 0.0)
        self.declare_parameter('send_torque', 0.0)
        self.declare_parameter('http_timeout', 0.08)
        self.declare_parameter('debug_csv_path', '')
        self.declare_parameter('debug_csv_period_sec', 0.0)
        self.declare_parameter('debug_stale_recheck_ms', 100.0)
        self.declare_parameter('stride_length', 0.024)
        self.declare_parameter('swing_height', 0.056)
        self.declare_parameter('duty_factor', 0.82)
        self.declare_parameter('diagonal_pair_delay_phase', 0.20)
        self.declare_parameter('front_stride_gain', 0.8)
        self.declare_parameter('rear_stride_gain', 0.95)
        self.declare_parameter('front_swing_height_gain', 1.22)
        self.declare_parameter('rear_swing_height_gain', 1.08)
        self.declare_parameter('front_x_bias', 0.002)
        self.declare_parameter('front_z_extend', -0.003)
        self.declare_parameter('front_swing_forward_unfold', 0.01)
        self.declare_parameter('front_thigh_delta_scale', 0.03)
        self.declare_parameter('front_calf_lift_extra', 0.15)
        self.declare_parameter('rear_thigh_delta_scale', 0.06)
        self.declare_parameter('rear_calf_lift_extra', 0.115)
        self.declare_parameter('front_calf_stance_push_amp', 0.03)
        self.declare_parameter('rear_calf_stance_push_amp', 0.05)
        self.declare_parameter('diagonal_push_boost', 0.004)
        self.declare_parameter('rear_push_during_front_swing_amp', 0.006)
        self.declare_parameter('front_push_during_rear_swing_amp', 0.003)
        self.declare_parameter('support_calf_hold_amp', 0.032)
        self.declare_parameter('pre_swing_unload_amp', 0.05)
        self.declare_parameter('pre_swing_support_boost_amp', 0.022)
        self.declare_parameter('rear_swing_body_x_shift', 0.05)
        self.declare_parameter('rear_swing_front_support_boost_amp', 0.050)
        self.declare_parameter('rear_swing_rear_support_hold_amp', 0.000)
        self.declare_parameter('rear_swing_rear_support_relief_amp', 0.012)
        self.declare_parameter('rear_swing_lateral_hip_amp', 0.003)
        self.declare_parameter('rear_swing_swing_leg_x_scale', 0.10)
        self.declare_parameter('rear_swing_front_x_shift_scale', 1.00)
        self.declare_parameter('rear_swing_opposite_rear_x_shift_scale', 0.18)
        self.declare_parameter('pre_swing_fraction', 0.3)
        self.declare_parameter('advance_start', 0.2)
        self.declare_parameter('advance_end', 0.82)
        self.declare_parameter('hip_default_scale', 0.45)
        self.declare_parameter('rear_hip_default_outward_offset', 0.025)
        self.declare_parameter('rear_thigh_default_back_offset', 0.035)
        self.declare_parameter('front_hip_swing_scale', 0.6)
        self.declare_parameter('rear_hip_swing_scale', 1.0)
        self.declare_parameter('front_thigh_swing_scale', 1.0)
        self.declare_parameter('rear_thigh_swing_scale', 1.0)
        self.declare_parameter('front_calf_swing_scale', 1.0)
        self.declare_parameter('rear_calf_swing_scale', 1.0)
        self.declare_parameter('support_hip_outward_amp', 0.003)
        self.declare_parameter('side_support_hip_amp', 0.008)
        self.declare_parameter('right_side_support_scale', 0.65)
        self.declare_parameter('left_side_support_scale', 1.00)
        self.declare_parameter('same_side_support_hip_scale', 0.45)
        self.declare_parameter('swing_hip_unload_amp', 0.0)
        self.declare_parameter('front_support_hip_scale', 0.08)
        self.declare_parameter('rear_support_hip_scale', 0.14)
        self.declare_parameter('opposite_side_boost', 0.0)
        self.declare_parameter('hip_body_y_sign', 1.0)
        self.declare_parameter('fr_swing_hip_inward_amp', 0.010)
        self.declare_parameter('front_touchdown_hip_counter_amp', 0.012)
        self.declare_parameter('front_touchdown_start', 0.66)
        self.declare_parameter('rear_touchdown_hip_counter_amp', 0.010)
        self.declare_parameter('rear_touchdown_start', 0.64)
        self.declare_parameter('max_target_rate_rad_s', 2.8)
        self.declare_parameter('max_delta', 0.9)
        self.declare_parameter('torque_warn_nm', 5.5)
        self.declare_parameter('thigh_length', 0.1560608)
        self.declare_parameter('calf_length', 0.1489418)
        self.motor_base_url = str(self.get_parameter('motor_base_url').value).rstrip('/')
        self.enable_send = bool(self.get_parameter('enable_send').value)
        self.motion_mode = str(self.get_parameter('motion_mode').value).strip().lower()
        self.gait_hz = float(self.get_parameter('gait_hz').value)
        self.step_hz = float(self.get_parameter('step_hz').value)
        self.stand_sec = float(self.get_parameter('stand_sec').value)
        self.warmup_sec = float(self.get_parameter('warmup_sec').value)
        self.stand_kp = float(self.get_parameter('stand_kp').value)
        self.stand_kd = float(self.get_parameter('stand_kd').value)
        self.send_kp = float(self.get_parameter('send_kp').value)
        self.send_kd = float(self.get_parameter('send_kd').value)
        self.send_speed = float(self.get_parameter('send_speed').value)
        self.send_torque = float(self.get_parameter('send_torque').value)
        self.http_timeout = float(self.get_parameter('http_timeout').value)
        self.debug_csv_path = str(self.get_parameter('debug_csv_path').value)
        self.debug_csv_period_sec = float(self.get_parameter('debug_csv_period_sec').value)
        self.debug_stale_recheck_ms = float(self.get_parameter('debug_stale_recheck_ms').value)
        self.stride_length = float(self.get_parameter('stride_length').value)
        self.swing_height = float(self.get_parameter('swing_height').value)
        self.duty_factor = float(self.get_parameter('duty_factor').value)
        self.diagonal_pair_delay_phase = float(self.get_parameter('diagonal_pair_delay_phase').value)
        self.front_stride_gain = float(self.get_parameter('front_stride_gain').value)
        self.rear_stride_gain = float(self.get_parameter('rear_stride_gain').value)
        self.front_swing_height_gain = float(self.get_parameter('front_swing_height_gain').value)
        self.rear_swing_height_gain = float(self.get_parameter('rear_swing_height_gain').value)
        self.front_x_bias = float(self.get_parameter('front_x_bias').value)
        self.front_z_extend = float(self.get_parameter('front_z_extend').value)
        self.front_swing_forward_unfold = float(self.get_parameter('front_swing_forward_unfold').value)
        self.front_thigh_delta_scale = float(self.get_parameter('front_thigh_delta_scale').value)
        self.front_calf_lift_extra = float(self.get_parameter('front_calf_lift_extra').value)
        self.rear_thigh_delta_scale = float(self.get_parameter('rear_thigh_delta_scale').value)
        self.rear_calf_lift_extra = float(self.get_parameter('rear_calf_lift_extra').value)
        self.front_calf_stance_push_amp = float(self.get_parameter('front_calf_stance_push_amp').value)
        self.rear_calf_stance_push_amp = float(self.get_parameter('rear_calf_stance_push_amp').value)
        self.diagonal_push_boost = float(self.get_parameter('diagonal_push_boost').value)
        self.rear_push_during_front_swing_amp = float(self.get_parameter('rear_push_during_front_swing_amp').value)
        self.front_push_during_rear_swing_amp = float(self.get_parameter('front_push_during_rear_swing_amp').value)
        self.support_calf_hold_amp = float(self.get_parameter('support_calf_hold_amp').value)
        self.pre_swing_unload_amp = float(self.get_parameter('pre_swing_unload_amp').value)
        self.pre_swing_support_boost_amp = float(self.get_parameter('pre_swing_support_boost_amp').value)
        self.rear_swing_body_x_shift = float(self.get_parameter('rear_swing_body_x_shift').value)
        self.rear_swing_front_support_boost_amp = float(self.get_parameter('rear_swing_front_support_boost_amp').value)
        self.rear_swing_rear_support_hold_amp = float(self.get_parameter('rear_swing_rear_support_hold_amp').value)
        self.rear_swing_rear_support_relief_amp = float(self.get_parameter('rear_swing_rear_support_relief_amp').value)
        self.rear_swing_lateral_hip_amp = float(self.get_parameter('rear_swing_lateral_hip_amp').value)
        self.rear_swing_swing_leg_x_scale = float(self.get_parameter('rear_swing_swing_leg_x_scale').value)
        self.rear_swing_front_x_shift_scale = float(self.get_parameter('rear_swing_front_x_shift_scale').value)
        self.rear_swing_opposite_rear_x_shift_scale = float(self.get_parameter('rear_swing_opposite_rear_x_shift_scale').value)
        self.pre_swing_fraction = float(self.get_parameter('pre_swing_fraction').value)
        self.advance_start = float(self.get_parameter('advance_start').value)
        self.advance_end = float(self.get_parameter('advance_end').value)
        self.hip_default_scale = float(self.get_parameter('hip_default_scale').value)
        self.rear_hip_default_outward_offset = float(self.get_parameter('rear_hip_default_outward_offset').value)
        self.rear_thigh_default_back_offset = float(self.get_parameter('rear_thigh_default_back_offset').value)
        self.front_hip_swing_scale = float(self.get_parameter('front_hip_swing_scale').value)
        self.rear_hip_swing_scale = float(self.get_parameter('rear_hip_swing_scale').value)
        self.front_thigh_swing_scale = float(self.get_parameter('front_thigh_swing_scale').value)
        self.rear_thigh_swing_scale = float(self.get_parameter('rear_thigh_swing_scale').value)
        self.front_calf_swing_scale = float(self.get_parameter('front_calf_swing_scale').value)
        self.rear_calf_swing_scale = float(self.get_parameter('rear_calf_swing_scale').value)
        self.support_hip_outward_amp = float(self.get_parameter('support_hip_outward_amp').value)
        self.side_support_hip_amp = float(self.get_parameter('side_support_hip_amp').value)
        self.right_side_support_scale = float(self.get_parameter('right_side_support_scale').value)
        self.left_side_support_scale = float(self.get_parameter('left_side_support_scale').value)
        self.same_side_support_hip_scale = float(self.get_parameter('same_side_support_hip_scale').value)
        self.swing_hip_unload_amp = float(self.get_parameter('swing_hip_unload_amp').value)
        self.front_support_hip_scale = float(self.get_parameter('front_support_hip_scale').value)
        self.fr_swing_hip_inward_amp = float(self.get_parameter('fr_swing_hip_inward_amp').value)
        self.front_touchdown_hip_counter_amp = float(self.get_parameter('front_touchdown_hip_counter_amp').value)
        self.front_touchdown_start = float(self.get_parameter('front_touchdown_start').value)
        self.rear_touchdown_hip_counter_amp = float(self.get_parameter('rear_touchdown_hip_counter_amp').value)
        self.rear_touchdown_start = float(self.get_parameter('rear_touchdown_start').value)
        self.rear_support_hip_scale = float(self.get_parameter('rear_support_hip_scale').value)
        self.opposite_side_boost = float(self.get_parameter('opposite_side_boost').value)
        self.hip_body_y_sign = float(self.get_parameter('hip_body_y_sign').value)
        self.max_target_rate_rad_s = float(self.get_parameter('max_target_rate_rad_s').value)
        self.max_delta = float(self.get_parameter('max_delta').value)
        self.torque_warn_nm = float(self.get_parameter('torque_warn_nm').value)
        self.thigh_length = float(self.get_parameter('thigh_length').value)
        self.calf_length = float(self.get_parameter('calf_length').value)
        self.mapper = JointSemanticMapper()
        self.motor_ids = self.mapper.get_real_motor_ids()
        self.real_joint_names = list(self.mapper.real_joint_names)
        self.policy_joint_names = self.mapper.get_policy_joint_names()
        self.default_policy = self.mapper.default_joint_angle.astype(np.float32).copy()
        self.apply_hip_default_scale()
        self.default_real = self.mapper.policy_target_to_real_target(self.default_policy, clamp=True).astype(np.float32)
        self.default_foot_xz = self.compute_default_foot_xz()
        self.last_target_policy = self.default_policy.copy()
        self.last_target_real = self.default_real.copy()
        self.start_time = time.time()
        self._last_update_time = self.start_time
        self._phase_acc = 0.0
        self._last_send_info_time = 0.0
        self.http_session = requests.Session()
        self.motor = MotorStateHttpInterface(base_url=self.motor_base_url, timeout=self.http_timeout, stale_recheck_ms=self.debug_stale_recheck_ms)
        self._debug_csv_file = None
        self._debug_csv_writer = None
        self._debug_sample_lock = threading.Lock()
        self._latest_debug_sample = None
        self._debug_stop_event = threading.Event()
        self._debug_thread = None
        self._last_feedback_warn_time = 0.0
        self.setup_debug_csv()
        self.start_debug_collector()
        self.pub_target = self.create_publisher(Float32MultiArray, '/mydog/fanfan_ik_target_real', 10)
        self.pub_phase = self.create_publisher(Float32MultiArray, '/mydog/fanfan_ik_phase', 10)
        self.get_logger().warn('Fanfan IK gait is open-loop. First run supported or hand-held, and be ready to cut power.')
        if self.enable_send:
            self.send_default_stand()
        else:
            self.get_logger().warn('enable_send=False: dry run only, no motor commands sent.')
        self.get_logger().info(f'Fanfan stable lift-first calf-dominant walk: mode={self.motion_mode}, step_hz={self.step_hz:.2f}, gait_hz={self.gait_hz:.1f}, stride={self.stride_length:.3f}m, swing={self.swing_height:.3f}m, duty={self.duty_factor:.2f}, pair_delay={self.diagonal_pair_delay_phase:.2f}, front_lift_gain={self.front_swing_height_gain:.2f}, front_unfold={self.front_swing_forward_unfold:.3f}m, front_thigh_scale={self.front_thigh_delta_scale:.2f}, calf_extra=({self.front_calf_lift_extra:.3f},{self.rear_calf_lift_extra:.3f}), push=({self.front_calf_stance_push_amp:.3f},{self.rear_calf_stance_push_amp:.3f}), hip_out={self.support_hip_outward_amp:.3f}, kp={self.send_kp:.1f}, kd={self.send_kd:.1f}, rear_hip_offset={self.rear_hip_default_outward_offset:.3f}, rear_thigh_back={self.rear_thigh_default_back_offset:.3f}, front_hip_scale={self.front_hip_swing_scale:.2f}, send={self.enable_send}')
        self.timer = self.create_timer(1.0 / max(self.gait_hz, 0.001), self.update)

    def apply_hip_default_scale(self):
        scale = min(1.0, max(0.0, self.hip_default_scale))
        if abs(scale - self.hip_default_scale) > 1e-06:
            self.get_logger().warn(f'hip_default_scale={self.hip_default_scale:.3f} clipped to {scale:.3f}')
        self.hip_default_scale = scale
        for leg in LEG_ORDER:
            i = LEG_START[leg]
            self.default_policy[i] *= scale
        for leg in ('RR', 'RL'):
            i = LEG_START[leg]
            self.default_policy[i] += HIP_OUTWARD_SIGNS[leg] * self.rear_hip_default_outward_offset
            self.default_policy[i + 1] += self.rear_thigh_default_back_offset

    def compute_default_foot_xz(self) -> dict[str, tuple[float, float]]:
        xz = {}
        for leg in LEG_ORDER:
            i = LEG_START[leg]
            thigh = float(self.default_policy[i + 1])
            calf = float(self.default_policy[i + 2])
            xz[leg] = self.forward_sagittal(thigh, calf)
        return xz

    def forward_sagittal(self, thigh: float, calf: float) -> tuple[float, float]:
        x = -self.thigh_length * math.sin(thigh) - self.calf_length * math.sin(thigh + calf)
        z = -self.thigh_length * math.cos(thigh) - self.calf_length * math.cos(thigh + calf)
        return (float(x), float(z))

    def inverse_sagittal(self, x: float, z: float) -> tuple[float, float]:
        x, z = self.clamp_reachable_xz(float(x), float(z))
        l1 = self.thigh_length
        l2 = self.calf_length
        cos_calf = (x * x + z * z - l1 * l1 - l2 * l2) / max(2.0 * l1 * l2, 1e-09)
        cos_calf = min(1.0, max(-1.0, cos_calf))
        calf = -math.acos(cos_calf)
        thigh = math.atan2(-x, -z) - math.atan2(l2 * math.sin(calf), l1 + l2 * math.cos(calf))
        return (float(thigh), float(calf))

    def clamp_reachable_xz(self, x: float, z: float) -> tuple[float, float]:
        r = math.hypot(x, z)
        max_r = self.thigh_length + self.calf_length - 1e-05
        min_r = abs(self.thigh_length - self.calf_length) + 1e-05
        if r < 1e-09:
            return (0.0, -min_r)
        if r > max_r:
            scale = max_r / r
            return (x * scale, z * scale)
        if r < min_r:
            scale = min_r / r
            return (x * scale, z * scale)
        return (x, z)

    @staticmethod
    def smoothstep_half_cos(s: float) -> float:
        s = min(1.0, max(0.0, float(s)))
        return 0.5 - 0.5 * math.cos(math.pi * s)

    @staticmethod
    def smootherstep(s: float) -> float:
        s = min(1.0, max(0.0, float(s)))
        return s * s * s * (10.0 - 15.0 * s + 6.0 * s * s)

    @classmethod
    def smooth_window(cls, s: float, edge: float=0.16) -> float:
        s = min(1.0, max(0.0, float(s)))
        edge = max(0.001, min(0.45, float(edge)))
        return cls.smootherstep(s / edge) * cls.smootherstep((1.0 - s) / edge)

    def send_default_stand(self):
        items = []
        for mid, pos in zip(self.motor_ids, self.default_real):
            items.append({'motor_id': int(mid), 'position': float(pos), 'speed': 0.0, 'torque': 0.0, 'kp': self.stand_kp, 'kd': self.stand_kd})
        payload = {'items': items, 'enable_first': True, 'stop_first': False}
        url = f'{self.motor_base_url}/api/rs04/motion_mode_run_batch'
        r = self.http_session.post(url, json=payload, timeout=max(self.http_timeout, 0.5))
        if r.status_code != 200:
            raise RuntimeError(f'default stand failed HTTP {r.status_code}: {r.text}')
        self.get_logger().info(f'Default stand sent: kp={self.stand_kp:.1f}, kd={self.stand_kd:.1f}')

    def update(self):
        now = time.time()
        dt = max(0.0, min(now - self._last_update_time, 0.25))
        self._last_update_time = now
        elapsed = now - self.start_time
        if elapsed < self.stand_sec:
            target_policy = self.default_policy.copy()
            phase = 0.0
            warm = 0.0
            leg_debug = self.default_leg_debug(phase, warm)
            self._phase_acc = 0.0
        else:
            gait_time = elapsed - self.stand_sec
            self._phase_acc = (self._phase_acc + self.step_hz * dt) % 1.0
            phase = self._phase_acc
            warm = min(1.0, gait_time / max(self.warmup_sec, 0.001))
            target_policy, leg_debug = self.build_target_policy(phase, warm)
        target_policy = self.apply_target_rate_limit(target_policy, dt)
        target_real = self.mapper.policy_target_to_real_target(target_policy, clamp=True)
        self.publish_array(self.pub_target, target_real)
        self.publish_array(self.pub_phase, np.array([phase, warm, self.step_hz], dtype=np.float32))
        sent = False
        if not self.enable_send:
            self.update_debug_sample(target_real, target_policy, phase, warm, leg_debug, sent)
            return
        delta = target_real - self.last_target_real
        max_delta = float(np.max(np.abs(delta)))
        if max_delta > self.max_delta:
            self.get_logger().warn(f'[SAFE] IK target jump too large: {max_delta:.3f} rad > {self.max_delta:.3f} rad. Skip send.')
            self.update_debug_sample(target_real, target_policy, phase, warm, leg_debug, sent)
            return
        sent = self.send_motion_batch(target_real)
        if sent:
            self.last_target_real = target_real.copy()
        self.update_debug_sample(target_real, target_policy, phase, warm, leg_debug, sent)


    def get_loop_offsets(self, duty: float = None) -> dict[str, float]:
        duty = self.duty_factor if duty is None else duty
        duty = min(max(float(duty), 0.74), 0.90)
        swing_fraction = max(0.08, 1.0 - duty)
        delay = max(float(self.diagonal_pair_delay_phase), swing_fraction + 0.02)
        delay = min(max(delay, 0.12), 0.32)
        starts = {
            'RR': 0.00,
            'FL': delay,
            'RL': 0.50,
            'FR': (0.50 + delay) % 1.0,
        }
        return {leg: ((1.0 - start) % 1.0) for leg, start in starts.items()}

    def get_leg_phase_value(self, leg: str, phase: float, duty: float = None) -> float:
        offsets = self.get_loop_offsets(duty=duty)
        return (phase + offsets[leg]) % 1.0

    def default_leg_debug(self, phase: float, warm: float) -> dict[str, dict[str, float]]:
        data = {}
        active = self.get_active_swing_leg(phase)
        for leg in LEG_ORDER:
            x, z = self.default_foot_xz[leg]
            data[leg] = {'leg_phase': self.get_leg_phase_value(leg, phase), 'stance': 1.0, 'swing': 0.0, 'x_foot': x, 'z_foot': z, 'warm': warm, 'swing_leg': active or 'none', 'hip_delta': 0.0, 'coordinated_push': 0.0, 'rear_swing_gate': 0.0, 'body_x_shift': 0.0, 'hip_swing_scale': self.front_hip_swing_scale if leg in ('FR', 'FL') else self.rear_hip_swing_scale, 'thigh_swing_scale': self.front_thigh_swing_scale if leg in ('FR', 'FL') else self.rear_thigh_swing_scale, 'calf_swing_scale': self.front_calf_swing_scale if leg in ('FR', 'FL') else self.rear_calf_swing_scale, 'thigh_ik': float(self.default_policy[LEG_START[leg] + 1]), 'calf_ik': float(self.default_policy[LEG_START[leg] + 2])}
        return data

    def build_target_policy(self, phase: float, warm: float):
        if self.motion_mode not in ('stable_calf_walk', 'calf_dominant_loop', 'loop_calf', 'loop_crawl'):
            self.get_logger().warn(f'Unknown motion_mode={self.motion_mode!r}; falling back to stable_calf_walk.')
            self.motion_mode = 'stable_calf_walk'
        return self.build_calf_dominant_loop_target_policy(phase, warm)

    def build_calf_dominant_loop_target_policy(self, phase: float, warm: float):
        q = self.default_policy.copy()
        leg_debug = {}
        duty = min(max(self.duty_factor, 0.74), 0.9)
        swing_fraction = max(0.08, 1.0 - duty)
        active_swing_leg = self.get_active_swing_leg(phase, duty=duty)
        pre_swing_leg = self.get_pre_swing_leg(phase, duty=duty)
        active_swing_phase = None
        if active_swing_leg is not None:
            active_swing_phase = self.get_leg_phase_value(active_swing_leg, phase, duty=duty)
        for leg in LEG_ORDER:
            leg_phase = self.get_leg_phase_value(leg, phase, duty=duty)
            is_front = leg in ('FR', 'FL')
            is_swing = leg_phase < swing_fraction
            i = LEG_START[leg]
            thigh_default = float(self.default_policy[i + 1])
            calf_default = float(self.default_policy[i + 2])
            target = self.compute_calf_dominant_target(leg=leg, leg_phase=leg_phase, duty=duty, swing_fraction=swing_fraction, thigh_default=thigh_default, calf_default=calf_default, active_swing_leg=active_swing_leg, pre_swing_leg=pre_swing_leg, active_swing_phase=active_swing_phase)
            thigh_target = target['thigh_target']
            calf_target = target['calf_target']
            swing_shape = target['swing_shape']
            stance_shape = target['stance_shape']
            touchdown_gate = self.compute_touchdown_gate(leg, leg_phase, swing_fraction)
            hip_delta = self.compute_hip_balance_delta(leg=leg, active_swing_leg=active_swing_leg, swing_shape=swing_shape, stance_shape=stance_shape, touchdown_gate=touchdown_gate)
            hip_scale = self.front_hip_swing_scale if is_front else self.rear_hip_swing_scale
            thigh_swing_scale = self.front_thigh_swing_scale if is_front else self.rear_thigh_swing_scale
            calf_swing_scale = self.front_calf_swing_scale if is_front else self.rear_calf_swing_scale
            q[i + 0] = self.default_policy[i + 0] + warm * hip_delta * hip_scale
            q[i + 1] = self.default_policy[i + 1] + warm * (thigh_target - self.default_policy[i + 1]) * thigh_swing_scale
            q[i + 2] = self.default_policy[i + 2] + warm * (calf_target - self.default_policy[i + 2]) * calf_swing_scale
            leg_debug[leg] = {'leg_phase': float(leg_phase), 'stance': float(0.0 if is_swing else 1.0), 'swing': float(1.0 if is_swing else 0.0), 'stance_shape': float(stance_shape), 'swing_shape': float(swing_shape), 'x_foot': float(target['x_actual']), 'z_foot': float(target['z_actual']), 'hip_delta': float(hip_delta), 'touchdown_gate': float(touchdown_gate), 'thigh_ik': float(target['thigh_ik']), 'calf_ik': float(target['calf_ik']), 'front_unfold': float(target['front_unfold']), 'coordinated_push': float(target.get('coordinated_push', 0.0)), 'pre_unload': float(target.get('pre_unload', 0.0)), 'pre_support_boost': float(target.get('pre_support_boost', 0.0)), 'swing_leg': active_swing_leg or 'none', 'pre_swing_leg': pre_swing_leg or 'none', 'thigh_target': float(thigh_target), 'calf_target': float(calf_target), 'hip_swing_scale': float(hip_scale), 'thigh_swing_scale': float(thigh_swing_scale), 'calf_swing_scale': float(calf_swing_scale)}
        return (q.astype(np.float32), leg_debug)

    def compute_calf_dominant_target(self, leg: str, leg_phase: float, duty: float, swing_fraction: float, thigh_default: float, calf_default: float, active_swing_leg: str | None=None, pre_swing_leg: str | None=None, active_swing_phase: float | None=None) -> dict[str, float]:
        x0, z0 = self.default_foot_xz[leg]
        is_front = leg in ('FR', 'FL')
        stride_gain = self.front_stride_gain if is_front else self.rear_stride_gain
        height_gain = self.front_swing_height_gain if is_front else self.rear_swing_height_gain
        thigh_scale = self.front_thigh_delta_scale if is_front else self.rear_thigh_delta_scale
        calf_lift_extra = self.front_calf_lift_extra if is_front else self.rear_calf_lift_extra
        calf_stance_push = self.front_calf_stance_push_amp if is_front else self.rear_calf_stance_push_amp
        stride = self.stride_length * stride_gain
        swing_h = self.swing_height * height_gain
        x_center = x0 + (self.front_x_bias if is_front else 0.0)
        z_center = z0 + (self.front_z_extend if is_front else 0.0)
        front_unfold = 0.0
        rear_swing_gate = 0.0
        if active_swing_leg in ('RR', 'RL') and active_swing_phase is not None:
            s_active = active_swing_phase / max(swing_fraction, 1e-06)
            rear_swing_gate = self.smootherstep(s_active / 0.28) * self.smootherstep((1.0 - s_active) / 0.28)
        if leg_phase < swing_fraction:
            s = leg_phase / max(swing_fraction, 1e-06)
            denom = max(1e-06, self.advance_end - self.advance_start)
            u = self.smootherstep((s - self.advance_start) / denom)
            lift_up = self.smootherstep(s / max(self.advance_start, 0.001))
            lift_down = self.smootherstep((1.0 - s) / max(1.0 - self.advance_end, 0.001))
            swing_shape = lift_up * lift_down
            stance_shape = 0.0
            stance_gate = 0.0
            front_unfold = self.front_swing_forward_unfold * swing_shape if is_front else 0.0
            x_des = x_center - 0.5 * stride + stride * u + front_unfold
            z_des = z_center + swing_h * swing_shape
        else:
            s = (leg_phase - swing_fraction) / max(duty, 1e-06)
            u = self.smootherstep(s)
            swing_shape = 0.0
            stance_shape = math.sin(math.pi * min(1.0, max(0.0, s))) ** 2
            stance_gate = self.smooth_window(s, edge=0.18)
            x_des = x_center + 0.5 * stride - stride * u
            z_des = z_center
        body_x_shift_applied = 0.0
        if active_swing_leg in ('RR', 'RL') and rear_swing_gate > 0.0:
            if leg == active_swing_leg:
                scale = self.rear_swing_swing_leg_x_scale
            elif leg in ('FR', 'FL'):
                scale = self.rear_swing_front_x_shift_scale
            else:
                scale = self.rear_swing_opposite_rear_x_shift_scale
            body_x_shift_applied = self.rear_swing_body_x_shift * rear_swing_gate * scale
            x_des -= body_x_shift_applied
        thigh_ik, calf_ik = self.inverse_sagittal(x_des, z_des)
        thigh_target = thigh_default + thigh_scale * (thigh_ik - thigh_default)
        calf_target = self.solve_calf_for_z(thigh_target, z_des, calf_default)
        coordinated_push = 0.0
        if active_swing_leg is not None and (not leg_phase < swing_fraction):
            if leg == DIAGONAL_PARTNER.get(active_swing_leg):
                coordinated_push += self.diagonal_push_boost
            if active_swing_leg in ('FR', 'FL') and leg in ('RR', 'RL'):
                coordinated_push += self.rear_push_during_front_swing_amp
            if active_swing_leg in ('RR', 'RL') and leg in ('FR', 'FL'):
                coordinated_push += self.front_push_during_rear_swing_amp
        support_hold = 0.0
        pre_unload = 0.0
        pre_support_boost = 0.0
        if not leg_phase < swing_fraction:
            if active_swing_leg is not None:
                support_hold = self.support_calf_hold_amp * stance_gate
                if active_swing_leg in ('RR', 'RL') and rear_swing_gate > 0.0:
                    if is_front:
                        support_hold += self.rear_swing_front_support_boost_amp * rear_swing_gate * stance_gate
                    elif leg != active_swing_leg:
                        support_hold += (
                            self.rear_swing_rear_support_hold_amp
                            - self.rear_swing_rear_support_relief_amp
                        ) * rear_swing_gate * stance_gate
            pre_window = max(0.02, min(0.5, self.pre_swing_fraction))
            pre_gate = self.smootherstep((s - (1.0 - pre_window)) / pre_window)
            if leg == pre_swing_leg:
                pre_unload = -self.pre_swing_unload_amp * pre_gate
            elif pre_swing_leg is not None:
                pre_support_boost = self.pre_swing_support_boost_amp * stance_gate
        calf_target += -calf_lift_extra * swing_shape + support_hold + pre_unload + pre_support_boost + (calf_stance_push + coordinated_push) * stance_shape * stance_gate
        x_actual, z_actual = self.forward_sagittal(thigh_target, calf_target)
        return {'x_des': float(x_des), 'z_des': float(z_des), 'x_actual': float(x_actual), 'z_actual': float(z_actual), 'swing_shape': float(swing_shape), 'stance_shape': float(stance_shape), 'front_unfold': float(front_unfold), 'coordinated_push': float(coordinated_push), 'support_hold': float(locals().get('support_hold', 0.0)), 'pre_unload': float(locals().get('pre_unload', 0.0)), 'pre_support_boost': float(locals().get('pre_support_boost', 0.0)), 'rear_swing_gate': float(rear_swing_gate), 'body_x_shift': float(body_x_shift_applied), 'thigh_ik': float(thigh_ik), 'calf_ik': float(calf_ik), 'thigh_target': float(thigh_target), 'calf_target': float(calf_target)}

    def solve_calf_for_z(self, thigh: float, z_des: float, calf_default: float) -> float:
        value = (-float(z_des) - self.thigh_length * math.cos(float(thigh))) / max(self.calf_length, 1e-09)
        value = min(1.0, max(-1.0, value))
        angle = math.acos(value)
        candidates = (angle - thigh, -angle - thigh)
        return float(min(candidates, key=lambda calf: abs(calf - calf_default)))

    def get_active_swing_leg(self, phase: float, duty: float=None) -> str | None:
        duty = self.duty_factor if duty is None else duty
        duty = min(max(float(duty), 0.74), 0.9)
        swing_fraction = max(0.08, 1.0 - duty)
        for leg in STABLE_SWING_ORDER:
            if self.get_leg_phase_value(leg, phase, duty=duty) < swing_fraction:
                return leg
        return None

    def get_pre_swing_leg(self, phase: float, duty: float=None) -> str | None:
        duty = self.duty_factor if duty is None else duty
        duty = min(max(float(duty), 0.74), 0.9)
        swing_fraction = max(0.08, 1.0 - duty)
        pre_window = max(0.02, min(0.5, self.pre_swing_fraction))
        start = max(swing_fraction, 1.0 - pre_window)
        for leg in STABLE_SWING_ORDER:
            p = self.get_leg_phase_value(leg, phase, duty=duty)
            if p >= start:
                return leg
        return None

    def compute_touchdown_gate(self, leg: str, leg_phase: float, swing_fraction: float) -> float:
        if leg_phase >= swing_fraction:
            return 0.0
        s = leg_phase / max(swing_fraction, 1e-06)
        if leg in ('FR', 'FL'):
            start = min(0.95, max(0.05, self.front_touchdown_start))
        elif leg in ('RR', 'RL'):
            start = min(0.95, max(0.05, self.rear_touchdown_start))
        else:
            return 0.0
        return self.smootherstep((s - start) / max(1.0 - start, 1e-06))

    def compute_hip_balance_delta(self, leg: str, active_swing_leg: str | None, swing_shape: float, stance_shape: float, touchdown_gate: float = 0.0) -> float:
        outward = HIP_OUTWARD_SIGNS[leg]
        is_front = leg in ('FR', 'FL')
        support_scale = self.front_support_hip_scale if is_front else self.rear_support_hip_scale
        if active_swing_leg is None:
            return outward * self.support_hip_outward_amp * support_scale * 0.2
        if leg == active_swing_leg:
            if is_front:
                counter = -outward * self.front_touchdown_hip_counter_amp * touchdown_gate
                if leg == 'FR':
                    counter += -abs(self.fr_swing_hip_inward_amp) * swing_shape
                return counter
            counter = -outward * self.rear_touchdown_hip_counter_amp * touchdown_gate
            return counter
        base = outward * support_scale * self.support_hip_outward_amp * (0.65 + 0.35 * stance_shape)
        swing_side = LEG_SIDE[active_swing_leg]
        preferred_support_side = -swing_side if self.hip_body_y_sign >= 0.0 else swing_side
        leg_side_scale = self.left_side_support_scale if LEG_SIDE[leg] > 0.0 else self.right_side_support_scale
        if LEG_SIDE[leg] == preferred_support_side:
            side_boost = self.side_support_hip_amp * leg_side_scale
        else:
            side_boost = self.side_support_hip_amp * self.same_side_support_hip_scale * leg_side_scale
        delta = base + outward * side_boost * (0.75 + 0.25 * stance_shape)
        if active_swing_leg in ('RR', 'RL') and LEG_SIDE[leg] == preferred_support_side:
            delta += outward * self.rear_swing_lateral_hip_amp * leg_side_scale
        return delta

    def apply_target_rate_limit(self, target_policy: np.ndarray, dt: float) -> np.ndarray:
        target_policy = np.asarray(target_policy, dtype=np.float32).reshape(12)
        if self.max_target_rate_rad_s <= 0.0 or dt <= 0.0:
            self.last_target_policy = target_policy.copy()
            return target_policy
        max_step = self.max_target_rate_rad_s * dt
        step = np.clip(target_policy - self.last_target_policy, -max_step, max_step)
        limited = self.last_target_policy + step
        limited = np.clip(limited, self.mapper.policy_lower_limit, self.mapper.policy_upper_limit)
        self.last_target_policy = limited.astype(np.float32).copy()
        return self.last_target_policy.copy()

    def send_motion_batch(self, target_real: np.ndarray) -> bool:
        items = []
        for i, mid in enumerate(self.motor_ids):
            items.append({'motor_id': int(mid), 'position': float(target_real[i]), 'speed': self.send_speed, 'torque': self.send_torque, 'kp': self.send_kp, 'kd': self.send_kd})
        payload = {'items': items, 'enable_first': False, 'stop_first': False}
        try:
            r = self.http_session.post(f'{self.motor_base_url}/api/rs04/motion_batch_fast', json=payload, timeout=self.http_timeout)
            if r.status_code != 200:
                self.get_logger().warn(f'[SEND] HTTP {r.status_code}: {r.text}')
                return False
            now = time.time()
            if now - self._last_send_info_time > 1.0:
                self._last_send_info_time = now
                self.get_logger().info(f'[SEND] fanfan IK ok | target_min={float(np.min(target_real)):.3f} target_max={float(np.max(target_real)):.3f} kp={self.send_kp:.1f} kd={self.send_kd:.1f}')
            return True
        except Exception as exc:
            self.get_logger().warn(f'[SEND] request failed: {exc}')
            return False

    def setup_debug_csv(self):
        path = self.debug_csv_path.strip()
        if not path:
            return
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._debug_csv_file = open(path, 'w', newline='')
        self._debug_csv_writer = csv.writer(self._debug_csv_file)
        self._debug_csv_writer.writerow(['time', 'elapsed', 'phase', 'warm', 'leg_phase', 'stance', 'swing', 'leg_name', 'joint_index', 'motor_id', 'joint_name', 'policy_joint_name', 'x_foot', 'z_foot', 'hip_delta', 'touchdown_gate', 'coordinated_push', 'rear_swing_gate', 'body_x_shift', 'thigh_ik', 'calf_ik', 'swing_leg', 'q_target_policy', 'q_target_real', 'q_current_real', 'q_error_real', 'torque_measured', 'temp', 'online', 'error_code', 'age_ms', 'sent', 'step_hz', 'stride_length', 'swing_height', 'duty_factor', 'kp', 'kd'])
        self._debug_csv_file.flush()
        self.get_logger().warn(f'[DEBUG_CSV] writing fanfan IK gait data to {path}')

    def start_debug_collector(self):
        if self._debug_csv_writer is None:
            return
        self._debug_thread = threading.Thread(target=self.debug_collect_loop, name='fanfan_ik_gait_logger', daemon=True)
        self._debug_thread.start()

    def debug_collect_loop(self):
        period = self.debug_csv_period_sec
        if period <= 0.0:
            period = 1.0 / max(self.gait_hz, 0.001)
        while not self._debug_stop_event.wait(period):
            with self._debug_sample_lock:
                sample = self._latest_debug_sample
                if sample is not None:
                    sample = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in sample.items()}
            if sample is None:
                continue
            self.write_debug_csv_sample(**sample)

    def update_debug_sample(self, target_real: np.ndarray, target_policy: np.ndarray, phase: float, warm: float, leg_debug: dict[str, dict[str, float]], sent: bool):
        if self._debug_csv_writer is None:
            return
        with self._debug_sample_lock:
            self._latest_debug_sample = {'target_real': np.asarray(target_real, dtype=np.float32).reshape(12).copy(), 'target_policy': np.asarray(target_policy, dtype=np.float32).reshape(12).copy(), 'phase': float(phase), 'warm': float(warm), 'leg_debug': {leg: dict(data) for leg, data in leg_debug.items()}, 'sent': bool(sent), 'stamp': time.time()}

    def write_debug_csv_sample(self, target_real: np.ndarray, target_policy: np.ndarray, phase: float, warm: float, leg_debug: dict[str, dict[str, float]], sent: bool, stamp: float):
        if self._debug_csv_writer is None:
            return
        now = time.time()
        try:
            snapshot = self.motor.get_latest()
        except Exception as exc:
            if now - self._last_feedback_warn_time > 1.0:
                self._last_feedback_warn_time = now
                self.get_logger().warn(f'[DEBUG_CSV] motor feedback read failed: {exc}')
            return
        target_real = np.asarray(target_real, dtype=np.float32).reshape(12)
        target_policy = np.asarray(target_policy, dtype=np.float32).reshape(12)
        q_real = np.asarray(snapshot.q_real, dtype=np.float32).reshape(12)
        torque = np.asarray(snapshot.torque, dtype=np.float32).reshape(12)
        max_abs_torque = float(np.max(np.abs(torque)))
        if max_abs_torque > self.torque_warn_nm and now - self._last_feedback_warn_time > 1.0:
            self._last_feedback_warn_time = now
            self.get_logger().warn(f'[TORQUE] measured max |tau|={max_abs_torque:.2f} Nm > {self.torque_warn_nm:.2f} Nm. Reduce step_hz/stride/swing or support the robot.')
        temp = np.asarray(snapshot.temp, dtype=np.float32).reshape(12)
        online = np.asarray(snapshot.online, dtype=bool).reshape(12)
        error_code = np.asarray(snapshot.error_code, dtype=np.int32).reshape(12)
        age_ms = np.asarray(snapshot.age_ms, dtype=np.float32).reshape(12)
        target_policy_real_order = np.zeros(12, dtype=np.float32)
        target_policy_real_order[self.mapper.policy_to_real_index] = target_policy
        elapsed = float(stamp) - self.start_time
        for real_i, (mid, real_name) in enumerate(zip(self.motor_ids, self.real_joint_names)):
            policy_i = int(np.where(self.mapper.policy_to_real_index == real_i)[0][0])
            policy_name = self.policy_joint_names[policy_i]
            leg = policy_name.split('_', 1)[0]
            leg_info = leg_debug.get(leg, {})
            self._debug_csv_writer.writerow([f'{now:.6f}', f'{elapsed:.6f}', f'{phase:.6f}', f'{warm:.6f}', f"{float(leg_info.get('leg_phase', 0.0)):.6f}", int(float(leg_info.get('stance', 0.0)) > 0.5), int(float(leg_info.get('swing', 0.0)) > 0.5), leg, int(real_i), f'0x{int(mid):02X}', real_name, policy_name, f"{float(leg_info.get('x_foot', 0.0)):.6f}", f"{float(leg_info.get('z_foot', 0.0)):.6f}", f"{float(leg_info.get('hip_delta', 0.0)):.6f}", f"{float(leg_info.get('touchdown_gate', 0.0)):.6f}", f"{float(leg_info.get('coordinated_push', 0.0)):.6f}", f"{float(leg_info.get('rear_swing_gate', 0.0)):.6f}", f"{float(leg_info.get('body_x_shift', 0.0)):.6f}", f"{float(leg_info.get('thigh_ik', 0.0)):.6f}", f"{float(leg_info.get('calf_ik', 0.0)):.6f}", str(leg_info.get('swing_leg', 'none')), f'{float(target_policy_real_order[real_i]):.6f}', f'{float(target_real[real_i]):.6f}', f'{float(q_real[real_i]):.6f}', f'{float(target_real[real_i] - q_real[real_i]):.6f}', f'{float(torque[real_i]):.6f}', f'{float(temp[real_i]):.3f}', int(online[real_i]), int(error_code[real_i]), f'{float(age_ms[real_i]):.3f}', int(bool(sent)), f'{self.step_hz:.6f}', f'{self.stride_length:.6f}', f'{self.swing_height:.6f}', f'{self.duty_factor:.6f}', f'{self.send_kp:.6f}', f'{self.send_kd:.6f}'])
        self._debug_csv_file.flush()

    @staticmethod
    def publish_array(pub, arr):
        msg = Float32MultiArray()
        msg.data = np.asarray(arr, dtype=np.float32).reshape(-1).tolist()
        pub.publish(msg)

    def destroy_node(self):
        try:
            self._debug_stop_event.set()
            if self._debug_thread is not None:
                self._debug_thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            if self._debug_csv_file is not None:
                self._debug_csv_file.flush()
                self._debug_csv_file.close()
        except Exception:
            pass
        try:
            self.motor.close()
        except Exception:
            pass
        try:
            self.http_session.close()
        except Exception:
            pass
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = FanfanIkGaitNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    try:
        rclpy.shutdown()
    except Exception:
        pass
if __name__ == '__main__':
    main()
