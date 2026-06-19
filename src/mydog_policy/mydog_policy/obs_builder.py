#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import math

import numpy as np

from .motor_state_interface import MotorStateHttpInterface
from .semantic_mapper import JointSemanticMapper


class ObsBuilder36:
    """
    Build the IsaacLab policy observation.

    Layout:
        obs[0:3]    base_lin_vel
        obs[3:6]    base_ang_vel
        obs[6:9]    projected_gravity
        obs[9:12]   velocity_commands
        obs[12:24]  joint_pos = q_abs_policy - default_joint_angle
        obs[24:36]  joint_vel
        obs[36:48]  last_action, when obs_dim is 48 or 50
        obs[48:50]  gait_phase = [sin(phase), cos(phase)], only when obs_dim is 50

    base_lin_vel_source:
        command    legacy mode: obs[0:3] = [cmd_vx, cmd_vy, 0]
        zero       legacy mode: obs[0:3] = [0, 0, 0]
        estimator  obs[0:9] comes from /mydog/state_estimator, so this builder
                   does not open /dev/myimu.
    """

    def __init__(
        self,
        motor_base_url="http://127.0.0.1:8000",
        base_lin_vel_source="command",
        state_estimator_timeout_sec=0.25,
        max_motor_age_ms=300.0,
        obs_dim=36,
        semantic_yaw_180=False,
        gait_phase_period=0.55,
    ):
        self.obs_dim = int(obs_dim)
        if self.obs_dim not in (36, 48, 50):
            raise ValueError("obs_dim must be 36, 48, or 50")

        self.semantic_yaw_180 = bool(semantic_yaw_180)
        self.gait_phase_period = float(gait_phase_period)
        if self.gait_phase_period <= 1e-6:
            raise ValueError("gait_phase_period must be positive")
        self.gait_phase_start_time = time.time()
        self.base_lin_vel_source = str(base_lin_vel_source).lower()
        if self.base_lin_vel_source not in ("command", "zero", "estimator"):
            raise ValueError("base_lin_vel_source must be 'command', 'zero', or 'estimator'")

        self.use_internal_imu = self.base_lin_vel_source != "estimator"
        self.state_estimator_timeout_sec = float(state_estimator_timeout_sec)

        self.imu = None
        if self.use_internal_imu:
            from .imu_serial_interface import ImuSerialInterface

            self.imu = ImuSerialInterface(
                port="/dev/myimu",
                read_hz=100.0,
            )

        self.motor = MotorStateHttpInterface(
            base_url=motor_base_url,
            timeout=0.08,
            stale_recheck_ms=max_motor_age_ms,
        )
        self.mapper = JointSemanticMapper()

        self.cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.last_action = np.zeros(12, dtype=np.float32)

        self.base_lin_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.base_ang_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.projected_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.state_estimator_stamp = 0.0
        self.state_estimator_valid = False

    def start(self):
        if not self.use_internal_imu:
            print("[ObsBuilder] estimator mode: using /mydog/state_estimator for obs[0:9].")
            return

        print("[ObsBuilder] starting IMU...")
        self.imu.start()

        ok = self._wait_imu_ready(timeout=3.0)
        if not ok:
            raise RuntimeError("IMU not ready. Please check /dev/myimu and imu_serial_interface.py")

        print("[ObsBuilder] IMU ready.")

    def stop(self):
        if self.imu is not None:
            self.imu.stop()

    def set_command(self, vx: float, vy: float, wz: float):
        self.cmd[:] = [vx, vy, wz]

        if self.base_lin_vel_source == "command":
            self.base_lin_vel[:] = [vx, vy, 0.0]
        elif self.base_lin_vel_source == "zero":
            self.base_lin_vel[:] = [0.0, 0.0, 0.0]

    def set_last_action(self, action_12):
        action = np.asarray(action_12, dtype=np.float32).reshape(-1)
        if action.shape[0] < 12:
            raise ValueError(f"last_action must have at least 12 floats, got {action.shape[0]}")
        self.last_action[:] = action[:12]

    def get_gait_phase_obs(self):
        elapsed = time.time() - self.gait_phase_start_time
        phase = (elapsed / self.gait_phase_period) % 1.0
        angle = 2.0 * math.pi * phase
        return np.array([math.sin(angle), math.cos(angle)], dtype=np.float32)

    def transform_policy_array_for_obs(self, arr_12):
        arr = np.asarray(arr_12, dtype=np.float32).reshape(12)
        if not self.semantic_yaw_180:
            return arr.astype(np.float32).copy()
        return self.swap_policy_legs_yaw_180(arr)

    @staticmethod
    def swap_policy_legs_yaw_180(arr_12):
        arr = np.asarray(arr_12, dtype=np.float32).reshape(12)
        swapped = np.zeros(12, dtype=np.float32)

        # Policy order is FR, FL, RR, RL. A 180-degree semantic transform maps:
        # FR <-> RL and FL <-> RR. Keep hip/thigh/calf inside each leg.
        swapped[0:3] = arr[9:12]    # FR <- RL
        swapped[3:6] = arr[6:9]     # FL <- RR
        swapped[6:9] = arr[3:6]     # RR <- FL
        swapped[9:12] = arr[0:3]    # RL <- FR
        return swapped

    def set_state_estimator(self, state_9):
        """
        Update external state estimator data.

        state_9 layout:
            [0:3] base_lin_vel
            [3:6] base_ang_vel
            [6:9] projected_gravity
        """
        state = np.asarray(state_9, dtype=np.float32).reshape(-1)
        if state.shape[0] < 9:
            raise ValueError(
                f"state estimator message must have at least 9 floats, got {state.shape[0]}"
            )

        self.base_lin_vel[:] = state[0:3]
        self.base_ang_vel[:] = state[3:6]
        self.projected_gravity[:] = state[6:9]
        self.state_estimator_stamp = time.time()
        self.state_estimator_valid = True

    def _wait_imu_ready(self, timeout=3.0) -> bool:
        start = time.time()

        while time.time() - start < timeout:
            s = self.imu.get_latest()
            if s.valid:
                return True
            time.sleep(0.01)

        return False

    def _get_base_state(self):
        if self.base_lin_vel_source == "estimator":
            age = time.time() - self.state_estimator_stamp
            valid = self.state_estimator_valid and age <= self.state_estimator_timeout_sec
            return (
                self.base_lin_vel.copy(),
                self.base_ang_vel.copy(),
                self.projected_gravity.copy(),
                valid,
            )

        base_ang_vel, projected_gravity, imu_valid = self.imu.get_policy_imu_obs()
        return (
            self.base_lin_vel.copy(),
            np.asarray(base_ang_vel, dtype=np.float32).copy(),
            np.asarray(projected_gravity, dtype=np.float32).copy(),
            bool(imu_valid),
        )

    def build_obs(self):
        base_lin_vel, base_ang_vel, projected_gravity, state_valid = self._get_base_state()
        if not state_valid:
            raise RuntimeError("IMU/state estimator data invalid")

        motor_snapshot = self.motor.get_latest()
        if not motor_snapshot.valid:
            raise RuntimeError("Motor state invalid")

        q_real = motor_snapshot.q_real
        dq_real = motor_snapshot.dq_real

        q_policy, dq_policy = self.mapper.real_to_policy_q_dq(
            q_real=q_real,
            dq_real=dq_real,
        )
        q_policy = self.transform_policy_array_for_obs(q_policy)
        dq_policy = self.transform_policy_array_for_obs(dq_policy)

        obs = np.zeros(self.obs_dim, dtype=np.float32)
        obs[0:3] = base_lin_vel
        obs[3:6] = base_ang_vel
        obs[6:9] = projected_gravity
        obs[9:12] = self.cmd
        obs[12:24] = q_policy
        obs[24:36] = dq_policy
        if self.obs_dim >= 48:
            obs[36:48] = self.last_action
        if self.obs_dim >= 50:
            obs[48:50] = self.get_gait_phase_obs()

        info = {
            "imu_valid": state_valid,
            "motor_valid": motor_snapshot.valid,
            "base_lin_vel": base_lin_vel.copy(),
            "base_ang_vel": base_ang_vel.copy(),
            "projected_gravity": projected_gravity.copy(),
            "cmd": self.cmd.copy(),
            "q_real": q_real.copy(),
            "dq_real": dq_real.copy(),
            "torque_real": motor_snapshot.torque.copy(),
            "temp_real": motor_snapshot.temp.copy(),
            "q_policy": q_policy.copy(),
            "dq_policy": dq_policy.copy(),
            "last_action": self.last_action.copy(),
            "gait_phase": self.get_gait_phase_obs(),
            "online": motor_snapshot.online.copy(),
            "error_code": motor_snapshot.error_code.copy(),
            "age_ms": motor_snapshot.age_ms.copy(),
        }

        return obs, info

    def print_debug(self, obs, info):
        print("=" * 90)
        print("obs.shape:", obs.shape)
        print("\n[0:3] base_lin_vel:")
        print(obs[0:3])
        print("\n[3:6] base_ang_vel:")
        print(obs[3:6])
        print("\n[6:9] projected_gravity:")
        print(obs[6:9])
        print("\n[9:12] velocity_commands:")
        print(obs[9:12])
        print("\n[12:24] q_policy / joint_pos:")
        self._print_policy_array(info["q_policy"])
        print("\n[24:36] dq_policy / joint_vel:")
        self._print_policy_array(info["dq_policy"])
        if obs.shape[0] >= 48:
            print("\n[36:48] last_action:")
            self._print_policy_array(info["last_action"])
        if obs.shape[0] >= 50:
            print("\n[48:50] gait_phase [sin, cos]:")
            print(info["gait_phase"])
        print("\nq_real, motor order = FR, FL, RL, RR:")
        self._print_real_array(info["q_real"])
        print("\ndq_real, motor order = FR, FL, RL, RR:")
        self._print_real_array(info["dq_real"])
        print("\nage_ms:")
        print(info["age_ms"])
        print("\nonline:")
        print(info["online"])

    def _print_policy_array(self, arr):
        names = self.mapper.get_policy_joint_names()
        for i, name in enumerate(names):
            print(f"  policy[{i:02d}] {name:16s}: {arr[i]:+.5f}")

    def _print_real_array(self, arr):
        ids = self.mapper.get_real_motor_ids()
        real_names = self.mapper.real_joint_names
        for i, mid in enumerate(ids):
            print(f"  real[{i:02d}] motor_id=0x{mid:02X} {real_names[i]:16s}: {arr[i]:+.5f}")


def main():
    builder = ObsBuilder36(motor_base_url="http://127.0.0.1:8000")
    builder.start()
    builder.set_command(0.10, 0.0, 0.0)

    try:
        while True:
            obs, info = builder.build_obs()
            builder.print_debug(obs, info)
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n[ObsBuilder] stopped by user.")

    finally:
        builder.stop()


if __name__ == "__main__":
    main()
