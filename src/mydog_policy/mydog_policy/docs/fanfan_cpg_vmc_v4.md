# Fanfan CPG + Light VMC v4 Node

This node is a reference-only hardware bring-up tool for the IsaacLab
`performance_soft_output_v2_light_vmc_balance_v4` diagonal trot candidate.

It does not run RL, does not load `.pt`/ONNX checkpoints, and does not use torque
control.  It only generates position targets:

```text
FastDiagonalTrot CPG
-> light VMC position offsets
-> rear late-swing / early-contact v4 guards
-> no-overshoot rate limiter
-> torque/current-aware position backoff
-> JointSemanticMapper policy->real motor order
-> existing HTTP position target batch
```

The default is safe dry-run:

```bash
ros2 run mydog_policy fanfan_cpg_vmc_v4_node --ros-args \
  -p enable_send:=false \
  -p test_mode:=air \
  -p duration_s:=5.0
```

Build:

```bash
cd ~/mydog_ros2_ws
colcon build --packages-select mydog_policy
source install/setup.bash
```

Air dry-run:

```bash
ros2 run mydog_policy fanfan_cpg_vmc_v4_node --ros-args \
  -p enable_send:=false \
  -p test_mode:=air \
  -p duration_s:=5.0 \
  -p dry_run_virtual_feedback:=true
```

`dry_run_virtual_feedback=true` is the default.  In dry-run, safety uses virtual
feedback from the previous command instead of the stationary real motor angles.
The CSV still records real motor feedback separately.  This prevents the
dry-run from falsely triggering torque backoff simply because the motors are not
being commanded.

Air send:

```bash
ros2 run mydog_policy fanfan_cpg_vmc_v4_node --ros-args \
  -p enable_send:=true \
  -p test_mode:=air \
  -p duration_s:=5.0
```

Touch / assist tests must be hand-supported:

```bash
ros2 run mydog_policy fanfan_cpg_vmc_v4_node --ros-args \
  -p enable_send:=true \
  -p test_mode:=touch \
  -p duration_s:=3.0

ros2 run mydog_policy fanfan_cpg_vmc_v4_node --ros-args \
  -p enable_send:=true \
  -p test_mode:=assist \
  -p duration_s:=3.0
```

Short free test is capped to 3 seconds unless `allow_long_free_test:=true`:

```bash
ros2 run mydog_policy fanfan_cpg_vmc_v4_node --ros-args \
  -p enable_send:=true \
  -p test_mode:=short_free \
  -p duration_s:=1.0
```

CSV defaults to:

```text
~/mydog_ros2_ws/src/mydog_policy/mydog_policy/docs/cpg_vmc_v4_YYYYMMDD_HHMMSS.csv
```

Use `-p csv_path:=...` to override.

Safety notes:

- `enable_send=false` by default.
- Dry-run uses virtual feedback by default; real sends always use real motor
  feedback for safety.
- `touch`, `assist`, and `short_free` require a valid IMU before sending.
- `assist` and `short_free` require motor feedback before sending.
- `enable_send=true` checks that the robot is close to the start stand pose
  unless `allow_start_from_any_pose:=true` is set.
- Safety stop performs a soft stop back to the default stand target.
- If torque/current/temp feedback is unavailable, CSV records `nan`; q-error based
  early-contact guard still works.
- URDF hip outward signs are fixed to `FR=-1, FL=+1, RR=-1, RL=+1`.
