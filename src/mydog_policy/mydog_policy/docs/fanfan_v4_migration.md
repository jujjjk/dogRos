# Fanfan V4 → ROS2 真机迁移说明

> 目标：把 IsaacLab 仿真中已验证的
> **FastDiagonalTrot(balanced) + Light VMC + safety_profile=`performance_soft_output_v2_light_vmc_balance_v4` + support_kp_level=`mid_soft`**
> 原样迁移到 ROS2 真机，而不是重新调一个新的真机踏步节点。
>
> 验收标准不是“节点能跑”，而是 ROS2 dry-run CSV 与 IsaacLab V4 golden CSV 对齐
> （`q_ref` / `q_cmd_final` / `phase` / `swing_mask` / `support_mask` / rear guard / `kp`/`kd` / rate filter）。

---

## 0. 文件清单

| 文件 | 作用 |
|---|---|
| `mydog_policy/fanfan_v4_migration_core.py` | 纯 Python(numpy) 迁移核心，**只在 sim_semantic 空间工作**。FK/IK、phase、swing/stance、Light VMC、yaw damping、rear guard、rate limiter、torque soft backoff 逐项迁移 |
| `mydog_policy/sim_real_semantic_bridge.py` | sim_semantic ↔ real_policy 语义桥（推导自 mapper + IsaacLab 语义）|
| `mydog_policy/fanfan_cpg_vmc_v4_migration_node.py` | ROS2 外壳，负责三空间转换 + IMU/反馈/发送/CSV/安全，不重新写 gait |
| `mydog_policy/tools/compare_v4_sim_ros2.py` | 把 golden CSV 与 ROS2 dry-run CSV 按 `relative_time` 对齐（**sim_semantic 空间**）并输出差异统计 |
| `mydog_policy/docs/fanfan_v4_migration.md` | 本文档 |
| `setup.py` | 已含 `fanfan_cpg_vmc_v4_migration_node` console script，无需改动 |

## 0.1 三个 joint 空间（必须分清楚）

| 空间 | 定义 | 谁用 |
|---|---|---|
| **sim_semantic** | IsaacLab / golden CSV 的空间（`q_ref_0~11` / `q_cmd_final_0~11`），policy 顺序 FR,FL,RR,RL，符号全 +1，零点 0 | V4 core 内部 + sim_compare |
| **real_policy** | `JointSemanticMapper.real_to_policy_abs_q_dq()` 返回的空间，policy 顺序但符号/零点由真机 mapper 定义 | ROS2 wrapper 中转 |
| **real_motor** | 真正发给电机 0x11~0x43 的空间，`mapper.policy_target_to_real_target()` | HTTP 发送 |

**桥的推导（不是猜）**：IsaacLab 部署契约 `q_real_motor = real_sign_isaac ⊙ q_sim`；mapper `q_real_policy = mapper.joint_sign ⊙ q_real_ordered`。代入得
`sign = mapper.joint_sign · real_sign_isaac`，`offset = mapper.joint_sign ⊙ (real_zero_isaac − mapper.real_zero)`。
当前仓库 `mapper.joint_sign == real_sign_isaac` 且零点均为 0，所以
**桥 = 恒等（sign=+1, offset=0, index=identity, roundtrip_error=0）**，即 sim_semantic ≡ real_policy。
桥仍保留显式 sign/offset/index 并在启动打印 + 自检，日后 mapper 改符号/零点会自动跟随。

> 结论：既然桥在当前语义下是恒等，`q_ref` 的 sim↔real_policy **不是空间问题**。
> 若 `q_ref_abs_diff_max` 仍大，应查 phase / warmup / support preload（下一轮，本轮不动 gait）。

## 0.2 数据链路

```
电机反馈 → mapper.real_to_policy_abs_q_dq → real_policy
        → bridge.real_policy_to_sim       → sim_semantic → core.step
core.step → q_cmd_final_sim
        → bridge.sim_to_real_policy        → real_policy
        → mapper.policy_target_to_real_target → real_motor → HTTP
sim_compare: core q_ref_sim / q_cmd_final_sim  ⟷  golden q_ref_* / q_cmd_final_*  (都在 sim_semantic)
```

core 的公开输入/输出变量名已改清楚：`q_actual_sim / q_cmd_final_sim / q_ref_sim / kp_sim / kd_sim ...`，
core 内部**不再调用** `JointSemanticMapper`。

迁移核心严格对应的 IsaacLab 源码：

```
scripts/environments/fanfan_reference_debug.py                       # golden CSV 生成 / safety_profile 覆盖 / CSV 列
.../fanfan_rl_cpg_residual/residual_action.py                        # FastDiagonalTrot + Light VMC + 输出滤波 + 安全链
.../fanfan_rl_cpg_residual/reference_gait.py                         # full sagittal FK/IK + base_phase
.../fanfan_rl_cpg_residual/flat_env_cfg.py                           # 任务默认站姿 / dt / reference_cfg
.../fanfan_a1_clean/fanfan_robot_cfg.py                              # FANFAN_TEXT_STAND_JOINT_POS + 站姿覆盖
.../fanfan_a1_clean/deploy_actions.py                               # DeployFilteredJointPositionAction
.../fanfan_a1_clean/rs01_motor_params.py                            # RS01 KP/KD/torque
```

---

## 1. 已迁移并锁定的真实参数

### 1.1 控制率 / FK·IK

- `dt = 0.02s`（velocity_env_cfg：`decimation=4`，`sim.dt=0.005` → 50 Hz）。
- full sagittal FK/IK：`thigh_length=0.1560608`，`calf_length=0.1489418`，`workspace_margin_m=0.005`
  （来自 `reference_gait.py` 与 `FanfanSmallHighFreqReferenceGaitCfg`），**不使用线性近似 IK**。

### 1.2 balanced preset（`trot_preset=balanced`）

| 参数 | 值 |
|---|---|
| `fast_trot_step_hz` | 1.15 |
| `fast_trot_duty_factor` | 0.61（→ swing_fraction=0.39）|
| `fast_trot_stride_length_m` | 0.022 |
| `fast_trot_front_swing_height_m` | 0.048 |
| `fast_trot_rear_swing_height_m` | 0.067 |
| `fast_trot_support_preload_z_m` | 0.0055（v4 覆盖 balanced 的 0.009）|
| `fast_trot_warmup_sec` | 2.0 |

### 1.3 support_kp_level = `mid_soft`（v2+ profile override）

| 阶段 | hip / thigh / calf Kp | Kd |
|---|---|---|
| swing | 50 / 80 / 80 | 5.0 |
| touchdown | 55 / 110 / 120 | 6.0 |
| early stance | 60 / 130 / 140 | 6.0 |
| support | 62 / 140 / 150 | 6.0 |

### 1.4 safety_profile = `performance_soft_output_v2_light_vmc_balance_v4`

- 输出滤波链：`enable_deploy_target_filter=True`、`rate_limit=True`、`accel_limit=False`、`torque_target_limit=True`、`action_delay=False`。
- rate limit：`sim_target_rate_limit=9.0`，`hip_target_rate_mul=7.5/9`，thigh/calf=1.0（`kd_scale=1` → `damping_scale=1`，所以 hip≈7.5、thigh/calf=9.0 rad/s）。
- torque soft backoff：`soft_output_start=10`、`soft_output_full=14`、guard `9.5/13.5`、`sim_hard_torque_budget=17`，`scale = 1 - 0.18·soft_t - 0.32·hard_t`。
- Light VMC（balance v3/v4）：`target_base_height=0.288`、`target_pitch=-0.04`、`z_sign=-1`、`pitch_sign=-1`、`enable_foot_placement=False`。
- yaw damping：`kp=0.0025`、`kd=0.006`、`hip_limit=0.007`、`rate=0.001`。
- rear preswing unload：`enable=True`、`z=0.0`（v4）。
- rear touchdown Kp/Kd ramp：`kp_scale=0.75`，hip/thigh/calf limit `58/125/130`，`kd=6.2`。
- rear late-swing clearance guard + descent softening：`phase 0.28~0.38`，`descent_scale=0.50`。
- rear early-contact guard：`force_threshold=10`、`phase 0.28~0.40`、`kp_scale=0.60`、`kd=6.5`、torque soft `9/13`。

> 核心对 preset / kp_level / safety_profile 做了硬校验：只接受上述组合，传入其它组合会直接报错，
> 避免“节点自己发明一个新步态”。

---

## 2. 关节顺序 / hip 符号（固定）

policy joint order：

```
FR_hip, FR_thigh, FR_calf,
FL_hip, FL_thigh, FL_calf,
RR_hip, RR_thigh, RR_calf,
RL_hip, RL_thigh, RL_calf
```

URDF hip outward signs（**禁止 legacy `(1,1,-1,1)`**）：

```
FR = -1.0, FL = +1.0, RR = -1.0, RL = +1.0
```

real ↔ policy 映射沿用 `semantic_mapper.JointSemanticMapper`：
`policy_to_real_index=(0,1,2,3,4,5,9,10,11,6,7,8)`，
`joint_sign=(-1,1,1, -1,-1,-1, 1,1,1, 1,-1,-1)`（policy 序）。
真机电机目标用 `mapper.policy_target_to_real_target(q_cmd_final)` 得到，CSV 记录
`raw_motor_target_0x11..0x43` 与 `semantic_to_motor_mapping_ok`。

---

## 3. default stand pose（sim_v4 vs mapper）

`stand_source` 参数：`sim_v4`(默认) / `mapper` / `fallback`。

- **sim_v4 default（level rear stand pose）**：四条腿 thigh=0.3491、calf=-0.7854，hip=±0.1571。
  来自 `FANFAN_TEXT_STAND_JOINT_POS` 经 `flat_env_cfg._set_rear_stand_pose` 把后腿
  thigh `0.2269→0.3491`、calf `-0.3491→-0.7854` 覆盖（FastDiagonalTrot 用
  `apply_default_pose_offsets=False`，不再叠加 offset）。
- **mapper default**：`JointSemanticMapper.default_joint_angle` 后腿仍是 0.2269 / -0.3491。

**默认站姿对比必须在 real_policy 空间做**（跨空间直接比没有意义）：

```
default_real_policy_diff = bridge.sim_to_real_policy(sim_v4_default_sim) - mapper_default_real_policy
```

当前桥是恒等，所以该差异 ≈ `0.4363 rad`（后腿 calf 差），**远大于 0.05 rad**。
因此 `stand_source=sim_v4` 时节点打印 WARNING：

```
真机必须先 stand_only 进入 sim_v4 default，不能直接从 mapper default 进入 gait。
```

CSV 记录 `sim_v4_default_sim_*`、`sim_v4_default_real_policy_*`、`mapper_default_real_policy_*`、
`default_real_policy_diff_*`、`default_real_policy_diff_max`，
以及 `bridge_enabled`、`bridge_roundtrip_error_max`、`sim_to_real_index_*`、`sim_to_real_sign_*`、`sim_to_real_offset_*`。

---

## 4. 真机传感器缺失替代规则

原则：**CPG reference 本体与仿真一致；真机传感器缺失只允许影响 VMC correction / early-contact / late-swing 替代量，不改 CPG reference。**

| 仿真量 | 真机缺失时 | CSV 记录 |
|---|---|---|
| base height | `height VMC = 0`（或 `use_base_height_estimate=true` 用 `base_height_estimate_m`）| `height_source = unavailable/estimated/measured` |
| foot contact force | rear early-contact 用摆动腿 `q_error` 替代（阈值 0.12 rad）| `early_contact_source = force/q_error/unavailable` |
| foot body z | rear late-swing guard 用 reference predicted foot height(FK) | `predicted_foot_height_*` |

dry-run（`dry_run_virtual_feedback=true`）下 IMU 视为水平、base_height=target、无 contact，
所以 Light VMC ≈ 0，`q_ref ≈ q_cpg`，便于和 golden 的 reference 对齐。

---

## 4.1 sim_compare = golden 输入回放（验证数学等价）

`test_mode=sim_compare` 且加载了 golden CSV 时，wrapper **把 golden 行里的 base 状态 / q_actual / dq / foot force 喂回 core**，
即“相同输入下检验迁移数学是否等价”，**不改 gait / VMC / 滤波数学**：

- `q_ref` 的 support/stance foot z 差异（约 5mm）来自 **Light VMC**：golden 有真实 base height/roll/pitch，dry-run 没有。
  喂入 golden base 状态后，core 复算出同样的 `vmc_foot_z_offset`（实测喂 pitch=0.05/height=0.300 → z offset ≈ 5.3mm），`q_ref` 即对齐。
- `q_cmd_final` 差异来自 **torque backoff 的 q_current / qd_current**：
  - 修正 1：core 的 PD 力矩补回了仿真的阻尼项 `tau = kp·(q_target−q_current) − kd·qd_current`（之前漏了 `−kd·qd`）。
  - 修正 2：sim_compare 喂入 golden 的 `q_actual`（作 q_current）和有限差分 `dq`（作 qd_current），backoff 即复刻 golden。
- off-by-one：sim 在 step *t* 用的是“上一步物理后”的状态（golden row *t−1*）算 `q_cmd[t]`，
  所以 INPUT 取 `rel−dt` 处的 golden 行，输出再对齐 `rel` 处。

> 真机 (air/touch/...) 仍用真实反馈；sim_compare 的回放只为验证数学端口正确。

## 5. 输出流水线（与仿真一致）

```
q_cpg_policy            纯 CPG（FK/IK，不含 VMC）
 + q_vmc_delta_policy   Light VMC(height/roll/pitch) + yaw damping + rear unload/guard
 = q_ref_policy         与 golden 的 q_ref 对齐
 → rate limiter         hip 7.5 / thigh·calf 9.0 rad/s（no-overshoot crossing 处理）
 → torque soft backoff  soft_output_v2（含 phase-switch guard + early-contact guard）
 = q_cmd_final_policy    与 golden 的 q_cmd_final 对齐
 → JointSemanticMapper  → raw_motor_target_0x11..0x43 → HTTP 电机
```

> 说明：仿真里 `q_cpg` 列与 `q_ref` 列数值相同（VMC 被折进 CPG 目标）。本迁移把 `q_cpg`
> 输出为“纯 CPG（未叠加 VMC）”更利于诊断；dry-run 下 VMC≈0，二者基本相等。验收只看
> `q_ref` 与 `q_cmd_final`。

---

## 6. 编译

```bash
cd ~/mydog_ros2_ws
colcon build --packages-select mydog_policy
source install/setup.bash
```

---

## 7. 先导出 IsaacLab golden CSV（金标准）

在 IsaacLab 机器上：

```bash
./isaaclab.sh -p scripts/environments/fanfan_reference_debug.py \
  --task Isaac-Velocity-Flat-FanfanRlCpgResidual-FastDiagonalTrot-SafeReference-v0 \
  --num_envs 1 \
  --duration 20 \
  --trot_preset balanced \
  --support_kp_level mid_soft \
  --safety_profile performance_soft_output_v2_light_vmc_balance_v4 \
  --output logs/reference_debug/fast_diagonal_trot_balanced_mid_soft_performance_soft_output_v2_light_vmc_balance_v4_golden.csv
```

把这个 CSV 拷到 Jetson，作为 `sim_compare_csv_path`。

---

## 8. 运行顺序（sim_compare 通过前禁止上真机）

### 8.1 先跑 sim_compare（不发电机）

```bash
ros2 run mydog_policy fanfan_cpg_vmc_v4_migration_node --ros-args \
  -p enable_send:=false \
  -p test_mode:=sim_compare \
  -p duration_s:=5.0 \
  -p dry_run_virtual_feedback:=true \
  -p stand_source:=sim_v4 \
  -p sim_compare_csv_path:=/path/to/..._golden.csv
```

然后离线比对（也可直接看节点打印的 `[SIM_COMPARE]` 结论）：

```bash
python3 src/mydog_policy/mydog_policy/tools/compare_v4_sim_ros2.py \
  --golden /path/to/..._golden.csv \
  --ros    /path/to/fanfan_v4_migration_sim_compare_*.csv \
  --start  0.3
```

### 8.2 普通 dry-run（air）

```bash
ros2 run mydog_policy fanfan_cpg_vmc_v4_migration_node --ros-args \
  -p enable_send:=false -p test_mode:=air -p duration_s:=5.0 \
  -p dry_run_virtual_feedback:=true -p stand_source:=sim_v4
```

### 8.3 通过后才允许上真机（依次）

```bash
# stand_only：当前 q_actual -> sim_v4 default 软插值 3~5s
ros2 run mydog_policy fanfan_cpg_vmc_v4_migration_node --ros-args \
  -p enable_send:=true -p test_mode:=stand_only -p duration_s:=4.0 -p stand_source:=sim_v4

# 架空
ros2 run mydog_policy fanfan_cpg_vmc_v4_migration_node --ros-args \
  -p enable_send:=true -p test_mode:=air -p duration_s:=8.0 \
  -p dry_run_virtual_feedback:=false -p stand_source:=sim_v4

# 轻触
ros2 run mydog_policy fanfan_cpg_vmc_v4_migration_node --ros-args \
  -p enable_send:=true -p test_mode:=touch -p duration_s:=6.0 \
  -p dry_run_virtual_feedback:=false -p stand_source:=sim_v4
```

`enable_send=true` 时：倒计时 3 秒；自动禁用 `dry_run_virtual_feedback`；用真实
`q_actual/dq/torque/temp` 做 safety；`touch/assist/short_free` 缺 IMU 拒绝启动；
`assist/short_free` 缺电机反馈拒绝启动；`require_stand_ready=true` 时若 `q_actual`
偏离 sim_v4 default 超过 `stand_ready_tol_rad` 则拒绝进入 gait。

---

## 9. test_mode

| 模式 | 发送 | VMC 比例 | 说明 |
|---|---|---|---|
| `sim_compare` | 否 | 1.0 | 只和 golden 对齐 |
| `air` | 可选 | 0.0 | 架空，验证方向/抬脚/连续性 |
| `touch` | 可选 | 0.25 | 脚尖轻触，人工扶住 |
| `assist` | 可选 | 0.5 | 半承重手扶 |
| `short_free` | 可选 | 1.0 | 极短放手，`duration_s<=3.0`（除非 `allow_long_free_test=true`）|
| `stand_only` | 可选 | 0.0 | 当前 q_actual → sim_v4 default 软插值，不启 CPG/VMC |

VMC 比例只缩放 VMC correction，不改 CPG reference。

---

## 10. 安全停止

触发条件：`roll>12°`、`pitch>12°`、`max_q_error>0.45rad`、`tau>17Nm`、`current>current_stop`、
`temp>75°C`、`communication_timeout`、`Ctrl+C`、`duration reached`。
停止时执行 **soft stop**：在 `soft_stop_sec` 内平滑插值回 sim_v4 default 站姿，不直接断 target。

---

## 11. 验收阈值

| 指标 | PASS | CAUTION | FAIL |
|---|---|---|---|
| `q_ref_abs_diff_max` | < 0.03 rad | 0.03~0.08 | > 0.08 |
| `q_cmd_abs_diff_max` | < 0.10 rad | 0.10~0.25 | > 0.25 |

`phase / leg_phase / swing_mask / support_mask` 必须对齐；`q_ref_policy` 紧对齐；
`q_cmd_final_policy` 尽量对齐；`default_joint_pos` 与 hip signs 对齐；
`fk_clearance_ref` 达到仿真量级（front≈0.048、rear≈0.067）；rear guard / 输出滤波差异
能被 CSV 字段解释清楚。**sim_compare 通过前不允许上真机。**

`compare_v4_sim_ros2.py` / 节点 summary 未通过时会打印差异来源排查清单：
default stand / phase·warmup / support preload / VMC input / contact force /
rate limiter / torque backoff / kp·kd / joint sign·order。
