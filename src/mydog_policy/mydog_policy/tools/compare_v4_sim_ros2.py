#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_v4_sim_ros2.py

把 IsaacLab V4 golden CSV (scripts/environments/fanfan_reference_debug.py 导出)
和 ROS2 dry-run CSV (fanfan_cpg_vmc_v4_migration_node.py 导出) 按 relative_time 对齐，
输出 q_ref / q_cmd_final / phase / swing_mask / support_mask / guard / kp / kd / clearance
的差异统计，并给出 PASS / CAUTION / FAIL 结论。

用法:
    python compare_v4_sim_ros2.py \
        --golden  /path/to/fast_diagonal_trot_balanced_mid_soft_performance_soft_output_v2_light_vmc_balance_v4_golden.csv \
        --ros     /path/to/fanfan_v4_migration_sim_compare_*.csv \
        [--out report.txt] [--start 0.5]

对齐规则:
    golden 时间列   = "time"          (从 0 开始)
    ros 时间列       = "relative_time"  (从 0 开始)
    对每个 ros 行, 取 golden 中 time 最近的一行 (nearest), 同时记录 phase_diff。
    不按 phase 对齐, 因为 warmup / soft start 会错配。

验收阈值:
    q_ref_abs_diff_max  < 0.03 rad     -> PASS
                        0.03 ~ 0.08    -> CAUTION
                        > 0.08         -> FAIL
    q_cmd_abs_diff_max  < 0.10 rad     -> PASS
                        0.10 ~ 0.25    -> CAUTION
                        > 0.25         -> FAIL

本脚本是纯 Python (numpy)，不依赖 ROS2 / IsaacLab。
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from typing import Optional

import numpy as np

# policy joint order index 0..11
POLICY_JOINT_NAMES = (
    "FR_hip", "FR_thigh", "FR_calf",
    "FL_hip", "FL_thigh", "FL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
)
POLICY_LEG_ORDER = ("FR", "FL", "RR", "RL")


# ----------------------------------------------------------------------------
# CSV 读取
# ----------------------------------------------------------------------------
def load_csv(path: str) -> tuple[list[str], list[dict]]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        rows = list(reader)
    return fields, rows


def _f(row: dict, key: str) -> float:
    v = row.get(key)
    if v is None or v == "":
        return float("nan")
    try:
        return float(v)
    except ValueError:
        return float("nan")


def col(rows: list[dict], key: str) -> np.ndarray:
    return np.asarray([_f(r, key) for r in rows], dtype=np.float64)


def col_exists(fields: list[str], key: str) -> bool:
    return key in fields


# ----------------------------------------------------------------------------
# golden / ros 列名映射
# ----------------------------------------------------------------------------
def golden_joint_matrix(rows: list[dict], prefix: str) -> np.ndarray:
    """golden 每关节列 prefix_0 .. prefix_11 -> (N,12)。"""
    return np.stack([col(rows, f"{prefix}_{i}") for i in range(12)], axis=1)


def golden_leg_matrix(rows: list[dict], prefix: str) -> np.ndarray:
    """golden 每腿列 prefix_0 .. prefix_3 -> (N,4) (FR,FL,RR,RL)。"""
    return np.stack([col(rows, f"{prefix}_{i}") for i in range(4)], axis=1)


def ros_joint_matrix(rows: list[dict], prefix: str) -> np.ndarray:
    """ros 每关节列 prefix_FR_hip .. prefix_RL_calf -> (N,12)。"""
    return np.stack([col(rows, f"{prefix}_{jn}") for jn in POLICY_JOINT_NAMES], axis=1)


def ros_leg_matrix(rows: list[dict], prefix: str) -> np.ndarray:
    """ros 每腿列 prefix_FR .. prefix_RL -> (N,4)。"""
    return np.stack([col(rows, f"{prefix}_{ln}") for ln in POLICY_LEG_ORDER], axis=1)


def ros_joint_prefix(fields: list[str], *candidates: str) -> str:
    """从多个候选前缀里挑第一个在 ros CSV 中存在的 (兼容 *_sim 与旧 *_policy)。"""
    for c in candidates:
        if f"{c}_FR_hip" in fields:
            return c
    return candidates[0]


# ----------------------------------------------------------------------------
# 统计
# ----------------------------------------------------------------------------
def diff_stats(name: str, a: np.ndarray, b: np.ndarray) -> dict:
    """a = ros, b = golden, 同形状 (N,) 或 (N,K)。"""
    d = np.abs(a - b)
    finite = np.isfinite(d)
    if not np.any(finite):
        return {"name": name, "valid": False, "n": 0, "max": float("nan"),
                "mean": float("nan"), "p95": float("nan")}
    dv = d[finite]
    return {
        "name": name,
        "valid": True,
        "n": int(dv.size),
        "max": float(np.max(dv)),
        "mean": float(np.mean(dv)),
        "p95": float(np.percentile(dv, 95.0)),
    }


def grade_q_ref(max_diff: float) -> str:
    if not math.isfinite(max_diff):
        return "N/A"
    if max_diff < 0.03:
        return "PASS"
    if max_diff <= 0.08:
        return "CAUTION"
    return "FAIL"


def grade_q_cmd(max_diff: float) -> str:
    if not math.isfinite(max_diff):
        return "N/A"
    if max_diff < 0.10:
        return "PASS"
    if max_diff <= 0.25:
        return "CAUTION"
    return "FAIL"


# ----------------------------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Compare IsaacLab V4 golden CSV vs ROS2 migration dry-run CSV.")
    ap.add_argument("--golden", required=True, help="IsaacLab V4 golden CSV path")
    ap.add_argument("--ros", required=True, help="ROS2 migration dry-run CSV path")
    ap.add_argument("--out", default="", help="optional report text output path")
    ap.add_argument("--start", type=float, default=0.0, help="ignore ros rows with relative_time < start (skip warmup)")
    args = ap.parse_args(argv)

    g_fields, g_rows = load_csv(args.golden)
    r_fields, r_rows = load_csv(args.ros)
    if not g_rows or not r_rows:
        print("[ERROR] empty CSV.", file=sys.stderr)
        return 2

    g_time = col(g_rows, "time")
    if not np.any(np.isfinite(g_time)):
        # 某些 golden 没有 time 列时退化用行号 * 0.02
        g_time = np.arange(len(g_rows), dtype=np.float64) * 0.02
    r_time = col(r_rows, "relative_time")

    # nearest-time 对齐: 对每个 ros 行选最近 golden 行
    aligned_g = []   # golden row index per ros row
    aligned_r = []   # ros row index
    for ri in range(len(r_rows)):
        rt = r_time[ri]
        if not math.isfinite(rt) or rt < args.start:
            continue
        gi = int(np.argmin(np.abs(g_time - rt)))
        aligned_g.append(gi)
        aligned_r.append(ri)
    if not aligned_r:
        print("[ERROR] no aligned rows (check --start / time columns).", file=sys.stderr)
        return 2
    g_sel = [g_rows[i] for i in aligned_g]
    r_sel = [r_rows[i] for i in aligned_r]

    lines = []

    def emit(text=""):
        lines.append(text)
        print(text)

    emit("=" * 78)
    emit("IsaacLab V4 golden  vs  ROS2 migration dry-run")
    emit(f"  golden : {args.golden}  rows={len(g_rows)}")
    emit(f"  ros    : {args.ros}  rows={len(r_rows)}")
    emit(f"  aligned rows = {len(r_sel)} (start>={args.start:.3f}s, nearest-time alignment)")
    emit("=" * 78)

    # ---- 时间 / phase ----
    g_t = np.asarray([g_time[i] for i in aligned_g])
    r_t = np.asarray([r_time[i] for i in aligned_r])
    emit(f"[TIME]  ros - golden  mean={np.mean(r_t - g_t):+.4f}s  max|.|={np.max(np.abs(r_t - g_t)):.4f}s")

    g_phase = col(g_sel, "base_phase")
    r_phase = col(r_sel, "phase")
    # phase 是周期量, 用环形差
    phase_d = np.abs((r_phase - g_phase + 0.5) % 1.0 - 0.5)
    phase_d = phase_d[np.isfinite(phase_d)]
    if phase_d.size:
        emit(f"[PHASE] |phase_diff| (wrap)  max={np.max(phase_d):.4f}  mean={np.mean(phase_d):.4f}")

    # ---- 收集对比项 ----
    stats = []

    # q_ref / q_cmd_final (核心验收): ros 用 sim_semantic 空间列 (*_sim)，golden 用 q_ref / q_cmd_final
    p_qref = ros_joint_prefix(r_fields, "q_ref_sim", "q_ref_policy")
    p_qcmd = ros_joint_prefix(r_fields, "q_cmd_final_sim", "q_cmd_final_policy")
    s_qref = diff_stats("q_ref(sim)", ros_joint_matrix(r_sel, p_qref), golden_joint_matrix(g_sel, "q_ref"))
    s_qcmd = diff_stats("q_cmd_final(sim)", ros_joint_matrix(r_sel, p_qcmd),
                        golden_joint_matrix(g_sel, "q_cmd_final"))
    stats.append(s_qref)
    stats.append(s_qcmd)

    # swing / support mask
    stats.append(diff_stats("swing_mask", ros_leg_matrix(r_sel, "swing_mask"), golden_leg_matrix(g_sel, "swing_mask")))
    stats.append(diff_stats("support_mask", ros_leg_matrix(r_sel, "support_mask"),
                            golden_leg_matrix(g_sel, "support_mask")))

    # leg_phase
    stats.append(diff_stats("leg_phase", ros_leg_matrix(r_sel, "leg_phase"), golden_leg_matrix(g_sel, "leg_phase")))

    # kp / kd
    stats.append(diff_stats("kp", ros_joint_matrix(r_sel, "kp"), golden_joint_matrix(g_sel, "kp")))
    stats.append(diff_stats("kd", ros_joint_matrix(r_sel, "kd"), golden_joint_matrix(g_sel, "kd")))

    # clearance (predicted foot lift): golden predicted_foot_height_0..3 vs ros predicted_foot_height_*
    stats.append(diff_stats("predicted_foot_height",
                            ros_leg_matrix(r_sel, "predicted_foot_height"),
                            golden_leg_matrix(g_sel, "predicted_foot_height")))

    # rear guard (RR=2, RL=3)
    g_late = golden_leg_matrix(g_sel, "rear_late_swing_guard_active")
    r_late = np.stack([col(r_sel, "rear_late_swing_guard_active_RR"),
                       col(r_sel, "rear_late_swing_guard_active_RL")], axis=1)
    stats.append(diff_stats("rear_late_swing_guard(RR,RL)", r_late, g_late[:, [2, 3]]))

    g_early = golden_leg_matrix(g_sel, "rear_early_contact_guard_active")
    r_early = np.stack([col(r_sel, "rear_early_contact_guard_active_RR"),
                        col(r_sel, "rear_early_contact_guard_active_RL")], axis=1)
    stats.append(diff_stats("rear_early_contact_guard(RR,RL)", r_early, g_early[:, [2, 3]]))

    # rate limiter delta
    stats.append(diff_stats("rate_limited_delta",
                            ros_joint_matrix(r_sel, "rate_limited_delta"),
                            golden_joint_matrix(g_sel, "rate_limit_delta")))

    # tau_est
    stats.append(diff_stats("tau_est", ros_joint_matrix(r_sel, "tau_est"),
                            golden_joint_matrix(g_sel, "tau_est")))

    emit("")
    emit(f"{'field':32s} {'n':>6s} {'max':>10s} {'mean':>10s} {'p95':>10s}")
    emit("-" * 78)
    for st in stats:
        if not st["valid"]:
            emit(f"{st['name']:32s} {'--':>6s} {'(missing columns)':>32s}")
            continue
        emit(f"{st['name']:32s} {st['n']:6d} {st['max']:10.4f} {st['mean']:10.4f} {st['p95']:10.4f}")
    emit("-" * 78)

    # ---- 结论 ----
    ref_grade = grade_q_ref(s_qref["max"]) if s_qref["valid"] else "N/A"
    cmd_grade = grade_q_cmd(s_qcmd["max"]) if s_qcmd["valid"] else "N/A"
    emit("")
    emit("[VERDICT]")
    emit(f"  q_ref_abs_diff_max   = {s_qref['max']:.4f} rad -> {ref_grade}   "
         f"(PASS<0.03, CAUTION<=0.08, FAIL>0.08)")
    emit(f"  q_cmd_abs_diff_max   = {s_qcmd['max']:.4f} rad -> {cmd_grade}   "
         f"(PASS<0.10, CAUTION<=0.25, FAIL>0.25)")

    if ref_grade != "PASS" or cmd_grade == "FAIL":
        emit("")
        emit("[DIFF SOURCES] 若未通过, 按以下清单排查:")
        emit("  - default stand 不一致 (检查 default_policy_diff_max / stand_source=sim_v4)")
        emit("  - phase / warmup 不一致 (检查 [PHASE] / [TIME], fast_trot_warmup_sec, step_hz)")
        emit("  - support preload 不一致 (fast_trot_support_preload_z_m / gate_max / ramp)")
        emit("  - VMC input 缺失 (height_source / IMU valid; dry-run 应 VMC≈0)")
        emit("  - contact force 缺失 (early_contact_source=q_error 替代)")
        emit("  - rate limiter 不一致 (sim_target_rate_limit / hip_target_rate_mul)")
        emit("  - torque backoff 不一致 (soft_output_start/full, q_actual 替代模型)")
        emit("  - kp/kd 不一致 (mid_soft profile / rear touchdown ramp)")
        emit("  - joint sign / order 不一致 (policy order, URDF hip signs)")

    emit("=" * 78)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[compare] report written -> {args.out}")

    # 退出码: PASS=0, CAUTION=0, FAIL=1
    if ref_grade == "FAIL" or cmd_grade == "FAIL":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
