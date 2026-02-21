#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import argparse
import numpy as np
import zarr
from tqdm import tqdm
import concurrent.futures

# make project importable
SCRIPT_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_PATH, "../../")))

from spatialmath.base import q2r, r2q
from spatialmath import SE3

from PyriteUtility.planning_control import compliance_helpers as ch
from PyriteUtility.spatial_math import spatial_utilities as su
from PyriteUtility.planning_control.filtering import LiveLPFilter


def nearest_index(sorted_ts: np.ndarray, t: float) -> int:
    """
    Return index of the closest timestamp in a sorted 1D array.
    Uses np.searchsorted for speed and stability.
    """
    if len(sorted_ts) == 0:
        raise ValueError("empty timestamp array")
    i = int(np.searchsorted(sorted_ts, t))
    if i <= 0:
        return 0
    if i >= len(sorted_ts):
        return len(sorted_ts) - 1
    # choose closer of i-1 and i
    if abs(sorted_ts[i] - t) < abs(sorted_ts[i - 1] - t):
        return i
    return i - 1


def filter_wrench(
    wrench: np.ndarray,
    mode: str,
    fs_hz: float,
    cutoff_hz: float,
    order: int,
    moving_avg_window: int,
) -> np.ndarray:
    """
    Filter wrench for stability of compliance estimation.
    - mode="lpf": LiveLPFilter
    - mode="moving_avg": per-dim moving average
    - mode="none": no filtering
    """
    wrench = np.asarray(wrench, dtype=np.float32)
    if mode == "none":
        return wrench

    if mode == "lpf":
        ft_filter = LiveLPFilter(fs=fs_hz, cutoff=cutoff_hz, order=order, dim=6)
        return np.array([ft_filter(y) for y in wrench], dtype=np.float32)

    if mode == "moving_avg":
        N = int(moving_avg_window)
        if N <= 1:
            return wrench
        out = np.zeros_like(wrench, dtype=np.float32)
        kernel = np.ones(N, dtype=np.float32) / float(N)
        # same-length convolution
        for i in range(6):
            out[:, i] = np.convolve(wrench[:, i], kernel, mode="same")
        return out

    raise ValueError(f"Unknown filter mode: {mode}")


def process_episode_right_only(
    ep_name: str,
    ep_data,
    # ids
    right_id: int,
    # estimator params
    k_max: float,
    k_min: float,
    f_low: float,
    f_high: float,
    dim: int,
    characteristic_length: float,
    vel_tol: float,
    # wrench filter params
    wrench_filter_mode: str,
    fs_hz: float,
    cutoff_hz: float,
    order: int,
    moving_avg_window: int,
    # velocity window
    half_window_size: int,
    plot=True,
    plot_every_n=20,
):
    """
    Compute and write:
      - ts_pose_virtual_target_{right_id}  (N,7)
      - stiffness_{right_id}              (N,)
    Left hand is skipped entirely.
    """

    # ---- required keys (right only) ----
    pose_key = f"ts_pose_fb_{right_id}"
    wrench_key = f"wrench_{right_id}"
    robot_ts_key = f"robot_time_stamps_{right_id}"
    wrench_ts_key = f"wrench_time_stamps_{right_id}"

    if pose_key not in ep_data:
        print(f"[{ep_name}] missing {pose_key}, skip.")
        return False
    if wrench_key not in ep_data:
        print(f"[{ep_name}] missing {wrench_key}, skip.")
        return False
    if robot_ts_key not in ep_data:
        print(f"[{ep_name}] missing {robot_ts_key}, skip.")
        return False
    if wrench_ts_key not in ep_data:
        print(f"[{ep_name}] missing {wrench_ts_key}, skip.")
        return False

    ts_pose_fb = np.asarray(ep_data[pose_key], dtype=np.float32)          # (N,7)
    wrench = np.asarray(ep_data[wrench_key], dtype=np.float32)            # (M,6)
    robot_time_stamps = np.asarray(ep_data[robot_ts_key], dtype=np.float64)   # (N,)
    wrench_time_stamps = np.asarray(ep_data[wrench_ts_key], dtype=np.float64) # (M,)

    if ts_pose_fb.shape[0] != robot_time_stamps.shape[0]:
        print(
            f"[{ep_name}] pose len {ts_pose_fb.shape[0]} != robot_ts len {robot_time_stamps.shape[0]}, skip."
        )
        return False
    if wrench.shape[0] != wrench_time_stamps.shape[0]:
        print(
            f"[{ep_name}] wrench len {wrench.shape[0]} != wrench_ts len {wrench_time_stamps.shape[0]}, skip."
        )
        return False

    # ensure timestamps are sorted (they should be)
    # If not sorted, sort with same permutation.
    if np.any(np.diff(wrench_time_stamps) < 0):
        perm = np.argsort(wrench_time_stamps)
        wrench_time_stamps = wrench_time_stamps[perm]
        wrench = wrench[perm]

    # ---- filter wrench (tool frame already, no transforms) ----
    wrench_f = filter_wrench(
        wrench=wrench,
        mode=wrench_filter_mode,
        fs_hz=fs_hz,
        cutoff_hz=cutoff_hz,
        order=order,
        moving_avg_window=moving_avg_window,
    )

    # ---- create estimator ----
    pe = ch.VirtualTargetEstimator(
        k_max, k_min, f_low, f_high, dim, characteristic_length, vel_tol
    )

    num_robot_steps = len(robot_time_stamps)
    ts_pose_virtual_target = np.zeros((num_robot_steps, 7), dtype=np.float32)
    stiffness = np.zeros((num_robot_steps,), dtype=np.float32)

    # ---- main loop ----
    for t in range(num_robot_steps):
        pose7_WT = ts_pose_fb[t]
        SE3_WT = SE3.Rt(q2r(pose7_WT[3:7]), pose7_WT[0:3], check=False)

        # nearest wrench index by timestamps
        iw = nearest_index(wrench_time_stamps, robot_time_stamps[t])
        wrench_T = wrench_f[iw]  # already in tool frame

        # velocity / motion proxy (pose difference over a window)
        id_start = max(0, t - half_window_size)
        id_end = min(num_robot_steps - 1, t + half_window_size)

        SE3_start = su.pose7_to_SE3(ts_pose_fb[id_start])
        SE3_end = su.pose7_to_SE3(ts_pose_fb[id_end])
        twist_diff = su.SE3_to_spt(su.SE3_inv(SE3_start) @ SE3_end)

        # estimate stiffness & virtual target offset
        if dim == 6:
            k, mat_TC, _flag_adjusted = pe.update(wrench_T, twist_diff)
            SE3_TC = SE3(mat_TC)
        else:
            k, pos_TC, _flag_adjusted = pe.update(wrench_T, twist_diff)
            SE3_TC = SE3.Rt(np.eye(3), pos_TC)

        SE3_WC = SE3_WT * SE3_TC

        ts_pose_virtual_target[t] = np.concatenate([SE3_WC.t, r2q(SE3_WC.R)]).astype(
            np.float32
        )
        stiffness[t] = float(k)

    # ---- write back to zarr ----
    ep_data[f"ts_pose_virtual_target_{right_id}"] = ts_pose_virtual_target
    ep_data[f"stiffness_{right_id}"] = stiffness

    if plot:
        import matplotlib.pyplot as plt
        from PyriteUtility.plotting.matplotlib_helpers import set_axes_equal

        plt.ion()
        fig = plt.figure()
        ax = plt.axes(projection="3d")

        ax.plot3D(
            ts_pose_fb[:, 0],
            ts_pose_fb[:, 1],
            ts_pose_fb[:, 2],
            color="red",
            marker="o",
            markersize=2,
            label="tool pose",
        )

        ax.plot3D(
            ts_pose_virtual_target[:, 0],
            ts_pose_virtual_target[:, 1],
            ts_pose_virtual_target[:, 2],
            color="blue",
            marker="o",
            markersize=2,
            label="virtual target",
        )

        # connect lines
        for i in range(0, num_robot_steps, plot_every_n):
            ax.plot3D(
                [ts_pose_fb[i, 0], ts_pose_virtual_target[i, 0]],
                [ts_pose_fb[i, 1], ts_pose_virtual_target[i, 1]],
                [ts_pose_fb[i, 2], ts_pose_virtual_target[i, 2]],
                color="black",
                linewidth=1,
            )

        ax.legend()
        set_axes_equal(ax)
        plt.title(f"Episode {ep_name}")
        plt.show()

        input("Press Enter to continue...")

    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to zarr dataset root (contains data/ and meta/).",
    )

    # which arm has force
    p.add_argument("--right_id", type=int, default=0)

    # estimator params (defaults copied from ACP-style settings; tune for 15Hz data/task)
    p.add_argument("--k_max", type=float, default=5000.0)
    p.add_argument("--k_min", type=float, default=200.0)
    p.add_argument("--f_low", type=float, default=0.5)
    p.add_argument("--f_high", type=float, default=5.0)
    p.add_argument("--dim", type=int, default=3, choices=[3, 6])
    p.add_argument("--characteristic_length", type=float, default=0.02)
    p.add_argument("--vel_tol", type=float, default=999.002)

    # wrench filtering
    p.add_argument("--wrench_filter_mode", type=str, default="lpf", choices=["lpf", "moving_avg", "none"])
    p.add_argument("--fs_hz", type=float, default=15.0, help="Sampling freq for wrench if using LPF (your case: 15Hz)")
    p.add_argument("--cutoff_hz", type=float, default=3.0)
    p.add_argument("--order", type=int, default=3)
    p.add_argument("--moving_avg_window", type=int, default=7, help="If mode=moving_avg, e.g. 7 frames ~0.47s at 15Hz")

    # motion window used to compute twist_diff
    p.add_argument("--half_window_size", type=int, default=3, help="Pose diff window half-size (frames). 3 => ~0.4s span at 15Hz")

    # multiprocessing
    p.add_argument("--num_workers", type=int, default=1)

    # Plot virtual target & stiffness (for debugging, not recommended for large datasets since it can be slow)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--plot_every_n", type=int, default=20)

    args = p.parse_args()

    buffer = zarr.open(args.dataset_path, mode="r+")
    data_group = buffer["data"]

    episodes = list(data_group.items())

    def _run_one(ep_name, ep_data):
        ok = process_episode_right_only(
            ep_name=ep_name,
            ep_data=ep_data,
            right_id=args.right_id,
            k_max=args.k_max,
            k_min=args.k_min,
            f_low=args.f_low,
            f_high=args.f_high,
            dim=args.dim,
            characteristic_length=args.characteristic_length,
            vel_tol=args.vel_tol,
            wrench_filter_mode=args.wrench_filter_mode,
            fs_hz=args.fs_hz,
            cutoff_hz=args.cutoff_hz,
            order=args.order,
            moving_avg_window=args.moving_avg_window,
            half_window_size=args.half_window_size,
            plot=args.plot,
            plot_every_n=args.plot_every_n,
        )
        return ok

    if args.num_workers <= 1:
        for ep_name, ep_data in tqdm(episodes, desc="Episodes"):
            _run_one(ep_name, ep_data)
    else:
        # NOTE: zarr groups are usually not safe to write from multiple processes simultaneously
        # unless using a process-safe store/locking. Safer to do single-process for write-back.
        # We'll warn and fall back to single-process.
        print(
            "[WARN] Multi-process write to the same zarr store is risky. "
            "Falling back to single-process. Set --num_workers 1."
        )
        for ep_name, ep_data in tqdm(episodes, desc="Episodes"):
            _run_one(ep_name, ep_data)

    print("Done! Added virtual target & stiffness for right hand only.")


if __name__ == "__main__":
    main()