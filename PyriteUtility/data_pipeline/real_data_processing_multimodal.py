import os
import sys
SCRIPT_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(SCRIPT_PATH, "../../"))

import re
import argparse
import shutil
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import cv2
import zarr
import concurrent.futures
from tqdm import tqdm


from PyriteUtility.computer_vision.imagecodecs_numcodecs import register_codecs, JpegXl
from PyriteUtility.data_pipeline.episode_data_buffer import (
    VideoData,
    EpisodeDataBuffer,
    img_copy,
)
from PyriteUtility.planning_control.filtering import LiveLPFilter

register_codecs()

def pick_time_zero(npz) -> float:
    """
    Choose a common t0 (seconds) to align all modalities.
    Prefer t_ref[0] if exists; otherwise choose the minimum among available *_t[0].
    """
    if "t_ref" in npz and len(npz["t_ref"]) > 0:
        return float(npz["t_ref"][0])

    candidates = []
    for k in [
        "t",
        "rgb2_t",
        "rgb3_t",
        "tactile_t",
        "force_torque_t",
        "vicon_tcp_right_t",
        "vicon_tcp_left_t",
        "fingertip_width_right_t",
        "fingertip_width_left_t",
    ]:
        if k in npz and len(npz[k]) > 0:
            candidates.append(float(npz[k][0]))
    if len(candidates) == 0:
        raise RuntimeError("No timestamp arrays found to determine t0.")
    return float(min(candidates))


def to_ms(npz, key: str, t0: float, expected_len: Optional[int] = None) -> np.ndarray:
    """
    Convert npz[key] timestamps (seconds) to milliseconds aligned by t0.
    Optionally check length.
    """
    if key not in npz:
        raise KeyError(f"missing key '{key}'")
    ts = npz[key].astype(np.float64).reshape(-1)
    if expected_len is not None and len(ts) != expected_len:
        raise ValueError(f"{key} length mismatch: {len(ts)} vs expected {expected_len}")
    return (ts - t0) * 1000.0


def safe_npz_load(path: Path):
    try:
        if path.stat().st_size < 100:
            return None, f"file too small: {path.stat().st_size} bytes"
        with open(path, "rb") as f:
            sig = f.read(2)
        if sig != b"PK":
            return None, f"bad signature: {sig!r} (not a zip/npz)"
        return np.load(path, allow_pickle=True), None
    except Exception as e:
        return None, f"np.load failed: {type(e).__name__}: {e}"


# def make_episode_id(ep_path: Path) -> str:
#     m = re.search(r"(\d+)", ep_path.stem)
#     if m:
#         return m.group(1)
#     return ep_path.stem

def make_episode_id(ep_path: Path) -> str:
    # 从 episode_YYYYMMDD_HHMMSS_... 中提取 YYYYMMDD_HHMMSS
    m = re.search(r"episode_(\d{8})_(\d{6})", ep_path.stem)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    return ep_path.stem


def decode_jpeg_dict_sequence(
    seq_obj: np.ndarray,
    resize_hw: Tuple[int, int],
    max_bad_ratio: float = 0.05,
) -> np.ndarray:
    H, W = resize_hw
    T = len(seq_obj)
    out = np.zeros((T, H, W, 3), dtype=np.uint8)

    bad = 0
    last_good = None
    for i in range(T):
        item = seq_obj[i]
        if not isinstance(item, dict) or "data" not in item:
            bad += 1
            out[i] = 0 if last_good is None else last_good
            continue

        img_bgr = cv2.imdecode(item["data"], cv2.IMREAD_COLOR)
        if img_bgr is None:
            bad += 1
            out[i] = 0 if last_good is None else last_good
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if (img_rgb.shape[0] != H) or (img_rgb.shape[1] != W):
            img_rgb = cv2.resize(img_rgb, (W, H), interpolation=cv2.INTER_AREA)
        out[i] = img_rgb
        last_good = img_rgb

    bad_ratio = bad / max(1, T)
    if bad_ratio > max_bad_ratio:
        raise RuntimeError(f"too many bad jpeg frames: bad_ratio={bad_ratio:.3f}")
    return out


def vicon_pose_to_pose7_pyrite(vicon_pose: np.ndarray) -> np.ndarray:
    """
    [x,y,z,qx,qy,qz,qw] -> [x,y,z,qw,qx,qy,qz]
    """
    out = vicon_pose.astype(np.float32).copy()
    q = out[:, 3:7].copy()
    out[:, 3] = q[:, 3]
    out[:, 4] = q[:, 0]
    out[:, 5] = q[:, 1]
    out[:, 6] = q[:, 2]
    return out


def write_tactile_dataset(ep_group, tactile_rgb, tactile_time_stamps_ms, compressor, max_workers=32):
    T, H, W, C = tactile_rgb.shape
    if "tactile" in ep_group:
        del ep_group["tactile"]
    tactile_arr = ep_group.require_dataset(
        "tactile",
        shape=(T, H, W, C),
        chunks=(1, H, W, C),
        dtype=np.uint8,
        compressor=compressor,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = set()
        for i in range(T):
            futures.add(executor.submit(img_copy, tactile_arr, i, tactile_rgb, i))
        completed, _ = concurrent.futures.wait(futures)
        for f in completed:
            if not f.result():
                raise RuntimeError("Failed to encode tactile frame!")
    ep_group["tactile_time_stamps"] = zarr.array(tactile_time_stamps_ms)


def convert_npz_dir_to_zarr(
    src_dir: Path,
    out_dir: Path,
    resize_hw: Tuple[int, int],
    base_right_in_board: np.ndarray,
    base_left_in_board: np.ndarray,
    overwrite: bool,
    max_bad_jpeg_ratio: float,
    max_workers: int,
    save_mp4: bool,
    save_mp4_fps: int,
):
    if overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    store = zarr.DirectoryStore(path=str(out_dir))
    root = zarr.open(store=store, mode="a")

    # # still 2 cameras / 2 arms for pose + rgb
    # id_list = [0, 1]  # 0: right, 1: left

    # 3 cameras: 2 wrist + 1 kinect
    id_list = [0, 1, 2]  # 0: rgb2(right), 1: rgb3(left), 2: kinect_rgb
    recorder = EpisodeDataBuffer(
        store_path=str(out_dir),
        camera_ids=id_list,
        max_workers=max_workers,
        save_video=save_mp4,
        save_video_fps=save_mp4_fps,
        data=root,
    )

    ep_files = sorted(src_dir.glob("episode*.npz"))
    if len(ep_files) == 0:
        ep_files = sorted(src_dir.glob("*.npz"))

    # meta
    episode_robot0_len = []
    episode_robot1_len = []
    episode_wrench0_len = []
    episode_rgb0_len = []
    episode_rgb1_len = []
    episode_rgb2_len = []
    episode_tactile_len = []

    skipped: List[Tuple[str, str]] = []

    tactile_compressor = JpegXl(level=80, numthreads=1)

    for ep_path in tqdm(ep_files, desc="Episodes"):
        ep_id = make_episode_id(ep_path)
        npz, err = safe_npz_load(ep_path)
        if npz is None:
            skipped.append((ep_id, err))
            continue

        if "t" not in npz:
            skipped.append((ep_id, "missing key 't'"))
            continue
        T = int(npz["t"].shape[0])
        if T <= 0:
            skipped.append((ep_id, "empty episode"))
            continue

        # unified timestamps (ms)

        # t = npz["t"].astype(np.float64).copy()
        # t = t - float(t[0])
        # t_ms = t * 1000.0
        try:
            t0 = pick_time_zero(npz)
        except Exception as e:
            skipped.append((ep_id, f"failed to pick t0: {e}"))
            continue

        # lengths per modality (we expect all to be T in your dataset, but we still check)
        T = int(npz["t"].shape[0])

        # per-modality timestamps (ms) aligned by same t0
        try:
            rgb0_t_ms = to_ms(npz, "rgb2_t", t0, expected_len=T)
            rgb1_t_ms = to_ms(npz, "rgb3_t", t0, expected_len=T)
            tactile_t_ms = to_ms(npz, "tactile_t", t0, expected_len=T)
            kinect_t_ms = to_ms(npz, "kinect_rgb_t", t0, expected_len=T)

            robot0_t_ms = to_ms(npz, "vicon_tcp_right_t", t0, expected_len=T)
            robot1_t_ms = to_ms(npz, "vicon_tcp_left_t", t0, expected_len=T)

            wrench_t_ms = to_ms(npz, "force_torque_t", t0, expected_len=T)

            width0_t_ms = to_ms(npz, "fingertip_width_right_t", t0, expected_len=T)
            width1_t_ms = to_ms(npz, "fingertip_width_left_t", t0, expected_len=T)
        except Exception as e:
            skipped.append((ep_id, f"timestamp convert error: {e}"))
            continue

        # decode images
        try:
            rgb0 = decode_jpeg_dict_sequence(npz["rgb2"], resize_hw, max_bad_ratio=max_bad_jpeg_ratio)
            rgb1 = decode_jpeg_dict_sequence(npz["rgb3"], resize_hw, max_bad_ratio=max_bad_jpeg_ratio)
            tactile_rgb = decode_jpeg_dict_sequence(npz["tactile"], resize_hw, max_bad_ratio=max_bad_jpeg_ratio)
            kinect_rgb = decode_jpeg_dict_sequence(npz["kinect_rgb"], resize_hw, max_bad_ratio=max_bad_jpeg_ratio)
        except Exception as e:
            skipped.append((ep_id, f"image decode failed: {e}"))
            continue

        # poses
        if ("vicon_tcp_right" not in npz) or ("vicon_tcp_left" not in npz):
            skipped.append((ep_id, "missing vicon_tcp_right/left"))
            continue
        pose7_r = vicon_pose_to_pose7_pyrite(npz["vicon_tcp_right"])
        pose7_l = vicon_pose_to_pose7_pyrite(npz["vicon_tcp_left"])
        if pose7_r.shape[0] != T or pose7_l.shape[0] != T:
            skipped.append((ep_id, "pose length mismatch"))
            continue
        pose7_r[:, 0:3] -= base_right_in_board.reshape(1, 3)
        pose7_l[:, 0:3] -= base_left_in_board.reshape(1, 3)

        # wrench only for right hand
        if "force_torque" not in npz:
            skipped.append((ep_id, "missing force_torque"))
            continue
        wrench0 = npz["force_torque"].astype(np.float32)
        if wrench0.shape != (T, 6):
            skipped.append((ep_id, f"force_torque shape mismatch: {wrench0.shape}"))
            continue
        # compute low-pass filtered wrench
        ft_filter = LiveLPFilter(
            fs=15,
            cutoff=3,
            order=3,
            dim=6,
        )
        wrench0_f = np.array([ft_filter(y) for y in wrench0])

        # fingertip widths
        if ("fingertip_width_right" not in npz) or ("fingertip_width_left" not in npz):
            skipped.append((ep_id, "missing fingertip widths"))
            continue
        w_right = npz["fingertip_width_right"].astype(np.float32).reshape(-1)
        w_left = npz["fingertip_width_left"].astype(np.float32).reshape(-1)
        if w_right.shape[0] != T or w_left.shape[0] != T:
            skipped.append((ep_id, "fingertip width length mismatch"))
            continue

        # --- write via EpisodeDataBuffer (rgb + pose + timestamps for both hands) ---
        # rgb_shapes = {0: rgb0.shape, 1: rgb1.shape}
        rgb_shapes = {0: rgb0.shape, 1: rgb1.shape, 2: kinect_rgb.shape}
        recorder.create_zarr_groups_for_episode(rgb_shapes, id_list, episode_id=ep_id)

        # visual_observations = {
        #     0: VideoData(rgb=rgb0, camera_id=0),
        #     1: VideoData(rgb=rgb1, camera_id=1),
        # }
        visual_observations = {
            0: VideoData(rgb=rgb0, camera_id=0),
            1: VideoData(rgb=rgb1, camera_id=1),
            2: VideoData(rgb=kinect_rgb, camera_id=2),
        }
        # visual_time_stamps = [None] * (max(id_list) + 1)
        # visual_time_stamps[0] = t_ms
        # visual_time_stamps[1] = t_ms
        visual_time_stamps = [None] * (max(id_list) + 1)
        visual_time_stamps[0] = rgb0_t_ms
        visual_time_stamps[1] = rgb1_t_ms
        visual_time_stamps[2] = kinect_t_ms
        recorder.save_video_for_episode(
            visual_observations=visual_observations,
            visual_time_stamps=visual_time_stamps,
            episode_id=ep_id,
        )

        # save pose & robot_time_stamps for BOTH arms using buffer
        recorder.save_low_dim_for_episode(
            ts_pose_command=[pose7_r, pose7_l],
            ts_pose_fb=[pose7_r, pose7_l],
            # robot_time_stamps=[t_ms, t_ms],
            robot_time_stamps=[robot0_t_ms, robot1_t_ms],
            episode_id=ep_id,
        )

        # --- now write wrench ONLY for id=0 directly into episode group ---
        ep_group = root["data"][f"episode_{ep_id}"]
        ep_group["wrench_0"] = zarr.array(wrench0)
        ep_group["wrench_filtered_0"] = zarr.array(wrench0_f)
        # ep_group["wrench_time_stamps_0"] = zarr.array(t_ms)
        ep_group["wrench_time_stamps_0"] = zarr.array(wrench_t_ms)

        # tactile
        write_tactile_dataset(
            ep_group=ep_group,
            tactile_rgb=tactile_rgb,
            # tactile_time_stamps_ms=t_ms,
            tactile_time_stamps_ms=tactile_t_ms,
            compressor=tactile_compressor,
            max_workers=max_workers,
        )

        # fingertip widths
        ep_group["fingertip_width_0"] = zarr.array(w_right)  # right
        ep_group["fingertip_width_1"] = zarr.array(w_left)   # left

        # meta
        episode_robot0_len.append(T)
        episode_robot1_len.append(T)
        episode_wrench0_len.append(T)
        episode_rgb0_len.append(T)
        episode_rgb1_len.append(T)
        episode_rgb2_len.append(T)
        episode_tactile_len.append(T)

    # write meta
    meta = root.create_group("meta", overwrite=True)
    meta["episode_robot0_len"] = zarr.array(np.array(episode_robot0_len, dtype=np.int64))
    meta["episode_robot1_len"] = zarr.array(np.array(episode_robot1_len, dtype=np.int64))
    meta["episode_wrench0_len"] = zarr.array(np.array(episode_wrench0_len, dtype=np.int64))
    meta["episode_rgb0_len"] = zarr.array(np.array(episode_rgb0_len, dtype=np.int64))
    meta["episode_rgb1_len"] = zarr.array(np.array(episode_rgb1_len, dtype=np.int64))
    meta["episode_rgb2_len"] = zarr.array(np.array(episode_rgb2_len, dtype=np.int64))
    meta["episode_tactile_len"] = zarr.array(np.array(episode_tactile_len, dtype=np.int64))

    print(f"\nAll done! Output zarr: {out_dir}")
    print(f"Converted episodes: {len(episode_robot0_len)} / {len(ep_files)}")
    if skipped:
        print("\nSkipped episodes (up to 50):")
        for name, reason in skipped[:50]:
            print(f"  - episode_{name}: {reason}")
        if len(skipped) > 50:
            print(f"  ... and {len(skipped)-50} more")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--resize_h", type=int, default=224)
    p.add_argument("--resize_w", type=int, default=224)
    p.add_argument("--base_right_in_board", type=float, nargs=3, default=[-0.375, -0.281, -0.025])
    p.add_argument("--base_left_in_board", type=float, nargs=3, default=[-0.375, 0.281, -0.025])
    p.add_argument("--max_bad_jpeg_ratio", type=float, default=0.05)
    p.add_argument("--max_workers", type=int, default=32)
    p.add_argument("--save_mp4", action="store_true")
    p.add_argument("--save_mp4_fps", type=int, default=60)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    convert_npz_dir_to_zarr(
        src_dir=Path(args.src_dir),
        out_dir=Path(args.out_dir),
        resize_hw=(args.resize_h, args.resize_w),
        base_right_in_board=np.array(args.base_right_in_board, dtype=np.float32),
        base_left_in_board=np.array(args.base_left_in_board, dtype=np.float32),
        overwrite=args.overwrite,
        max_bad_jpeg_ratio=args.max_bad_jpeg_ratio,
        max_workers=args.max_workers,
        save_mp4=args.save_mp4,
        save_mp4_fps=args.save_mp4_fps,
    )


if __name__ == "__main__":
    main()