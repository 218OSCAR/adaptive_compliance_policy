import sys
import os

SCRIPT_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(SCRIPT_PATH, "../../../"))

import copy
from typing import Dict, Optional, Union, List

import numpy as np
import torch
import zarr
from threadpoolctl import threadpool_limits
from tqdm import trange, tqdm
from filelock import FileLock
from einops import rearrange, reduce

from diffusion_policy.common.normalize_util import (
    array_to_stats,
    concatenate_normalizer,
    get_identity_normalizer_from_stat,
    get_image_identity_normalizer,
    get_range_normalizer_from_stat,
)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.dataset.base_dataset import BaseDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer

import PyriteUtility.spatial_math.spatial_utilities as su
from PyriteUtility.computer_vision.imagecodecs_numcodecs import register_codecs

register_codecs()


# =========================
# raw -> obs
# =========================
def raw_to_obs_cable_mounting(
    raw_data: Union[zarr.Group, Dict[str, np.ndarray]],
    episode_data: dict,
    shape_meta: dict,
):
    """
    Convert shape_meta.raw -> shape_meta.obs for cable_mounting.

    Differences vs original raw_to_obs():
    - Adds fingertip_width_{id} into obs (low_dim)
    - Left hand wrench may not exist (skip if missing)
    - tactile is treated as rgb key (kept as compressed array in memory) if declared in shape_meta.raw
    """
    episode_data["obs"] = {}

    # ---- keep rgb (including tactile) as compressed arrays ----
    for key, attr in shape_meta["raw"].items():
        type_ = attr.get("type", "low_dim")
        if type_ == "rgb":
            episode_data["obs"][key] = raw_data[key]

    # ---- low-dim per robot ----
    for rid in shape_meta["id_list"]:
        pose7_fb_key = f"ts_pose_fb_{rid}"
        if pose7_fb_key not in raw_data:
            raise KeyError(f"Missing {pose7_fb_key} in raw data.")

        pose7_fb = raw_data[pose7_fb_key]
        pose9_fb = su.SE3_to_pose9(su.pose7_to_SE3(pose7_fb))

        episode_data["obs"][f"robot{rid}_eef_pos"] = pose9_fb[..., :3]
        episode_data["obs"][f"robot{rid}_eef_rot_axis_angle"] = pose9_fb[..., 3:]

        # optional: abs pose keys
        if f"robot{rid}_abs_eef_pos" in shape_meta.get("obs", {}):
            episode_data["obs"][f"robot{rid}_abs_eef_pos"] = pose9_fb[..., :3]
            episode_data["obs"][f"robot{rid}_abs_eef_rot_axis_angle"] = pose9_fb[..., 3:]

        # wrench is optional per robot (right only in your dataset)
        wrench_key = f"wrench_{rid}"
        if wrench_key in raw_data and f"robot{rid}_eef_wrench" in shape_meta.get("obs", {}):
            episode_data["obs"][f"robot{rid}_eef_wrench"] = raw_data[wrench_key][:]
        # else: do nothing (left hand has no wrench)

        # fingertip width (required by you)

        # ftw_key = f"fingertip_width_{rid}"
        # if ftw_key in raw_data:
        #     episode_data["obs"][ftw_key] = raw_data[ftw_key][:]
        # else:
        #     # If missing, still fail loudly: you said you want both hands widths.
        #     raise KeyError(f"Missing {ftw_key} in raw data.")

        ftw_raw_key = f"fingertip_width_{rid}"
        ftw_obs_key = f"robot{rid}_fingertip_width"
        if ftw_raw_key in raw_data and ftw_obs_key in shape_meta.get("obs", {}):
            ftw = raw_data[ftw_raw_key][:]
            # ✅ 保证是 (T, 1)
            if ftw.ndim == 1:
                ftw = ftw.reshape(-1, 1)
            else:
                ftw = ftw.reshape(ftw.shape[0], 1)
            episode_data["obs"][ftw_obs_key] = ftw.astype(np.float32)
        else:
            raise KeyError(
                f"Missing {ftw_raw_key} in raw or {ftw_obs_key} not declared in shape_meta.obs"
            )
        
        if "tactile_time_stamps" in raw_data and "tactile_time_stamps" in shape_meta.get("raw", {}):
            episode_data["obs"]["tactile_time_stamps"] = raw_data["tactile_time_stamps"][:]
        # timestamps (optional; only copy if exist in raw and declared in shape_meta.raw)
        for ts_key in [f"rgb_time_stamps_{rid}", f"robot_time_stamps_{rid}", f"wrench_time_stamps_{rid}"]:
            if ts_key in raw_data and ts_key in shape_meta.get("raw", {}):
                episode_data["obs"][ts_key] = raw_data[ts_key][:]


# =========================
# raw -> action (30D)
# =========================
def raw_to_action30_cable_mounting(
    raw_data: Union[zarr.Group, Dict[str, np.ndarray]],
    episode_data: dict,
    right_id: int = 0,
    left_id: int = 1,
):
    """
    Build action with dim=30:
      - right (20): pose9_command + pose9_virtual_target + stiffness(1) + fingertip_width(1)
      - left  (10): pose9_command + fingertip_width(1)

    action_time_stamps uses robot_time_stamps_{right_id}.
    """
    # ---- right ----
    ts_pose7_cmd_r = raw_data[f"ts_pose_command_{right_id}"][:]
    ts_pose9_cmd_r = su.SE3_to_pose9(su.pose7_to_SE3(ts_pose7_cmd_r))

    ts_pose7_vt_r = raw_data[f"ts_pose_virtual_target_{right_id}"][:]
    ts_pose9_vt_r = su.SE3_to_pose9(su.pose7_to_SE3(ts_pose7_vt_r))

    stiffness_r = raw_data[f"stiffness_{right_id}"][:].astype(np.float32).reshape(-1, 1)
    width_r = raw_data[f"fingertip_width_{right_id}"][:].astype(np.float32).reshape(-1, 1)

    right_action = np.concatenate([ts_pose9_cmd_r, ts_pose9_vt_r, stiffness_r, width_r], axis=-1)
    assert right_action.shape[1] == 20

    # ---- left ----
    ts_pose7_cmd_l = raw_data[f"ts_pose_command_{left_id}"][:]
    ts_pose9_cmd_l = su.SE3_to_pose9(su.pose7_to_SE3(ts_pose7_cmd_l))
    width_l = raw_data[f"fingertip_width_{left_id}"][:].astype(np.float32).reshape(-1, 1)

    left_action = np.concatenate([ts_pose9_cmd_l, width_l], axis=-1)
    assert left_action.shape[1] == 10

    # ---- trim to shortest length ----
    L = min(right_action.shape[0], left_action.shape[0])
    right_action = right_action[:L]
    left_action = left_action[:L]

    episode_data["action"] = np.concatenate([right_action, left_action], axis=-1).astype(np.float32)
    assert episode_data["action"].shape[1] == 30

    # timestamps
    # ts_key = f"robot_time_stamps_{right_id}"
    # if ts_key in raw_data:
    #     episode_data["action_time_stamps"] = raw_data[ts_key][:L]
    # else:
    #     # fallback: if not present, create 0..L-1
    #     episode_data["action_time_stamps"] = np.arange(L, dtype=np.float64)
    # timestamps: align action to rgb0 time axis (since sampler queries on rgb0)
    if "rgb_time_stamps_0" in raw_data:
        episode_data["action_time_stamps"] = raw_data["rgb_time_stamps_0"][:L]
    else:
        ts_key = f"robot_time_stamps_{right_id}"
        episode_data["action_time_stamps"] = raw_data[ts_key][:L]


# =========================
# action_sample (30D)
# =========================
def action30_to_action_sample(
    action_sparse: np.ndarray,  # (T, 30)
    right_id: int = 0,
    left_id: int = 1,
):
    """
    Convert action sequence to relative, like ACP's action19_to_action_sample:

    Layout (T,30):
      right:  0:9   pose9
              9:18  pose9_vt
              18:19 stiffness
              19:20 width
      left:   20:29 pose9
              29:30 width

    Relative transform:
      - For right: pose9 and pose9_vt are transformed by inv(SE3(pose9[0])).
        stiffness/width unchanged.
      - For left: pose9 transformed by inv(SE3(left_pose9[0])) OR by inv(right_pose9[0])?

    Here we choose: each arm relative to its own initial pose (more consistent with bimanual symmetry).
    If you prefer "both relative to right initial pose", tell me and I’ll switch.
    """
    action_processed = {"sparse": {}}
    T, D = action_sparse.shape
    assert D == 30

    # ----- right -----
    pose9_r = action_sparse[:, 0:9]
    pose9_vt_r = action_sparse[:, 9:18]
    stiffness_r = action_sparse[:, 18:19]
    width_r = action_sparse[:, 19:20]

    SE3_r = su.pose9_to_SE3(pose9_r)
    SE3_vt_r = su.pose9_to_SE3(pose9_vt_r)
    SE3_r0_inv = su.SE3_inv(SE3_r[0])

    pose9_r_rel = su.SE3_to_pose9(SE3_r0_inv @ SE3_r)
    pose9_vt_r_rel = su.SE3_to_pose9(SE3_r0_inv @ SE3_vt_r)
    right_rel = np.concatenate([pose9_r_rel, pose9_vt_r_rel, stiffness_r, width_r], axis=-1)

    # ----- left -----
    pose9_l = action_sparse[:, 20:29]
    width_l = action_sparse[:, 29:30]

    SE3_l = su.pose9_to_SE3(pose9_l)
    SE3_l0_inv = su.SE3_inv(SE3_l[0])
    pose9_l_rel = su.SE3_to_pose9(SE3_l0_inv @ SE3_l)
    left_rel = np.concatenate([pose9_l_rel, width_l], axis=-1)

    out = np.concatenate([right_rel, left_rel], axis=-1)
    assert out.shape == (T, 30)
    action_processed["sparse"] = out.astype(np.float32)
    return action_processed


# ============================================================
# Dataset
# ============================================================
class CableMountingDataset(BaseDataset):
    """
    A dataset compatible with ACP-style pipeline but customized for:
      - tactile rgb key
      - only right wrench
      - fingertip widths in obs
      - asymmetric action dim=30 (right 20, left 10)
    """

    def __init__(
        self,
        shape_meta: dict,
        dataset_path: str,
        sparse_query_frequency_down_sample_steps: int = 1,
        action_padding: bool = False,
        temporally_independent_normalization: bool = False,
        seed: int = 42,
        val_ratio: float = 0.0,
        normalize_wrench: bool = False,
        right_id: int = 0,
        left_id: int = 1,
    ):
        self.shape_meta = shape_meta
        self.id_list = shape_meta["id_list"]
        self.right_id = int(right_id)
        self.left_id = int(left_id)
        self.temporally_independent_normalization = temporally_independent_normalization
        self.threadpool_limits_is_applied = False
        self.normalize_wrench = normalize_wrench
        self.sparse_query_frequency_down_sample_steps = sparse_query_frequency_down_sample_steps
        self.action_padding = action_padding
        self.action_type = "sparse"

        # ---- load zarr store into memory ----
        print("[CableMountingDataset] loading data into store")
        with zarr.DirectoryStore(dataset_path) as directory_store:
            replay_buffer_raw = ReplayBuffer.copy_from_store(
                src_store=directory_store, dest_store=zarr.MemoryStore()
            )

        # ---- build converted replay buffer ----
        print("[CableMountingDataset] raw -> obs/action conversion")
        replay_buffer = self.raw_episodes_conversion(replay_buffer_raw, shape_meta)


        # debug logs to check conversion
        print("[DEBUG] converted replay_buffer data keys:", list(replay_buffer["data"].keys())[:10])
        print("[DEBUG] raw meta keys:", replay_buffer_raw["meta"].keys())
        for k in replay_buffer_raw["meta"].keys():
            try:
                v = replay_buffer_raw["meta"][k]
                if hasattr(v, "shape"):
                    print("[DEBUG] meta", k, "shape", v.shape, "dtype", getattr(v, "dtype", None))
            except Exception as e:
                print("[DEBUG] meta", k, "read error:", e)

        # ---- train/val mask ----
        val_mask = get_val_mask(
            n_episodes=replay_buffer_raw.n_episodes, val_ratio=val_ratio, seed=seed
        )
        train_mask = ~val_mask
        self.val_mask = val_mask

        # ---- sampler ----
        print("[CableMountingDataset] creating SequenceSampler")
        sampler = SequenceSampler(
            shape_meta=shape_meta,
            replay_buffer=replay_buffer,
            sparse_query_frequency_down_sample_steps=sparse_query_frequency_down_sample_steps,
            episode_mask=train_mask,
            action_padding=action_padding,
            obs_to_obs_sample=self._obs_to_obs_sample_wrapper,
            action_to_action_sample=self._action_to_action_sample_wrapper,
            id_list=self.id_list,
        )

        self.replay_buffer = replay_buffer
        self.sampler = sampler

    def raw_episodes_conversion(self, replay_buffer_raw: ReplayBuffer, shape_meta: dict):
        replay_buffer = {"data": {}, "meta": replay_buffer_raw["meta"]}

        for ep in replay_buffer_raw["data"].keys():
            replay_buffer["data"][ep] = {}

            # obs
            raw_to_obs_cable_mounting(
                replay_buffer_raw["data"][ep], replay_buffer["data"][ep], shape_meta
            )

            # action (30D)
            raw_to_action30_cable_mounting(
                replay_buffer_raw["data"][ep],
                replay_buffer["data"][ep],
                right_id=self.right_id,
                left_id=self.left_id,
            )

        return replay_buffer

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            shape_meta=self.shape_meta,
            replay_buffer=self.replay_buffer,
            sparse_query_frequency_down_sample_steps=self.sparse_query_frequency_down_sample_steps,
            episode_mask=self.val_mask,
            action_padding=self.action_padding,
            obs_to_obs_sample=self._obs_to_obs_sample_wrapper,
            action_to_action_sample=self._action_to_action_sample_wrapper,
            id_list=self.id_list,
        )
        # follow original behavior (invert so that get_validation_dataset().val_mask is train mask)
        val_set.val_mask = ~self.val_mask
        return val_set

    # ---- wrappers to match SequenceSampler signature ----
    def _obs_to_obs_sample_wrapper(self, obs_sparse, shape_meta, reshape_mode, id_list, ignore_rgb=False):
        # reuse your existing common_type_conversions behavior,
        # but we must modify wrench-relative processing to be robust when left has no wrench.
        return obs_to_obs_sample_cable_mounting(
            obs_sparse=obs_sparse,
            shape_meta=shape_meta,
            reshape_mode=reshape_mode,
            id_list=id_list,
            right_id=self.right_id,
            left_id=self.left_id,
            ignore_rgb=ignore_rgb,
        )

    def _action_to_action_sample_wrapper(self, action_sparse, id_list):
        # ignore id_list, because action layout is fixed 30
        return action30_to_action_sample(
            action_sparse=action_sparse,
            right_id=self.right_id,
            left_id=self.left_id,
        )

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        """
        Compute normalizer for:
          - sparse obs low_dim keys (including fingertip_width)
          - action (30D) with correct slices
        """
        sparse_normalizer = LinearNormalizer()

        # Gather all data
        self.sampler.ignore_rgb(True)
        dataloader = torch.utils.data.DataLoader(
            dataset=self,
            batch_size=64,
            num_workers=32,
        )

        data_cache_sparse = {}
        for batch in tqdm(dataloader, desc="iterating dataset to get normalization"):
            # obs low_dim only
            for key in self.shape_meta["sample"]["obs"]["sparse"].keys():
                if key not in batch["obs"]["sparse"]:
                    continue
                if self.shape_meta["obs"][key]["type"] == "low_dim":
                    data_cache_sparse.setdefault(key, []).append(copy.deepcopy(batch["obs"]["sparse"][key]))
            # action
            data_cache_sparse.setdefault("action", []).append(copy.deepcopy(batch["action"]["sparse"]))

        self.sampler.ignore_rgb(False)

        # Concatenate
        for key in list(data_cache_sparse.keys()):
            data_cache_sparse[key] = np.concatenate(data_cache_sparse[key])  # (B, T, D)
            if not self.temporally_independent_normalization:
                data_cache_sparse[key] = rearrange(data_cache_sparse[key], "B T ... -> (B T) (...)")

        # ---- action normalizer for 30D ----
        act = data_cache_sparse["action"]  # (N, 30)
        if act.shape[-1] != 30:
            raise RuntimeError(f"Expected action dim=30, got {act.shape[-1]}")

        action_norms = []

        # right pose pos (0:3) range
        action_norms.append(get_range_normalizer_from_stat(array_to_stats(act[..., 0:3])))
        # right pose rot (3:9) identity
        action_norms.append(get_identity_normalizer_from_stat(array_to_stats(act[..., 3:9])))
        # right vt pos (9:12) range
        action_norms.append(get_range_normalizer_from_stat(array_to_stats(act[..., 9:12])))
        # right vt rot (12:18) identity
        action_norms.append(get_identity_normalizer_from_stat(array_to_stats(act[..., 12:18])))
        # right stiffness (18:19) range
        action_norms.append(get_range_normalizer_from_stat(array_to_stats(act[..., 18:19])))
        # right width (19:20) range
        action_norms.append(get_range_normalizer_from_stat(array_to_stats(act[..., 19:20])))

        # left pose pos (20:23) range
        action_norms.append(get_range_normalizer_from_stat(array_to_stats(act[..., 20:23])))
        # left pose rot (23:29) identity
        action_norms.append(get_identity_normalizer_from_stat(array_to_stats(act[..., 23:29])))
        # left width (29:30) range
        action_norms.append(get_range_normalizer_from_stat(array_to_stats(act[..., 29:30])))

        sparse_normalizer["action"] = concatenate_normalizer(action_norms)

        # ---- obs normalizer ----
        for key in self.shape_meta["sample"]["obs"]["sparse"].keys():
            if key not in self.shape_meta["obs"]:
                continue
            typ = self.shape_meta["obs"][key]["type"]

            if typ == "low_dim":
                if key not in data_cache_sparse:
                    # e.g. left wrench missing
                    continue
                stat = array_to_stats(data_cache_sparse[key])

                if "eef_pos" in key:
                    this_norm = get_range_normalizer_from_stat(stat)
                elif "rot_axis_angle" in key:
                    this_norm = get_identity_normalizer_from_stat(stat)
                elif "wrench" in key:
                    this_norm = get_range_normalizer_from_stat(stat) if self.normalize_wrench else get_identity_normalizer_from_stat(stat)
                elif "fingertip_width" in key:
                    this_norm = get_range_normalizer_from_stat(stat)
                else:
                    raise RuntimeError(f"Unsupported low_dim key for normalizer: {key}")

                sparse_normalizer[key] = this_norm

            elif typ == "rgb":
                sparse_normalizer[key] = get_image_identity_normalizer()

            elif typ == "timestamp":
                continue
            else:
                raise RuntimeError(f"Unsupported obs type: {typ} for key={key}")

        return sparse_normalizer

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if not self.threadpool_limits_is_applied:
            threadpool_limits(1)
            self.threadpool_limits_is_applied = True

        obs_dict, action_array = self.sampler.sample_sequence(idx)
        torch_data = {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "action": dict_apply(action_array, torch.from_numpy),
        }
        return torch_data


# ============================================================
# Robust obs_to_obs_sample for missing left wrench + tactile
# ============================================================
def obs_rgb_preprocess_generic(obs: dict, obs_output: dict, reshape_mode: str, shape_meta: dict):
    """
    Minimal copy of your obs_rgb_preprocess but generic:
    - processes any key in shape_meta.obs with type=rgb (including 'tactile')
    - expects raw stored as (T,H,W,C)
    - outputs (T,C,H,W) float32 [0,1]
    """
    from PyriteUtility.computer_vision.computer_vision_utility import get_image_transform

    for key, attr in shape_meta["obs"].items():
        if attr.get("type", "low_dim") != "rgb":
            continue
        co, ho, wo = attr["shape"]
        imgs_in = obs[key]
        t, hi, wi, ci = imgs_in.shape
        assert ci == co

        out_imgs = imgs_in
        if (ho != hi) or (wo != wi):
            if reshape_mode == "reshape":
                tf = get_image_transform(input_res=(wi, hi), output_res=(wo, ho), bgr_to_rgb=False)
                out_imgs = np.stack([tf(x) for x in imgs_in])
            elif reshape_mode == "check":
                raise AssertionError(f"[obs_rgb_preprocess] Require {ho}x{wo}, got {hi}x{wi}")

        if out_imgs.dtype in (np.uint8, np.int32):
            out_imgs = out_imgs.astype(np.float32) / 255.0

        obs_output[key] = np.moveaxis(out_imgs, -1, 1)  # THWC -> TCHW


def obs_to_obs_sample_cable_mounting(
    obs_sparse: dict,
    shape_meta: dict,
    reshape_mode: str,
    id_list: list,
    right_id: int = 0,
    left_id: int = 1,
    ignore_rgb: bool = False,
):
    """
    Similar to common_type_conversions.obs_to_obs_sample, but:
    - supports tactile as rgb key
    - supports missing left wrench (skips relative wrench transform if missing)
    """
    obs_processed = {"sparse": {}}

    # 1) RGB (rgb_0, rgb_1, tactile, ...)
    if not ignore_rgb:
        obs_rgb_preprocess_generic(obs_sparse, obs_processed["sparse"], reshape_mode, shape_meta)

    # 2) Copy low-dim keys directly (float32)
    for key, attr in shape_meta["obs"].items():
        if attr.get("type", "low_dim") == "low_dim":
            if key in obs_sparse:
                obs_processed["sparse"][key] = obs_sparse[key].astype(np.float32)

    # 3) Make pose relative per arm; wrench relative if present
    base_SE3_WT = {}
    for rid in id_list:
        pos_key = f"robot{rid}_eef_pos"
        rot_key = f"robot{rid}_eef_rot_axis_angle"
        if pos_key not in obs_processed["sparse"] or rot_key not in obs_processed["sparse"]:
            continue

        SE3_WT = su.pose9_to_SE3(
            np.concatenate([obs_processed["sparse"][pos_key], obs_processed["sparse"][rot_key]], axis=-1)
        )
        base = SE3_WT[-1]
        base_SE3_WT[rid] = base
        SE3_rel = su.SE3_inv(base) @ SE3_WT
        pose9_rel = su.SE3_to_pose9(SE3_rel)

        obs_processed["sparse"][pos_key] = pose9_rel[..., :3]
        obs_processed["sparse"][rot_key] = pose9_rel[..., 3:]

        # wrench transform if exists
        wrench_key = f"robot{rid}_eef_wrench"
        if wrench_key in obs_sparse and wrench_key in obs_processed["sparse"]:
            # Convert wrench into base frame
            SE3_i_base = su.SE3_inv(SE3_rel)[-1]
            wrench = su.transpose(su.SE3_to_adj(SE3_i_base)) @ np.expand_dims(obs_sparse[wrench_key], -1)
            obs_processed["sparse"][wrench_key] = np.squeeze(wrench).astype(np.float32)

    return obs_processed