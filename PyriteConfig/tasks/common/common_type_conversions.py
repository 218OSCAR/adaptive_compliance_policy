import sys
import os

SCRIPT_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(SCRIPT_PATH, "../../../"))

import PyriteUtility.spatial_math.spatial_utilities as su
from PyriteUtility.computer_vision.computer_vision_utility import get_image_transform
from PyriteUtility.common import dict_apply

import numpy as np
from typing import Union, Dict, Optional
import zarr
import torch

##
## raw: keys used in the dataset. Each key contains data for a whole episode
## obs: keys used in inference. Needs some pre-processing before sending to the policy NN.
## obs_preprocessed: obs with normalized rgb keys. len = whole episode
## obs_sample: len = obs horizon, pose computed relative to current pose (id = -1)
## action: pose command in world frame. len = whole episode
## action_sample: len = action horizon, pose computed relative to current pose (id = 0)


def _key_exists(data, key: str) -> bool:
    """Return True if key exists in data (works for dict, zarr.Group, etc.)."""
    try:
        _ = data[key]
        return True
    except (KeyError, IndexError):
        return False


def raw_to_obs(
    raw_data: Union[zarr.Group, Dict[str, np.ndarray]],
    episode_data: dict,
    shape_meta: dict,
):
    """convert shape_meta.raw data to shape_meta.obs.

    This function keeps image data as compressed zarr array in memory, while loads and decompresses
    low dim data.

    Args:
      raw_data: input, has keys from shape_meta.raw, each value is an ndarray of shape (t, ...)
      episode_data: output dictionary that matches shape_meta.obs
    """
    episode_data["obs"] = {}
    # obs.rgb and timestamp keys: copy directly from raw_data
    for key, attr in shape_meta["raw"].items():
        type = attr.get("type", "low_dim")
        if type == "rgb":
            episode_data["obs"][key] = raw_data[key]
        elif type == "timestamp":
            # Copy timestamp keys through (handles tactile_time_stamps etc.)
            # Use .get() so missing keys (not yet in obs_raw) are silently skipped
            val = raw_data[key] if hasattr(raw_data, '__getitem__') and _key_exists(raw_data, key) else None
            if val is not None:
                episode_data["obs"][key] = val if not hasattr(val, '__getitem__') else val[:]

    # obs.low_dim: load entry, convert to obs.low_dim
    for id in shape_meta["id_list"]:
        pose7_fb = raw_data[f"ts_pose_fb_{id}"]
        T = pose7_fb.shape[0]

        pose9_fb = su.SE3_to_pose9(su.pose7_to_SE3(pose7_fb))

        episode_data["obs"][f"robot{id}_eef_pos"] = pose9_fb[..., :3]
        episode_data["obs"][f"robot{id}_eef_rot_axis_angle"] = pose9_fb[..., 3:]

        # optional: abs
        if "robot0_abs_eef_pos" in shape_meta["obs"].keys():
            episode_data["obs"][f"robot{id}_abs_eef_pos"] = pose9_fb[..., :3]
            episode_data["obs"][f"robot{id}_abs_eef_rot_axis_angle"] = pose9_fb[..., 3:]

        # fingertip width → robot{id}_fingertip_width
        if f"robot{id}_fingertip_width" in shape_meta["obs"]:
            fw_key = f"fingertip_width_{id}"
            fw = raw_data[fw_key][:] if _key_exists(raw_data, fw_key) else np.zeros((T, 1), dtype=np.float64)
            episode_data["obs"][f"robot{id}_fingertip_width"] = fw

        # wrench: only for robots that have wrench in shape_meta
        has_wrench = f"robot{id}_eef_wrench" in shape_meta["obs"]
        wrench_key = f"wrench_{id}"
        wt_key = f"wrench_time_stamps_{id}"
        if has_wrench and _key_exists(raw_data, wrench_key):
            episode_data["obs"][f"robot{id}_eef_wrench"] = raw_data[wrench_key][:]
            episode_data["obs"][wt_key] = raw_data[wt_key][:] if _key_exists(raw_data, wt_key) else np.zeros(T)
        else:
            # No wrench sensor for this robot — fill zeros so downstream code stays consistent
            episode_data["obs"][f"robot{id}_eef_wrench"] = np.zeros((T, 6), dtype=np.float64)
            episode_data["obs"][wt_key] = np.zeros(T, dtype=np.float64)

        # robot-indexed timestamps
        episode_data["obs"][f"rgb_time_stamps_{id}"] = raw_data[f"rgb_time_stamps_{id}"][:]
        episode_data["obs"][f"robot_time_stamps_{id}"] = raw_data[f"robot_time_stamps_{id}"][:]


def raw_to_action9(
    raw_data: Union[zarr.Group, Dict[str, np.ndarray]],
    episode_data: dict,
    id_list: list,
):
    """Convert shape_meta.raw data to shape_meta.action.
    Note: if relative action is used, the relative pose still needs to be computed every time a sample
    is made. This function only converts the whole episode, and does not know what pose to be relative to.

    Args:
        raw_data: input, has keys from shape_meta.raw, each value is an ndarray of shape (t, ...)
        episode_data: output dictionary that has an 'action' field that matches shape_meta.action
    """
    action = []
    action_lens = []
    for id in id_list:
        # action: assemble from low_dim
        action_lens.append(raw_data[f"ts_pose_command_{id}"].shape[0])
        ts_pose7_command = raw_data[f"ts_pose_command_{id}"][:]
        ts_pose9_command = su.SE3_to_pose9(su.pose7_to_SE3(ts_pose7_command))
        action.append(ts_pose9_command)

    action_len = min(action_lens)
    action = [x[:action_len] for x in action]

    episode_data["action"] = np.concatenate(action, axis=-1)
    assert episode_data["action"].shape[1] == 9 or episode_data["action"].shape[1] == 18

    # action timestamps is set according to robot 0
    episode_data["action_time_stamps"] = raw_data["robot_time_stamps_0"][:action_len]


def raw_to_action19(
    raw_data: Union[zarr.Group, Dict[str, np.ndarray]],
    episode_data: dict,
    id_list: list,
):
    """Convert shape_meta.raw data to shape_meta.action.
    Note: if relative action is used, the relative pose still needs to be computed every time a sample
    is made. This function only converts the whole episode, and does not know what pose to be relative to.

    Args:
        raw_data: input, has keys from shape_meta.raw, each value is an ndarray of shape (t, ...)
        episode_data: output dictionary that has an 'action' field that matches shape_meta.action
    """
    action = []
    action_lens = []
    for id in id_list:
        # action: assemble from low_dim
        action_lens.append(raw_data[f"ts_pose_command_{id}"].shape[0])
        ts_pose7_command = raw_data[f"ts_pose_command_{id}"][:]
        ts_pose9_command = su.SE3_to_pose9(su.pose7_to_SE3(ts_pose7_command))
        ts_pose7_virtual_target = raw_data[f"ts_pose_virtual_target_{id}"][:]
        ts_pose9_virtual_target = su.SE3_to_pose9(
            su.pose7_to_SE3(ts_pose7_virtual_target)
        )
        stiffness = raw_data[f"stiffness_{id}"][:][:, np.newaxis]
        action.append(
            np.concatenate(
                [ts_pose9_command, ts_pose9_virtual_target, stiffness], axis=-1
            )
        )
    # action: trim to the shortest length
    action_len = min(action_lens)
    action = [x[:action_len] for x in action]

    episode_data["action"] = np.concatenate(action, axis=-1)
    assert (
        episode_data["action"].shape[1] == 19 or episode_data["action"].shape[1] == 38
    )

    # action timestamps is set according to robot 0
    episode_data["action_time_stamps"] = raw_data["robot_time_stamps_0"][:action_len]


def obs_rgb_preprocess(
    obs: dict,
    obs_output: dict,
    reshape_mode: str,
    shape_meta: dict,
):
    """Pre-process the rgb data in the obs dictionary as inputs to policy network.

    This function does the following to the rgb keys in the obs dictionary:
    * Unpack/unzip it, if the rgb data is still stored as a compressed zarr array (not recommended)
    * Reshape the rgb image, or just check its shape.
    * Convert uint8 (0~255) to float32 (0~1)
    * Move its axes from THWC to TCHW.
    Since this function unpacks the whole key, it should only be used for online inference.
    If used in training, so the data length is the obs horizon instead of the whole episode len.

    Args:
        obs: dict with keys from shape_meta.obs
        obs_output: dict with the same keys but processed images
        reshape_mode: One of 'reshape', 'check', or 'none'.
        shape_meta: the shape_meta from task.yaml
    """
    obs_shape_meta = shape_meta["obs"]
    for key, attr in obs_shape_meta.items():
        type = attr.get("type", "low_dim")
        shape = attr.get("shape")
        if type == "rgb":
            this_imgs_in = obs[key]
            t, hi, wi, ci = this_imgs_in.shape
            co, ho, wo = shape
            assert ci == co
            out_imgs = this_imgs_in
            if (ho != hi) or (wo != wi):
                if reshape_mode == "reshape":
                    tf = get_image_transform(
                        input_res=(wi, hi), output_res=(wo, ho), bgr_to_rgb=False
                    )
                    out_imgs = np.stack([tf(x) for x in this_imgs_in])
                elif reshape_mode == "check":
                    print(
                        f"[obs_rgb_preprocess] shape check failed! Require {ho}x{wo}, get {hi}x{wi}"
                    )
                    assert False
            if this_imgs_in.dtype == np.uint8 or this_imgs_in.dtype == np.int32:
                out_imgs = out_imgs.astype(np.float32) / 255

            # THWC to TCHW
            obs_output[key] = np.moveaxis(out_imgs, -1, 1)


def sparse_obs_to_obs_sample(
    obs_sparse: dict,  # each key: (T, D)
    shape_meta: dict,
    reshape_mode: str,
    id_list: list,
    ignore_rgb: bool = False,
):
    """Prepare a sample of sparse obs as inputs to policy network.

    After packing an obs dictionary with keys according to shape_meta.sample.obs.sparse, with
    length corresponding to the correct horizons, pass it to this function to get it ready
    for the policy network.

    It does two things:
        1. RGB: unpack, reshape, normalize, turn into float
        2. low dim: convert pose to relative pose, turn into float

    Args:
        obs_sparse: dict with keys from shape_meta['sample']['obs']['sparse']
        shape_meta: the shape_meta from task.yaml
        reshape_mode: One of 'reshape', 'check', or 'none'.
        ignore_rgb: if True, skip the rgb keys. Used when computing normalizers.
    return:
        sparse_obs_processed: dict with keys from shape_meta['sample']['obs']['sparse']
        base_SE3: the initial pose used for relative pose calculation
    """
    sparse_obs_processed = {}
    assert len(obs_sparse) > 0
    if not ignore_rgb:
        obs_rgb_preprocess(obs_sparse, sparse_obs_processed, reshape_mode, shape_meta)

    # copy all low dim keys
    for key, attr in shape_meta["obs"].items():
        type = attr.get("type", "low_dim")
        if type == "low_dim":
            sparse_obs_processed[key] = obs_sparse[key].astype(
                np.float32
            )  # astype() makes a copy

    # generate relative pose
    base_SE3_WT = []
    for id in id_list:
        # convert pose to mat
        SE3_WT = su.pose9_to_SE3(
            np.concatenate(
                [
                    sparse_obs_processed[f"robot{id}_eef_pos"],
                    sparse_obs_processed[f"robot{id}_eef_rot_axis_angle"],
                ],
                axis=-1,
            )
        )

        # solve relative obs
        base_SE3_WT.append(SE3_WT[-1])
        SE3_base_i = su.SE3_inv(base_SE3_WT[id]) @ SE3_WT

        pose9_relative = su.SE3_to_pose9(SE3_base_i)
        sparse_obs_processed[f"robot{id}_eef_pos"] = pose9_relative[..., :3]
        sparse_obs_processed[f"robot{id}_eef_rot_axis_angle"] = pose9_relative[..., 3:]

        # solve relative wrench (only for robots that have a wrench sensor)
        wrench_key = f"robot{id}_eef_wrench"
        if wrench_key in shape_meta.get("obs", {}) and wrench_key in obs_sparse:
            SE3_i_base = su.SE3_inv(SE3_base_i)[-1]
            wrench = su.transpose(su.SE3_to_adj(SE3_i_base)) @ np.expand_dims(
                obs_sparse[wrench_key], -1
            )
            sparse_obs_processed[wrench_key] = np.squeeze(wrench)

        # double check the shape
        for key, attr in shape_meta["sample"]["obs"]["sparse"].items():
            sparse_obs_horizon = attr["horizon"]
            if shape_meta["obs"][key]["type"] == "low_dim":
                assert len(sparse_obs_processed[key].shape) == 2  # (T, D)
                assert sparse_obs_processed[key].shape[0] == sparse_obs_horizon
            else:
                if not ignore_rgb:
                    assert len(sparse_obs_processed[key].shape) == 4  # (T, C, H, W)
                    assert sparse_obs_processed[key].shape[0] == sparse_obs_horizon

    return sparse_obs_processed, base_SE3_WT


def obs_to_obs_sample(
    obs_sparse: dict,  # each key: (T, D)
    shape_meta: dict,
    reshape_mode: str,
    id_list: list,
    ignore_rgb: bool = False,
):
    """Prepare a sample of obs as inputs to policy network.

    After packing an obs dictionary with keys according to shape_meta.obs, with
    length corresponding to the correct horizons, pass it to this function to get it ready
    for the policy network.

    It does two things:
        1. RGB: unpack, reshape, normalize, turn into float
        2. low dim: convert pose to relative pose, turn into float

    Args:
        obs_sparse: dict with keys from shape_meta['sample']['obs']['sparse']
        shape_meta: the shape_meta from task.yaml
        reshape_mode: One of 'reshape', 'check', or 'none'.
        ignore_rgb: if True, skip the rgb keys. Used when computing normalizers.
    """
    obs_processed = {"sparse": {}}
    obs_processed["sparse"], base_pose_mat = sparse_obs_to_obs_sample(
        obs_sparse, shape_meta, reshape_mode, id_list, ignore_rgb
    )
    return obs_processed


def action9_to_action_sample(
    action_sparse: np.ndarray,  # (T, D), D = 9
    id_list: list,
):
    """Prepare a sample of actions as labels for the policy network.

    This function is used in training. It takes a sample of actions (len = action_horizon)
    and convert the poses in it to be relative to the current pose (id = 0).

    """
    action_processed = {"sparse": {}}
    T, D = action_sparse.shape
    assert D == 9

    # generate relative pose
    # convert pose to mat
    pose9 = action_sparse
    SE3 = su.pose9_to_SE3(pose9)

    # solve relative obs
    base_SE3 = SE3[0]
    SE3_relative = su.SE3_inv(base_SE3) @ SE3

    pose9_relative = su.SE3_to_pose9(SE3_relative)
    action_processed["sparse"] = pose9_relative

    # double check the shape
    assert action_processed["sparse"].shape == (T, D)

    return action_processed


def action19_to_action_sample(
    action_sparse: np.ndarray,  # (T, D), D = 19 or 38
    id_list: list,
):
    """Prepare a sample of actions as labels for the policy network.

    This function is used in training. It takes a sample of actions (len = action_horizon)
    and convert the poses in it to be relative to the current pose (id = 0).

    """
    action_processed = {"sparse": {}}
    T, D = action_sparse.shape
    if len(id_list) == 1:
        assert D == 19
    else:
        assert D == 38

    def action19_preprocess(action: np.ndarray):
        # generate relative pose
        # convert pose to mat
        pose9 = action[:, 0:9]
        pose9_vt = action[:, 9:18]
        stiffness = action[:, 18:19]
        SE3 = su.pose9_to_SE3(pose9)
        SE3_vt = su.pose9_to_SE3(pose9_vt)

        # solve relative obs
        SE3_WBase_inv = su.SE3_inv(SE3[0])
        SE3_relative = SE3_WBase_inv @ SE3
        SE3_vt_relative = SE3_WBase_inv @ SE3_vt
        pose9_relative = su.SE3_to_pose9(SE3_relative)
        pose9_vt_relative = su.SE3_to_pose9(SE3_vt_relative)

        return np.concatenate([pose9_relative, pose9_vt_relative, stiffness], axis=-1)

    if len(id_list) == 1:
        action_processed["sparse"] = action19_preprocess(action_sparse)
    else:
        action_processed["sparse"] = np.concatenate(
            [
                action19_preprocess(action_sparse[:, :19]),
                action19_preprocess(action_sparse[:, 19:38]),
            ],
            axis=-1,
        )

    # double check the shape
    assert action_processed["sparse"].shape == (T, D)
    return action_processed


def action9_postprocess(
    action: np.ndarray,
    env_obs: Dict[str, np.ndarray],
):
    """Convert policy outputs from relative pose to world frame pose
    Used in online inference
    """
    # convert poses to mat
    current_SE3 = su.pose9_to_SE3(
        np.concatenate(
            [env_obs[f"robot0_eef_pos"][-1], env_obs[f"robot0_eef_rot_axis_angle"][-1]],
            axis=-1,
        )
    )

    action_pose9 = action[..., 0:9]
    action_SE3 = su.pose9_to_SE3(action_pose9)

    action_SE3_absolute = current_SE3 @ action_SE3

    # return pose matrices
    return action_SE3_absolute


def action19_postprocess(
    action: np.ndarray, current_SE3: list, id_list: list, fix_orientation=False
):
    """Convert policy outputs from relative pose to world frame pose
    Used in online inference
    """

    action_SE3_absolute = [np.array] * len(id_list)
    action_SE3_vt_absolute = [np.array] * len(id_list)
    stiffness = [0] * len(id_list)

    for id in id_list:
        action_pose9 = action[..., 19 * id + 0 : 19 * id + 9]
        action_pose9_vt = action[..., 19 * id + 9 : 19 * id + 18]
        stiffness[id] = action[..., 19 * id + 18]
        action_SE3 = su.pose9_to_SE3(action_pose9)
        action_SE3_vt = su.pose9_to_SE3(action_pose9_vt)

        action_SE3_absolute[id] = current_SE3[id] @ action_SE3
        action_SE3_vt_absolute[id] = current_SE3[id] @ action_SE3_vt

        # print(f"fix_orientation: {fix_orientation}")
        if fix_orientation:
            action_SE3_absolute[id][:, :3, :3] = current_SE3[:3, :3]
            action_SE3_vt_absolute[id][:, :3, :3] = current_SE3[:3, :3]

    # return pose matrices
    return action_SE3_absolute, action_SE3_vt_absolute, stiffness


def action30_postprocess(
    action: np.ndarray,
    current_SE3: list,
    id_list: list,
):
    """Convert 30-D policy output to world-frame SE3 matrices.

    Asymmetric bimanual layout (cable_mounting task):

      id=0 — compliant arm (has virtual target + stiffness):
        action[...,  0: 9]  target pose9
        action[...,  9:18]  virtual-target pose9
        action[..., 18:19]  stiffness scalar
        action[..., 19:20]  gripper (not decoded into SE3)

      id=1 — position-only arm (no virtual target, no stiffness):
        action[..., 20:29]  target pose9
        action[..., 29:30]  gripper (not decoded into SE3)

    Both pose9 sequences are expressed *relative* to the corresponding arm's
    current EE pose (current_SE3[id]).  This function reverses that transform
    to recover world-frame SE3 matrices.

    Args:
        action:      (T, 30) raw network output
        current_SE3: list[id] → (4, 4) current EE pose in world frame,
                     as returned by sparse_obs_to_obs_sample → base_SE3_WT
        id_list:     robot ids, must be [0, 1] for this layout

    Returns:
        target_mats  list[id] → (T, 4, 4) absolute target SE3
        vt_mats      list[id] → (T, 4, 4) absolute virtual-target SE3;
                                vt_mats[1] == target_mats[1] (position-only)
        stiffnesses  list[id] → (T,) scalar, or None for position-only arm
        grippers     list[id] → (T,) gripper width command
    """
    assert action.shape[-1] == 30, f"Expected 30-D action, got {action.shape[-1]}"

    n = max(id_list) + 1
    target_mats = [None] * n
    vt_mats     = [None] * n
    stiffnesses = [None] * n
    grippers    = [None] * n

    # --- id=0 : compliant arm ---
    SE3_0    = su.pose9_to_SE3(action[...,  0:9])
    SE3_vt_0 = su.pose9_to_SE3(action[...,  9:18])
    target_mats[0] = current_SE3[0] @ SE3_0
    vt_mats[0]     = current_SE3[0] @ SE3_vt_0
    stiffnesses[0] = action[..., 18]          # (T,)
    grippers[0]    = action[..., 19]          # (T,) gripper width

    # --- id=1 : position-only arm ---
    SE3_1 = su.pose9_to_SE3(action[..., 20:29])
    target_mats[1] = current_SE3[1] @ SE3_1
    vt_mats[1]     = target_mats[1]           # no virtual target
    stiffnesses[1] = None                      # no stiffness control
    grippers[1]    = action[..., 29]          # (T,) gripper width

    return target_mats, vt_mats, stiffnesses, grippers


def action30_to_action_sample(
    action_sparse: np.ndarray,  # (T, 30)
    id_list: list,
) -> dict:
    """Prepare a 30-D action sample as training label.

    Converts absolute poses to poses relative to the first timestep of each
    arm independently.  Stiffness and gripper values pass through unchanged.

    Layout mirrors action30_postprocess (see above).
    """
    action_processed = {"sparse": {}}
    T, D = action_sparse.shape
    assert D == 30, f"Expected 30-D action, got {D}"

    def _preprocess_compliant(a20: np.ndarray) -> np.ndarray:
        """Relativise pose9 and vt_pose9; pass stiffness and gripper through."""
        pose9     = a20[:, 0:9]
        pose9_vt  = a20[:, 9:18]
        stiffness = a20[:, 18:19]
        gripper   = a20[:, 19:20]

        SE3    = su.pose9_to_SE3(pose9)
        SE3_vt = su.pose9_to_SE3(pose9_vt)

        SE3_base_inv = su.SE3_inv(SE3[0])
        SE3_rel      = SE3_base_inv @ SE3
        SE3_vt_rel   = SE3_base_inv @ SE3_vt

        return np.concatenate([
            su.SE3_to_pose9(SE3_rel),
            su.SE3_to_pose9(SE3_vt_rel),
            stiffness,
            gripper,
        ], axis=-1)  # (T, 20)

    def _preprocess_position_only(a10: np.ndarray) -> np.ndarray:
        """Relativise pose9; pass gripper through."""
        pose9   = a10[:, 0:9]
        gripper = a10[:, 9:10]

        SE3 = su.pose9_to_SE3(pose9)
        SE3_base_inv = su.SE3_inv(SE3[0])
        SE3_rel = SE3_base_inv @ SE3

        return np.concatenate([
            su.SE3_to_pose9(SE3_rel),
            gripper,
        ], axis=-1)  # (T, 10)

    action_processed["sparse"] = np.concatenate([
        _preprocess_compliant(action_sparse[:, :20]),      # id=0
        _preprocess_position_only(action_sparse[:, 20:]),  # id=1
    ], axis=-1)

    assert action_processed["sparse"].shape == (T, D)
    return action_processed
