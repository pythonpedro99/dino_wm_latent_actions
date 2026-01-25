import pickle
from pathlib import Path
from typing import Callable, Optional, Any, Dict, Sequence, Union
from collections import OrderedDict

import torch
import numpy as np
import decord
from decord import VideoReader
from einops import rearrange

from .traj_dset import TrajDataset, TrajSlicerDataset

decord.bridge.set_bridge("torch")

# === Stats for subset 10k ===
#   split: train
#     - rel_actions.pth [RAW]: samples=2,214,000 feat_dim=2 (scale=1.0, stride=1)
#       mean: [-0.36953 ,  0.328818]
#       std : [14.843125, 14.722512]
#     - rel_actions.pth [SCALED/STRIDED]: samples=2,214,000 feat_dim=2 (scale=100.0, stride=1)
#       mean: [-0.003695,  0.003288]
#       std : [0.148431, 0.147225]
#     - states.pth: samples=2,214,000 feat_dim=5 (scale=1.0, stride=1)
#       mean: [115.863365, 148.13002 , 123.305336, 139.50023 ,   1.080451]
#       std : [135.9158  , 162.18025 , 132.06897 , 147.15077 ,   1.735476]
#     - states_constant.pth: missing

ACTION_MEAN = torch.tensor([-0.003695,  0.003288], dtype=torch.float32)
ACTION_STD  = torch.tensor([0.148431, 0.147225], dtype=torch.float32)

STATE_MEAN = torch.tensor(
    [115.863365, 148.13002 , 123.305336, 139.50023 ,   1.080451],
    dtype=torch.float32,
)
STATE_STD = torch.tensor(
    [135.9158  , 162.18025 , 132.06897 , 147.15077 ,   1.735476],
    dtype=torch.float32,
)


class PushTDataset(TrajDataset):
    """
    PushT dataset that returns vision + action + state.
    Proprio is not used and not returned.
    """

    def __init__(
        self,
        *,
        n_rollout: Optional[int],
        transform: Optional[Callable],
        data_path: str,
        normalize_action: bool,
        relative: bool,
        action_scale: float,
        with_velocity: bool,  # affects state only
    ):
        self.data_path = Path(data_path)
        self.transform = transform
        self.relative = relative
        self.normalize_action = normalize_action
        self.with_velocity = with_velocity
        # ---- per-worker VideoReader cache (each DataLoader worker has its own dataset instance) ----
        self._reader_cache = OrderedDict()
        self._reader_cache_max = 8  # tune: 4–16 depending on open file limits / locality


        # ---- load states (required for planning/evaluator success) ----
        self.states = torch.load(self.data_path / "states.pth").float()

        # ---- load actions ----
        if relative:
            self.actions = torch.load(self.data_path / "rel_actions.pth")
        else:
            self.actions = torch.load(self.data_path / "abs_actions.pth")
        self.actions = self.actions.float() / float(action_scale)

        # ---- sequence lengths ----
        with open(self.data_path / "seq_lengths.pkl", "rb") as f:
            self.seq_lengths = pickle.load(f)

        # ---- shapes optional ----
        shapes_file = self.data_path / "shapes.pkl"
        if shapes_file.exists():
            with open(shapes_file, "rb") as f:
                self.shapes = pickle.load(f)
        else:
            # fall back to number of trajectories in states
            self.shapes = ["T"] * len(self.states)

        # ---- truncate rollouts ----
        n = int(n_rollout) if n_rollout is not None else len(self.states)
        self.states = self.states[:n]
        self.actions = self.actions[:n]
        self.seq_lengths = self.seq_lengths[:n]
        self.shapes = self.shapes[:n]

        # ---- optional velocity appended to state ----
        if self.with_velocity:
            velocities = torch.load(self.data_path / "velocities.pth").float()
            velocities = velocities[:n]
            self.states = torch.cat([self.states, velocities], dim=-1)

        # ---- dims expected by downstream code ----
        self.action_dim = int(self.actions.shape[-1])
        self.state_dim = int(self.states.shape[-1])
        self.proprio_dim = 0  # explicitly unused

        # ---- normalize actions ----
        if normalize_action:
            self.action_mean = ACTION_MEAN
            self.action_std = ACTION_STD
            # state stats are useful for evaluators/preprocessors that expect them
            self.state_mean = STATE_MEAN[: self.state_dim]
            self.state_std = STATE_STD[: self.state_dim]
        else:
            self.action_mean = torch.zeros(self.action_dim, dtype=torch.float32)
            self.action_std = torch.ones(self.action_dim, dtype=torch.float32)
            self.state_mean = torch.zeros(self.state_dim, dtype=torch.float32)
            self.state_std = torch.ones(self.state_dim, dtype=torch.float32)

        self.actions = (self.actions - self.action_mean) / self.action_std

        print(f"Loaded {n} rollouts (vision + action + state) from {self.data_path}")

    def get_seq_length(self, idx: int) -> int:
        return int(self.seq_lengths[idx])
    
    def _get_reader(self, idx: int):
        key = int(idx)
        if key in self._reader_cache:
            self._reader_cache.move_to_end(key)
            return self._reader_cache[key]

        vid_dir = self.data_path / "obses"
        candidates = [
            vid_dir / f"episode_{key:04d}.mp4",  # 4-pad (train, maybe)
            vid_dir / f"episode_{key:03d}.mp4",  # 3-pad (val, per your note)
        ]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            raise FileNotFoundError(
                f"No video found for episode {key}. Tried: " + ", ".join(p.name for p in candidates)
            )

        reader = VideoReader(str(path), num_threads=1)

        self._reader_cache[key] = reader
        self._reader_cache.move_to_end(key)
        if len(self._reader_cache) > self._reader_cache_max:
            self._reader_cache.popitem(last=False)

        return reader


    def get_visual(self, idx: int, frames: Sequence[int]) -> torch.Tensor:
        # Ensure frames is a concrete list for decord
        frames = list(map(int, frames))

        reader = self._get_reader(idx)
        image = reader.get_batch(frames)  # THWC torch tensor (decord bridge)
        image = image / 255.0
        image = rearrange(image, "T H W C -> T C H W")

        if self.transform:
            image = self.transform(image)

        return image


    def get_all_actions(self) -> torch.Tensor:
        result = []
        for i in range(len(self.seq_lengths)):
            T = int(self.seq_lengths[i])
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)

    def get_frames(self, idx: int, frames: Sequence[int]):
        frames = list(map(int, frames))  # important when frames is range()

        reader = self._get_reader(idx)

        act = self.actions[idx, frames]
        state = self.states[idx, frames]
        shape = self.shapes[idx]

        image = reader.get_batch(frames)
        image = image / 255.0
        image = rearrange(image, "T H W C -> T C H W")

        if self.transform:
            image = self.transform(image)

        obs = {"visual": image}
        info: Dict[str, Any] = {"shape": shape}
        return obs, act, state, info


    def __getitem__(self, idx: int):
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def __len__(self) -> int:
        return len(self.seq_lengths)

    def preprocess_imgs(self, imgs):
        if isinstance(imgs, np.ndarray):
            raise NotImplementedError
        if isinstance(imgs, torch.Tensor):
            return rearrange(imgs, "b h w c -> b c h w") / 255.0
        raise TypeError(f"Unsupported type for imgs: {type(imgs)}")


def load_pusht_slice_train_val(
    *,
    transform,
    data_path: str,
    n_rollout: int,
    num_hist: int,
    num_pred: int,
    frameskip: int,
    normalize_action: bool,
    action_scale: float,
    relative: bool,
    with_velocity: bool,
    process_actions: str,
):
    train_dset = PushTDataset(
        n_rollout=n_rollout,
        transform=transform,
        data_path=data_path + "/train",
        normalize_action=normalize_action,
        relative=relative,
        action_scale=action_scale,
        with_velocity=with_velocity,
    )

    val_dset = PushTDataset(
        n_rollout=n_rollout,
        transform=transform,
        data_path=data_path + "/val",
        normalize_action=normalize_action,
        relative=relative,
        action_scale=action_scale,
        with_velocity=with_velocity,
    )

    num_frames = num_hist + num_pred
    
    train_slices = TrajSlicerDataset(
    train_dset,
    num_frames=num_frames,
    frameskip=frameskip,
    process_actions=process_actions,
    )

    val_slices = TrajSlicerDataset(
        val_dset,
        num_frames=num_frames,
        frameskip=frameskip,
        process_actions=process_actions,
    )


    datasets = {"train": train_slices, "valid": val_slices}
    traj_dset = {"train": train_dset, "valid": val_dset}
    return datasets, traj_dset
