#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
from pathlib import Path

import torch

import data_core

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import check_delta_timestamps, get_delta_indices
from lerobot.utils.constants import HF_LEROBOT_HOME


class DataCoreDataset(torch.utils.data.Dataset):
    """LeRobot-compatible dataset backed by data-core-rust with COW caching.

    Drop-in replacement for LeRobotDataset that pre-decodes video to raw RGB
    pixels and serves them from a frozen lock-free cache. Forked DataLoader
    workers share pixel memory via Linux copy-on-write.

    Pass ``use_data_core=True`` in DatasetConfig (or via ``--dataset.use_data_core=true``
    on the CLI) to activate this path in ``make_dataset``.
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms=None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 1e-4,
        preload_episodes: int | None = None,
        augmentations: list[dict] | None = None,
        **ignored_kwargs,
    ):
        self.repo_id = repo_id
        self.root = Path(root) if root else HF_LEROBOT_HOME / repo_id
        self.image_transforms = image_transforms

        self.meta = LeRobotDatasetMetadata(repo_id=repo_id, root=self.root)

        # Open with raw pixels for zero-copy COW caching
        self.lazy = data_core.LazyDataset.open(str(self.root)).with_raw_pixels()

        # Register Rust-side augmentations (applied on the Arrow IR before the
        # data crosses into Python). Order matters: each entry is a dict
        # ``{"name": "...", "params": {...}}``; ``params`` is JSON-serialized
        # and passed to ``LazyDataset.add_augmentation``.
        if augmentations:
            for aug in augmentations:
                name = aug["name"]
                params = aug.get("params", {})
                self.lazy.add_augmentation(name, json.dumps(params))
                logging.info(f"DataCoreDataset: registered aug '{name}' with params={params}")

        # Resolve delta timestamps → integer frame offsets
        if delta_timestamps is not None:
            check_delta_timestamps(delta_timestamps, self.meta.fps, tolerance_s)
            self.delta_indices = get_delta_indices(delta_timestamps, self.meta.fps)
        else:
            self.delta_indices = None

        # Map full LeRobot camera key → bare camera name used by data-core
        # e.g. "observation.images.image" → "image"
        self._cam_key_to_name: dict[str, str] = {
            k: k.removeprefix("observation.images.")
            for k in self.meta.camera_keys
        }

        # Compute image and action offsets for get_sample_raw
        self._image_offsets = [0]
        self._action_offsets = [0]
        if self.delta_indices:
            for key in self.meta.camera_keys:
                if key in self.delta_indices:
                    self._image_offsets = self.delta_indices[key]
                    break
            if "action" in self.delta_indices:
                self._action_offsets = self.delta_indices["action"]

        # Resolve which episodes to use: explicit list > preload limit > all
        if episodes is not None:
            ep_list = list(episodes)
        elif preload_episodes is not None:
            ep_list = list(range(min(preload_episodes, self.meta.total_episodes)))
        else:
            ep_list = list(range(self.meta.total_episodes))
        self.episodes = ep_list

        # Build flat list of global frame indices for the selected episodes.
        # Trim the last max_action_offset frames from each episode to prevent
        # action lookups from crossing into unpreloaded episodes.
        max_action_offset = max(self._action_offsets) if self._action_offsets else 0
        ep_meta = self.meta.episodes
        frame_indices = []
        for ep_idx in ep_list:
            start = ep_meta["dataset_from_index"][ep_idx]
            end = ep_meta["dataset_to_index"][ep_idx]
            safe_end = max(start, end - max_action_offset)
            frame_indices.extend(range(start, safe_end))
        self._frame_indices = frame_indices

        # Preload episodes into frozen COW cache
        logging.info(f"DataCoreDataset: preloading {len(self.episodes)} episodes with raw pixels...")
        self.lazy.preload_episodes(self.episodes)
        logging.info("DataCoreDataset: preload complete")

    # ------------------------------------------------------------------
    # Properties expected by lerobot_train.py and EpisodeAwareSampler
    # ------------------------------------------------------------------

    @property
    def num_frames(self) -> int:
        return len(self._frame_indices)

    @property
    def num_episodes(self) -> int:
        return len(self.episodes) if self.episodes is not None else self.meta.total_episodes

    @property
    def fps(self) -> int:
        return self.meta.fps

    # ------------------------------------------------------------------
    # PyTorch Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        global_idx = self._frame_indices[idx]

        # Fetch raw sample from Rust frozen cache
        # Images come from get_frame_windowed (handles .rgb columns),
        # actions come from get_frame (only needs frame.action, not images)
        sample = self.lazy.get_sample_raw(
            global_idx, self._image_offsets, self._action_offsets
        )

        result: dict[str, torch.Tensor] = {}

        # --- Images: raw HWC uint8 bytes → float32 CHW tensor ---
        for feature_key, cam_name in self._cam_key_to_name.items():
            if cam_name not in sample["images"]:
                continue
            raw_bytes, w, h, n_frames = sample["images"][cam_name]
            tensor = torch.frombuffer(bytearray(raw_bytes), dtype=torch.uint8)
            tensor = tensor.reshape(n_frames, h, w, 3).permute(0, 3, 1, 2).contiguous()
            tensor = tensor.float() / 255.0
            if n_frames == 1:
                tensor = tensor.squeeze(0)  # [3, H, W]
            if self.image_transforms is not None:
                tensor = self.image_transforms(tensor)
            result[feature_key] = tensor

        # --- State ---
        if "state" in sample and sample["state"]:
            state = torch.tensor(sample["state"], dtype=torch.float32)
            if state.dim() == 2 and state.shape[0] == 1:
                state = state.squeeze(0)  # [state_dim]
            result["observation.state"] = state

        # --- Action ---
        if "action" in sample and sample["action"]:
            action = torch.tensor(sample["action"], dtype=torch.float32)
            if len(self._action_offsets) == 1:
                action = action.squeeze(0) if action.dim() == 2 else action
            result["action"] = action

        # --- Scalar metadata ---
        result["timestamp"] = torch.tensor(sample.get("timestamp", 0.0), dtype=torch.float32)
        result["frame_index"] = torch.tensor(sample.get("frame_index", 0), dtype=torch.int64)
        result["episode_index"] = torch.tensor(sample.get("episode_index", 0), dtype=torch.int64)
        result["index"] = torch.tensor(global_idx, dtype=torch.int64)
        result["task_index"] = torch.tensor(0, dtype=torch.int64)

        # --- Action padding mask ---
        if self.delta_indices and "action" in self.delta_indices:
            chunk_size = len(self.delta_indices["action"])
            result["action_is_pad"] = torch.zeros(chunk_size, dtype=torch.bool)

        return result
