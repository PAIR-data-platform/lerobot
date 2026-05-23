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

import io
import logging
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

import data_core

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import check_delta_timestamps, get_delta_indices
from lerobot.utils.constants import HF_LEROBOT_HOME


class DataCoreStreamingDataset(torch.utils.data.Dataset):
    """LeRobot-compatible dataset with Rust on-demand video decode.

    Unlike DataCoreDataset (which preloads all pixels to RAM), this streams
    frames from video on-demand via Rust's ffmpeg decoder. Episodes are cached
    as JPEG bytes (~100MB/ep at 480x640) rather than raw pixels (~4GB/ep),
    allowing hundreds of episodes at any resolution.

    Pass ``use_data_core_streaming=True`` in DatasetConfig (or via
    ``--dataset.use_data_core_streaming=true`` on the CLI) to activate.
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms=None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 1e-4,
        cache_limit: int | None = None,
        **ignored_kwargs,
    ):
        self.repo_id = repo_id
        self.root = Path(root) if root else HF_LEROBOT_HOME / repo_id
        self.image_transforms = image_transforms

        self.meta = LeRobotDatasetMetadata(repo_id=repo_id, root=self.root)

        # Open WITHOUT raw pixels — episodes cached as JPEG (small footprint)
        # ~119MB/ep at 480×640×4cams vs ~4GB/ep with raw pixels
        self.lazy = data_core.LazyDataset.open(str(self.root))

        # Resolve delta timestamps -> integer frame offsets
        if delta_timestamps is not None:
            check_delta_timestamps(delta_timestamps, self.meta.fps, tolerance_s)
            self.delta_indices = get_delta_indices(delta_timestamps, self.meta.fps)
        else:
            self.delta_indices = None

        # Map full LeRobot camera key -> bare camera name used by data-core
        self._cam_key_to_name: dict[str, str] = {
            k: k.removeprefix("observation.images.")
            for k in self.meta.camera_keys
        }

        # Compute image and action offsets
        self._image_offsets = [0]
        self._action_offsets = [0]
        if self.delta_indices:
            for key in self.meta.camera_keys:
                if key in self.delta_indices:
                    self._image_offsets = self.delta_indices[key]
                    break
            if "action" in self.delta_indices:
                self._action_offsets = self.delta_indices["action"]

        # Resolve episodes
        if episodes is not None:
            ep_list = list(episodes)
        else:
            ep_list = list(range(self.meta.total_episodes))
        self.episodes = ep_list

        # Build frame indices with boundary trimming
        max_action_offset = max(self._action_offsets) if self._action_offsets else 0
        ep_meta = self.meta.episodes
        frame_indices = []
        for ep_idx in ep_list:
            start = ep_meta["dataset_from_index"][ep_idx]
            end = ep_meta["dataset_to_index"][ep_idx]
            safe_end = max(start, end - max_action_offset)
            frame_indices.extend(range(start, safe_end))
        self._frame_indices = frame_indices

        # No upfront preload — episodes decode on-demand and cache in Rust's
        # internal LRU. Use num_workers=0 to avoid fork (FFmpeg hangs after fork).
        # A background thread warms the cache while training runs.
        self._cache_ready = threading.Event()
        self._bg_thread = threading.Thread(
            target=self._background_warmup,
            daemon=True,
        )
        self._bg_thread.start()

        logging.info(
            f"DataCoreStreamingDataset: {len(self.episodes)} episodes, "
            f"{len(self._frame_indices)} frames (streaming, background warmup started)"
        )

    def _background_warmup(self):
        """Progressively warm the Rust episode cache in a background thread."""
        ep_meta = self.meta.episodes
        for i, ep_idx in enumerate(self.episodes):
            try:
                start_frame = ep_meta["dataset_from_index"][ep_idx]
                self.lazy.get_frame(start_frame)
            except Exception as e:
                logging.warning(f"Background warmup: episode {ep_idx} failed: {e}")
            if (i + 1) % 10 == 0:
                logging.info(f"Background warmup: {i + 1}/{len(self.episodes)} episodes cached")
        self._cache_ready.set()
        logging.info("Background warmup: all episodes cached")

    # ------------------------------------------------------------------
    # Properties expected by lerobot_train.py and EpisodeAwareSampler
    # ------------------------------------------------------------------

    @property
    def num_frames(self) -> int:
        return len(self._frame_indices)

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)

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
        num_frames = self.lazy.num_frames

        # Get main frame — triggers on-demand episode decode + caching
        frame = self.lazy.get_frame(global_idx)

        result: dict[str, torch.Tensor] = {}

        # --- Images: JPEG bytes -> PIL decode -> float32 CHW [0,1] ---
        for feature_key, cam_name in self._cam_key_to_name.items():
            tensors = []
            for offset in self._image_offsets:
                if offset == 0:
                    img_bytes = frame.image(cam_name)
                else:
                    target = max(0, min(global_idx + offset, num_frames - 1))
                    other = self.lazy.get_frame(target)
                    img_bytes = other.image(cam_name)

                if img_bytes is None:
                    continue

                img = Image.open(io.BytesIO(bytes(img_bytes)))
                tensors.append(TF.to_tensor(img))  # float32 CHW [0,1]

            if not tensors:
                continue

            tensor = tensors[0] if len(tensors) == 1 else torch.stack(tensors)
            if self.image_transforms is not None:
                tensor = self.image_transforms(tensor)
            result[feature_key] = tensor

        # --- State ---
        state = frame.state
        if state is not None:
            result["observation.state"] = torch.tensor(state, dtype=torch.float32)

        # --- Action (collect across offsets) ---
        actions = []
        for offset in self._action_offsets:
            if offset == 0:
                actions.append(frame.action)
            else:
                target = max(0, min(global_idx + offset, num_frames - 1))
                other = self.lazy.get_frame(target)
                actions.append(other.action)
        action = torch.tensor(actions, dtype=torch.float32)
        if len(self._action_offsets) == 1:
            action = action.squeeze(0)
        result["action"] = action

        # --- Scalar metadata ---
        result["timestamp"] = torch.tensor(frame.timestamp, dtype=torch.float32)
        result["frame_index"] = torch.tensor(frame.frame_index, dtype=torch.int64)
        result["episode_index"] = torch.tensor(frame.episode_index, dtype=torch.int64)
        result["index"] = torch.tensor(global_idx, dtype=torch.int64)
        result["task_index"] = torch.tensor(0, dtype=torch.int64)

        # --- Action padding mask ---
        if self.delta_indices and "action" in self.delta_indices:
            chunk_size = len(self.delta_indices["action"])
            result["action_is_pad"] = torch.zeros(chunk_size, dtype=torch.bool)

        return result
