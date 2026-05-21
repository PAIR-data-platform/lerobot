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

from pathlib import Path

import torch
import torchvision.io as tvio

import data_core

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import check_delta_timestamps, get_delta_indices
from lerobot.utils.constants import HF_LEROBOT_HOME


def _decode_jpeg(jpeg_bytes: bytes) -> torch.Tensor:
    buf = torch.frombuffer(bytearray(jpeg_bytes), dtype=torch.uint8)
    return tvio.decode_image(buf, tvio.ImageReadMode.RGB).float() / 255.0  # [3, H, W]


class DataCoreDataset(torch.utils.data.Dataset):
    """LeRobot-compatible dataset backed by data-core-rust's LazyDataset.

    Drop-in replacement for LeRobotDataset that uses a Rust-backed lazy loader
    for faster episode-level caching and on-the-fly augmentation support.
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
        **ignored_kwargs,
    ):
        self.repo_id = repo_id
        self.root = Path(root) if root else HF_LEROBOT_HOME / repo_id
        self.episodes = episodes
        self.image_transforms = image_transforms

        # Reuse LeRobot's metadata loader (reads info.json, stats.json, episodes.parquet)
        self.meta = LeRobotDatasetMetadata(repo_id=repo_id, root=self.root)

        # Open Rust-backed lazy dataset — near-instant, loads only metadata
        self.lazy = data_core.LazyDataset.open(str(self.root))

        # Resolve delta timestamps → integer frame offsets
        if delta_timestamps is not None:
            check_delta_timestamps(delta_timestamps, self.meta.fps, tolerance_s)
            self.delta_indices = get_delta_indices(delta_timestamps, self.meta.fps)
        else:
            self.delta_indices = None

        # Build flat list of global frame indices (respects episode subset filter)
        if episodes is not None:
            ep_meta = self.meta.episodes
            frame_indices = []
            for ep_idx in episodes:
                start = ep_meta["dataset_from_index"][ep_idx]
                end = ep_meta["dataset_to_index"][ep_idx]
                frame_indices.extend(range(start, end))
            self._frame_indices = frame_indices
        else:
            self._frame_indices = list(range(self.meta.total_frames))

        # Map full LeRobot camera key → bare camera name used by data-core Frame API
        # e.g. "observation.images.image" → "image"
        self._cam_key_to_name: dict[str, str] = {
            k: k.removeprefix("observation.images.")
            for k in self.meta.camera_keys
        }

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

        # Compute the union of all frame offsets we need for this sample
        all_offsets_set: set[int] = {0}
        if self.delta_indices:
            for offsets in self.delta_indices.values():
                all_offsets_set.update(offsets)
        all_offsets = sorted(all_offsets_set)

        total = self.meta.total_frames

        # Fetch frames; episode is cached in Rust after first access
        loaded: dict[int, data_core.Frame] = {
            offset: self.lazy.get_frame(max(0, min(total - 1, global_idx + offset)))
            for offset in all_offsets
        }

        result: dict[str, torch.Tensor] = {}

        for feature_key, feature_info in self.meta.features.items():
            dtype = feature_info["dtype"]

            # These scalar keys are written below from the current frame
            if dtype in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
                continue

            # Determine which offsets this key uses
            if self.delta_indices and feature_key in self.delta_indices:
                use_offsets = self.delta_indices[feature_key]
            else:
                use_offsets = [0]

            selected = [loaded[o] for o in use_offsets]
            n = len(selected)

            if dtype in ("video", "image"):
                cam = self._cam_key_to_name[feature_key]
                imgs = [_decode_jpeg(f.image(cam)) for f in selected]
                tensor = torch.stack(imgs)          # [n, 3, H, W]
                if n == 1:
                    tensor = tensor.squeeze(0)      # [3, H, W]
                if self.image_transforms is not None:
                    tensor = self.image_transforms(tensor)
                result[feature_key] = tensor

            elif dtype == "float32":
                if feature_key == "action":
                    vecs = [torch.tensor(f.action, dtype=torch.float32) for f in selected]
                elif feature_key == "observation.state":
                    vecs = [torch.tensor(f.state or [], dtype=torch.float32) for f in selected]
                else:
                    continue  # unrecognised float key — skip gracefully
                tensor = torch.stack(vecs)          # [n, dim]
                if n == 1:
                    tensor = tensor.squeeze(0)      # [dim]
                result[feature_key] = tensor

        # Scalar metadata from the current (offset-0) frame
        cur = loaded[0]
        result["timestamp"] = torch.tensor(cur.timestamp, dtype=torch.float64)
        result["frame_index"] = torch.tensor(cur.frame_index, dtype=torch.int64)
        result["episode_index"] = torch.tensor(cur.episode_index, dtype=torch.int64)
        result["index"] = torch.tensor(global_idx, dtype=torch.int64)
        result["task_index"] = torch.tensor(cur.task_index, dtype=torch.int64)

        # Action padding mask — always all-False here; rely on EpisodeAwareSampler
        # with drop_n_last_frames to avoid requesting frames beyond episode end.
        if self.delta_indices and "action" in self.delta_indices:
            chunk_size = len(self.delta_indices["action"])
            result["action_is_pad"] = torch.zeros(chunk_size, dtype=torch.bool)

        return result
