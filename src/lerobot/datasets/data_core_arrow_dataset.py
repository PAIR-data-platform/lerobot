#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""LeRobot dataset wrapping data-core-rust for the uint8 + GPU-convert path.

Research artifact. Returns **uint8 CHW** image tensors so the downstream training
loop can defer the ``.float().div_(255.0)`` step to the GPU (4× cheaper PCIe
transfer than f32). This does **not** plug into lerobot's stock training loop —
``NormalizerProcessorStep`` expects float32. Use this dataset directly from
benchmarks (e.g. ``scripts/d6_arrow_microbench.py``) or after adding GPU-side
conversion to the trainer's device-transfer step.

Two preload modes (see ``raw_pixels`` ctor arg):

- ``raw_pixels=False`` (default) — Episodes preloaded as **JPEG bytes**
  (~119 MB/ep at 480×640×4 cams; full 50-ep aloha fits in ~6 GB). Per-batch,
  Rust's ``get_sample_raw`` rayon-parallel libjpeg-decodes the requested
  frames into HWC uint8 bytes. No RAM ceiling.

- ``raw_pixels=True`` — Episodes preloaded as **raw RGB pixels** (~4 GB/ep at
  480×640×4 cams; ~7 of 50 aloha episodes fit on a 32 GB box). Per-batch,
  ``get_sample_raw`` returns the cached bytes directly with no decode. Fastest
  per-batch latency.

Background: an earlier ``mode="f32"`` variant called ``get_frame_windowed_f32``
to convert HWC→CHW + uint8→f32 + /255 in Rust. It was a regression vs the
existing COW path because the 4× larger tensor dominated H2D bandwidth. That
mode was removed; see ``results/d6_arrow_dataloader_benchmark.md`` for numbers.
"""

import logging
from pathlib import Path

import torch

import data_core

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import check_delta_timestamps, get_delta_indices
from lerobot.utils.constants import HF_LEROBOT_HOME


class DataCoreArrowDataset(torch.utils.data.Dataset):
    """Returns uint8 CHW images for downstream GPU-side conversion.

    ``raw_pixels=False`` (default): JPEG cache, Rust libjpeg decode per batch,
    no RAM ceiling. ``raw_pixels=True``: raw RGB pixel cache, ~4 GB/ep at
    480×640×4 cams, cap ``preload_episodes`` to fit RAM.
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
        raw_pixels: bool = False,
        output_dtype: str = "uint8",
        **ignored_kwargs,
    ):
        if output_dtype not in ("uint8", "float32"):
            raise ValueError(f"output_dtype must be 'uint8' or 'float32', got {output_dtype!r}")
        self.repo_id = repo_id
        self.root = Path(root) if root else HF_LEROBOT_HOME / repo_id
        self.image_transforms = image_transforms
        self.raw_pixels = raw_pixels
        self.output_dtype = output_dtype

        self.meta = LeRobotDatasetMetadata(repo_id=repo_id, root=self.root)

        self.lazy = data_core.LazyDataset.open(str(self.root))
        if raw_pixels:
            self.lazy = self.lazy.with_raw_pixels()

        if delta_timestamps is not None:
            check_delta_timestamps(delta_timestamps, self.meta.fps, tolerance_s)
            self.delta_indices = get_delta_indices(delta_timestamps, self.meta.fps)
        else:
            self.delta_indices = None

        self._cam_key_to_name: dict[str, str] = {
            k: k.removeprefix("observation.images.") for k in self.meta.camera_keys
        }

        self._image_offsets = [0]
        self._action_offsets = [0]
        if self.delta_indices:
            for key in self.meta.camera_keys:
                if key in self.delta_indices:
                    self._image_offsets = self.delta_indices[key]
                    break
            if "action" in self.delta_indices:
                self._action_offsets = self.delta_indices["action"]

        if episodes is not None:
            ep_list = list(episodes)
        elif preload_episodes is not None:
            ep_list = list(range(min(preload_episodes, self.meta.total_episodes)))
        else:
            ep_list = list(range(self.meta.total_episodes))
        self.episodes = ep_list

        max_action_offset = max(self._action_offsets) if self._action_offsets else 0
        ep_meta = self.meta.episodes
        frame_indices: list[int] = []
        for ep_idx in ep_list:
            start = ep_meta["dataset_from_index"][ep_idx]
            end = ep_meta["dataset_to_index"][ep_idx]
            safe_end = max(start, end - max_action_offset)
            frame_indices.extend(range(start, safe_end))
        self._frame_indices = frame_indices

        mode_tag = "raw-pixel" if raw_pixels else "JPEG"
        logging.info(
            f"DataCoreArrowDataset: preloading {len(self.episodes)} {mode_tag} episodes..."
        )
        self.lazy.preload_episodes(self.episodes)
        logging.info("DataCoreArrowDataset: preload complete")

    @property
    def num_frames(self) -> int:
        return len(self._frame_indices)

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)

    @property
    def fps(self) -> int:
        return self.meta.fps

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        global_idx = self._frame_indices[idx]
        sample = self.lazy.get_sample_raw(
            global_idx, self._image_offsets, self._action_offsets
        )

        result: dict[str, torch.Tensor] = {}

        for feature_key, cam_name in self._cam_key_to_name.items():
            if cam_name not in sample["images"]:
                continue
            raw_bytes, w, h, n_frames = sample["images"][cam_name]
            tensor = torch.frombuffer(bytearray(raw_bytes), dtype=torch.uint8)
            tensor = tensor.reshape(n_frames, h, w, 3).permute(0, 3, 1, 2).contiguous()
            if n_frames == 1:
                tensor = tensor.squeeze(0)
            if self.output_dtype == "float32":
                # Trainer-compat: pay the CPU cast in the worker so downstream
                # NormalizerProcessorStep receives float32 [0,1]. Trades the
                # 4× PCIe-bandwidth win for drop-in trainer compatibility.
                tensor = tensor.to(torch.float32).div_(255.0)
            if self.image_transforms is not None:
                tensor = self.image_transforms(tensor)
            result[feature_key] = tensor

        if "state" in sample and sample["state"]:
            state = torch.tensor(sample["state"], dtype=torch.float32)
            if state.dim() == 2 and state.shape[0] == 1:
                state = state.squeeze(0)
            result["observation.state"] = state

        if "action" in sample and sample["action"]:
            action = torch.tensor(sample["action"], dtype=torch.float32)
            if len(self._action_offsets) == 1 and action.dim() == 2:
                action = action.squeeze(0)
            result["action"] = action

        result["timestamp"] = torch.tensor(sample.get("timestamp", 0.0), dtype=torch.float32)
        result["frame_index"] = torch.tensor(sample.get("frame_index", 0), dtype=torch.int64)
        result["episode_index"] = torch.tensor(sample.get("episode_index", 0), dtype=torch.int64)
        result["index"] = torch.tensor(global_idx, dtype=torch.int64)
        result["task_index"] = torch.tensor(0, dtype=torch.int64)

        if self.delta_indices and "action" in self.delta_indices:
            chunk_size = len(self.delta_indices["action"])
            result["action_is_pad"] = torch.zeros(chunk_size, dtype=torch.bool)

        return result
