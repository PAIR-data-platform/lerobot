#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""MimicGen environment wrapper for LeRobot.

Wraps robosuite environments registered by the ``mimicgen`` package into the
standard ``gymnasium.Env`` interface expected by the LeRobot eval pipeline.

The observation structure mirrors the LIBERO wrapper so that the same
processor step and policy pipelines can be reused.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from lerobot.processor import RobotObservation


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ACTION_DIM = 7
ACTION_LOW = -1.0
ACTION_HIGH = 1.0

# Default max episode steps per task family (conservative upper bounds).
# Callers can override via episode_length in the config.
DEFAULT_MAX_EPISODE_STEPS = 400

# Human-readable task descriptions for language-conditioned policies.
TASK_DESCRIPTIONS: dict[str, str] = {
    "Coffee_D0": "Pick up the coffee pod and place it in the coffee machine",
    "Coffee_D1": "Pick up the coffee pod and place it in the coffee machine",
    "Coffee_D2": "Pick up the coffee pod and place it in the coffee machine",
    "Stack_D0": "Stack cube A on top of cube B",
    "Stack_D1": "Stack cube A on top of cube B",
    "StackThree_D0": "Stack three cubes on top of each other",
    "StackThree_D1": "Stack three cubes on top of each other",
    "Square_D0": "Place the square nut on the square peg",
    "Square_D1": "Place the square nut on the square peg",
    "Square_D2": "Place the square nut on the square peg",
    "Threading_D0": "Thread the needle through the tripod",
    "Threading_D1": "Thread the needle through the tripod",
    "Threading_D2": "Thread the needle through the tripod",
    "ThreePieceAssembly_D0": "Assemble three pieces together on the base",
    "ThreePieceAssembly_D1": "Assemble three pieces together on the base",
    "ThreePieceAssembly_D2": "Assemble three pieces together on the base",
    "HammerCleanup_D0": "Pick up the hammer and place it in the drawer",
    "HammerCleanup_D1": "Pick up the hammer and place it in the drawer",
    "MugCleanup_D0": "Pick up the mug and place it in the drawer",
    "MugCleanup_D1": "Pick up the mug and place it in the drawer",
    "MugCleanup_O1": "Pick up the mug and place it in the drawer",
    "MugCleanup_O2": "Pick up the mug and place it in the drawer",
    "NutAssembly_D0": "Assemble the nuts onto the pegs",
    "PickPlace_D0": "Pick objects and place them in the correct bins",
    "Kitchen_D0": "Complete the kitchen task sequence",
    "Kitchen_D1": "Complete the kitchen task sequence",
    "CoffeePreparation_D0": "Prepare coffee from start to finish",
    "CoffeePreparation_D1": "Prepare coffee from start to finish",
}

# Tasks that use a non-Panda default robot
NON_PANDA_TASKS: dict[str, dict[str, str]] = {
    "NutAssembly_D0": {"robots": "Sawyer"},
    "PickPlace_D0": {"robots": "Sawyer"},
}


def _parse_camera_names(camera_name: str | Sequence[str]) -> list[str]:
    """Normalize camera_name into a non-empty list of strings."""
    if isinstance(camera_name, str):
        cams = [c.strip() for c in camera_name.split(",") if c.strip()]
    elif isinstance(camera_name, (list, tuple)):
        cams = [str(c).strip() for c in camera_name if str(c).strip()]
    else:
        raise TypeError(f"camera_name must be str or sequence, got {type(camera_name).__name__}")
    if not cams:
        raise ValueError("camera_name resolved to an empty list.")
    return cams


def get_mimicgen_dummy_action():
    """No-op action: zero delta EEF + gripper close."""
    return [0, 0, 0, 0, 0, 0, -1]


def _load_init_states(path: str | Path) -> dict:
    """Load init states saved by the conversion script."""
    payload = torch.load(path, weights_only=False)  # nosec B614
    return payload


class MimicGenEnv(gym.Env):
    """Gymnasium wrapper around a MimicGen / robosuite environment."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        env_name: str,
        init_states_path: str | Path | None = None,
        camera_name: str | Sequence[str] = "agentview,robot0_eye_in_hand",
        camera_name_mapping: dict[str, str] | None = None,
        obs_type: str = "pixels_agent_pos",
        render_mode: str = "rgb_array",
        observation_height: int = 84,
        observation_width: int = 84,
        render_height: int | None = None,
        render_width: int | None = None,
        max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
        control_freq: int = 20,
        num_steps_wait: int = 10,
    ):
        super().__init__()
        self.env_name = env_name
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.observation_height = observation_height
        self.observation_width = observation_width
        # render_height/width default to observation_*; override for nicer videos.
        self.render_height = render_height if render_height is not None else observation_height
        self.render_width = render_width if render_width is not None else observation_width
        self.num_steps_wait = num_steps_wait
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

        self.camera_name = _parse_camera_names(camera_name)
        if camera_name_mapping is None:
            camera_name_mapping = {
                "agentview": "image",
                "robot0_eye_in_hand": "wrist_image",
            }
        self.camera_name_mapping = camera_name_mapping

        # Task metadata
        self.task = env_name
        self.task_description = TASK_DESCRIPTIONS.get(env_name, env_name)

        # Load init states for deterministic eval
        self._init_states = None
        self._model_files = None
        if init_states_path is not None:
            payload = _load_init_states(init_states_path)
            self._init_states = payload["states"]  # (N, state_dim)
            self._model_files = payload.get("model_files")

        # Create robosuite environment
        self._env = self._make_robosuite_env(env_name, control_freq)

        # Observation / action spaces
        images = {}
        for cam in self.camera_name:
            mapped = self.camera_name_mapping[cam]
            images[mapped] = spaces.Box(
                low=0, high=255,
                shape=(self.observation_height, self.observation_width, 3),
                dtype=np.uint8,
            )

        if self.obs_type == "pixels":
            self.observation_space = spaces.Dict({"pixels": spaces.Dict(images)})
        elif self.obs_type == "pixels_agent_pos":
            self.observation_space = spaces.Dict({
                "pixels": spaces.Dict(images),
                # Flat 8D state: eef_pos(3) + axis_angle(3) + gripper_qpos(2)
                "agent_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float64),
            })
        else:
            raise ValueError(f"Unsupported obs_type: {self.obs_type}")

        self.action_space = spaces.Box(
            low=ACTION_LOW, high=ACTION_HIGH, shape=(ACTION_DIM,), dtype=np.float32
        )

    def _make_robosuite_env(self, env_name: str, control_freq: int):
        """Create the underlying robosuite environment."""
        import mimicgen_envs  # noqa: F401 — registers MimicGen tasks with robosuite
        import robosuite
        from robosuite.controllers import load_controller_config

        robot_kwargs = NON_PANDA_TASKS.get(env_name, {})
        robots = robot_kwargs.get("robots", "Panda")

        controller_config = load_controller_config(default_controller="OSC_POSE")

        env = robosuite.make(
            env_name=env_name,
            robots=robots,
            has_renderer=False,
            has_offscreen_renderer=True,
            ignore_done=True,
            use_camera_obs=True,
            use_object_obs=True,
            control_freq=control_freq,
            controller_configs=controller_config,
            camera_names=self.camera_name,
            camera_heights=self.observation_height,
            camera_widths=self.observation_width,
            reward_shaping=False,
            # render_collision_mesh / render_visual_mesh exist as kwargs but
            # robosuite 1.4.1's _reset_internal sets them in a way that doesn't
            # stick on the offscreen render context actually used by
            # sim.render() — eval videos still showed collision geoms
            # overlapping visual geoms (yellow/green/blue Panda). The explicit
            # vopt fixup below is what actually works.
            render_collision_mesh=False,
            render_visual_mesh=True,
        )
        env.reset()
        return env

    def _fix_render_vopt(self):
        """Workaround for robosuite 1.4.1 offscreen-renderer vopt bug.

        Robosuite's _reset_internal sets vopt.geomgroup on the offscreen
        render context, but in practice the context used by sim.render()
        ends up with all groups enabled. Collision geoms then render on top
        of visual geoms as colored patches on the Panda. See
        ARISE-Initiative/robosuite#636 and wandb eval frames from Mrinal's
        Coffee_D0 baseline.

        Force a render to ensure the offscreen context exists, then set
        geomgroup explicitly. Robosuite's XML convention: group 0 = collision,
        group 1 = visual. This must be called after every robosuite reset
        because reset overwrites vopt.
        """
        try:
            _ = self._env.sim.render(
                camera_name=self.camera_name[0],
                width=self.observation_width,
                height=self.observation_height,
            )
            ctx = self._env.sim._render_context_offscreen
            if ctx is not None:
                ctx.vopt.geomgroup[0] = 0  # collision off
                ctx.vopt.geomgroup[1] = 1  # visual on
                for i in range(2, 6):
                    ctx.vopt.geomgroup[i] = 0
        except Exception as e:
            import warnings
            warnings.warn(f"mimicgen: failed to fix offscreen render vopt: {e}")

    def _format_raw_obs(self, raw_obs: dict) -> RobotObservation:
        """Extract and restructure raw robosuite observations."""
        from scipy.spatial.transform import Rotation

        images = {}
        for cam in self.camera_name:
            img_key = f"{cam}_image"
            # Flip vertically to match the training data's mujoco-raw orientation.
            # robosuite's `_get_observations()` returns right-side-up images while
            # the recorded HDF5 demos store mujoco-raw upside-down — train/eval
            # mismatch confirmed via row-mean RGB diff. Without this, the policy
            # sees flipped visual input at eval and silently degrades.
            images[self.camera_name_mapping[cam]] = np.ascontiguousarray(raw_obs[img_key][::-1])

        if self.obs_type == "pixels":
            return {"pixels": images}

        # Build flat 8D state: eef_pos(3) + axis_angle(3) + gripper_qpos(2)
        eef_pos = raw_obs["robot0_eef_pos"]
        eef_quat = raw_obs["robot0_eef_quat"]
        gripper_qpos = raw_obs["robot0_gripper_qpos"]
        axis_angle = Rotation.from_quat(eef_quat).as_rotvec()
        agent_pos = np.concatenate([eef_pos, axis_angle, gripper_qpos]).astype(np.float64)

        return {"pixels": images, "agent_pos": agent_pos}

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)
        self._elapsed_steps = 0

        idx = None
        if self._init_states is not None:
            idx = int(self.np_random.integers(len(self._init_states)))

        # robosuite's placement samplers / lighting / contact-solver caches use
        # the global np.random under the hood and aren't seeded by gym. Scope a
        # global-RNG window around the reset so we get bit-identical sims for
        # the same gym seed without dirtying numpy's RNG for the rest of the
        # process (dataloaders, augmentations, MPPI noise, etc.).
        np_state = np.random.get_state()
        try:
            if seed is not None:
                np.random.seed(seed)

            # Restore the per-demo XML model when available. Source demos randomize
            # object placements per demo; without restoring the matching XML the
            # set_state below loads qpos into a slightly different model and the
            # resulting eef pose / rendered scene drifts from the recorded demo.
            if (
                idx is not None
                and self._model_files is not None
                and idx < len(self._model_files)
                and self._model_files[idx]
            ):
                self._env.reset()
                xml = self._model_files[idx]
                if hasattr(self._env, "edit_model_xml"):
                    xml = self._env.edit_model_xml(xml)
                else:
                    from robosuite.utils.mjcf_utils import postprocess_model_xml
                    xml = postprocess_model_xml(xml)
                self._env.reset_from_xml_string(xml)
                self._env.sim.reset()
            else:
                self._env.reset()

            if idx is not None:
                self._env.sim.set_state_from_flattened(self._init_states[idx].numpy())
                self._env.sim.forward()

            raw_obs = self._env._get_observations()

            # Let the simulation settle.
            for _ in range(self.num_steps_wait):
                raw_obs, _, _, _ = self._env.step(get_mimicgen_dummy_action())
        finally:
            np.random.set_state(np_state)

        # robosuite's reset overwrites the offscreen render context's vopt;
        # re-apply our visual-geom-only setting so eval renders look correct.
        self._fix_render_vopt()

        observation = self._format_raw_obs(raw_obs)
        info = {"is_success": False, "init_state_idx": idx}
        return observation, info

    def step(self, action: np.ndarray) -> tuple[RobotObservation, float, bool, bool, dict[str, Any]]:
        if action.ndim != 1:
            raise ValueError(
                f"Expected 1-D action (shape ({ACTION_DIM},)), got shape {action.shape}"
            )
        raw_obs, reward, done, info = self._env.step(action)
        self._elapsed_steps += 1

        # robosuite 1.4.x uses _check_success() (bool), newer versions use is_success() (dict)
        if hasattr(self._env, "is_success"):
            success_dict = self._env.is_success()
            is_success = success_dict.get("task", False) if isinstance(success_dict, dict) else bool(success_dict)
        else:
            is_success = self._env._check_success()
        terminated = done or is_success
        truncated = self._elapsed_steps >= self._max_episode_steps and not terminated

        info.update({
            "task": self.task,
            "done": done,
            "is_success": is_success,
        })

        observation = self._format_raw_obs(raw_obs)

        if terminated or truncated:
            info["final_info"] = {
                "task": self.task,
                "done": bool(done),
                "is_success": bool(is_success),
            }

        return observation, reward, terminated, truncated, info

    def render(self):
        img = self._env.sim.render(
            height=self.render_height,
            width=self.render_width,
            camera_name=self.camera_name[0],
        )
        return img[::-1]  # robosuite renders upside-down

    def close(self):
        self._env.close()


# ---------------------------------------------------------------------------
# Vectorized env factory
# ---------------------------------------------------------------------------


def _make_env_fns(
    *,
    env_name: str,
    n_envs: int,
    camera_names: list[str],
    init_states_path: str | Path | None,
    gym_kwargs: Mapping[str, Any],
    max_episode_steps: int,
) -> list[Callable[[], MimicGenEnv]]:
    """Build n_envs factory callables for a single MimicGen task."""

    def _make_env(**kwargs) -> MimicGenEnv:
        return MimicGenEnv(
            env_name=env_name,
            camera_name=camera_names,
            init_states_path=init_states_path,
            max_episode_steps=max_episode_steps,
            **kwargs,
        )

    return [partial(_make_env, **gym_kwargs) for _ in range(n_envs)]


def create_mimicgen_envs(
    task: str,
    n_envs: int,
    gym_kwargs: dict[str, Any] | None = None,
    camera_name: str | Sequence[str] = "agentview,robot0_eye_in_hand",
    init_states_path: str | Path | None = None,
    env_cls: Callable[[Sequence[Callable[[], Any]]], Any] | None = None,
    episode_length: int | None = None,
) -> dict[str, dict[int, Any]]:
    """Create vectorized MimicGen environments.

    Args:
        task: Comma-separated list of MimicGen task names (e.g. "Coffee_D0,Stack_D0").
        n_envs: Number of parallel envs per task.
        gym_kwargs: Extra kwargs forwarded to MimicGenEnv constructor.
        camera_name: Camera name(s) for observations.
        init_states_path: Path to init_states.pt for deterministic eval.
        env_cls: Vectorized env class (e.g. SyncVectorEnv).
        episode_length: Max steps per episode (overrides default).

    Returns:
        {task_name: {0: vec_env}} mapping.
    """
    if env_cls is None or not callable(env_cls):
        raise ValueError("env_cls must be a callable.")
    if not isinstance(n_envs, int) or n_envs <= 0:
        raise ValueError(f"n_envs must be positive; got {n_envs}.")

    gym_kwargs = dict(gym_kwargs or {})
    camera_names = _parse_camera_names(camera_name)
    task_names = [t.strip() for t in str(task).split(",") if t.strip()]
    if not task_names:
        raise ValueError("`task` must contain at least one MimicGen task name.")

    max_steps = episode_length if episode_length is not None else DEFAULT_MAX_EPISODE_STEPS

    print(f"Creating MimicGen envs | tasks={task_names} | n_envs(per task)={n_envs}")

    out: dict[str, dict[int, Any]] = defaultdict(dict)
    for tname in task_names:
        fns = _make_env_fns(
            env_name=tname,
            n_envs=n_envs,
            camera_names=camera_names,
            init_states_path=init_states_path,
            gym_kwargs=gym_kwargs,
            max_episode_steps=max_steps,
        )
        out[tname][0] = env_cls(fns)
        print(f"  Built vec env | task={tname} | n_envs={n_envs}")

    return {t: dict(m) for t, m in out.items()}
