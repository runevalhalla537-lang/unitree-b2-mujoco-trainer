from __future__ import annotations

import os
from dataclasses import dataclass

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


@dataclass
class B2EnvConfig:
    xml_path: str
    target_lin_vel: float = 0.5
    max_episode_steps: int = 2000
    frame_skip: int = 4
    action_scale: float = 0.3
    fall_height_threshold: float = 0.18


class B2MuJoCoEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, cfg: B2EnvConfig, render_mode: str | None = None):
        super().__init__()
        if not os.path.exists(cfg.xml_path):
            raise FileNotFoundError(f"B2 xml not found: {cfg.xml_path}")

        self.cfg = cfg
        self.model = mujoco.MjModel.from_xml_path(cfg.xml_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self._step_count = 0

        self.nq = self.model.nq
        self.nv = self.model.nv
        self.nu = self.model.nu

        obs_dim = self.nq + self.nv + 3  # qpos + qvel + base lin vel xyz
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.nu,), dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        base_lin = qvel[:3] if len(qvel) >= 3 else np.zeros(3, dtype=np.float64)
        return np.concatenate([qpos, qvel, base_lin]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self._step_count = 0

        # small randomization around default pose
        self.data.qpos[:] += np.random.normal(0, 0.002, size=self.data.qpos.shape)
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

        return self._get_obs(), {}

    def step(self, action):
        self._step_count += 1
        action = np.asarray(action, dtype=np.float64)
        action = np.clip(action, -1.0, 1.0)

        # joint-delta style control (placeholder; tune with real B2 actuator setup)
        if self.nu > 0:
            self.data.ctrl[:] = action * self.cfg.action_scale

        for _ in range(self.cfg.frame_skip):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()

        # reward: forward velocity tracking + posture stability + ctrl penalty
        vx = float(self.data.qvel[0]) if self.nv > 0 else 0.0
        vel_reward = 1.0 - abs(vx - self.cfg.target_lin_vel)

        base_height = float(self.data.qpos[2]) if self.nq > 2 else 0.3
        upright_bonus = 0.5 if base_height > self.cfg.fall_height_threshold else -1.0

        ctrl_penalty = 0.001 * float(np.sum(np.square(action)))

        reward = vel_reward + upright_bonus - ctrl_penalty

        terminated = base_height < self.cfg.fall_height_threshold
        truncated = self._step_count >= self.cfg.max_episode_steps

        info = {
            "vx": vx,
            "base_height": base_height,
            "vel_reward": vel_reward,
            "upright_bonus": upright_bonus,
            "ctrl_penalty": ctrl_penalty,
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            return None
        return None

    def close(self):
        return None
