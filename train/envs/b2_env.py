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
    # fraction of ctrl range used, 0..1
    action_scale: float = 0.35
    fall_height_threshold: float = 0.18
    reset_noise_qpos: float = 0.002
    ctrl_penalty_weight: float = 0.002


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

        # Capture actuator control ranges from XML (B2 has 12 motors)
        if self.nu > 0:
            self.ctrl_min = self.model.actuator_ctrlrange[:, 0].copy()
            self.ctrl_max = self.model.actuator_ctrlrange[:, 1].copy()
            self.ctrl_mid = 0.5 * (self.ctrl_min + self.ctrl_max)
            self.ctrl_half = 0.5 * (self.ctrl_max - self.ctrl_min)
        else:
            self.ctrl_min = self.ctrl_max = self.ctrl_mid = self.ctrl_half = np.array([], dtype=np.float64)

        # Optional home keyframe support (present in Unitree b2.xml)
        self.home_key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")

        obs_dim = self.nq + self.nv + 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.nu,), dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        base_lin = qvel[:3] if len(qvel) >= 3 else np.zeros(3, dtype=np.float64)
        return np.concatenate([qpos, qvel, base_lin]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0

        if self.home_key >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_key)
        else:
            mujoco.mj_resetData(self.model, self.data)

        # small randomization around default pose for robustness
        if self.cfg.reset_noise_qpos > 0:
            self.data.qpos[:] += np.random.normal(0, self.cfg.reset_noise_qpos, size=self.data.qpos.shape)
        self.data.qvel[:] = 0
        if self.nu > 0:
            self.data.ctrl[:] = self.ctrl_mid
        mujoco.mj_forward(self.model, self.data)

        return self._get_obs(), {}

    def step(self, action):
        self._step_count += 1
        action = np.asarray(action, dtype=np.float64)
        action = np.clip(action, -1.0, 1.0)

        # torque/motor command in actuator ctrlrange
        if self.nu > 0:
            scaled = self.ctrl_mid + action * (self.ctrl_half * self.cfg.action_scale)
            self.data.ctrl[:] = np.clip(scaled, self.ctrl_min, self.ctrl_max)

        for _ in range(self.cfg.frame_skip):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()

        # reward: track forward speed, keep body up, penalize large controls
        vx = float(self.data.qvel[0]) if self.nv > 0 else 0.0
        vel_reward = 1.0 - abs(vx - self.cfg.target_lin_vel)

        base_height = float(self.data.qpos[2]) if self.nq > 2 else 0.3
        upright_bonus = 0.5 if base_height > self.cfg.fall_height_threshold else -1.0

        ctrl_penalty = self.cfg.ctrl_penalty_weight * float(np.sum(np.square(action)))
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
        return None

    def close(self):
        return None
