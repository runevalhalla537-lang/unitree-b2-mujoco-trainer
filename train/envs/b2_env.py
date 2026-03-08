from __future__ import annotations

import os
from dataclasses import dataclass

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R


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
    smooth_penalty_weight: float = 0.002
    orient_penalty_weight: float = 1.0
    alive_bonus: float = 0.5
    stand_pose_penalty_weight: float = 2.0


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
        self.prev_action = np.zeros(self.model.nu, dtype=np.float64)

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

        # Map each actuator to its joint qpos index; used for stand-pose reward shaping.
        self.actuator_joint_qpos_idx = []
        for i in range(self.nu):
            jid = int(self.model.actuator_trnid[i, 0])
            qadr = int(self.model.jnt_qposadr[jid])
            self.actuator_joint_qpos_idx.append(qadr)
        self.actuator_joint_qpos_idx = np.asarray(self.actuator_joint_qpos_idx, dtype=np.int32)

        self.home_joint_qpos = np.zeros(self.nu, dtype=np.float64)
        if self.home_key >= 0 and self.nu > 0:
            home_q = self.model.key_qpos[self.home_key]
            self.home_joint_qpos = home_q[self.actuator_joint_qpos_idx].copy()

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
        self.prev_action[:] = 0
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

        # reward: track forward speed, keep body up+upright, penalize large/sudden controls
        vx = float(self.data.qvel[0]) if self.nv > 0 else 0.0
        vel_reward = 1.0 - abs(vx - self.cfg.target_lin_vel)

        base_height = float(self.data.qpos[2]) if self.nq > 2 else 0.3
        alive = self.cfg.alive_bonus if base_height > self.cfg.fall_height_threshold else -1.0

        # quaternion in qpos[3:7] for free base (w,x,y,z)
        roll = pitch = 0.0
        if self.nq >= 7:
            q = self.data.qpos[3:7]
            quat_xyzw = np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)
            roll, pitch, _ = R.from_quat(quat_xyzw).as_euler("xyz", degrees=False)
        orient_penalty = self.cfg.orient_penalty_weight * (abs(roll) + abs(pitch))

        ctrl_penalty = self.cfg.ctrl_penalty_weight * float(np.sum(np.square(action)))
        smooth_penalty = self.cfg.smooth_penalty_weight * float(np.sum(np.square(action - self.prev_action)))

        stand_pose_penalty = 0.0
        if self.nu > 0:
            joint_q = self.data.qpos[self.actuator_joint_qpos_idx]
            stand_pose_penalty = self.cfg.stand_pose_penalty_weight * float(np.mean(np.abs(joint_q - self.home_joint_qpos)))

        self.prev_action = action.copy()

        reward = vel_reward + alive - orient_penalty - ctrl_penalty - smooth_penalty - stand_pose_penalty

        terminated = base_height < self.cfg.fall_height_threshold
        truncated = self._step_count >= self.cfg.max_episode_steps

        info = {
            "vx": vx,
            "base_height": base_height,
            "vel_reward": vel_reward,
            "alive": alive,
            "orient_penalty": orient_penalty,
            "ctrl_penalty": ctrl_penalty,
            "smooth_penalty": smooth_penalty,
            "stand_pose_penalty": stand_pose_penalty,
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        return None

    def close(self):
        return None
