from __future__ import annotations

import argparse
import time

import mujoco.viewer
import mujoco
from stable_baselines3 import PPO

import sys
from pathlib import Path

# allow importing env from train/
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "train"))

from envs.b2_env import B2EnvConfig, B2MuJoCoEnv  # noqa: E402


def force_stand_pose(env: B2MuJoCoEnv):
    """Reset env state into an explicit standing-like pose for gait evaluation."""
    mujoco.mj_resetData(env.model, env.data)

    # base pose: x,y,z + unit quaternion
    if env.model.nq >= 7:
        if getattr(env, "home_base_qpos", None) is not None:
            env.data.qpos[:7] = env.home_base_qpos
        else:
            env.data.qpos[:7] = [0.0, 0.0, 0.33, 1.0, 0.0, 0.0, 0.0]

    # joint pose from home profile when available
    if env.nu > 0 and hasattr(env, "actuator_joint_qpos_idx") and hasattr(env, "home_joint_qpos"):
        env.data.qpos[env.actuator_joint_qpos_idx] = env.home_joint_qpos
        env.data.ctrl[:] = env.ctrl_mid

    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to policy.zip")
    ap.add_argument("--xml", required=True, help="Path to B2 XML")
    ap.add_argument("--steps", type=int, default=3000)
    args = ap.parse_args()

    cfg = B2EnvConfig(xml_path=args.xml)
    env = B2MuJoCoEnv(cfg, render_mode="human")

    model = PPO.load(args.model)

    obs, _ = env.reset()
    force_stand_pose(env)
    obs = env._get_obs()

    # Launch native MuJoCo viewer for visible playback on Windows.
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        for _ in range(args.steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            viewer.sync()
            if terminated or truncated:
                obs, _ = env.reset()
            time.sleep(0.01)

    env.close()


if __name__ == "__main__":
    main()
