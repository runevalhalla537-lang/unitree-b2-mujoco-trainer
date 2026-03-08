from __future__ import annotations

import argparse
import time

from stable_baselines3 import PPO

import sys
from pathlib import Path

# allow importing env from train/
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "train"))

from envs.b2_env import B2EnvConfig, B2MuJoCoEnv  # noqa: E402


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
    for _ in range(args.steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
        time.sleep(0.01)

    env.close()


if __name__ == "__main__":
    main()
