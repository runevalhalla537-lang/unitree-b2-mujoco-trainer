from __future__ import annotations

import argparse
import os
import time
from dataclasses import fields

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from envs.b2_env import B2EnvConfig, B2MuJoCoEnv


def load_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    env_raw = dict(cfg["env"])
    allowed = {f.name for f in fields(B2EnvConfig)}
    env_filtered = {k: v for k, v in env_raw.items() if k in allowed}
    env_cfg = B2EnvConfig(**env_filtered, xml_path=cfg["xml_path"])

    env = Monitor(B2MuJoCoEnv(env_cfg))

    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(os.path.dirname(__file__), "runs", ts)
    os.makedirs(run_dir, exist_ok=True)

    ppo_cfg = cfg["train"]
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=os.path.join(run_dir, "tb"),
        n_steps=ppo_cfg["n_steps"],
        batch_size=ppo_cfg["batch_size"],
        learning_rate=ppo_cfg["learning_rate"],
        gamma=ppo_cfg["gamma"],
        gae_lambda=ppo_cfg["gae_lambda"],
        clip_range=ppo_cfg["clip_range"],
        ent_coef=ppo_cfg["ent_coef"],
        vf_coef=ppo_cfg["vf_coef"],
    )

    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint_cb = CheckpointCallback(
        save_freq=20_000,
        save_path=ckpt_dir,
        name_prefix="policy_step",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    model.learn(total_timesteps=ppo_cfg["total_timesteps"], progress_bar=False, callback=checkpoint_cb)

    out_model = os.path.join(run_dir, "policy.zip")
    model.save(out_model)

    latest_link = os.path.join(os.path.dirname(__file__), "runs", "latest")
    if os.path.islink(latest_link) or os.path.exists(latest_link):
        try:
            os.remove(latest_link)
        except OSError:
            pass
    os.symlink(run_dir, latest_link)

    print(f"Saved model: {out_model}")


if __name__ == "__main__":
    main()
