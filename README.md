# Unitree B2 MuJoCo Trainer (Windows Sim + Spark Training)

Starter scaffold for:
- **Training** on Spark (headless MuJoCo + PPO)
- **Simulation / playback** on Windows (visual eval)

## Architecture

- `train/` runs on Spark (Linux)
- `sim/` runs on Windows
- `assets/` stores B2 MJCF/meshes (not committed if proprietary)

## Quick start

## 1) Add B2 model assets

Recommended (auto-fetch from Unitree upstream):

```bash
bash scripts/fetch_unitree_b2_assets.sh
```

This copies `unitree_robots/b2/*` into `assets/b2/`.
Main XML path becomes:

`assets/b2/b2.xml`

See `B2_PROFILE.md` for joint/control alignment assumptions.

## 2) Validate model before training (recommended)

```bash
cd unitree-b2-mujoco-trainer/train
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python ../scripts/validate_b2_model.py --xml ../assets/b2/b2.xml
```

## 3) Spark setup (training)

```bash
cd unitree-b2-mujoco-trainer/train
source .venv/bin/activate
python train_ppo.py --config configs/base.yaml
# or stability-first profile:
python train_ppo.py --config configs/stability.yaml
# or stand-first profile (recommended first stage):
python train_ppo.py --config configs/stand.yaml
```

Suggested curriculum:
1. `stage_a_stand.yaml` (learn to stand/upright with smooth actions)
2. `stage_b_shift.yaml` (tiny forward drift + weight shifting)
3. `stability.yaml` (slow controlled motion)
4. `base.yaml` (faster tracking)

Example:
```bash
python train_ppo.py --config configs/stage_a_stand.yaml
```

Final model saves to `train/runs/<timestamp>/policy.zip`.
Interim checkpoints save every ~20k steps to `train/runs/<timestamp>/checkpoints/`.

## 3) Windows setup (visual eval)

```powershell
cd sim
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python windows_eval.py --model ..\train\runs\latest\policy.zip --xml ..\assets\b2\b2.xml
```

## Notes

- This scaffold uses **placeholder B2 dynamics assumptions** until exact joint map and limits are confirmed from Unitree docs/assets.
- Start with standing/velocity tracking before rough terrain.
- Add domain randomization early for sim2real stability.

## Next steps

1. Validate B2 joint names/order in MJCF
2. Tune action scaling + reward weights
3. Add curriculum (`flat -> mild uneven -> rough`)
4. Add deployment adapter for Unitree runtime control loop
