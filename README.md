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
Place your MuJoCo XML at:

`assets/b2/b2.xml`

Update paths inside XML for meshes/textures as needed.

## 2) Spark setup (training)

```bash
cd unitree-b2-mujoco-trainer/train
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_ppo.py --config configs/base.yaml
```

Checkpoints save to `train/runs/<timestamp>/`.

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
