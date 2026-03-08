# B2 Profile (Unitree MuJoCo Alignment)

This project aligns with Unitree's MuJoCo B2 model from:
- https://github.com/unitreerobotics/unitree_mujoco/tree/main/unitree_robots/b2

## Joint layout (from upstream b2.xml)

Actuated joints (12):
1. FR_hip_joint
2. FR_thigh_joint
3. FR_calf_joint
4. FL_hip_joint
5. FL_thigh_joint
6. FL_calf_joint
7. RR_hip_joint
8. RR_thigh_joint
9. RR_calf_joint
10. RL_hip_joint
11. RL_thigh_joint
12. RL_calf_joint

Free base joint exists (`floating_base_joint`) and is not directly actuated.

## Control assumptions

- Uses MuJoCo `motor` actuators with `ctrlrange` from upstream XML.
- Env actions are normalized `[-1, 1]` and scaled into actuator ranges.
- `action_scale` is a safety fraction of actuator range (default `0.35`).

## Reset behavior

- If keyframe `home` exists, reset uses `mj_resetDataKeyframe(..., home)`.
- Otherwise, falls back to `mj_resetData`.
- Small qpos noise is applied after reset for robustness.

## Current training objective (baseline)

- Forward velocity tracking (`target_lin_vel`)
- Upright/stability term from base height
- Control-effort penalty

This is a starter objective. Add gait symmetry, foot slip penalties, and energy terms for production training.

## Recommended next upgrades

1. Add privileged observations for teacher policy.
2. Add terrain curriculum and randomization.
3. Add action latency/noise for sim2real.
4. Add contact-quality rewards (foot clearance/slip).
