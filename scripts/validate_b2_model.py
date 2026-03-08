#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys

import mujoco

EXPECTED_ACTUATED_JOINTS = [
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True, help="Path to B2 MuJoCo XML (e.g. assets/b2/b2.xml)")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)

    # Collect motor->joint mapping in model order
    actuated = []
    for i in range(model.nu):
        trnid = model.actuator_trnid[i, 0]
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(trnid))
        aname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        actuated.append((aname, jname))

    found_joints = [j for _, j in actuated]

    ok = True
    if model.nu != 12:
        print(f"FAIL: expected 12 actuators, found {model.nu}")
        ok = False

    if found_joints != EXPECTED_ACTUATED_JOINTS:
        print("FAIL: actuated joint order mismatch")
        print("Expected:")
        for x in EXPECTED_ACTUATED_JOINTS:
            print("  -", x)
        print("Found:")
        for x in found_joints:
            print("  -", x)
        ok = False

    # Keyframe presence
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_id < 0:
        print("WARN: no 'home' keyframe found; reset will use default mj_resetData")

    # Print summary
    print("Model summary:")
    print(f"  nq={model.nq}, nv={model.nv}, nu={model.nu}")
    print("  actuators:")
    for aname, jname in actuated:
        print(f"    - {aname} -> {jname}")

    if ok:
        print("PASS: B2 model structure matches expected training profile")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
