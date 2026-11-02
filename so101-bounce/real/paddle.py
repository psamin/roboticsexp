"""Clamp the paddle in the gripper (or let it go). Only the gripper moves; the arm joints are not touched.

  uv run python real/paddle.py grip      # opens the jaws; put the handle in (blade face up when the arm points
                                         # forward); press Enter; the jaws clamp and keep clamping after exit
  uv run python real/paddle.py release   # opens the jaws so you can take the paddle out

Every other real/ script keeps the gripper clamped and only turns the arm joints off at the end.
"""
import sys
import time

import arm as A


def main(cmd):
    if not A.USE_GRIPPER:
        raise SystemExit("use_gripper is false in real/config.yaml: the paddle is fixed in the jaws, nothing to do")
    b = A.open_bus()
    try:
        b.sync_write("Goal_Position", {"gripper": A.read_pos(b)["gripper"]})
        b.enable_torque("gripper")
        b.sync_write("Goal_Position", {"gripper": A.GRIP_OPEN})
        time.sleep(1.0)
        if cmd == "grip":
            input("Jaws open. Put the paddle handle between them (flat sides of the handle against the jaws), hold it "
                  "there, press Enter...")
            contact, goal = A.clamp_paddle(b)
            time.sleep(0.5)
            print(f"clamped: jaws touched the handle at {contact:.0f}, holding at {goal:.0f} with limited force "
                  f"(0 = fully closed). Tug the paddle gently: it should not move.")
        else:
            print("jaws open; take the paddle out")
    finally:
        b.disconnect(disable_torque=False)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "grip")
