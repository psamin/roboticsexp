"""Install probe: boots Isaac Lab headless, prints the APIs this project relies on, and test-drives the Jackal."""
from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

import faulthandler
import math
import os
import sys
import traceback

faulthandler.dump_traceback_later(240, repeat=True)   # print where we are if anything hangs

import isaaclab
import isaaclab.sim as sim_utils
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sensors import ContactSensorCfg, ContactSensorData
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from pxr import Usd, UsdPhysics

def main():
    print("isaaclab", getattr(isaaclab, "__version__", "?"), "torch", torch.__version__, torch.cuda.get_device_name(0))
    print("ISAAC_NUCLEUS_DIR", ISAAC_NUCLEUS_DIR)
    print("ContactSensorData fields:", [f for f in dir(ContactSensorData) if not f.startswith("_")])
    print("ContactSensorCfg fields:", [f for f in dir(ContactSensorCfg) if not f.startswith("_")])

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120, device="cuda:0"))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    usd = f"{ISAAC_NUCLEUS_DIR}/Robots/Clearpath/Jackal/jackal.usd"
    cfg = ArticulationCfg(
        prim_path="/World/Jackal",
        spawn=sim_utils.UsdFileCfg(usd_path=usd),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.05)),
        actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0.0, damping=50.0)},
    )
    robot = Articulation(cfg)

    import omni.usd
    stage = omni.usd.get_context().get_stage()
    for p in Usd.PrimRange(stage.GetPrimAtPath("/World/Jackal")):
        apis = [a for a in p.GetAppliedSchemas() if "Physics" in a or "Physx" in a]
        if p.IsA(UsdPhysics.Joint) or apis:
            extra = ""
            if p.IsA(UsdPhysics.RevoluteJoint):
                j = UsdPhysics.RevoluteJoint(p)
                extra = f" axis={j.GetAxisAttr().Get()} b0={j.GetBody0Rel().GetTargets()} b1={j.GetBody1Rel().GetTargets()}"
            print(f"  {p.GetPath()} [{p.GetTypeName()}] {apis}{extra}")

    sim.reset()
    print("joints:", robot.joint_names)
    print("bodies:", robot.body_names)
    print("masses:", [round(float(m), 3) for m in robot.root_physx_view.get_masses()[0]])
    print("total mass:", float(robot.root_physx_view.get_masses()[0].sum()))
    print("default joint damping/stiffness:", robot.data.joint_damping[0].tolist(), robot.data.joint_stiffness[0].tolist())
    print("effort limits:", robot.data.joint_effort_limits[0].tolist() if hasattr(robot.data, "joint_effort_limits") else "?")
    print("vel limits:", robot.data.joint_vel_limits[0].tolist() if hasattr(robot.data, "joint_vel_limits") else "?")
    for _ in range(60):
        robot.write_data_to_sim(); sim.step(); robot.update(sim.get_physics_dt())
    print("settled root pos:", robot.data.root_pos_w[0].tolist(), "quat:", robot.data.root_quat_w[0].tolist())

    wheel_ids = [i for i, n in enumerate(robot.joint_names) if "wheel" in n]
    for target_w in (10.0, 20.0, -20.0):
        p0 = robot.data.root_pos_w[0].clone()
        tgt = torch.zeros_like(robot.data.joint_vel)
        tgt[:, wheel_ids] = target_w
        robot.set_joint_velocity_target(tgt)
        for k in range(120):
            robot.write_data_to_sim(); sim.step(); robot.update(sim.get_physics_dt())
            if k in (11, 29, 59, 119):
                v = robot.data.root_lin_vel_b[0].tolist()
                print(f"  w={target_w} t={(k+1)/120:.2f}s base-frame lin vel={[round(x,3) for x in v]} wheel vel={[round(float(x),1) for x in robot.data.joint_vel[0, wheel_ids]]}")
        d = robot.data.root_pos_w[0] - p0
        q = robot.data.root_quat_w[0]
        yaw = math.degrees(math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))
        print(f"  w={target_w}: moved {d.tolist()} in 1 s, yaw {yaw:.2f} deg")
    print("PROBE_OK")


if __name__ == "__main__":
    code = 0
    try:
        main()
    except Exception:
        traceback.print_exc()
        code = 1
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(code)   # Kit can hang in shutdown; exit hard once results are printed
