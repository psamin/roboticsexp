"""Third round (CPU, seconds):
 A. moving paddle (mocap, +2 m/s up) vs falling ball (-3 m/s): COR of relative
    velocity and tunneling, vs dt and contact stiffness.
 B. fluidcoef blunt-drag value needed for Cd~0.5 (terminal velocity check).
 C. spin transfer: ball with 0.5 m/s horizontal velocity hits a fixed paddle;
    shell vs solid inertia; condim 3 vs 6; friction 0.2 vs 1.0. Compare with
    the rigid-body rolling prediction v_t' = v_t * (m r^2/I)/(1 + m r^2/I).
"""
import numpy as np
import mujoco

G = 9.81; R = 0.02; M = 0.0027
Z = 0.0517  # damping ratio for e = 0.85


def xml(dt, ball, pad="", opt="", mocap=False, z0=0.1):
    padxml = (f'<body name="pad" mocap="true" pos="0 0 0"><geom type="box" size="0.08 0.075 0.005" {pad}/></body>'
              if mocap else f'<geom name="pad" type="box" size="0.08 0.075 0.005" {pad}/>')
    return f"""
<mujoco><option timestep="{dt}" integrator="implicitfast" {opt}/>
<worldbody>{padxml}
<body name="ball" pos="0 0 {z0}"><freejoint/>
<geom type="sphere" size="{R}" mass="{M}" {ball}/></body></worldbody></mujoco>"""


print("== A. moving paddle +2 m/s, ball -3 m/s (relative 5 m/s), e_target 0.85 ==")
print("expected ball v_out = 0.85*5 + 2 = 6.25 m/s")
for dt in [0.0005, 0.001, 0.002, 0.005]:
    out = []
    for k in [2.5e4, 1e5, 4e5]:
        b = 2 * Z * np.sqrt(k)
        m = mujoco.MjModel.from_xml_string(xml(dt, f'solref="{-k} {-b:.2f}"', mocap=True, z0=0.005 + R + 0.0001 + 3 * 0.0))
        d = mujoco.MjData(m)
        # start ball just above paddle, separated by one step of closing
        d.qpos[2] = 0.005 + R + 5.0 * dt * 0.99
        d.qvel[2] = -3.0
        vpad = 2.0
        sep = None
        for i in range(int(0.1 / dt)):
            d.mocap_pos[0, 2] += vpad * dt
            mujoco.mj_step(m, d)
            gap = d.qpos[2] - d.mocap_pos[0, 2] - 0.005 - R
            if i > 2 and d.ncon == 0 and gap > 0:
                sep = d.qvel[2]; break
        tunneled = d.qpos[2] < d.mocap_pos[0, 2]
        out.append(f"k={k:.0e}: " + ("TUNNEL" if tunneled else f"v_out={sep:.2f}" if sep is not None else "stuck"))
    print(f"dt={dt}: " + " | ".join(out))

print("\n== B. fluid drag calibration (ellipsoid model) ==")
for fc in ["0.5 0.25 1.5 1.0 1.0", "0.25 0.25 1.5 1.0 1.0", "0.25 0.25 0 1.0 1.0"]:
    m = mujoco.MjModel.from_xml_string(xml(0.001, f'fluidshape="ellipsoid" fluidcoef="{fc}"',
                                           opt='density="1.204" viscosity="1.8e-5"', z0=2000))
    d = mujoco.MjData(m)
    for _ in range(30000):
        mujoco.mj_step(m, d)
    vt = -d.qvel[2]
    print(f"fluidcoef={fc}: v_terminal={vt:.2f} m/s, implied Cd={2*M*G/(1.204*np.pi*R**2*vt**2):.2f}")

print("\n== C. spin transfer on a fixed paddle (vz_in=-2.4 m/s, vx_in=0.5 m/s) ==")
for inert in ["solid", "shell"]:
    for condim in [3, 6]:
        for mu in [0.2, 1.0]:
            sh = 'shellinertia="true"' if inert == "shell" else ""
            k = 1e5; b = 2 * Z * np.sqrt(k)
            fr = f'friction="{mu} 0.005 0.0001"'
            ballg = f'solref="{-k} {-b:.2f}" condim="{condim}" {fr} {sh}'
            m = mujoco.MjModel.from_xml_string(xml(0.0005, ballg, pad=f'condim="{condim}" {fr}'))
            d = mujoco.MjData(m)
            d.qpos[2] = 0.005 + R + 0.001; d.qvel[2] = -2.4; d.qvel[0] = 0.5
            touched = False
            for _ in range(400):
                mujoco.mj_step(m, d)
                if d.ncon > 0: touched = True
                if touched and d.ncon == 0: break
            I = m.body_inertia[1][0]
            q = M * R ** 2 / I
            pred = 0.5 * q / (1 + q)
            print(f"{inert:5s} I/mr2={I/(M*R*R):.3f} condim={condim} mu={mu}: vx_out={d.qvel[0]:.3f} "
                  f"(rolling pred {pred:.3f}), spin wy={d.qvel[4]:.1f} rad/s, vz_out={d.qvel[2]:.2f}")
