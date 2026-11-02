"""Measure the effective coefficient of restitution of a 2.7 g, 40 mm ball
dropped on a fixed 10 mm-thick box ("paddle") in MuJoCo, for several solref
settings, timesteps and impact speeds. CPU, seconds.

COR = v_up(after separation) / v_down(just before first contact).
"""
import sys
import numpy as np
import mujoco

G = 9.81
R = 0.02
M = 0.0027

XML = """
<mujoco>
  <option timestep="{dt}" integrator="{integ}" gravity="0 0 -9.81" {extra_opt}/>
  <worldbody>
    <geom name="paddle" type="box" size="0.08 0.075 0.005" pos="0 0 0" {pad_sol}/>
    <body name="ball" pos="0 0 {z0}">
      <freejoint/>
      <geom name="ball" type="sphere" size="{R}" mass="{M}" {ball_sol} {ball_extra}/>
    </body>
  </worldbody>
</mujoco>
"""


def run(dt, solref, v_in, integ="implicitfast", solimp=None, extra_opt="",
        ball_extra="", n_bounces=1):
    h = v_in ** 2 / (2 * G)
    z0 = 0.005 + R + h
    sol = f'solref="{solref[0]} {solref[1]}"'
    if solimp is not None:
        sol += ' solimp="' + " ".join(map(str, solimp)) + '"'
    xml = XML.format(dt=dt, integ=integ, extra_opt=extra_opt, pad_sol="",
                     z0=z0, R=R, M=M, ball_sol=sol, ball_extra=ball_extra)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    v_before = None
    in_contact = False
    cors, pens, durs = [], [], []
    t_contact = 0
    max_pen = 0.0
    prev_vz = 0.0
    for _ in range(int(5.0 / dt)):
        mujoco.mj_step(m, d)
        vz = d.qvel[2]
        pen = 0.005 + R - d.qpos[2]
        if d.ncon > 0 and not in_contact:
            in_contact = True
            v_before = prev_vz
            t_contact = 0
            max_pen = 0.0
        if in_contact:
            t_contact += 1
            max_pen = max(max_pen, pen)
            if d.ncon == 0:
                in_contact = False
                cors.append(-vz / v_before)
                pens.append(max_pen)
                durs.append(t_contact * dt)
                if len(cors) >= n_bounces:
                    break
        prev_vz = vz
    if not cors:
        return None
    return np.mean(cors), np.max(pens), np.mean(durs), cors


def cor_to_zeta(e):
    le = np.log(e)
    return -le / np.sqrt(np.pi ** 2 + le ** 2)


if __name__ == "__main__":
    print("mujoco", mujoco.__version__)
    e_target = 0.85
    z = cor_to_zeta(e_target)
    print(f"target e={e_target}: damping ratio zeta={z:.4f}")

    print("\n== A. positive solref (timeconst, dampratio=zeta) ==")
    print("dt      tc      v_in  COR    maxpen_mm  contact_ms")
    for dt in [0.0005, 0.001, 0.002, 0.005]:
        for tc in [0.002, 0.005, 0.01, 0.02]:
            for v in [1.0, 2.0, 3.0]:
                r = run(dt, (tc, round(z, 4)), v)
                if r is None:
                    print(f"{dt:<7} {tc:<7} {v:<5} no-bounce"); continue
                print(f"{dt:<7} {tc:<7} {v:<5} {r[0]:.3f}  {r[1]*1e3:8.2f}  {r[2]*1e3:8.2f}")

    print("\n== B. direct solref (-k, -b), b = 2*zeta*sqrt(k) ==")
    print("dt      k        b       v_in  COR    maxpen_mm  contact_ms")
    for dt in [0.0005, 0.001, 0.002, 0.005]:
        for k in [1e4, 1e5, 1e6]:
            b = 2 * z * np.sqrt(k)
            for v in [1.0, 2.0, 3.0]:
                r = run(dt, (-k, -round(b, 2)), v)
                if r is None:
                    print(f"{dt:<7} {k:<8.0e} {b:<7.2f} {v:<5} no-bounce"); continue
                print(f"{dt:<7} {k:<8.0e} {b:<7.2f} {v:<5} {r[0]:.3f}  {r[1]*1e3:8.2f}  {r[2]*1e3:8.2f}")

    print("\n== C. defaults: solref 0.02 1 (Menagerie-style) ==")
    for dt in [0.002, 0.005]:
        r = run(dt, (0.02, 1), 2.0)
        print(dt, "no-bounce" if r is None else f"COR={r[0]:.3f} pen={r[1]*1e3:.2f}mm")

    print("\n== D. solimp sensitivity (direct k=1e5, dt=0.001, v=2) ==")
    k = 1e5; b = round(2 * z * np.sqrt(k), 2)
    for si in [(0.9, 0.95, 0.001, 0.5, 2), (0.99, 0.99, 0.001, 0.5, 2),
               (0.0, 0.95, 0.001, 0.5, 2), (0.95, 0.99, 0.002, 0.5, 2)]:
        r = run(0.001, (-k, -b), 2.0, solimp=si)
        print(si, "no-bounce" if r is None else f"COR={r[0]:.3f} pen={r[1]*1e3:.2f}mm dur={r[2]*1e3:.2f}ms")

    print("\n== E. repeated bounces, energy drift (direct k=1e5, e=0.85 vs e=1 b=0) ==")
    for bb in [b, 0.0]:
        r = run(0.001, (-k, -bb), 2.0, n_bounces=8)
        print(f"b={bb}: CORs", np.round(r[3], 3))

    print("\n== F. Euler vs implicitfast vs RK4 (direct k=1e5, dt=0.001, v=2) ==")
    for integ in ["Euler", "implicitfast", "RK4"]:
        r = run(0.001, (-k, -b), 2.0, integ=integ)
        print(integ, f"COR={r[0]:.3f}")
