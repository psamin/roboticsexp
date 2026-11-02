"""Follow-up checks:
 A. positive solref set on BOTH geoms (avoids the solmix averaging with the
    paddle's default 0.02 1), COR vs dt.
 B. MJX (JAX, CPU) vs MuJoCo C for the direct-format ball.
 C. MuJoCo fluid model: terminal velocity of a 2.7 g / 40 mm ball and Magnus force.
"""
import numpy as np
import mujoco

G = 9.81; R = 0.02; M = 0.0027


def model(dt, ball_sol, pad_sol="", extra_opt="", ball_extra="", z0=0.3, integ="implicitfast"):
    xml = f"""
<mujoco>
  <option timestep="{dt}" integrator="{integ}" {extra_opt}/>
  <worldbody>
    <geom name="paddle" type="box" size="0.08 0.075 0.005" {pad_sol}/>
    <body name="ball" pos="0 0 {z0}">
      <freejoint/>
      <geom name="ball" type="sphere" size="{R}" mass="{M}" {ball_sol} {ball_extra}/>
    </body>
  </worldbody>
</mujoco>"""
    return mujoco.MjModel.from_xml_string(xml)


def cor_c(m, steps):
    d = mujoco.MjData(m)
    prev, inc, vb = 0.0, False, None
    for _ in range(steps):
        mujoco.mj_step(m, d)
        if d.ncon > 0 and not inc:
            inc, vb = True, prev
        if inc and d.ncon == 0:
            return -d.qvel[2] / vb
        prev = d.qvel[2]


def zeta(e):
    le = np.log(e); return -le / np.sqrt(np.pi ** 2 + le ** 2)


z = zeta(0.85)
print("== A. positive solref on both geoms, e_target=0.85, zeta=%.4f, v_in=2 m/s ==" % z)
h = 2.0 ** 2 / (2 * G); z0 = 0.005 + R + h
for dt in [0.0005, 0.001, 0.002, 0.005]:
    row = []
    for tc in [0.005, 0.01, 0.02, 0.05]:
        s = f'solref="{tc} {z:.4f}"'
        e = cor_c(model(dt, s, s, z0=z0), int(2 / dt))
        row.append(f"tc={tc}: {e:.3f}")
    print(f"dt={dt}: " + "  ".join(row))

print("\n== A2. same, priority=1 on ball instead of setting paddle ==")
for dt in [0.001, 0.002]:
    s = f'solref="0.02 {z:.4f}" priority="1"'
    print(dt, f"{cor_c(model(dt, s, z0=z0), int(2/dt)):.3f}")

print("\n== B. MJX vs C (direct solref -1e5 -32.67, dt=0.001, v_in=2) ==")
try:
    import jax, jax.numpy as jnp
    from mujoco import mjx
    for sol in ['solref="-100000 -32.67"', f'solref="0.02 {z:.4f}"']:
        m = model(0.001, sol, sol, z0=z0)
        ec = cor_c(m, 2000)
        mx = mjx.put_model(m)
        dx = mjx.put_data(m, mujoco.MjData(m))
        def body(dx, _):
            dx = mjx.step(mx, dx)
            return dx, (dx.qvel[2], dx.qpos[2])
        _, (vz, pz) = jax.jit(lambda d: jax.lax.scan(body, d, None, length=600))(dx)
        vz = np.array(vz); pz = np.array(pz)
        contact = pz < (0.005 + R)
        i0 = np.argmax(contact); i1 = i0 + np.argmax(~contact[i0:])
        print(sol, f"C COR={ec:.3f}  MJX COR={-vz[i1]/vz[i0-1]:.3f}  (jax {jax.__version__}, {jax.default_backend()})")
except Exception as ex:
    print("MJX check failed:", repr(ex))

print("\n== C. fluid model: terminal velocity and Magnus ==")
for label, opt, gx in [
        ("inertia-box (density only)", 'density="1.204" viscosity="1.8e-5"', ""),
        ("ellipsoid fluidshape", 'density="1.204" viscosity="1.8e-5"', 'fluidshape="ellipsoid"')]:
    m = model(0.001, "", "", extra_opt=opt, ball_extra=gx, z0=1000)
    d = mujoco.MjData(m)
    for _ in range(20000):
        mujoco.mj_step(m, d)
    vt = -d.qvel[2]
    cd = 2 * M * G / (1.204 * np.pi * R ** 2 * vt ** 2)
    # Magnus: horizontal throw v=5 m/s along x, spin 100 rad/s about y (topspin-ish)
    m2 = model(0.0005, "", "", extra_opt=opt + ' gravity="0 0 0"', ball_extra=gx, z0=1000)
    d2 = mujoco.MjData(m2)
    d2.qvel[0] = 5.0; d2.qvel[4] = 100.0
    mujoco.mj_forward(m2, d2)
    f = d2.qfrc_passive[:3].copy()
    print(f"{label}: v_terminal={vt:.2f} m/s -> implied Cd={cd:.2f}; at v=5 m/s, w=100 rad/s: "
          f"F_drag={-f[0]*1e3:.2f} mN, F_lift(z)={f[2]*1e3:.3f} mN, gravity={M*G*1e3:.1f} mN")
