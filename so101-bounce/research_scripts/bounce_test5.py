"""Spin transfer with explicit <pair solreffriction> damping (shell ball, k=1e5, e~0.85)."""
import numpy as np, mujoco
R=0.02; M=0.0027; Z=0.0517; k=1e5; b=2*Z*np.sqrt(k)
def run(srf, mu, dt=0.0005, vx=0.5):
    pair=f'<contact><pair geom1="ball" geom2="pad" condim="3" friction="{mu} {mu} 0.005 0.0001 0.0001" solref="{-k} {-b:.2f}" solreffriction="{srf}"/></contact>'
    x=f"""<mujoco><option timestep="{dt}" integrator="implicitfast" cone="elliptic"/>
<worldbody><geom name="pad" type="box" size="0.08 0.075 0.005"/>
<body name="ball" pos="0 0 0.026"><freejoint/><geom name="ball" type="sphere" size="{R}" mass="{M}" shellinertia="true"/></body></worldbody>{pair}</mujoco>"""
    m=mujoco.MjModel.from_xml_string(x); d=mujoco.MjData(m); d.qvel[2]=-2.4; d.qvel[0]=vx; t=False
    for _ in range(int(0.2/dt)):
        mujoco.mj_step(m,d); t|=d.ncon>0
        if t and d.ncon==0: break
    return d.qvel[0], d.qvel[4], d.qvel[2]
print("shell rolling prediction for vx_in=0.5: vx_out=0.300, wy=15.0; vx_in=2.0 (oblique, 40 deg): sliding regime")
for srf in ["0 0", "-1 -300", "-1 -1000", "-1 -3000", "0.002 1"]:
    for mu in [0.2, 1.0]:
        for dt in [0.0005, 0.001]:
            vx,wy,vz=run(srf,mu,dt); print(f"solreffriction={srf:9s} mu={mu} dt={dt}: vx_out={vx:.3f} wy={wy:5.1f} vz_out={vz:.2f}")
print("oblique vx_in=2.0, mu=0.2, solreffriction -1 -3000:", ["%.3f"%v for v in run("-1 -3000",0.2,0.0005,2.0)],
      " Coulomb-sliding prediction vx_out = 2.0 - mu*(1+e)*2.4 = %.3f"%(2.0-0.2*(1+0.845)*2.4))
