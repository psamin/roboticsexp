"""Spin transfer vs friction-cone settings (shell ball, k=1e5, dt=0.5 ms)."""
import numpy as np, mujoco
R=0.02; M=0.0027; Z=0.0517; k=1e5; b=2*Z*np.sqrt(k)
def run(opt, mu, pair=""):
    x=f"""<mujoco><option timestep="0.0005" integrator="implicitfast" {opt}/>
<worldbody><geom name="pad" type="box" size="0.08 0.075 0.005" condim="3" friction="{mu} 0.005 0.0001"/>
<body name="ball" pos="0 0 0.026"><freejoint/><geom name="ball" type="sphere" size="{R}" mass="{M}" shellinertia="true"
 solref="{-k} {-b:.2f}" condim="3" friction="{mu} 0.005 0.0001"/></body></worldbody>{pair}</mujoco>"""
    m=mujoco.MjModel.from_xml_string(x); d=mujoco.MjData(m); d.qvel[2]=-2.4; d.qvel[0]=0.5; t=False
    for _ in range(400):
        mujoco.mj_step(m,d); t|=d.ncon>0
        if t and d.ncon==0: break
    return d.qvel[0], d.qvel[4], d.qvel[2]
print("rolling prediction (shell): vx_out=0.300, wy=15.0 rad/s")
for opt in ['', 'cone="elliptic"', 'cone="elliptic" impratio="10"', 'cone="elliptic" impratio="100"']:
    for mu in [0.2, 0.5, 1.0]:
        vx,wy,vz=run(opt,mu); print(f"{opt or 'default (pyramidal, impratio=1)':32s} mu={mu}: vx_out={vx:.3f} wy={wy:5.1f} vz_out={vz:.2f}")
pair='<contact><pair geom1="ball" geom2="pad" condim="3" friction="1 1 0.005 0.0001 0.0001" solref="%g %g" solreffriction="-1e6 -50"/></contact>'%(-k,-b)
vx,wy,vz=run('cone="elliptic"',1.0,pair); print(f"explicit pair, solreffriction=-1e6 -50, elliptic mu=1: vx_out={vx:.3f} wy={wy:5.1f} vz_out={vz:.2f}")
