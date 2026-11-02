import math
g=9.81
H=[0.05,0.10,0.15,0.20]; E=[0.75,0.85,0.90]
print("== A. steady bounce (apex h above impact plane)")
print("h_cm  v_imp  T_flight  steps@30/50/100  f_bounce")
for h in H:
    v=math.sqrt(2*g*h); T=2*v/g
    print(f"{h*100:4.0f}  {v:.3f}  {T*1000:6.1f}ms  {T*30:5.1f}/{T*50:5.1f}/{T*100:5.1f}  {1/T:.2f}Hz")
print("v_paddle = v(1-e)/(1+e) [m/s];  sinusoid at bounce freq: min amplitude A=v_p/w (pk-pk 2A), peak accel A w^2")
for h in H:
    v=math.sqrt(2*g*h); T=2*v/g; w=2*math.pi/T
    row=[]
    for e in E:
        vp=v*(1-e)/(1+e); A=vp/w
        row.append(f"e={e}: vp={vp:.3f} A={A*1000:.1f}mm pkpk={2*A*1000:.1f}mm apk={A*w*w:.2f}")
    print(f"h={h*100:.0f}cm  "+" | ".join(row))
print("== lateral: landing drift per deg tilt (8h*th), apex drift (4h*th); tilt to recentre 1 cm in one bounce (dx/8h)")
for h in H:
    th=math.radians(1)
    print(f"h={h*100:.0f}: landing {8*h*th*1000:.1f} mm/deg, apex {4*h*th*1000:.1f} mm/deg, 1cm-correct tilt {math.degrees(0.01/(8*h)):.2f} deg, per encoder step(0.088deg) {8*h*math.radians(0.088)*1000:.2f} mm, per 0.5deg backlash {8*h*math.radians(0.5)*1000:.1f} mm")
print("== B. launch from rest: v_sep=sqrt(2gh); stroke to separation d1=v^2/(2a_up); paddle overshoot after sep d2=v^2/(2a_dn)")
for h in H:
    v=math.sqrt(2*g*h)
    s=[]
    for aup in (10,20):
        for adn in (1.2*g,2*g):
            s.append(f"aup={aup} adn={adn/g:.1f}g: d1={v*v/(2*aup)*100:.1f} d2={v*v/(2*adn)*100:.1f} tot={(v*v/(2*aup)+v*v/(2*adn))*100:.1f}cm t_acc={v/aup*1000:.0f}ms")
    print(f"h={h*100:.0f}cm v_sep={v:.3f} m/s; "+" ; ".join(s))
    print(f"     wrist-hinge joint speed at sep for r=0.15/0.25 m: {v/0.15:.1f}/{v/0.25:.1f} rad/s ; elbow-translation (L2=0.135): {v/0.135:.1f} rad/s")
print("launch apex feasible if v_sep <= r*w_max: r*w for w=3.7 (5V est), 5.2 (fw cap):", [f"r={r}: {r*3.7:.2f}/{r*5.2:.2f}" for r in (0.135,0.15,0.25,0.30)])
print("required joint decel for >g at lever r:", [f"r={r}: {g/r:.0f} rad/s^2" for r in (0.135,0.15,0.18,0.25,0.30,0.39)])
acc254=254*8.7*math.pi/180
print(f"fw Acceleration=254 cap = {acc254:.1f} rad/s^2 -> paddle accel at r=0.15/0.25: {acc254*0.15:.1f}/{acc254*0.25:.1f} m/s^2")
print("== C. torques")
TAU={'5V(7.4V servo,est)':13.75*0.0981,'6V(7.4V servo)':16.5*0.0981,'7.4V(7.4V servo)':19.5*0.0981,'12V(C018)':30*0.0981}
for k,v in TAU.items(): print(f"stall {k}: {v:.2f} N*m")
JA=(0.0216,0.0261)  # BAM armature range m1..m6
mw,rw=0.079,0.0386; mgp,rg=0.099,0.086; mfa,rfa=0.104,0.098; L2=0.135; L1=0.116
for mp in (0.15,0.17,0.19):
  for dp in (0.12,0.15,0.18,0.25,0.30):
    tw=g*(mw*rw+mgp*rg+mp*dp)
    Jw=mw*rw**2+mgp*rg**2+3e-5+4e-5+mp*dp**2+mp*0.157**2/12
    # pose A: upper arm vertical, forearm horizontal fwd, paddle horizontal fwd
    te=g*(mfa*rfa+mw*(L2+rw)+mgp*(L2+rg)+mp*(L2+dp))
    ts=te  # upper arm CoM on vertical line through shoulder axis
    # pose X: arm fully extended horizontally (worst case)
    tsx=g*(0.103*0.09+mfa*(L1+rfa)+mw*(L1+L2+rw)+mgp*(L1+L2+rg)+mp*(L1+L2+dp))
    print(f"mp={mp} dp={dp}: wrist_flex static {tw:.3f} Nm, J_load {Jw*1e3:.2f}e-3 (+armature ~{JA[0]*1e3:.0f}-{JA[1]*1e3:.0f}e-3); elbow=shoulder static poseA {te:.3f}; shoulder fully-extended {tsx:.3f}")
print("-- wrist hinge dynamics (paddle contact at r=dp), mp=0.17")
mp=0.17
for dp in (0.15,0.25):
  tw=g*(mw*rw+mgp*rg+mp*dp); Jw=mw*rw**2+mgp*rg**2+7e-5+mp*dp**2+mp*0.157**2/12
  for ja in JA:
    J=Jw+ja
    for k,ts in TAU.items():
      a_brake=(ts+tw)/J; a_up=(ts-tw)/J
      print(f" dp={dp} Jarm={ja} {k}: alpha_up_max={a_up:.0f} alpha_brake_max={a_brake:.0f} rad/s^2 -> paddle {a_up*dp:.1f}/{a_brake*dp:.1f} m/s^2 ; torque for 4.4 m/s^2 sinus stroke = {J*4.4/dp+tw:.2f} Nm")
print("-- elbow translation, pose A, wrist counter-rotates (wrist armature also reflected), mp=0.17")
for dp in (0.15,0.25):
  for ja in JA:
    Je=mfa*rfa**2+1.6e-4+(mw+mgp+mp)*L2**2+ja
    te=g*(mfa*rfa+mw*(L2+rw)+mgp*(L2+rg)+mp*(L2+dp))
    for k,ts in TAU.items():
      ab=(ts+te)/Je; au=(ts-te)/Je
      print(f" dp={dp} Jarm={ja} {k}: elbow static {te:.2f}Nm; alpha_up={au:.0f} alpha_brake={ab:.0f} -> paddle up {au*L2:.1f} brake {ab*L2:.1f} m/s^2; wrist servo needs {ja*ab:.2f} Nm just to spin its rotor at that alpha")
print("== D. linearised joint loop from BAM m1 (P=16, D ignored)")
kt,R,eg,egr,fv=1.1776,2.4787,0.166,1.005,0.0599
for V in (5.0,7.4):
  K=kt*(V/R)*16*eg*egr; B=kt*kt/R+fv
  for J in (0.024,0.030,0.035):
    wn=math.sqrt(K/J); z=B/(2*math.sqrt(K*J))
    print(f"V={V}: K={K:.2f} Nm/rad, B={B:.2f} Nms/rad, J={J}: wn={wn:.1f} rad/s ({wn/2/math.pi:.2f} Hz), zeta={z:.2f}; duty saturates at err {math.degrees(0.97/(16*eg*egr)):.1f} deg")
print(f"BAM m1 implied 'stall' kt*0.97*7.4/R = {kt*0.97*7.4/R:.2f} Nm vs datasheet 1.91")
print("== E. bus timing at 1 Mbps (10 bits/byte)")
n=6
tx_r=2+1+1+1+1+1+n+1; rx=8*n; tx_w=2+1+1+1+1+1+n*3+1
print(f"sync_read Present_Position x6: {tx_r}+{rx} bytes = {(tx_r+rx)*10/1e3:.2f} ms wire; with Present_Velocity too (4 B each): {(tx_r+(10*n))*10/1e3:.2f} ms; sync_write Goal_Position x6: {tx_w} B = {tx_w*10/1e3:.2f} ms")
print("LeRobot failure timeout per packet ~ tx*len + 3*tx + 50 ms")
