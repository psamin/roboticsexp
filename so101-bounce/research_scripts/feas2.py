import math, cmath
g=9.81
kt,R,eg,egr,fv=1.1776,2.4787,0.166,1.005,0.0599
print("2nd-order servo model (BAM m1 linearised, D-term ignored) + dead time 5/15/30 ms")
for V,P in ((5.0,16),(7.4,16),(5.0,32),(7.4,32)):
  K=kt*(V/R)*P*eg*egr; B=kt*kt/R+fv
  for J in (0.027,0.033):
    out=[]
    for f in (2.5,3.5,5.0):
      w=2*math.pi*f; G=K/(K-J*w*w+1j*B*w)
      ph=math.degrees(cmath.phase(G)); 
      out.append(f"{f}Hz |G|={abs(G):.2f} ph={ph:.0f}deg (+{-360*f*0.015:.0f}deg per 15ms delay)")
    print(f"V={V} P={P} J={J}: wn={math.sqrt(K/J)/2/math.pi:.2f}Hz  "+"; ".join(out))
print("pumping: paddle speed to go from apex h1 to h2 in one hit, v_p=(v2-e v1)/(1+e)")
for h1,h2 in ((0.02,0.05),(0.05,0.10),(0.10,0.15),(0.15,0.20)):
  v1,v2=math.sqrt(2*g*h1),math.sqrt(2*g*h2)
  print(f" {h1*100:.0f}->{h2*100:.0f}cm: "+", ".join(f"e={e}: {(v2-e*v1)/(1+e):.2f} m/s" for e in (0.75,0.85,0.9)))
print("drop-and-hit launch: paddle drops with a>g so ball floats; ball falls hf before hit; paddle speed needed for apex h")
for hf in (0.01,0.02):
  vin=math.sqrt(2*g*hf)
  for h in (0.05,0.10,0.15,0.20):
    vout=math.sqrt(2*g*h)
    print(f" fall {hf*100:.0f}cm (v_in={vin:.2f}) -> apex {h*100:.0f}cm: "+", ".join(f"e={e}: v_p={(vout-e*vin)/(1+e):.2f}" for e in (0.75,0.85,0.9)))
print("gap growth when paddle accel = -k g for t: gap = 0.5 (k-1) g t^2")
for k in (1.2,1.5):
  print(f" k={k}: t=50ms gap {0.5*(k-1)*g*0.05**2*1000:.1f} mm, t=80ms {0.5*(k-1)*g*0.08**2*1000:.1f} mm")
print("ball impact angular impulse on wrist: m_b(1+e)v * r")
for h in (0.05,0.2):
  v=math.sqrt(2*g*h); Jimp=0.0027*1.85*v
  print(f" h={h*100:.0f}: linear impulse {Jimp*1000:.1f} mN*s; at r=0.25 -> {Jimp*0.25*1000:.2f} mN*m*s; dOmega with J=0.033: {Jimp*0.25/0.033:.3f} rad/s; paddle eff. mass at r=0.25 ~ J/r^2={0.033/0.0625:.2f} kg vs ball 0.0027")
print("grip moment demand: paddle CoM ~0.105 m beyond grip, m=0.17")
for a in (0,2.5,10,14):
  print(f" a_blade={a}: M = m(g+a)d = {0.17*(9.81+a)*0.105:.2f} N*m")
print("velocity quantisation from 1-step finite difference:", [f"{f}Hz: {math.radians(0.088)*f:.2f} rad/s" for f in (50,100,200)])
