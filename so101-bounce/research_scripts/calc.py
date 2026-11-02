import math, numpy as np
g=9.81; D=0.040
print("== flight per apex height ==")
for h in [0.05,0.10,0.20,0.30]:
    T=2*math.sqrt(2*h/g); v=math.sqrt(2*g*h)
    fr={f:T*f for f in (30,60,120,187)}
    print(f"h={h*100:.0f}cm T={T*1000:.0f}ms v_impact={v:.2f}m/s frames:"+" ".join(f"{f}fps={n:.1f}" for f,n in fr.items()))
    print("   blur mm at impact for exposure 33/16/8/4/2/1 ms:", [round(v*t,1) for t in (33.3,16.7,8,4,2,1)])
print("== C920 focal length estimate from dFOV 78deg @16:9 ==")
diag=math.hypot(16,9); f1080=(1920/2)*(diag/16)/math.tan(math.radians(39))
print(f"f@1080p={f1080:.0f}px HFOV={2*math.degrees(math.atan(960/f1080)):.1f} VFOV={2*math.degrees(math.atan(540/f1080)):.1f}")
modes={"1280x720":f1080*1280/1920,"640x360(16:9)":f1080*640/1920,"640x480(4:3 crop of 1440x1080)":f1080*640/1440}
for name,f in modes.items():
    print(f"  {name}: f={f:.0f}px")
    for Z in (0.6,0.75,0.9):
        d=f*D/Z
        # ball rises dh vertically; camera looks down 60deg -> range shrinks by dh*sin60, image shift dh*cos60
        for dh in (0.05,0.10):
            Z2=Z-dh*math.sin(math.radians(60)); d2=f*D/Z2
            shift=f*dh*math.cos(math.radians(60))/Z
            print(f"     Z={Z}: d={d:.1f}px; +{dh*100:.0f}cm height -> diam +{d2-d:.2f}px, image shift {shift:.0f}px; mm/px lateral={Z/f*1000:.2f}")
    # height noise from diameter noise 0.5/1px
    Z=0.75; d=f*D/Z
    for sd in (0.5,1.0):
        print(f"     Z=0.75 sigma_d={sd}px -> sigma_range={Z*sd/d*100:.1f}cm -> sigma_height={Z*sd/d/math.sin(math.radians(60))*100:.1f}cm")
print("== side camera at 0.5m, f=604px: mm/px", 0.5/604*1000)
print("== wrist/paddle cam f=400px assumed: ball diameter px at 5/10/20cm:", [round(400*D/z) for z in (0.05,0.10,0.20)])
print("== apex & impact-time error from velocity error ==")
for v in (0.99,1.4,1.98):
    for sv in (0.02,0.1,0.27):
        print(f"  v={v} sigma_v={sv}: sigma_apex={v*sv/g*100:.1f}cm sigma_t_impact~{sv/g*1000:.0f}ms")
print("== LS velocity noise: sigma_p, N frames at fps ==")
for sp,label in ((0.02,'overhead diam 2cm'),(0.005,'overhead 5mm'),(0.001,'side 1mm')):
    for fps,N in ((30,4),(60,8),(120,17)):
        dt=1/fps; sv=sp*math.sqrt(12/(N*(N*N-1)))/dt
        print(f"  {label} fps={fps} N={N}: sigma_v={sv*100:.1f}cm/s")
print("== negative-acceleration window (unverified formula) -2g(1+e^2)/(1+e)^2 < a < 0 ==")
for e in (0.7,0.8,0.87,0.9):
    print(f"  e={e}: a_min={-2*(1+e*e)/(1+e)**2:.2f} g")
print("== flat-paddle lateral drift Monte Carlo ==")
rng=np.random.default_rng(0)
R=0.055 # usable radius (blade ~7.5cm radius minus ball radius/margin)
for h in (0.05,0.10,0.20):
    T=2*math.sqrt(2*h/g); v=math.sqrt(2*g*h)
    for tilt_deg in (0.5,1.0,2.0):
        for a in (1.0,0.6):
            sdv=(1+0.85)*v*math.sin(math.radians(tilt_deg))  # lateral dv per bounce from tilt error
            n_trials=4000; res=[]
            for _ in range(n_trials):
                x=np.zeros(2); vv=np.zeros(2); k=0
                while k<500:
                    vv=a*vv+rng.normal(0,sdv/math.sqrt(2),2)
                    x=x+vv*T; k+=1
                    if np.linalg.norm(x)>R: break
                res.append(k)
            print(f"  h={h*100:.0f}cm tilt_err={tilt_deg}deg damping={a}: median bounces before leaving {R*100:.1f}cm radius = {int(np.median(res))} (p10={int(np.percentile(res,10))}, p90={int(np.percentile(res,90))})")
