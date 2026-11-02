import time, numpy as np, cv2
cv2.setNumThreads(1)
def frames(w,h,n=200):
    rng=np.random.default_rng(0)
    bg=np.full((h,w,3),235,np.uint8)  # white board
    out=[]
    for i in range(n):
        f=bg.copy(); f+=rng.integers(0,8,f.shape,dtype=np.uint8)
        cx=int(w*0.3+w*0.4*(i/n)); cy=int(h*0.5)
        cv2.circle(f,(cx,cy),max(3,w//64),(0,120,255),-1)  # orange ball (BGR)
        out.append(f)
    return out
def hsv_track(f):
    hsv=cv2.cvtColor(f,cv2.COLOR_BGR2HSV)
    m=cv2.inRange(hsv,(5,120,120),(25,255,255))
    m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    n,lab,st,cen=cv2.connectedComponentsWithStats(m)
    if n>1:
        k=1+np.argmax(st[1:,cv2.CC_STAT_AREA]); return cen[k]
def hsv_roi(f,c,r=48):
    x,y=int(c[0]),int(c[1]); x0,y0=max(0,x-r),max(0,y-r)
    return hsv_track(f[y0:y+r,x0:x+r])
for (w,h) in [(320,240),(640,480),(1280,720)]:
    F=frames(w,h)
    # MJPEG decode cost
    jp=[cv2.imencode('.jpg',f,[cv2.IMWRITE_JPEG_QUALITY,85])[1] for f in F]
    t=time.perf_counter(); [cv2.imdecode(j,cv2.IMREAD_COLOR) for j in jp]; dec=(time.perf_counter()-t)/len(F)*1e3
    t=time.perf_counter(); [hsv_track(f) for f in F]; full=(time.perf_counter()-t)/len(F)*1e3
    c=(w*0.5,h*0.5)
    t=time.perf_counter(); [hsv_roi(f,c) for f in F]; roi=(time.perf_counter()-t)/len(F)*1e3
    mog=cv2.createBackgroundSubtractorMOG2(history=200,detectShadows=False)
    t=time.perf_counter(); [mog.apply(f) for f in F]; mg=(time.perf_counter()-t)/len(F)*1e3
    # simple static-background absdiff in a region (hand detector proxy)
    ref=F[0].astype(np.int16)
    t=time.perf_counter()
    for f in F:
        d=cv2.absdiff(f,F[0]); g=cv2.cvtColor(d,cv2.COLOR_BGR2GRAY); _,m=cv2.threshold(g,30,255,cv2.THRESH_BINARY); cnt=cv2.countNonZero(m)
    ad=(time.perf_counter()-t)/len(F)*1e3
    print(f"{w}x{h}: jpeg-decode {dec:.2f} ms | HSV+open+CC full {full:.2f} ms | HSV ROI 96px {roi:.3f} ms | MOG2 {mg:.2f} ms | absdiff+thresh {ad:.2f} ms")
