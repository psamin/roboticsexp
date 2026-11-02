import re, zlib, sys
data = open(sys.argv[1],'rb').read()
out=[]
for m in re.finditer(rb'stream\r?\n(.*?)\r?\nendstream', data, re.S):
    s=m.group(1)
    try: d=zlib.decompress(s)
    except Exception: continue
    # extract text in TJ/Tj
    for t in re.finditer(rb'\[(.*?)\]\s*TJ|\((.*?)\)\s*Tj', d, re.S):
        if t.group(1) is not None:
            parts=re.findall(rb'\((.*?)(?<!\\)\)|(-?\d+\.?\d*)', t.group(1))
            line=''
            for p,n in parts:
                if p: line+=p.decode('latin1')
                elif n and float(n)<-200: line+=' '
            out.append(line)
        else:
            out.append(t.group(2).decode('latin1'))
    out.append('\n')
print(' '.join(out))
