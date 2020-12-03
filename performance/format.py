import numpy as np

x = np.fromfile("perf.txt", sep=' ')
x = x.tolist()
if len(x) % 33 != 0:
    x.extend([0.0]*( 33-len(x)%33 ))

y = np.array(x).reshape((-1,33))
cols = y[:,0]
z = y[:,1:].reshape((-1,4,2,4))
z = z.transpose((2,0,3,1)) # H/D, sz, dtype, asymm

hst=open("host.txt", "w")
dev=open("device.txt", "w")
hst.write("# N <float 1 8 125 1000> <dbl 1 8 125 1000> <cplx float 1 8 125 1000> <cplx dbl 1 8 125 1000>\n")
dev.write("# N <float 1 8 125 1000> <dbl 1 8 125 1000> <cplx float 1 8 125 1000> <cplx dbl 1 8 125 1000>\n")
for i in range(len(y)):
  hst.write("%5d "%cols[i] + " ".join("%f"%u for u in z[0,i].reshape(-1)) + "\n")
  dev.write("%5d "%cols[i] + " ".join("%f"%u for u in z[1,i].reshape(-1)) + "\n")
hst.close()
dev.close()
