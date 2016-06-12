import numpy as np
from numpy import linalg
import scipy
import matplotlib.pyplot as plt


#306

x,y,z= np.loadtxt("datos_clasificacion.dat",skiprows=1,unpack=True)
n=z.size


M=np.array([np.ones(n),x,y]).T


#beta= np.dot(linalg.inv(np.dot(M.T,M)),np.dot(M.T,z)) #  Beta= Inv(Mt,M)*Mt * z
beta=linalg.lstsq(M,z)[0]


index_1= np.where(z==1.0)[0]
index_2= np.where(z==2.0)[0]
plt.plot(x[index_1],y[index_1],'ro')
plt.plot(x[index_2],y[index_2],'bo')



N=160
r=np.linspace(-4.0,12.0,num=N)
xr, yr = np.meshgrid(r, r)
xr=xr.flatten()
yr=yr.flatten()
zr= np.dot( np.array([np.ones(N*N),xr,yr]).T,beta)
i1=np.where(zr<1.5)[0]
i2=np.where(zr>=1.5)[0]
plt.plot(xr[i1],yr[i1],'r+')
plt.plot(xr[i2],yr[i2],'b+')
plt.show()

