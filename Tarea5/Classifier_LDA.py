import numpy as np
from numpy import linalg
import scipy
import matplotlib.pyplot as plt




x,y,z= np.loadtxt("datos_clasificacion.dat",skiprows=1,unpack=True)
n=z.size

index_1= np.where(z==1.0)[0]
index_2= np.where(z==2.0)[0]
plt.plot(x[index_1],y[index_1],'ro')
plt.plot(x[index_2],y[index_2],'bo')


Pi_1= float(index_1)/float(n)
Pi_2= float(index_2)/float(n)

mu_1=np.array([np.average(x[index_1]) ,np.average(y[index_1])])
mu_2=np.array([np.average(x[index_2]) ,np.average(y[index_2])])



plt.show()

