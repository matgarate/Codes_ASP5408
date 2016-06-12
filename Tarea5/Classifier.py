import numpy as np
import matplotlib.pyplot as plt



x,y,clase= np.loadtxt("datos_clasificacion.dat",skiprows=1,unpack=True)


index_1= np.where(clase==1)[0]
index_2= np.where(clase==2)[0]

plt.plot(x[index_1],y[index_1],'ro')
plt.plot(x[index_2],y[index_2],'bo')

plt.show()
