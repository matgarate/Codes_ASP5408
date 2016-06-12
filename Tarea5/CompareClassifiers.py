import numpy as np
import matplotlib.pyplot as plt

from Classifier_Methods import *
from Classifier_MethodsExtra import *

x,y,z= np.loadtxt("datos_clasificacion.dat",skiprows=1,unpack=True)
n=z.size


index_1= np.where(z==1.0)[0]
index_2= np.where(z==2.0)[0]

x1,y1= x[index_1],y[index_1]
x2,y2= x[index_2],y[index_2]



N=160
def PlotContours(xr,yr,zr,title):
	plt.title(title)	
	i1=np.where(zr<1.5)[0]
	i2=np.where(zr>=1.5)[0]
	plt.plot(xr[i1],yr[i1],'r+')
	plt.plot(xr[i2],yr[i2],'b+')

	plt.plot(x1,y1,'ro')
	plt.plot(x2,y2,'bo')

	aux_x=xr.reshape(N,N)	
	aux_y=yr.reshape(N,N)
	zr[i1]=1.0
	zr[i2]=2.0
	aux_z=zr.reshape(N,N)
	plt.contour(aux_x,aux_y,aux_z, 3,colors='k')




N=160
r=np.linspace(-4.0,12.0,num=N)
xr, yr = np.meshgrid(r, r)
xr=xr.flatten()
yr=yr.flatten()

#Methods Parte a)
z_lsq = LeastSquare(x,y,z,xr,yr,1)
z_lsq_quad=LeastSquare(x,y,z,xr,yr,2)
z_lda = LDA(x1,y1,x2,y2,xr,yr)
z_qda = QDA(x1,y1,x2,y2,xr,yr)

plt.figure(1)
PlotContours(xr,yr,z_lsq,"Least Squares Order 1")
plt.figure(2)
PlotContours(xr,yr,z_lsq_quad,"Least Squares Order 2")
plt.figure(3)
PlotContours(xr,yr,z_lda,"Linear Discriminant Analysis")
plt.figure(4)
PlotContours(xr,yr,z_qda,"Quadratic Discriminant Analysis")

#Methods Parte b)


#Methods Parte c)


plt.show()

