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
	plt.contour(aux_x,aux_y,aux_z, 1,colors='k',linewidths=3)


	


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


Confussion_Score(z,LeastSquare(x,y,z,x,y,1) ,"LSQ")
Confussion_Score(z,LeastSquare(x,y,z,x,y,2) ,"LSQ Quadratic")
Confussion_Score(z,LDA(x1,y1,x2,y2,x,y) ,"LDA")
Confussion_Score(z,QDA(x1,y1,x2,y2,x,y) ,"QDA")


plt.figure(0)
plt.title('Data')
plt.plot(x1,y1,'ro')
plt.plot(x2,y2,'bo')

plt.figure(1)
PlotContours(xr,yr,z_lsq,"Least Squares Order 1")
plt.figure(2)
PlotContours(xr,yr,z_lsq_quad,"Least Squares Order 2")
plt.figure(3)
PlotContours(xr,yr,z_lda,"Linear Discriminant Analysis")
plt.figure(4)
PlotContours(xr,yr,z_qda,"Quadratic Discriminant Analysis")

#Methods Parte b)

z_k1=KNeigh(x,y,z,xr,yr,1)
z_k15=KNeigh(x,y,z,xr,yr,15)

z_svc=SVC(x,y,z,xr,yr)
#z_lasso=LassoQuadratic(x,y,z,xr,yr)


Confussion_Score(z,KNeigh(x,y,z,x,y,1) ,"NN-1")
Confussion_Score(z,KNeigh(x,y,z,x,y,15) ,"NN-15")
#Confussion_Score(z,LassoQuadratic(x,y,z,x,y) ,"Quadratic Lasso")
Confussion_Score(z,SVC(x,y,z,x,y) ,"SVC")


plt.figure(5)
PlotContours(xr,yr,z_k1,"1 Nearest Neighbohr")

plt.figure(6)
PlotContours(xr,yr,z_k15,"15 Nearest Neighbohr")

plt.figure(7)
#PlotContours(xr,yr,z_lasso,"Quadratic Lasso")
PlotContours(xr,yr,z_svc,"SVC")


#Methods Parte c)
z_bayes=BayesClassifier(float(index_1.size)/float(n),float(index_2.size)/float(n),np.array([2,3]),np.array([6,6]),np.array([[5,-2],[-2,5]]),np.array([[1,0],[0,1]]),xr,yr)
Confussion_Score(z,BayesClassifier(float(index_1.size)/float(n),float(index_2.size)/float(n),np.array([2,3]),np.array([6,6]),np.array([[5,-2],[-2,5]]),np.array([[1,0],[0,1]]),x,y) ,"Bayes Classifier")
plt.figure(8)
PlotContours(xr,yr,z_bayes,"Bayes Classifier")



plt.show()

