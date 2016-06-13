import numpy as np
from numpy import linalg

def Discriminant_LDA(vector,Pi,mu,cov):
	inv_cov= linalg.inv(cov)	
	return np.dot(vector.T,np.dot(inv_cov,mu))-0.5* np.dot(mu.T,np.dot(inv_cov,mu)) + np.log(Pi)

def Discriminant_QDA(vector,Pi,mu,cov):
	inv_cov= linalg.inv(cov)	
	det_cov= linalg.det(cov)
	d=[]
	for v in vector.T:
		x= v - mu
		d.append( -0.5*np.dot(x.T,np.dot(inv_cov,x)) )
	d=np.array(d)+np.log(Pi) -0.5*np.log(det_cov)
	return d

def CoeffMatrix(x,y,order):
	n=x.size

	if order==0:
		M=[np.ones(n)]
	if order==1:
		M=[np.ones(n),x,y]
	if order==2:
		M=[np.ones(n),x,y,np.square(x),np.square(y),np.multiply(x,y)]
	return np.array(M).T


def LeastSquare(x,y,z,xr,yr,order):
	M=CoeffMatrix(x,y,order)

	beta=linalg.lstsq(M,z)[0]
	print "Params = "+ str(np.round(beta,2))
	return np.dot(CoeffMatrix(xr,yr,order),beta)


def LDA(x1,y1,x2,y2,xr,yr):
	k=2
	n=x1.size + x2.size
	Pi_1= float(x1.size)/float(n)
	Pi_2= float(x2.size)/float(n)

	mu_1=np.array([np.average(x1) ,np.average(y1)])
	mu_2=np.array([np.average(x2) ,np.average(y2)])

	a = np.array([x1-mu_1[0],y1-mu_1[1]])
	b = np.array([x2-mu_2[0],y2-mu_2[1]])
	cov = (np.dot(a,a.T)+np.dot(b,b.T))/(n-k)

	Delta_1= Discriminant_LDA(np.array([xr,yr]), Pi_1,mu_1,cov)
	Delta_2= Discriminant_LDA(np.array([xr,yr]), Pi_2,mu_2,cov)

	zr=np.ones(xr.size)
	Delta_1= Discriminant_LDA(np.array([xr,yr]), Pi_1,mu_1,cov)
	Delta_2= Discriminant_LDA(np.array([xr,yr]), Pi_2,mu_2,cov)
	zr[np.where((Delta_2-Delta_1)>=0.0)[0]]=2
	return zr

def QDA(x1,y1,x2,y2,xr,yr):
	n=x1.size + x2.size
	Pi_1= float(x1.size)/float(n)
	Pi_2= float(x2.size)/float(n)

	mu_1=np.array([np.average(x1) ,np.average(y1)])
	mu_2=np.array([np.average(x2) ,np.average(y2)])
	
	cov_1 = np.cov(x1,y1)
	cov_2 = np.cov(x2,y2)

	zr=np.ones(xr.size)
	Delta_1= Discriminant_QDA(np.array([xr,yr]), Pi_1,mu_1,cov_1)
	Delta_2= Discriminant_QDA(np.array([xr,yr]), Pi_2,mu_2,cov_2)

	zr[np.where((Delta_2-Delta_1)>=0.0)[0]]=2
	return zr


