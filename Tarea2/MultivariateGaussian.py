import numpy as np
from numpy.linalg import cholesky
from numpy.random import multivariate_normal as mn
import matplotlib.pyplot as plt


Dim=2
Lenght=5.0

Factor= 2.0*Lenght/(float(Dim-1))
Index=np.arange(Dim)


#This vector g can be used to describe our covariance matrix completely
#Covariance[i,j] = g[|i-j|]
g=np.exp(-0.5*np.square(Factor*Index))

Zero=np.zeros(Dim)
Identity=np.identity(Dim)
Sigma=np.zeros((Dim,Dim))
for i in range(Dim):
	Sigma[i,i]=g[0]
	for j in range(i):
		m=i-j
		Sigma[i,j]=g[m]
		Sigma[j,i]=g[m]
L=cholesky(Sigma)


print Sigma

PointNum=3000

X=mn(Zero,Sigma,PointNum).T
S= np.dot( mn(Zero,Identity,PointNum), L ).T

plt.title('Bivariate Gaussian Distributions')
plt.plot(X[0],X[1],'bo',markersize=5.0,label='N(0,Sigma)')
plt.plot(S[0],S[1],'ro',markersize=5.0,label='LS with (S~N(0,I))' )
plt.axis('equal')
plt.legend()

plt.show()




'''
#Get the cholesky decomposition
L=np.zeros((Dim,Dim))
for i in range(Dim):
	s=0.0
	for k in range(i-1):
		s +=  L[i,k]**2
	L[i,i]=np.sqrt(1.0-s)
	for j in range(i):
		s=0.0		
		for k in range(j-1):
			s += L[i,k]*L[j,k]
		L[i,j]=(g[i-j] - s)/L[j,j]
'''



