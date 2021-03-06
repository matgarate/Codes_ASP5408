import numpy as np
import matplotlib.pyplot as plt

from scipy.special import ndtr as NormCDF
from scipy.special import ndtri as NormCDFInv

num=2000				#Number of random variables to generate
u= np.random.uniform(0.0,1.0,num)	#Uniform random variables
X=NormCDFInv(u)				#Inverse of F(u)
x=np.linspace(0.0,3.0,num=num)		#Evenly spaced grid
CDF_X=[]

#How many values of X have lower values than x
for i in xrange(num):
	CDF_X.append(np.where(X<=x[i])[0].size)
CDF_X=np.divide(CDF_X,float(num))


plt.title('Random variables following: X=InvF(U)')
plt.plot(x,NormCDF(x),'b',label='Normal Distribution CDF')
plt.plot(x,CDF_X,'r',label='CDF of X ')
plt.legend(loc=4)
plt.show()
