import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.stats import chisquare
from numpy import linalg
from numpy import random
from numpy.polynomial.polynomial import polyval


sigma= 30.0*10**(-6)
PolyCoeff= [0.0 , 0.02 , -0.07 , 0.12 , -0.11 ,  0.04]
time,flux,err= np.loadtxt("datos.dat",unpack=True, skiprows=1)


N=time.size
SimNum=1000
BasePoly= polyval(time,PolyCoeff)		#Polynomial model using derived coefficients
Noise= random.normal(0.0,sigma,(SimNum,N))	#Gaussian Noise
Sample=BasePoly+Noise				#Sample of PolyModel Noise.

chi2,pval= chisquare( Sample.T ,(np.zeros((SimNum,N)) + BasePoly).T)



Nbins=20
Histogram=np.histogram(pval,range=(0.0,1.0),bins=Nbins)


XBins=np.linspace(0.0,1.0,num=Nbins)
plt.title('P-value distribution.\n')
plt.plot(XBins,Histogram[0],'ro-')
plt.xlabel('p-value')
plt.ylabel('count')

plt.show()



'''
plt.xlabel('Time')
plt.ylabel('Flux')
plt.plot(time,flux,'ro', label='Data')
plt.plot(time,BasePoly,'b', label='Poly Model')
plt.legend()
'''
