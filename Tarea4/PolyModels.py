import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.stats import chisquare
from numpy import linalg
from numpy import random
from numpy.polynomial.polynomial import polyval


def CoeffMatrix(x,n):
	N=x.size
	exponents=np.arange(n)
	M=[]
	for k in range(N):
		M.append( np.power(np.ones(n)*x[k],exponents))
	return np.array(M)	



sigma= 30.0*10**(-6)
PolyCoeff= [1.0 , 0.02 , -0.07 , 0.12 , -0.11 ,  0.04]

#time,flux,err= np.loadtxt("datos.dat",unpack=True, skiprows=1)
time=np.linspace(0.0,1.0,num=300)

N=time.size
SimNum=1000
BasePoly= polyval(time,PolyCoeff)		#Polynomial model using derived coefficients
Noise= random.normal(0.0,sigma,(SimNum,N))	#Gaussian Noise
Sample=BasePoly+Noise				#Sample of PolyModel Noise.

chi2_list=[]

for P in range(0,11):
	ParamAdjust= linalg.lstsq( CoeffMatrix(time,P+1),Sample.T)[0]	#Adjust P-degree poly to Samples
	PolyAdjust= polyval(time,ParamAdjust)				#Generate Poly Adjust
	chi2,pval= chisquare( Sample.T ,PolyAdjust.T)			#Chisquare foreach poly adjust.
	chi2_list.append(np.average(chi2))

	
	Nbins=100
	Histogram=np.histogram(pval,range=(0.0,1.0),bins=Nbins)

	plt.figure(1)
	XBins=np.linspace(0.0,1.0,num=Nbins)
	plt.title('P-value distribution.\n'+ str(P)+'-Degree Polynomial Fit')
	plt.plot(XBins,Histogram[0],'ro-')
	plt.xlabel('p-value')
	plt.ylabel('count')
	plt.show()

plt.figure(2)
plt.title('Chi-square Test Average values')
plt.xlabel('Fitted Polynomial Degree')
plt.ylabel('ChiSquare') 
plt.plot(np.arange(0,11),chi2_list)
plt.show()
