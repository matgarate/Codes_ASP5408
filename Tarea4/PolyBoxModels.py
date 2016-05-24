import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.stats import chi2 as chisquare
from numpy import linalg
from numpy import random
from numpy.polynomial.polynomial import polyval


tI=0.4
tIV=0.7

def box_model(time,delta):
	box_flux=np.ones(time.size)
	index=np.intersect1d(np.where(time>=tI)[0],np.where(time<=tIV)[0])
	box_flux[index]=1.0-delta
	return box_flux


def CoeffMatrix(x,n):
	N=x.size
	exponents=np.arange(n)
	M=[]
	for k in range(N):
		M.append( np.power(np.ones(n)*x[k],exponents))
	return np.array(M)	


P=6

sigma= 30.0*10**(-6)
PolyCoeff= [0.0 , 0.02 , -0.07 , 0.12 , -0.11 ,  0.04]
delta=0.00011
#time,flux,err= np.loadtxt("datos.dat",unpack=True, skiprows=1)
time=np.linspace(0.0,1.0,num=300)

index_transit=np.intersect1d(np.where(time>=tI)[0],np.where(time<=tIV)[0])
index_notransit=np.union1d(np.where(time<tI)[0],np.where(time>tIV)[0])

N=time.size
SimNum=1000
BaseModel= polyval(time,PolyCoeff)+box_model(time,delta)#Polynomial+Box model using derived coefficients
Noise= random.normal(0.0,sigma,(SimNum,N))		#Gaussian Noise
Sample=BaseModel+Noise					#Sample of PolyBoxModel + Noise.

pval=[]

for flux in Sample:
	b= linalg.lstsq(CoeffMatrix(time[index_notransit],P),flux[index_notransit])[0]
	#Fit the depth of the transit by substracting the polynomial
	a= linalg.lstsq(CoeffMatrix(time[index_transit],1),flux[index_transit]-polyval(time[index_transit],b))[0]

	delta=-a[0]
	b[0]=b[0]-1.0
	model=box_model(time,delta)+polyval(time,b)

	residuals= (flux-model)/sigma

	chi2= np.sum(np.square(residuals))
	dof= N- 6 -1
	red_chi2=chi2/float(dof)
	pval.append(chisquare.sf(chi2, dof))



	
Nbins=100
Histogram=np.histogram(pval,range=(0.0,1.0),bins=Nbins)
plt.figure(1)
XBins=np.linspace(0.0,1.0,num=Nbins)
plt.title('P-value distribution.')
plt.plot(XBins,Histogram[0],'ro-')
plt.xlabel('p-value')
plt.ylabel('count')
plt.show()


