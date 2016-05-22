import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.stats import chisquare
from numpy import linalg
from numpy.polynomial.polynomial import polyval

tI=0.4
tIV=0.7
sigma=30.0*10**(-6)

P=6

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



#Read Data and indentify transit invervals
time,flux,err= np.loadtxt("datos.dat",unpack=True, skiprows=1)
index_transit=np.intersect1d(np.where(time>=tI)[0],np.where(time<=tIV)[0])
index_notransit=np.union1d(np.where(time<tI)[0],np.where(time>tIV)[0])

#Fit a polynomial for the regions outside the transit
b= linalg.lstsq(CoeffMatrix(time[index_notransit],P),flux[index_notransit])[0]
#Fit the depth of the transit by substracting the polynomial
a= linalg.lstsq(CoeffMatrix(time[index_transit],1),flux[index_transit]-polyval(time[index_transit],b))[0]


delta=-a[0]
b[0]=b[0]-1.0
model=box_model(time,delta)+polyval(time,b)

residuals= (flux-model)/sigma
print "Delta: " + str(delta)
print "Poly Coefficients: " + str(b)


chi2,p_value =chisquare(flux,model)
print "Chi-square Test: " + str(chi2)
print "p-value: "+ str(p_value)


'''
#chi2= np.sum(np.divide(np.square(model-flux),model))
chi2= np.sum(np.square(residuals))
print "Chi-square Test: " + str(chi2)

#p_value = np.prod(2* (1.0-norm.cdf(np.fabs(residuals) ) ))
p_value = np.average(2* (1.0-norm.cdf(np.fabs(residuals) ) ))
print "p-value: "+ str(p_value)
'''

plt.xlabel('Time')
plt.ylabel('Flux')
plt.plot(time,flux,'ro', label='Data')
plt.plot(time,model,'b', label='Box+Poly Model')
plt.legend()
plt.show()


