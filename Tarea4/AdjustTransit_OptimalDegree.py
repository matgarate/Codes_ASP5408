import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.stats import chisquare
from numpy import linalg
from numpy.polynomial.polynomial import polyval

from sklearn.cross_validation import KFold

tI=0.4
tIV=0.7
sigma=30.0*10**(-6)

P=5

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


print "P & AIC & BIC"

Delta=[]
for P in range(0,7):
	#Fit a polynomial for the regions outside the transit
	b= linalg.lstsq(CoeffMatrix(time[index_notransit],P+1),flux[index_notransit])[0]
	#Fit the depth of the transit by substracting the polynomial
	a= linalg.lstsq(CoeffMatrix(time[index_transit],1),flux[index_transit]-polyval(time[index_transit],b))[0]

	delta=-a[0]
	Delta.append(delta)
	b[0]=b[0]-1.0
	model=box_model(time,delta)+polyval(time,b)

	plt.title(str(P)+" degree poly fit")
	plt.xlabel('Time')
	plt.ylabel('Flux')
	plt.plot(time,flux,'ro', label='Data')
	plt.plot(time,model,'b', label='Box+Poly Model')
	plt.legend()
	

	LogLike= -np.log(1.0/np.sqrt(2*3.1415*sigma*sigma))*np.sum(np.square(flux-model))/(2.0*sigma*sigma)
	AIC=2.0*P-2.0*LogLike
	BIC=np.log(time.size)*P -2.0*LogLike
	
	print str(P)+" & "+str(AIC)+" & "+str(BIC)

	plt.show()


#Kfold
#Removing the depth for simplicity
flux[index_transit]+=Delta[4]
model[index_transit]+=Delta[4]


fold=5
kf = KFold(time.size, n_folds=fold)
MSE=[]
PolyParam=[]


print "P & K-fold MSE"

for P in range(0,8):
	MSE.append(0.0)
	PolyParam.append(np.zeros(P+1))
	for train_index, test_index in kf:
		time_train, time_test = time[train_index], time[test_index]
		flux_train, flux_test = flux[train_index], flux[test_index]
		
		k_param= linalg.lstsq(CoeffMatrix(time_train,P+1),flux_train)[0]
		PolyParam[P]= PolyParam[P]+ k_param/float(fold)
		set_error= np.sum(np.square(flux_test - polyval(time_test,k_param)))
		MSE[P]+=set_error/float(time.size)
	
	print str(P)+" & "+str(MSE[P])

