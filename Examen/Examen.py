import numpy as np
import matplotlib.pyplot as plt

from numpy import linalg
from sklearn.decomposition import PCA
from sklearn.cross_validation import KFold

import batman
import george
import emcee
import corner

######################################################
#
#	GLOBALS
#
######################################################

NCOMP=6
NSAMP=9
NFEAT=100


######################################################
#
#	READ AND PREPARE DATA
#
######################################################
time, flux_target= np.loadtxt("dataset_7.dat",usecols=(0,1),unpack=True,skiprows=1)
flux_comp= np.loadtxt("dataset_7.dat",usecols=(2,3,4,5,6,7,8,9,10),unpack=True,skiprows=1)

#CONVERT TO LOGSCALE
time= time/24.0
flux_target=np.log(flux_target)
flux_comp=np.log(flux_comp)

#SUBSTRACT MEAN FLUX
flux_average=np.average(flux_comp,axis=0)
for i in range(NSAMP):
	flux_comp[i]=flux_comp[i]-flux_average
flux_target=flux_target-flux_average


######################################################
#
#	PRINCIPAL COMPONENT ANALYSIS
#
######################################################


#OBTAIN AND FIT PCA COMPONENTS TO TRAINING SAMPLES
pca = PCA(n_components=NCOMP)
flux_sample_param= pca.fit_transform(flux_comp)
signals=pca.components_
variance_ratio=pca.explained_variance_ratio_

print flux_sample_param

print variance_ratio

flux_fit=[]
for i in range(NSAMP):
	a=np.zeros(NFEAT)
	for j in range(NCOMP):
		a=a+ signals[j]*flux_sample_param[i][j]
	flux_fit.append(a)


#Models, Probabilities and MCMC based on http://dan.iel.fm/george/current/user/model/
######################################################
#
#	MODELS: TRANSIT + SIGNAL + BASE
#
######################################################


def model_transit(params, t):
	rp,a,inc =params

	#From http://astro.uchicago.edu/~kreidberg/batman/quickstart.html
	params = batman.TransitParams()
	params.t0 = 0.                       #time of inferior conjunction
	params.per = 0.78884                 #orbital period
	params.rp = rp 	                     #planet radius (in units of stellar radii)
	params.a = a 	                     #semi-major axis (in units of stellar radii)
	params.inc = inc 	             #orbital inclination (in degrees)
	params.ecc = 0.                      #eccentricity
	params.w = 90.                       #longitude of periastron (in degrees)
	params.u = [0.1, 0.3]                #limb darkening coefficients
	params.limb_dark = "quadratic"       #limb darkening model

	m = batman.TransitModel(params, t)    
	bat_flux = np.log(m.light_curve(params))
    	return bat_flux

def model_signal(params, t):
	signal_sum=np.zeros(t.size)
	for i in range(params.size):
		signal_sum= signal_sum+signals[i]*params[i]
    	return signal_sum

def model1(base_flux, transit_par, signal_par,t):
	return base_flux + model_signal(signal_par,t)+ model_transit(transit_par, t)

######################################################
#
#	PRIOR - LIKELIHOOD - POSTERIOR
#
######################################################

def ReturnParams(p):
	base_flux=p[0]
	transit_par=np.array([p[1],p[2],p[3]])
	signal_par=np.array(p[4:p.size])
	return base_flux,transit_par,signal_par

def lnlike1(sig,base_flux, transit_par, signal_par, t, y,index):
	m=model1(base_flux,transit_par,signal_par,t)
	return -0.5 * np.sum(((y[index] - m[index])/sig) ** 2)

def lnprior1(base_flux, transit_par, signal_par):
	rp,a,inc= transit_par
	signal_inrange=True
	for i in range(signal_par.size):
		if(signal_par[i]<-0.1 or signal_par[i]>0.1):
			signal_inrange=False
    	if (-0.01 < base_flux < 0.01 and  0.01 < rp < 0.5 and 1.0 < a < 30.0 and 50 < inc < 90 and signal_inrange):
       		return 0.0
    	return -np.inf

def lnprob1(p, t, y,sigma,index):
	#sigma=0.0001
	
	base_flux,transit_par,signal_par=ReturnParams(p)
    	lp = lnprior1(base_flux,transit_par,signal_par)
    	return lp + lnlike1(sigma,base_flux,transit_par,signal_par, t, y,index) if np.isfinite(lp) else -np.inf

######################################################
#
#	MONTECARLO MARKOV CHAIN (K-FOLD VALIDATION)
#
######################################################

def MCMC(nwalkers,iter_fac,num_signals, t, y, sigma, index):
	data=[t,y,sigma,index]
	#data=[time,flux]

	#sigma, floor_flux, rp,a,i, signal_params
	initial=[0, 0.1, 8.0, 85.0]
	delta_steps=[1e-4,1e-3,1e-1,1e-2]
	for i in range(num_signals):
		initial.append(0.0)
		delta_steps.append(1e-2)

	initial=np.array(initial)
	delta_steps=np.array(delta_steps)
	ndim = len(initial)

	p0 = [np.array(initial) + np.multiply(delta_steps, np.random.randn(ndim))
	      for i in xrange(nwalkers)]
	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob1, args=data)
	
	p0, _, _ = sampler.run_mcmc(p0, 500*iter_fac)
	sampler.reset()
	sampler.run_mcmc(p0, 1500*iter_fac)
	samples = sampler.flatchain
	return samples

Walkers=42
Iter=1
sigma=0.0001

DoKFold=True
opt_nsignal=3
if DoKFold:
	kf = KFold(time.size, n_folds=4)
	MSE=[]
	for nsignal in range(NCOMP+1):
		print nsignal
		MSE.append(0.0)
		for train_index, test_index in kf:
			samples_train=MCMC(Walkers,Iter,nsignal,time,flux_target,sigma,train_index)
			base_flux, transit_par, signal_par=ReturnParams(np.average(samples_train,axis=0) )

			test_model=model1(base_flux,transit_par,signal_par, time)
			MSE[nsignal]= MSE[nsignal]+ np.sum(np.square(test_model-flux_target)[test_index])
	MSE=np.array(MSE)/NFEAT
	print MSE
	opt_nsignal= np.argmin(MSE)
	print opt_nsignal


samples= MCMC(Walkers*2,Iter*2,opt_nsignal,time,flux_target,sigma,np.arange(NFEAT))
SquareResiduals=[]
for i in range(len(samples)):
	c,trans,sigpar=ReturnParams(samples[i])
	SquareResiduals.append(np.sum(np.square(model1(c,trans,sigpar,time)-flux_target)))

MCMC_estimator=samples[np.argmin(SquareResiduals)]
MCMC_variance= np.min(SquareResiduals)/float(NFEAT-1)

print MCMC_estimator
print MCMC_variance

######################################################
#
#	PLOTS
#
######################################################

plt.figure(0)
plt.title("Average Samples Flux ")
plt.xlabel("Time (days)")
plt.ylabel("Log Flux")
plt.plot(time,flux_average)

plt.figure(1)
plt.title("Flux Target")
plt.xlabel("Time (days)")
plt.ylabel("Log Relative Flux")
plt.plot(time,flux_target)

plt.figure(2)
plt.title('Principal Components')
for i in range(NCOMP):
	plt.subplot("32"+str(i+1))
	if i>=4:
		plt.xlabel("Time (days)")
	if i==0 or i==2 or i==4:
		plt.ylabel("Log Flux (Normalized)")


	plt.title('Signal '+str(i+1)+ ' - Explained Variance: '+str(np.round( pca.explained_variance_ratio_[i]*100,1  ))+'%' )
	plt.plot(time,signals[i], label='VarExp: '+str(pca.explained_variance_ratio_[i]))
#plt.legend()

plt.figure(3)
plt.title('Samples(Red) - PCA Fit (Blue)')
for i in range(NSAMP):
	plt.subplot("33"+str(i+1))	
	if i>=6:
		plt.xlabel("Time (days)")
	if i==0 or i==3 or i==6:
		plt.ylabel("Log Relative Flux")
	plt.plot(time,flux_comp[i],'r-')
	plt.plot(time,flux_fit[i],'b-')


plt.figure(4)
plt.title("MCMC Estimation")
plt.xlabel("Time (days)")
plt.ylabel("Log Relative Flux")

for s in samples[np.random.randint(len(samples), size=1000)]:
	base_flux, transit_par, signal_par=ReturnParams(s)
    	plt.plot(time, model1(base_flux,transit_par,signal_par, time), color="#4682b4", alpha=0.3)

base_flux, transit_par, signal_par=ReturnParams(MCMC_estimator)
plt.plot(time,flux_target,'k-',label='Star Flux')
plt.plot(time,model1(base_flux,transit_par,signal_par, time),'r-',label='Model Flux')
plt.legend()



corner.corner(samples, labels=["c", "rp","a","inc","a1","a2","a3"],truths=MCMC_estimator)



plt.show()
