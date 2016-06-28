import numpy as np
import matplotlib.pyplot as plt

from numpy import linalg
from sklearn.decomposition import PCA
from sklearn.cross_validation import KFold

import batman
import george
import emcee

######################################################
#
#	GLOBALS
#
######################################################

NCOMP=5
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
flux_param= pca.fit_transform(flux_comp)
signals=pca.components_

flux_fit=[]
for i in range(NSAMP):
	a=np.zeros(NFEAT)
	for j in range(NCOMP):
		a=a+ signals[j]*flux_param[i][j]
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
	signal_sum=np.zeros(NFEAT)
	for i in range(params.size):
		signal_sum=signals[i]*params[i]
    	return signal_sum

def model1(base_flux, transit_par, signal_par,t):
	return base_flux + model_signal(signal_par,t)+ model_transit(transit_par, t)

######################################################
#
#	PRIOR - LIKELIHOOD - POSTERIOR
#
######################################################

def ReturnParams(p):
	sigma=p[0]
	base_flux=p[1]
	transit_par=np.array([p[2],p[3],p[4]])
	signal_par=np.array(p[5:p.size])
	return sigma,base_flux,transit_par,signal_par

def lnlike1(sig,base_flux, transit_par, signal_par, t, y):
	m=model1(base_flux,transit_par,signal_par,t)
	return -0.5 * np.sum(((y - m)/sig) ** 2)

def lnprior1(sig,base_flux, transit_par, signal_par):
	rp,a,inc= transit_par
	signal_inrange=True
	for i in range(signal_par.size):
		if(signal_par[i]<-20.0 or signal_par[i]>20.0):
			signal_inrange=False
    	if (0.00005<sig<0.002 and -0.01 < base_flux < 0.01 and  0.01 < rp < 0.5 and 0.01 < a < 15 and 50 < inc < 90 and signal_inrange):
       		return 0.0
    	return -np.inf

def lnprob1(p, t, y):
	#p[0]=0.0001	#Fixing sigma to the known value
	sigma,base_flux,transit_par,signal_par=ReturnParams(p)
    	lp = lnprior1(sigma,base_flux,transit_par,signal_par)
    	return lp + lnlike1(sigma,base_flux,transit_par,signal_par, t, y) if np.isfinite(lp) else -np.inf

######################################################
#
#	MONTECARLO MARKOV CHAIN
#
######################################################

nwalkers=56
data=[time,flux_target]

fold=5

num_sigpar=5
kf = KFold(time.size, n_folds=fold)
MSE=[]

#sigma, floor_flux, rp,a,i, signal_params
initial=[0.0001,0, 0.1, 8.0, 85.0]
delta_steps=[1e-5,1e-4,1e-3,1e-3,1e-2]
for i in range(num_sigpar):
	initial.append(0.0)
	delta_steps.append(1e-2)

initial=np.array(initial)
delta_steps=np.array(delta_steps)
ndim = len(initial)

p0 = [np.array(initial) + np.multiply(delta_steps, np.random.randn(ndim))
      for i in xrange(nwalkers)]

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob1, args=data)
p0, _, _ = sampler.run_mcmc(p0, 500)
sampler.reset()
sampler.run_mcmc(p0, 2000)
samples = sampler.flatchain
print samples[-1]


######################################################
#
#	PLOTS
#
######################################################

plt.figure(0)
plt.title("Average Samples Flux ")
plt.xlabel("Time")
plt.ylabel("Flux (Logscale)")
plt.plot(time,flux_average)

plt.figure(1)
plt.title("Flux Target")
plt.xlabel("Time")
plt.ylabel("Relative Flux (Logscale)")
plt.plot(time,flux_target)

plt.figure(2)
plt.title('Principal Components')
plt.xlabel("Time")
plt.ylabel("Relative Flux (Logscale)")
for i in range(NCOMP):
	plt.plot(time,signals[i], label='VarExp: '+str(pca.explained_variance_ratio_[i]))
plt.legend()

plt.figure(3)
plt.title('Samples(Red) - PCA Fit (Blue)')
for i in range(NSAMP):
	plt.subplot("33"+str(i))
	plt.plot(time,flux_comp[i],'r-')
	plt.plot(time,flux_fit[i],'b-')


plt.figure(4)
for s in samples[np.random.randint(len(samples), size=64)]:
	sigma,base_flux, transit_par, signal_par=ReturnParams(s)
    	plt.plot(time, model1(base_flux,transit_par,signal_par, time), color="#4682b4", alpha=0.3)

sigma,base_flux, transit_par, signal_par=ReturnParams(samples[-1])
plt.plot(time,flux_target,'k-')
plt.plot(time,model1(base_flux,transit_par,signal_par, time),'r-')

plt.show()
