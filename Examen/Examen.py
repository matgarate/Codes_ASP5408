import numpy as np
import matplotlib.pyplot as plt

from numpy import linalg
from sklearn.decomposition import PCA
from sklearn.cross_validation import KFold

import batman
import george
import emcee



NCOMP=3
NSAMP=9
NFEAT=100

time, flux_target= np.loadtxt("dataset_7.dat",usecols=(0,1),unpack=True,skiprows=1)
flux_comp= np.loadtxt("dataset_7.dat",usecols=(2,3,4,5,6,7,8,9,10),unpack=True,skiprows=1)

#CONVERT TO LOGSCALE
time= time/24.0
flux_target=np.log(flux_target)
flux_comp=np.log(flux_comp)



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

#SUBSTRACT MEAN FLUX
flux_average=np.average(flux_comp,axis=0)
for i in range(NSAMP):
	flux_comp[i]=flux_comp[i]-flux_average
flux_target=flux_target-flux_average






#Based on http://dan.iel.fm/george/current/user/model/
def model1(params, t):
	c,rp,a,i, p1,p2,p3= params

	#based on http://astro.uchicago.edu/~kreidberg/batman/quickstart.html

	params = batman.TransitParams()
	params.t0 = 0.                       #time of inferior conjunction
	params.per = 0.78884                 #orbital period
	params.rp = rp 	                     #planet radius (in units of stellar radii)
	params.a = a 	                     #semi-major axis (in units of stellar radii)
	params.inc = i 	                     #orbital inclination (in degrees)
	params.ecc = 0.                      #eccentricity
	params.w = 90.                       #longitude of periastron (in degrees)
	params.u = [0.1, 0.3]                #limb darkening coefficients
	params.limb_dark = "quadratic"       #limb darkening model

	m = batman.TransitModel(params, t)    
	bat_flux = np.log(m.light_curve(params))

	signal_sum=np.zeros(NFEAT)
	signal_par=[p1,p2,p3]
	for i in range(NCOMP):
		signal_sum=signals[i]*signal_par[i]
    	return c + signal_sum + bat_flux
def lnlike1(p, t, y, yerr):
    return -0.5 * np.sum(((y - model1(p, t))/yerr) ** 2)
def lnprior1(p):
    c,rp,a,i, p1,p2,p3 = p
    if (-0.01 < c < 0.01 and  0.01 < rp < 0.5 and 0.01 < a < 15 and 50 < i < 90 and -2 < p1 < 2 and -2 < p2 < 2 and -2 < p3 < 2):
        return 0.0
    return -np.inf
def lnprob1(p, t, y, yerr):
    lp = lnprior1(p)
    return lp + lnlike1(p, t, y, yerr) if np.isfinite(lp) else -np.inf


nwalkers=42
data=[time,flux_target,0.0001]
initial = np.array([0, 0.1, 8.0, 87.0, 0.0,0.0,0.0])
ndim = len(initial)
p0 = [np.array(initial) + 1e-3 * np.random.randn(ndim)
      for i in xrange(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob1, args=data)

print("Running burn-in...")
p0, _, _ = sampler.run_mcmc(p0, 500)
sampler.reset()
print("Running production...")
sampler.run_mcmc(p0, 1000)


#GRAFICOS PCA
plt.figure(0)
plt.title("Flux Target")
plt.xlabel("Time")
plt.ylabel("Relative Flux (Logscale)")
plt.plot(time,flux_target)

plt.figure(1)
plt.title("Average Samples Flux ")
plt.xlabel("Time")
plt.ylabel("Relative Flux (Logscale)")
plt.plot(time,flux_average)


plt.figure(2)
plt.title('Principal Components')
plt.xlabel("Time")
plt.ylabel("Relative Flux (Logscale)")
for i in range(NCOMP):
	plt.plot(time,signals[i], label='VarExp: '+str(pca.explained_variance_ratio_[i]))
plt.legend()

plt.figure(3)
plt.title('Samples(Red) -PCA Fit (Blue)')
for i in range(NSAMP):
	plt.subplot("33"+str(i))
	plt.plot(time,flux_comp[i],'r-')
	plt.plot(time,flux_fit[i],'b-')


plt.figure(4)
samples = sampler.flatchain
for s in samples[np.random.randint(len(samples), size=64)]:
    plt.plot(time, model1(s, time), color="#4682b4", alpha=0.3)

plt.plot(time,flux_target,'k-')
plt.plot(time,model1(samples[-1], time),'r-')



plt.show()
