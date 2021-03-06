import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

PersonNum=33	#Number of persons per poll
DataX=18	#In how many polls, "Desired_X people" picked the option 1?



size=1000
r=np.linspace(0,1,num=size)
dr=1.0/float(size)

#P(X|r)= Binomial Distribution
PX_r= binom.pmf(DataX,PersonNum,r)

#Probability P(X) of finding DataX without assuming a value of r 
#(This is the Riemann integration for the denominator in the Bayes Theorem)
PX= np.sum(PX_r*dr)

#P(r|x=18)
Pr_X= PX_r/PX

#Check the normalization integrating over r
Norm_Pr_X=np.sum(Pr_X*dr)
Max_r= r[np.argmax(Pr_X)]


CDF=[]
for i in xrange(size):
	CDF.append(np.sum(Pr_X[0:i+1]*dr))

print "At r=0.5, CDF(0.5) = "+str(round(CDF[size/2],2))


plt.figure(1)
plt.title("PDF of P(r|X=18)\nNorm is "+ str(Norm_Pr_X)+"\nMaximum is at r: "+str(np.round(Max_r,2)))
plt.xlabel("r")
plt.ylabel("P(r|X)")
plt.plot(r,Pr_X)

plt.figure(2)
plt.title("CDF of P(r|X=18)")
plt.xlabel("r")
plt.ylabel("CDF(r|X)")
plt.plot(r,CDF)


plt.show()




