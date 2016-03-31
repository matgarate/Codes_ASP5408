import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

#In these poll simulator people can pick the option 0 or the option 1, both with equal probability.
#We simulate PollNum polls, each with PersonNum persons answering. 
#We compare the final count with a binomial distribution with B(N=PersonNum,p=0.5)

PollNum=1000	#Number of polls simulated
PersonNum=33	#Number of persons per poll

Desired_X=18	#In how many polls, "Desired_X people" picked the option 1?

#PollList[poll_id][person_id]
PollList=np.random.randint(2,size=[PollNum,PersonNum])

#Sum the values of each poll. The sum gives us how many persons picked the answer with value of 1.
PollResults=np.sum(PollList,axis=1)
Histogram=np.histogram(PollResults,range=(0,PersonNum+1),bins=PersonNum+1,normed=True)

Binomial=binom(PersonNum,0.5)

plt.title("SyntheticPoll\n NPolls: "+str(PollNum)+" - NPersons: "+str(PersonNum))
plt.xlabel("People who picked the option 1 (Num)")
plt.ylabel("Poll Fraction (%)")
plt.xlim([0,33])

AnswerCount=np.arange(PersonNum+1)
plt.plot(AnswerCount,Histogram[0],'bo-',label='Synthetic Distribution')
plt.plot(AnswerCount,Binomial.pmf(AnswerCount),'r--',label='Binomial PMF')
plt.legend()

print "In the "+str(Histogram[0][Desired_X]*100.0)+" % of the polls, "+ str(Desired_X)+" picked the option 1."


plt.show()
