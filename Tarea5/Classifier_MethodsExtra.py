import numpy as np
from numpy import linalg
import sklearn
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier

from sklearn import svm

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from Classifier_Methods import *



def KNeigh(x,y,z,xr,yr,k_neigh):
	In=np.array([x,y]).T
	neigh = KNeighborsClassifier(n_neighbors=k_neigh)
	neigh.fit(In, z)

	Test=np.array([xr,yr]).T
	return neigh.predict(Test)


def SVC(x,y,z,xr,yr):
	In=np.array([x,y]).T
	clf = svm.SVC(kernel='poly',degree=2)
	clf.fit(In,z)
	
	Test=np.array([xr,yr]).T
	return clf.predict(Test)


def LassoQuadratic(x,y,z,xr,yr):
	M=CoeffMatrix(x,y,2)
	beta=linalg.lstsq(M,z)[0]
	

	b0=np.average(z)
	t=np.average(np.fabs(beta[np.arange(1,beta.size)]))
	In=np.array([x,y,np.square(x),np.square(y),np.multiply(x,y)]).T
	clf = linear_model.Lasso(alpha=0.5)
	clf.fit(In, z-b0)
	beta_lasso=[b0]
	for f in clf.coef_:
		beta_lasso.append(f)
	

	'''
	t=np.average(beta)
	clf = linear_model.Lasso(alpha=t)
	clf.fit(M, z)
	beta_lasso=clf.coef_
	'''

	return np.dot(CoeffMatrix(xr,yr,2),beta_lasso)




def Confussion_Score(z_true,z_pred,name):
	print name
	z_pred[np.where(z_pred>=1.5)[0]]=2
	z_pred[np.where(z_pred<1.5)[0]]=1
	
	print "Misclassification Rate: "+ str(1.0-accuracy_score(z_true,z_pred))
	print "Confusion Matrix:"
	print confusion_matrix(z_true, z_pred)



def BayesClassifier(Pi_1,Pi_2,mu_1,mu_2,cov_1,cov_2,xr,yr):
	zr=np.ones(xr.size)
	Delta_1= Discriminant_QDA(np.array([xr,yr]), Pi_1,mu_1,cov_1)
	Delta_2= Discriminant_QDA(np.array([xr,yr]), Pi_2,mu_2,cov_2)
	zr[np.where((Delta_2-Delta_1)>=0.0)[0]]=2
	return zr	
	
	

