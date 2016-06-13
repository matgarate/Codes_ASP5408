import numpy as np
import scipy
import sklearn

from sklearn.neighbors import KNeighborsClassifier



def KNeigh(x,y,z,xr,yr,k_neigh):
	In=np.array([x,y]).T
	neigh = KNeighborsClassifier(n_neighbors=k_neigh)
	neigh.fit(In, z)

	Test=np.array([xr,yr]).T
	return neigh.predict(Test)
