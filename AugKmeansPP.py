'''
Implement the Augmented k-means algorithm as described in the arXiv article #1705.07592
Improved Clustering with Augmented k-means, by J. Andrew Howe

Copyright (C) 2016 J. Andrew Howe; see below
'''
import numpy as np
import numpy.random as rnd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression


def AugKMeans(data, Khat, convgcrit, maxiters, clust_means = None, AugMe = True, randseed = None):
	'''
	Augmented K-means, wherein the k-means agorithm (initialized
	with k-means++) is augmented by an extra step to fit a logistic
	regression model that's used to exclude certain observations from
	being used to compute the cluster centroids for the next iteration.
	-----
	data = nxp array of data
	Khat = integer number of groups to find in data
	convgcrit = float convergence criteria used to compare successive
		values of the total sum of squared distances
	maxiter = inter maximum number iterations Aug Kmeans can consume
	AugMe = (optional) boolean flag True = Augment, False = don't
	clust_means (optional) Khatxp array of Khat p-length initial cluster means
	randseed = (optional) integer seed for randomizer - only used for seeding,
		so if you pass clust_means this is not used
	-----
	clusts = n-length array of cluster labels, labeled as 0,1,...
	sumdist = array of total sum of squared distances for each iteration
	Excld = list of boolean selection arrays of observations excluded
		by logistic regression Augmentation in each iteration
	'''
	# initialize stuff
	n = data.shape[0]
	ks = range(Khat)
	iters = 0
	Sumdist = []
	excld = []
	if AugMe:
		LR = LogisticRegression()
	
	if clust_means is None:
		# initialize centroids with Kmeans++ initialization
		clust_means = KPPInit(data,Khat,randseed)	
	cents_hist = [clust_means]
		
	# main driver loop
	while True:
		iters += 1
		# initialize more stuff
		clusts = np.zeros(n,dtype=int)			# cluster numbers
		clust_dists = np.zeros(n,dtype=float)	# distance from each obs. to its clust mean
		# cluster assignment & squared distance calculation
		for i in range(n):
			# compute all dists
			dists = [SqDist(data[i,:],m) for m in clust_means]
			# get the minimum distance
			clust_dists[i] = min(dists)
			# find the min and assign the cluster
			clusts[i] = dists.index(clust_dists[i])
		
		if AugMe:
			augpred = LR.fit(X = data, y = clusts).predict_proba(X = data)
			# for each observation, get the 2 largest cluster probabilities ...
			bigs = [a[np.argsort(a)[-2:]] for a in augpred]
			# ... and compute their ratio; if it is <=1.5, this observation is not solidly classified
			keepme = np.array([b[1]/b[0]>1.5 for b in bigs])
			excld.append(~keepme)
		else:
			keepme = np.ones(n,dtype=bool)
	 
		# recompute the means *excluding* the mismatched observations
		clust_means = np.tile(np.mean(data,axis=0,keepdims=True),(Khat,1))
		for k in ks:
			if np.sum((clusts==k) & keepme)==0:
				# no certainty on this cluster, so just set it's mean to the global data average
				pass 
			else:
				clust_means[k,:] = np.mean(data[(clusts==k) & keepme,:],axis=0)
		cents_hist.append(clust_means)

		# get the sum
		Sumdist.append(sum(clust_dists))
		# now test, if we can
		if ((len(Sumdist) > 1) and (abs(Sumdist[-1]-Sumdist[-2]) < convgcrit)) or (iters > maxiters):
			break
	
	return clusts, Sumdist, cents_hist, excld


def SqDist(x1,x2):
	'''
	squared Euclidian distance - could be changed to something else
	'''
	return sum((x1 - x2)**2)


def KPPInit(data, Khat, randseed = None):
	'''
	Initializer for KMeans++ from
	Arthur, D. and Vassilvitskii, S. (2007). "k-means++: the advantages of careful seeding"
	Proceedings of the eighteenth annual ACM-SIAM symposium on Discrete algorithms.
	Society for Industrial and Applied Mathematics Philadelphia, PA, USA. pp. 1027–1035.
	-----
	data = nxp array of data
	Khat = (optional) integer number of groups to find in data
	randseed = (optional) integer seed for randomizer
	-----
	centroids = Khatxp array of Khat p-length cluster centroids
	'''
	# Kmeans++ initialization
	(n,p) = data.shape
	# setup to hold centers	
	centroids = np.zeros((Khat,p),dtype=float)
	# randomly select the first centroid
	if randseed is not None:
		rnd.seed(randseed)
	centroids[0,:] = rnd.permutation(data)[0,:]
	# iteratively add centroids
	for ks in range(1,Khat):
		dists = np.zeros(n,dtype=float)
		for ind,dat in enumerate(data):
			# get the squared distance to the nearest centroid
			dists[ind] = min([SqDist(dat,centroids[r,:]) for r in range(ks)])
		# now sample from the observations proportional to the
		# squared distances (further = more likely)
		# first: convert to cumulative probabilities which are upper bounds for bins
		cumprobs = np.cumsum(dists/np.sum(dists))
		# second: get a random uniform variate and figure out where it falls
		seld = (rnd.rand() <= cumprobs)
		# third: the selected sample corresponds to the first upper bound > than the random
		centroids[ks,:] = data[seld,:][0,:]
	# these *are* the (cent)roids you're looking for
	return centroids

'''
Copyright (C) 2016 J. Andrew Howe

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''
