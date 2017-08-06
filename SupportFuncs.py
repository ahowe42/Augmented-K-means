'''
Data support functions for clustering modeling.

Copyright (C) 2016 J. Andrew Howe; see below
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.ion()

def ScatterGrpd(data, labels, names = None):
	'''
	Draw either a 2d or 3d scatter plot of grouped data, with each
	group using a different color & marker. If cluster names are passed
	in, these will be annotated on the plot; otherwise the group labels
	will be annotated - either way at the cluster means.
	-----
	data = nxp array of data, where p is either 2 or 3-p
	labels = n-length array_like of cluster labels, labeled as 0,1,...
	names = (optional) k-length array_like of cluster names
	'''
	uniK = np.unique(labels)
	p = data.shape[1]
	
	# prepare cluster "names" if none passed
	if names is None:
		names = ['Cluster %d'%k for k in uniK]
	
	# setup for the different colors & markers
	col = 'bgrmk'; col_num = len(col)
	mrk = '.s*o^+<x>vd'; mrk_num = len(mrk)
	# modularly cycle through them
	colmrk_use = [(col[k%col_num],mrk[k%mrk_num]) for k in range(len(uniK))]
	
	# plot
	fig = plt.figure()
	if p == 3:
		ax = fig.add_subplot(111, projection='3d')
	plt.hold('on')
	for k in uniK:
		# get the indices of this cluster & the means
		ind = (labels == k)
		mn = np.mean(data[ind,:],axis=0)
		# scatter plot & labels cluster centers
		if p == 2:
			plt.scatter(data[ind,0],data[ind,1], color=colmrk_use[k][0], marker=colmrk_use[k][1])
			plt.gca().text(mn[0],mn[1],'*'+names[k])
		elif p == 3:
			ax.scatter(data[ind,0],data[ind,1],data[ind,2], color=colmrk_use[k][0], marker=colmrk_use[k][1])
			ax.text(mn[0],mn[1],mn[2],'*'+names[k])
	plt.axis('tight')
	plt.hold('off')

def LoadClustData(datafile, datatype):
	''' Prepare data for modeling - either real or simulated.
	If real (0), the first column of datafile will be used as labels,
	which should be labeled as 0,1,2,.... If simulated (1), the file
	structure should be:
	row 1:        k followed by (p-1) 0s tab separated
	row 2:        tab separated mean vector
	row 3-p+2:    tab separated covariance matrix
	row p+3:      beta tab mixing proportion
	must have (k-1)*(p+2) additional rows in the same format.	
	Either way, the file must be tab delimited with no headers.
	-----
	datafile = string filename to import
	datatype = scalar 0=real, 1=simulated
	-----
	if datatype = 0, returns
		data = nxp array of simulated data
		labels = n-length array of cluster labels
	else
		mixpropors = k-length array of mixing proportions (must sum to 1)
		betas = k-length array of beta shape parameters
		meanvecs = kxp matrix of k p-length mean vectors
		covrmats = kxpxp array of k pxp covariance matrices
	'''
	# get the data
	datinput = np.loadtxt(datafile)
	
	if datatype == 0:
		# real data
		labels = datinput[:,0]
		data = datinput[:,1:]
		# wow, that was easy, ... done!
		return data, labels
	else:
		# simulated data
		K = int(datinput[0,0])		# number clusters
		p = datinput.shape[1]		# number dimensions
		rng = range(1,K*(p+2),p+2)	# bounds through cluster parameter sets
		# pre-allocate the vectors of simulation parameters
		meanvecs = np.zeros((K,p),dtype=float)
		covrmats = np.zeros((K,p,p),dtype=float)		
		# fill in the mixing proportions & betas
		betas = datinput[rng,0]
		mixpropors = datinput[rng,1]/np.sum(datinput[rng,1])
		# get the mean vectors & covariance matrices
		for i,r in enumerate(rng):
			# could easily do this with list comprehension ...
			meanvecs[i,:] = datinput[r+1,:]
			# ... but would then need another list comprehension here
			covrmats[i,:] = datinput[r+2:(r+p+2),:]
		# done!
		return mixpropors, betas, meanvecs, covrmats


def SimClustData(n, mixpropors, betas, meanvecs, covrmats, Scatter=False):
	'''
	This will simulate n observations from a multivariate PE distribution
	in multiple clusters using the parameters passed. If the passed n &
	mixing proportions results in a non-integer number of observations in
	a cluster, that clusters n is rounded.
	-----
	n = integer total number observations
	mixpropors = k-length array_like of mixing proportions (must sum to 1)
	betas = k-length array_like of beta shape parameters
	meanvecs = kxp matrix of k p-length mean vectors
	covrmats = kxpxp array of k pxp covariance matrices
	scatter = (optional) boolean True = create a grouped scatter plot;
		False = don't (only possible if p = 2 or 3)
	-----
	data = nxp array of simulated data
	labels = n-length array of cluster labels, labeled as 0,1,...
	'''
	# get number of observations per cluster & create the labels
	nks = np.asarray(np.round(n*mixpropors),dtype=int)
	K = len(mixpropors)
	labels = np.repeat(range(K),nks)
	
	# do the simulating
	data = np.zeros((0,meanvecs.shape[1]),dtype=float)	# have to init as a (0,p) array for vstack to work
	for k in range(K):
		data = np.vstack((data,MVPexpRnd(nks[k],betas[k],meanvecs[k,:],covrmats[k,:,:])))
		
	# scatter plot if allowed and requested
	p = data.shape[1]
	if Scatter and ((p == 2) or (p == 3)):
		ScatterGrpd(data,labels)
	# done!
	return data,labels


def MVPexpRnd(n, beta, mu, Sigma, randseed = None):
	'''
	Generate n samples from a p-variate PE distribution, using
	the parameters passed in. p must be >=2.
	-----
	n = integer number observations
	beta = float shape parameter
	mu = p-length array_like mean vector
	Sigma = pxp covariance matrix
	randseed = (optional) integer seed for randomizer
	-----
	data = nxp array of observations simulated from the PE distribution
	'''
	# set the randomizer seed if requested
	if randseed is not None:
		rnd.seed(randseed)	
	p = len(mu)
	# Generate n-by-p random points uniformly distributed on
	# the surface of a hypersphere of p dimension
	a = np.random.randn(n,p)
	a1 = np.sqrt(np.sum(a**2,axis=1,keepdims=True))
	Unif = a / a1	# divide each observation by the square root of its norm

	# Generate n Gamma variates with shape param p/2/B and scale param 2
	# then scale to the power of 1/2/B
	Gama = np.random.gamma(shape = p/2/beta, scale = 2, size = (n,1))**(1/2/beta)
	
	# Cholesky Factorization of Sigma
	A = np.linalg.cholesky(Sigma);

	# Generate the PE matrix - this makes heavy use of
	# numpy's ability to broadcase together arrays
	return np.dot(A.T,Unif.T).T*Gama + mu


def ClustsAlign(trues, estim, ks):
	'''
	A problem in clustering is that the cluster labels from an algorithm can be
	different than the real cluster labels, i.e. [2,0,1] instead of [0,1,2].
	This attempts to edit the estimated cluster labels so that they align with
	the true labels.
	-----
	trues = n-length array-like of true labels
	estim = n-length array-like of estimated labels
	ks = array_like unique values in trues
	-----
	new_estim = n-length array-like of aligned estimated labels
	'''
	K = len(ks)					# number unique true labels
	uniK = np.unique(estim)		# unique estimated labels
	# save a copy that we can mess around with
	new_estim = estim + 1000
	# loop through each estimated cluster and figure out the majority true cluster
	for k in uniK:
		tru_est = trues[estim == k]
		# get the frequency with which each label is present
		preps = [np.sum(tru_est == k)/len(tru_est) for k in ks]
		truth = max(preps)
		truthk = ks[preps.index(truth)]
		# we need to change this k in estim for a new k, since the majority in trues
		# is different, with a higher frequency then even
		if (truth > 1/K) and (truthk != k):
			new_estim[estim==k] = 1000+truthk
	return new_estim - 1000

def SimandScat(datafile,n):
	'''
	This simply called LoadClustData, then SimClustData with the plot flag on.
	It's a helper function for lazy people like myself :-).
	'''
	pis,betas,mus,sigmas = LoadClustData(datafile,1)	
	data,labels = SimClustData(n,pis,betas,mus,sigmas,(len(mus[0])==2) or (len(mus[0])==3))
	return data,labels
	
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
