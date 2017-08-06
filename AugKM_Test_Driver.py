'''
Compare the performance of Augmented K-means++ against K-means++
  
Copyright (C) 2016 J. Andrew Howe; see below
'''
import datetime as dat
import numpy as np
# get my clustering modeling data support functions
from SupportFuncs import *
# get the actual augmented k-means algorithm
from AugKmeansPP import *

# real / simulated data parameters
datafile = input('Please enter the name of the data file (include path if not cwd): ')
datatype = int(input('Simulated data (1) or Real data (0)? '))
if datatype == 1:
	n = int(input('How many simulated observations? '))

if datatype == 1:
	pis, betas, mus, sigmas = LoadClustData(datafile,type)
	ks = np.arange(len(pis))
else:
	data,labels = LoadClustData(datafile,datatype)
	n = data.shape[0]
	ks = np.unique(labels)

# K-means parameters
Khat = len(ks)
convgcrit = 0.00001
maxiters = 100

# Monte Carlo simulation
MCsims = 1000
printgran = 10
MCres = np.zeros((MCsims,8),dtype=float)
arr_res = []
for sim in range(MCsims):
	# talk a little, maybe
	if sim%printgran == 0:
		print('Sim %d of %d'%(sim+1,MCsims))
		
	# get the data, maybe
	if datatype == 1:
		data,labels = SimClustData(n, pis, betas, mus, sigmas)
	
	# set the seed
	tim = dat.datetime.now()
	randseed = tim.hour*10000+tim.minute*100+tim.second+tim.microsecond
	
	# initialize the centroids
	clust_means = KPPInit(data,Khat,randseed)

	# run Kmeans
	nw = dat.datetime.now()
	clustsK,SK,jnk,jnk = AugKMeans(data,Khat,convgcrit,maxiters,clust_means,AugMe = False)
	MCres[sim,3] = (dat.datetime.now() - nw).microseconds
	MCres[sim,0] = np.sum(ClustsAlign(labels,clustsK,ks) == labels)/n
	MCres[sim,1] = len(SK) 
	MCres[sim,2] = SK[-1]

	# run augmented Kmeans
	nw = dat.datetime.now()
	clustsA,SA,jnk,Excl = AugKMeans(data,Khat,convgcrit,maxiters,clust_means,AugMe = True)
	MCres[sim,7] = (dat.datetime.now() - nw).microseconds
	MCres[sim,4] = np.sum(ClustsAlign(labels,clustsA,ks) == labels)/n
	MCres[sim,5] = len(SA)
	MCres[sim,6] = SA[-1]
	
	# talk a little, maybe
	if sim%printgran == 0:
		print('\tClassification winner = %s'%(('Reg.','Aug.')[MCres[sim,4]>MCres[sim,0]]))
		
	# save the other arrays
	arr_res.append([SK,SA,Excl])

# summarize results
print('%s data used: %s (n=%d), %d MC sims'%(['Real','Simulated'][datatype],datafile,n,MCsims))
# compare the classification rates
crates_bst = np.argmax(MCres[:,[0,4]],axis=1) # better than
crates_beq = MCres[:,4] >= MCres[:,0]		# not worse than
# compare the number iterations
iters_bst = np.argmin(MCres[:,[1,5]],axis=1)	# better than
iters_beq = MCres[:,5] <= MCres[:,1]		# not worse than
# compare the final SSE
errs_bst = np.argmin(MCres[:,[2,6]],axis=1) 	# better than
errs_beq = MCres[:,6] <= MCres[:,2]		# not worse than
# compare the execution times
time_bst = np.argmin(MCres[:,[3,7]],axis=1)	# better than
time_beq = MCres[:,7] <= MCres[:,3]		# not worse than

# talk
print('Augmented Better Frequencies\n\tClassification: %0.2f%%\n\tIterations: %5.2f%%\n\tErrors: %0.2f'\
	%(100*np.sum(crates_bst==1)/MCsims,100*np.sum(iters_bst==1)/MCsims,100*np.sum(errs_bst==1)/MCsims))
print('Augmented Better or Equal Frequencies\n\tClassification: %0.2f%%\n\tIterations: %5.2f%%\n\tErrors: %0.2f'\
	%(100*np.sum(crates_beq==1)/MCsims,100*np.sum(iters_beq==1)/MCsims,100*np.sum(errs_beq==1)/MCsims))
# mean performance when Augmented is better / worst
clrts = np.mean(MCres[crates_bst==1,4] - MCres[crates_bst==1,0])
iters = np.mean(MCres[iters_bst==1,1] - MCres[iters_bst==1,5])
print('When Augmented is Better, Avg. Benefit:\n\tClassification = %0.2f%%\n\tIterations = %0.2f'\
	%(100*clrts,iters))	
clrtsw = np.mean(MCres[crates_beq==0,0] - MCres[crates_beq==0,4])
itersw = np.mean(MCres[iters_beq==0,5] - MCres[iters_beq==0,1])
print('When Augmented is Worse, Avg. Detriment:\n\tClassification = %0.2f%%\n\tIterations = %0.2f'\
	%(100*clrtsw,itersw))	
# mean performances
mns = np.mean(MCres,axis=0)
print('     K-means Average: Class. Rate: %0.2f%%, Num. Iters.: %5.2f, Convg. Err.: %0.2f, Exec. Time (s): %0.2f'\
    %(100*mns[0],mns[1],mns[2],mns[3]/100000))
print('Aug. K-means Average: Class. Rate: %0.2f%%, Num. Iters.: %5.2f, Convg. Err.: %0.2f, Exec. Time (s): %0.2f'\
    %(100*mns[4],mns[5],mns[6],mns[7]/100000))
				
# talk about how many excluded observations in augmentation
#excls = [sum(r[2]) for r in arr_res]

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
