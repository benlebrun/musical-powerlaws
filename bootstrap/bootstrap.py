import numpy as np
from math import *
from random import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from distributions import Exponential, Lognormal, Stretched_exponential, Powerlaw_with_cutoff, Powerlaw, Positive_lognormal

def KS(S, P):
	''' KS statistic '''
	assert len(P) == len(S)
	return max(list(map(lambda x: abs(S[x]-P[x]), range(0,len(P)))))

def bootstrap(dist, semi_parametric=True, niter=1e3, quiet=False, dest=False):
	'''
	Implementaion of bootstrap goodness-of-fit for heavy-tailed distributions.
	Based on plpva.py, see http://tuvalu.santafe.edu/~aaronc/powerlaws/.

	Parameters:
	- dist: the distribution object we wish to generate a gof statistic for
	- semi_parametric: if TRUE sample semiparametrically like Clauset et al. (2009) 
					   else sample all n from model distribution (parametrically)
	- niter: number of bootstrap iterations
	- quiet: whether to print some distribution statistics

	Output: csv file with statistics for each synthetic sample

	Note on parametric bootstraping: To bootstrap parametrically, the distribution being boostraped cannot have a lower bound xmin.
	This is the case since the model CDF is dependent on the value of xmin. i.e. P[X <= x] is only defined for x >= xmin. 
	So if P[X = x] where x < xmin is 0, the probability of sampling x will also be 0.  
	So to sample parametrically, the value of xmin must be set to the minimal value of the empirical distribution.
	'''
	niter = int(niter)
	if not quiet:
		print('\t Number of iterations: %i' % (niter))
		print('\t Method: %s' 
				% 'semiparametric' if semi_parametric == True else 'parametric', end='\n \n')

	X = dist.data_original
	xmin = int(dist.xmin)
	xmax = int(dist.xmax)
	means = []
	nontail = list(filter(lambda x: x<xmin, X))
	n_nontail = len(nontail)
	tail = list(filter(lambda x: x>=xmin, X))
	n_tail = len(tail)
	n = len(X)
	# following plpva.py, we generate the model distribution with length 20*empirical xmax
	mmax = xmax*20
	KS = dist.D

	for B in tqdm(range(0, niter)):
		if semi_parametric == True:
			'''
			semi-parametric method to build a synthetic distribution with similar behaviour as original data
			i.e. with probability p of sampling from the model distribution above x_min
			and 1 - p from the empirical below x_min, where p = n_tail/n
			'''
			# probability of sampling from model
			p_model = (n_tail/float(n))
			# p of sampling below xmin
			p_non = 1 - p_model

			# calculate number of samples to draw from below x_min
			n_non = 0
			for i in range(0,n):
				if random() > p_model: 
					n_non += 1
			# randomly sample from nontail data n_non times
			nondata = [nontail[int(floor(n_nontail*random()))] for i in range(0,n_non)]

			# number of samples to draw from model distribution
			n_model = n-n_non

			# get cdf of model above x_min
			cdf = dist.cdf(data=np.array([x for x in range(xmin, mmax)]))
		
			# 0's to pad front of cdf 
			# must be done to properly index: e.g. cdf[10] must be P(X <= 10)
			# note that we append xmin-1 zeros since powerlaw library returns P(X < x) not P(X <= x)
			zeros = np.zeros(xmin-1)
			# append 0's to front and 1.0 to end (since pl returns P[X < x])
			# and flatten to 1D
			cdf = np.array(sum([list(zeros), list(cdf), [1.]], []))
			#print(list(cdf))

			# build list of ps of length n_model to randomly sample
			# from cdf of model distribution
			model_p = [random() for i in range(0,n_model)]
			model_p.sort()

			# build synthetic dataset by sampling cdf 
			# the idea here is to sample index i n times 
			# where n is the number of random p's (generated above) 
			# that are less than P[X <= i] (and greater than P[X <= i-1]). 
			# we begin with i = xmin and continue until the sample is equal to
			# the empirical sample size
			c, k = 0, 0
			modeldata = []
			for i in range(xmin, mmax):
				# c-k is the number of i's to sample 
				# c is therefore the last index j with x_j = i
				while c<n_model and model_p[c]<=cdf[i]:
					c+=1
				# for k to c append i 
				for k in range(k,c):
					modeldata.append(i)
				# update index
				k=c
				# stop when n = n_model
				if k >= n_model: break

			# concatenate and flatten to one dimension
			synthetic_data = np.array(sum([nondata, modeldata], []))
			synthetic_data.sort()
		else:
			''' parametric sampling '''
			try:
				assert xmin == np.min(dist.data_original)
			except AssertionError:
				print('Invalid value for xmin. To sample parametrically, there must not be a lower bound on distribution behaviour.')
				sys.exit()

			n_model = n

			cdf = dist.cdf(data=np.array([x for x in range(xmin, mmax)]))

			model_p = [random() for i in range(0,n)]
			model_p.sort()

			# build synthetic dataset by sampling cdf 
			# the idea here is to sample index i n times 
			# where n is the number of random p's (generated above) 
			# that are less than P[X <= i] (and greater than P[X <= i-1]). 
			# we begin with i = xmin and continue until the sample is equal to
			# the empirical sample size
			c, k = 0, 0
			modeldata = []
			for i in range(xmin, mmax):
				# c-k is the number of i's to sample 
				# c is therefore the last index j with x_j = i
				while c<n_model and model_p[c] <= cdf[i]:
					c+=1
				# for k to c append i 
				for j in range(k,c):
					modeldata.append(i)
				# update index
				k=c
				# stop when n = n_model
				if k >= n_model: break

			# concatenate and flatten to one dimension
			synthetic_data = modeldata
			np.sort(synthetic_data)
			

		# fit synthetic data
		syn_dist = dist.get_obj()
		syn_fit = syn_dist(synthetic_data)
		stats = [np.mean(synthetic_data), np.std(synthetic_data), np.median(synthetic_data), syn_fit.D, syn_fit.xmin]
		stats.extend(syn_fit.get_parameters())
		means.append(stats)

		if isinstance(dest, str):
			# write periodically to temp file in case of error 
			with open('temp_%s.txt' % dest.partition('.csv')[0], 'a+') as f:
				for stat in stats[:-1]:
					f.write(str(stat) + '\t')
				f.write(str(stats[-1])+'\n')
				f.close()

	return means

def main(btype, source, dest, niter=1e3):
	import pandas as pd 
	ranks = pd.read_csv(source)['ranks']

	print('\n Fitting data to distribution...' , end=' ')
	if btype == 'semiparametric' or btype == 'semi-parametric':
		btype=True
	else:
		btype=False

	fit = Powerlaw(ranks)
	print('done.')

	print('\t Estimated xmin =', fit.xmin)
	for param, param_name in zip(fit.get_parameters(), fit.get_parameter_names()):
		print('\t Estimated %s =' % param_name, param)
	print('\t D =', fit.D)

	print('\n Beginning bootstrap')
	results = bootstrap(fit, semi_parametric=btype, niter=niter, dest=dest)

	cols = ['mean','std','median', 'D','xmin']
	cols.extend(fit.get_parameter_names())
	pd.DataFrame(results, columns=cols).to_csv(dest, index=False)

if __name__=='__main__':
	import sys

	if len(sys.argv) == 4:
		main(sys.argv[1], sys.argv[2], sys.argv[3])
	elif len(sys.argv) == 5:
		main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
	else:
		print('Invalid input. Enter bootstrap type (semi-parametric or parametric), data source file, output file destination, and (optionally) the number of iterations.') 
		sys.exit()
		 
















	

