import powerlaw as pl
import numpy as np
from math import *

'''
This script essentially acts as a wrapper to the powerlaw library. see: https://github.com/jeffalstott/powerlaw.
It is used in the bootstrap.py script in order to fit and test alternative distributions for goodness-of-fit.

Each distribution's parameters are estimated via MLE, and we generate an estimate for xmin by minimizing
the KS distance between the hypothesized CDF and the empirical CDF. We can also set a value for xmin in which case
no KS minimizing will occur.

Original powerlaw library objects, along with their functions and attributes, are stored in the 'fit' attribute
of each distribution object. 
'''

class Lognormal:
	def __init__(self, data, xmin=False):
		self.data_original = np.array(data)
		self.xmin = xmin
		self.xmax = max(self.data_original)
		self.mu = False
		self.sigma = False
		self.Ds = False
		self.fit = self._fit(self.xmin)
		self.data = self.fit.data
		self.D = self.fit.lognormal.D
		
	def _fit(self, xmin):
		'''
		Estimate paramters by minimizing KS distance
		'''
		if not xmin:
			estimates = []
			for xmin_ in np.unique(self.data_original)[:-1]:
				fit=pl.Fit(data=self.data_original, xmin=xmin_, discrete=True)
				estimates.append(fit)
			estimates.sort(key=lambda dist: dist.lognormal.D)
			Ds = [[e, e.lognormal.xmin, e.lognormal.D] for e in estimates]
			Ds.sort(key=lambda k: k[-1])
			self.Ds = Ds
			best_fit = estimates[0]
			self.xmin = best_fit.xmin
		else:
			best_fit = pl.Fit(data=self.data_original, xmin=xmin, discrete=True)
		self.mu = best_fit.lognormal.mu
		self.sigma = best_fit.lognormal.sigma
		return best_fit
	
	def cdf(self, data=False):
		if isinstance(data, bool):
			data = np.unique(self.data)
		return self.fit.lognormal.cdf(data=data)

	def get_parameter_names(self):
		return ['mu', 'sigma']

	def get_parameters(self):
		return [self.mu, self.sigma]

	def get_obj(selfm, xmin=False):
		if xmin:
			return lambda data: Lognormal(data[0], data[1])
		else:
			return lambda data: Lognormal(data)

class Positive_lognormal:
	def __init__(self, data, xmin=False):
		self.data_original = np.array(data)
		self.xmin = xmin
		self.xmax = max(self.data_original)
		self.mu = False
		self.sigma = False
		self.Ds = False
		self.fit = self._fit(self.xmin)
		self.data = self.fit.data
		self.D = self.fit.lognormal_positive.D
		
	def _fit(self, xmin):
		'''
		Estimate paramters by minimizing KS distance
		'''
		if not xmin:
			estimates = []
			for xmin_ in np.unique(self.data_original)[:-1]:
				fit=pl.Fit(data=self.data_original, xmin=xmin_, discrete=True)
				estimates.append(fit)
			estimates.sort(key=lambda dist: dist.lognormal_positive.D)
			best_fit = estimates[0]
			self.xmin = best_fit.xmin
		else:
			best_fit = pl.Fit(data=self.data_original, xmin=xmin, discrete=True)
		self.mu = best_fit.lognormal_positive.mu
		self.sigma = best_fit.lognormal_positive.sigma
		return best_fit
	
	def cdf(self, data=False):
		if isinstance(data, bool):
			data = np.unique(self.data)
		return self.fit.lognormal_positive.cdf(data=data)

	def get_parameter_names(self):
		return ['mu', 'sigma']

	def get_parameters(self):
		return [self.mu, self.sigma]

	def get_obj(selfm, xmin=False):
		if xmin:
			return lambda data: Positive_lognormal(data[0], data[1])
		else:
			return lambda data: Positive_lognormal(data)
	
class Exponential:
	def __init__(self, data, xmin=False):
		self.data_original = np.array(data)
		self.xmin = xmin
		self.xmax = max(self.data_original)
		self.xmin = xmin
		self.Lambda = False
		self.Ds = False
		self.fit = self._fit(self.xmin)
		self.data = self.fit.data
		self.D = self.fit.exponential.D
		
	def _fit(self, xmin):
		'''
		Estimate paramters by minimizing KS distance
		'''
		if not xmin:
			estimates = []
			for xmin_ in np.unique(self.data_original)[:-1]:
				fit=pl.Fit(data=self.data_original, xmin=xmin_, discrete=True)
				estimates.append(fit)
			estimates.sort(key=lambda dist: dist.exponential.D)
			Ds = [[e, e.exponential.xmin, e.exponential.D] for e in estimates]
			Ds.sort(key=lambda k: k[-1])
			self.Ds = Ds
			best_fit = Ds[0][0]
			self.xmin = best_fit.xmin
		else:
			best_fit = pl.Fit(data=self.data_original, xmin=xmin, discrete=True)
		self.Lambda = best_fit.exponential.Lambda
		return best_fit
	
	def cdf(self, data=False):
		if isinstance(data, bool):
			data = np.unique(self.data)
		return self.fit.exponential.cdf(data=data)

	def get_parameter_names(self):
		return ['lambda']

	def get_parameters(self):
		return [self.Lambda]

	def get_obj(self, xmin=False):
		if xmin:
			return lambda data: Exponential(data[0], data[1])
		else:
			return lambda data: Exponential(data)
	
class Stretched_exponential:
	def __init__(self, data, xmin=False):
		self.data_original = np.array(data)
		self.xmin = xmin
		self.xmax = max(self.data_original)
		self.Lambda = False
		self.beta = False
		self.Ds = False
		self.fit = self._fit(self.xmin)
		self.data = self.fit.data
		self.D = self.fit.stretched_exponential.D
		
	def _fit(self, xmin):
		'''
		Estimate paramters by minimizing KS distance
		'''
		if not xmin:
			estimates = []
			for xmin_ in np.unique(self.data_original)[:-1]:
				fit=pl.Fit(data=self.data_original, xmin=xmin_, discrete=True)
				estimates.append(fit)
			estimates.sort(key=lambda dist: dist.stretched_exponential.D)
			Ds = [[e, e.stretched_exponential.xmin, e.stretched_exponential.D] for e in estimates]
			Ds.sort(key=lambda k: k[-1])
			self.Ds = Ds
			best_fit = estimates[0]
			self.xmin = best_fit.xmin
		else:
			best_fit = pl.Fit(data=self.data_original, xmin=xmin, discrete=True)
		self.Lambda = best_fit.stretched_exponential.Lambda
		self.beta = best_fit.stretched_exponential.beta
		return best_fit
	
	def cdf(self, data=False):
		if isinstance(data, bool):
			data = np.unique(self.data)
		return self.fit.stretched_exponential.cdf(data=data)

	def get_parameter_names(self):
		return ['lambda', 'beta']

	def get_parameters(self):
		return [self.Lambda, self.beta]

	def get_obj(self, xmin=False):
		if xmin:
			return lambda data: Stretched_exponential(data[0], data[1])
		else:
			return lambda data: Stretched_exponential(data)
	
class Powerlaw_with_cutoff:
	def __init__(self, data, xmin=False, xmin_range=False):
		self.data_original = np.array(data)
		self.xmin = xmin
		self.xmax = max(self.data_original)
		self.Lambda = False
		self.alpha = False
		self.Ds = False
		self.xmin_range = xmin_range
		self.fit = self._fit(self.xmin, self.xmin_range)
		self.data = self.fit.data
		self.D = self.fit.truncated_power_law.D
		
	def _fit(self, xmin, xmin_range=False):
		'''
		Estimate paramters by minimizing KS distance
		'''
		if not xmin:
			if not xmin_range:
				xmins = np.unique(self.data_original)[:-1]
			else:
				# to save time, we can set an xmin estimate range
				# when bootstraping. the range should be based on the
				# estimate for xmin of the hypothesized model
				xmins = xmin_range

			estimates = []
			for xmin_ in xmins:
				try:
					fit = pl.Fit(data=self.data_original, xmin=xmin_, discrete=True)
				except ZeroDivisionError:
					fit = np.nan
				estimates.append(fit)
			try:
				estimates.sort(key=lambda dist: dist.truncated_power_law.D)
				best_fit = estimates[0]
			except ZeroDivisionError:
				Ds = []
				for est in estimates:
					try:
						D = est.truncated_power_law.D
					except ZeroDivisionError:
						D = np.nan
					Ds.append([est, D])
				Ds.sort(key=lambda l: l[-1])
				self.Ds = Ds
				best_fit = Ds[0][0]

			self.xmin = best_fit.xmin
		else:
			best_fit = pl.Fit(data=self.data_original, xmin=xmin, discrete=True)
		self.Lambda = best_fit.truncated_power_law.Lambda
		self.alpha = best_fit.truncated_power_law.alpha

		return best_fit
	
	def cdf(self, data=False):
		# if no data specified, return support
		if isinstance(data, bool):
			data = np.unique(self.data)
		return self.fit.truncated_power_law.cdf(data=data)

	def get_parameter_names(self):
		return ['alpha', 'lambda']

	def get_parameters(self):
		return [self.alpha, self.Lambda]

	def get_obj(self, xmin=False):
		if xmin:
			return lambda data: Powerlaw_with_cutoff(data[0], data[1])
		else:
			return lambda data: Powerlaw_with_cutoff(data)
	

class Powerlaw:
	def __init__(self, data, xmin=False):
		self.data_original = np.array(data)
		self.xmin = xmin
		self.xmax = max(self.data_original)
		self.alpha = False
		self.fit = self._fit(self.xmin)
		self.Ds = self.fit.Ds
		self.data = self.fit.data
		self.D = self.fit.D
		
	def _fit(self, xmin):
		'''
		Estimate paramters by minimizing KS distance
		'''
		if not xmin:
			best_fit = pl.Fit(data=self.data_original, discrete=True)
		else:
			best_fit = pl.Fit(data=self.data_original, xmin=xmin, discrete=True)

		self.xmin = best_fit.xmin
		self.alpha = best_fit.alpha
		
		return best_fit
	
	def cdf(self, data=False):
		if isinstance(data, bool):
			data = np.unique(self.data)
		return self.fit.power_law.cdf(data=data)

	def get_parameter_names(self):
		return ['alpha']

	def get_parameters(self):
		return [self.alpha]

	def get_obj(self):
		return lambda data: Powerlaw(data)

