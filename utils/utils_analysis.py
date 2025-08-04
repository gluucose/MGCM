import warnings

warnings.filterwarnings('ignore')
import re
import scipy
import numpy as np
import matplotlib as mpl

# from scipy import interp
mpl.rcParams['axes.linewidth'] = 3


def natural_sort(l):
	convert = lambda text: int(text) if text.isdigit() else text.lower()
	alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
	return sorted(l, key=alphanum_key)


# ***************************  Survival Prediction  ***************************
def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)


def hazard2grade(hazard, p):
	for i in range(len(p)):
		if hazard < p[i]:
			return i
	return len(p)


def CI_pm(data, confidence=0.95):
	a = 1.0 * np.array(data)
	n = len(a)
	m, se = np.mean(a), scipy.stats.sem(a)
	h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
	return str("{0:.4f} ± ".format(m) + "{0:.3f}".format(h))


def CI_interval(data, confidence=0.95):
	a = 1.0 * np.array(data)
	n = len(a)
	m, se = np.mean(a), scipy.stats.sem(a)
	h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
	return str("{0:.3f}, ".format(m - h) + "{0:.3f}".format(m + h))


def p(n):
	def percentile_(x):
		return np.percentile(x, n)
	
	percentile_.__name__ = 'p%s' % n
	return percentile_
