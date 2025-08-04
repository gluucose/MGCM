import math
import warnings
warnings.filterwarnings('ignore')

import torch.nn as nn
from torch.nn import init
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test
from torch.utils.data._utils.collate import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ***************************  Regularization  ***************************

def regularize_weights(model, reg_type=None):
	l1_reg = None
	
	for W in model.parameters():
		if l1_reg is None:
			l1_reg = torch.abs(W).sum()
		else:
			l1_reg = l1_reg + torch.abs(W).sum()  # torch.abs(W).sum() is equivalent to W.norm(1)
	return l1_reg


def regularize_path_weights(model, reg_type=None):
	l1_reg = None
	
	for W in model.classifier.parameters():
		if l1_reg is None:
			l1_reg = torch.abs(W).sum()
		else:
			l1_reg = l1_reg + torch.abs(W).sum()  # torch.abs(W).sum() is equivalent to W.norm(1)
	
	for W in model.linear.parameters():
		if l1_reg is None:
			l1_reg = torch.abs(W).sum()
		else:
			l1_reg = l1_reg + torch.abs(W).sum()  # torch.abs(W).sum() is equivalent to W.norm(1)
	
	return l1_reg


def regularize_MM_weights(model, reg_type=None):
	l1_reg = None
	
	if model.module.__hasattr__('omic_net'):
		for W in model.module.omic_net.parameters():
			if l1_reg is None:
				l1_reg = torch.abs(W).sum()
			else:
				l1_reg = l1_reg + torch.abs(W).sum()  # torch.abs(W).sum() is equivalent to W.norm(1)
	
	if model.module.__hasattr__('encoder1'):
		for W in model.module.encoder1.parameters():
			if l1_reg is None:
				l1_reg = torch.abs(W).sum()
			else:
				l1_reg = l1_reg + torch.abs(W).sum()  # torch.abs(W).sum() is equivalent to W.norm(1)
	
	if model.module.__hasattr__('encoder2'):
		for W in model.module.encoder2.parameters():
			if l1_reg is None:
				l1_reg = torch.abs(W).sum()
			else:
				l1_reg = l1_reg + torch.abs(W).sum()  # torch.abs(W).sum() is equivalent to W.norm(1)
	
	if model.module.__hasattr__('classifier'):
		for W in model.module.classifier.parameters():
			if l1_reg is None:
				l1_reg = torch.abs(W).sum()
			else:
				l1_reg = l1_reg + torch.abs(W).sum()  # torch.abs(W).sum() is equivalent to W.norm(1)
	
	return l1_reg


def regularize_MM_omic(model, reg_type=None):
	l1_reg = None
	if hasattr(model.module, 'omic_net'):
		for W in model.module.omic_net.parameters():
			if torch.isnan(W).any() or torch.isinf(W).any():
				print("Invalid value detected in W")
				return None
			if l1_reg is None:
				l1_reg = torch.abs(W).sum()
			else:
				l1_reg = l1_reg + torch.abs(W).sum()
	return l1_reg


# ***************************   Network Initialization  ***************************

def print_model(model, optimizer):
	print(model)
	
	print("Model's state_dict:")
	for param_tensor in model.state_dict():
		print(param_tensor, "\t", model.state_dict()[param_tensor].size())
	
	print("optimizer's state_dict:")
	for var_name in optimizer.state_dict():
		print(var_name, "\t", optimizer.state_dict()[var_name])


def init_weights(net, init_type='orthogonal', init_gain=0.02):
	"""
	Parameters:
		net (network)   -- The network to initialize
		init_type (str) -- The method to initialize: normal | xavier | kaiming | orthogonal
		init_gain (float)    -- Scale factor: normal, xavier and orthogonal.
	"""
	
	def init_func(m):
		classname = m.__class__.__name__
		if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
			if init_type == 'normal':
				init.normal_(m.weight.data, 0.0, init_gain)
			elif init_type == 'xavier':
				init.xavier_normal_(m.weight.data, gain=init_gain)
			elif init_type == 'kaiming':
				init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
			elif init_type == 'orthogonal':
				init.orthogonal_(m.weight.data, gain=init_gain)
			else:
				raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
			if hasattr(m, 'bias') and m.bias is not None:
				init.constant_(m.bias.data, 0.0)
		elif classname.find('BatchNorm2d') != -1:
			init.normal_(m.weight.data, 1.0, init_gain)
			init.constant_(m.bias.data, 0.0)
	
	print('initialize network with %s' % init_type)
	net.apply(init_func)


def init_max_weights(module):
	for m in module.modules():
		if type(m) == nn.Linear:
			stdv = 1. / math.sqrt(m.weight.size(1))
			m.weight.data.normal_(0, stdv)
			m.bias.data.zero_()


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
	"""
	Parameters:
		net (network)      -- The network to initialize
		init_type (str)    -- The method to initialize: normal | xavier | kaiming | orthogonal
		init_gain (float)  -- Scale factor: normal, xavier and orthogonal.
		gpu_ids (int list) -- The location where the GPU is running: e.g., 0,1,2
	"""
	if len(gpu_ids) > 0:
		assert (torch.cuda.is_available())
		net.to(gpu_ids[0])
		net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
	
	if init_type != 'max' and init_type != 'none':
		print("Init Type:", init_type)
		init_weights(net, init_type, init_gain=init_gain)
	elif init_type == 'none':
		print("Init Type: Not initializing networks.")
	elif init_type == 'max':
		print("Init Type: Self-Normalizing Weights")
	return net


def define_act_layer(act_type='Tanh'):
	if act_type == 'Tanh':
		act_layer = nn.Tanh()
	elif act_type == 'ReLU':
		act_layer = nn.ReLU()
	elif act_type == 'Sigmoid':
		act_layer = nn.Sigmoid()
	elif act_type == 'LSM':
		act_layer = nn.LogSoftmax(dim=1)
	elif act_type == "none":
		act_layer = None
	else:
		raise NotImplementedError('activation layer [%s] is not found' % act_type)
	return act_layer


# ***************************  NLL SurvLoss  ***************************
def nll_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
	batch_size = len(Y)
	Y = Y.view(batch_size, 1).long()  # ground truth bin, 1,2,...,k
	c = c.view(batch_size, 1).float()  # censorship status, 0 or 1
	if S is None:
		S = torch.cumprod(1 - hazards, dim=1)  # surival is cumulative product of 1 - hazards
	# without padding, S(0) = S[0], h(0) = h[0]
	S_padded = torch.cat([torch.ones_like(c), S], 1)
	
	# S(-1) = 0, all patients are alive from (-inf, 0) by definition
	# after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
	# h[y] = h(1), S[1] = S(1)
	
	uncensored_loss = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(
		torch.gather(hazards, 1, Y).clamp(min=eps)))
	censored_loss = - c * torch.log(torch.gather(S_padded, 1, Y + 1).clamp(min=eps))
	
	neg_l = censored_loss + uncensored_loss
	loss = (1 - alpha) * neg_l + alpha * uncensored_loss
	loss = loss.mean()
	return loss


class NLLSurvLoss(object):
	def __init__(self, alpha=0.15):
		self.alpha = alpha
	
	def __call__(self, hazards, S, Y, c, alpha=None):
		if alpha is None:
			return nll_loss(hazards, S, Y, c, alpha=self.alpha)
		else:
			return nll_loss(hazards, S, Y, c, alpha=alpha)


# ***************************  CrossEntropy SurvLoss  ***************************
def ce_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
	batch_size = len(Y)
	Y = Y.view(batch_size, 1)  # ground truth bin, 1,2,...,k
	c = c.view(batch_size, 1).float()  # censorship status, 0 or 1
	if S is None:
		S = torch.cumprod(1 - hazards, dim=1)  # surival is cumulative product of 1 - hazards
	# without padding, S(0) = S[0], h(0) = h[0]
	# after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
	# h[y] = h(1)
	# S[1] = S(1)
	S_padded = torch.cat([torch.ones_like(c), S], 1)
	
	reg = -(1 - c) * (
				torch.log(torch.gather(S_padded, 1, Y) + eps) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
	ce_l = - c * torch.log(torch.gather(S, 1, Y).clamp(min=eps)) - (1 - c) * torch.log(
		1 - torch.gather(S, 1, Y).clamp(min=eps))
	
	loss = (1 - alpha) * ce_l + alpha * reg
	loss = loss.mean()
	return loss


class CrossEntropySurvLoss(object):
	def __init__(self, alpha=0.15):
		self.alpha = alpha
	
	def __call__(self, hazards, S, Y, c, alpha=None):
		if alpha is None:
			return ce_loss(hazards, S, Y, c, alpha=self.alpha)
		else:
			return ce_loss(hazards, S, Y, c, alpha=alpha)


# ***************************  Cox SurvLoss  ***************************
class CoxSurvLoss(object):
	def __call__(hazards, S, c, **kwargs):
		current_batch_len = len(S)
		R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
		for i in range(current_batch_len):
			for j in range(current_batch_len):
				R_mat[i, j] = S[j] >= S[i]
		
		R_mat = torch.FloatTensor(R_mat).to(device)
		theta = hazards.np.reshape(-1)
		exp_theta = torch.exp(theta)
		loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta * R_mat, dim=1))) * (1 - c))
		return loss_cox


# ***************************  Survival Prediction  ***************************
def accuracy(output, labels):
	preds = output.max(1)[1].type_as(labels)
	correct = preds.eq(labels).double()
	correct = correct.sum()
	return correct / len(labels)


def accuracy_cox(hazardsdata, labels):
	# This accuracy is based on estimated survival events against true survival events
	median = np.median(hazardsdata)
	hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
	hazards_dichotomize[hazardsdata > median] = 1
	correct = np.sum(hazards_dichotomize == labels)
	return correct / len(labels)


def cox_log_rank(hazardsdata, labels, survtime_all):
	median = np.median(hazardsdata)
	hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
	hazards_dichotomize[hazardsdata > median] = 1
	idx = hazards_dichotomize == 0
	T1 = survtime_all[idx]
	T2 = survtime_all[~idx]
	E1 = labels[idx]
	E2 = labels[~idx]
	results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
	pvalue_pred = results.p_value
	return (pvalue_pred)


def CIndex(hazards, labels, survtime_all):
	concord = 0.
	total = 0.
	N_test = labels.shape[0]
	for i in range(N_test):
		if labels[i] == 1:
			for j in range(N_test):
				if survtime_all[j] > survtime_all[i]:
					total += 1
					if hazards[j] < hazards[i]:
						concord += 1
					elif hazards[j] < hazards[i]:
						concord += 0.5
	return (concord / total)


def CIndex_lifeline(hazards, labels, survtime_all):
	return (concordance_index(survtime_all, -hazards, labels))
