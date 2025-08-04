from torch.optim import lr_scheduler

from models.model_graph_convolution import *
from models.model_bidirection_mamba import *
from models.model_interactive_mamba import *

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"


def define_net(opt, k):
	act = define_act_layer(act_type=opt.act_type)
	if opt.model_name == "mgcm":
		net = mgcm(opt=opt, act=act, k=k)
	elif opt.model_name == "lowomic":
		net = LowGraph(opt, opt.omic_dim, opt.dropout_gene, opt.alpha)
	elif opt.model_name == "midomic":
		net = MidGraph(opt, opt.omic_dim, opt.dropout_gene, opt.alpha)
	elif opt.model_name == "highomic":
		net = HighGraph(opt, opt.omic_dim, opt.dropout_gene, opt.alpha)
	else:
		raise NotImplementedError('model [%s] is not implemented' % opt.model)
	return init_net(net, opt.init_type, opt.init_gain, opt.gpu_ids)


def define_optimizer(opt, model):
	if opt.optimizer_type == 'adabound':
		optimizer = torch.adabound.AdaBound(model.parameters(), lr=opt.lr, final_lr=opt.final_lr)
	elif opt.optimizer_type == 'adam':
		optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2),
		                             weight_decay=opt.weight_decay)
	elif opt.optimizer_type == 'adagrad':
		optimizer = torch.optim.Adagrad(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay,
		                                initial_accumulator_value=0.1)
	else:
		raise NotImplementedError('initialization method [%s] is not implemented' % opt.optimizer)
	return optimizer


def define_reg(opt, model):
	if opt.reg_type == 'none':
		loss_reg = 0
	elif opt.reg_type == 'path':
		loss_reg = regularize_path_weights(model=model)
	elif opt.reg_type == 'mm':
		loss_reg = regularize_MM_weights(model=model)
	elif opt.reg_type == 'all':
		loss_reg = regularize_weights(model=model)
	elif opt.reg_type == 'omic':
		loss_reg = regularize_MM_omic(model=model)
	else:
		raise NotImplementedError('reg method [%s] is not implemented' % opt.reg_type)
	return loss_reg


def define_scheduler(opt, optimizer):
	if opt.lr_policy == 'linear':
		def lambda_rule(epoch):
			lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
			return lr_l
		scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
	elif opt.lr_policy == 'exp':
		scheduler = lr_scheduler.ExponentialLR(optimizer, 0.1, last_epoch=-1)
	elif opt.lr_policy == 'step':
		scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
	elif opt.lr_policy == 'plateau':
		scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
	elif opt.lr_policy == 'cosine':
		scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
	else:
		return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
	return scheduler


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


# ***************  MGCM Model  ***************

class mgcm(nn.Module):
	def __init__(self, opt, act, k, n_classes=4):
		super(mgcm, self).__init__()
		
		# Gene Network: LowGraph [B, N, 128], MidGraph [B, N, 256], HighGraph [B, N, 512]
		self.gene_net = MidGraph(opt=opt, omic_dim=opt.omic_dim, dropout_gene=opt.dropout_gene, alpha=opt.alpha)
		
		# Path Layer: Select the corresponding dimension based on the Gene Network
		self.path_layer = nn.Linear(opt.path_in, opt.path_out)
		
		self.BiFusion = BiMamba(dim=opt.fusion_dim)
		self.TriFusion = TriMamba(dim=opt.fusion_dim)
		self.PathBiSS = BiSSBlock(hidden_dim=opt.path_out, drop_path=0., norm_layer=nn.LayerNorm, attn_drop_rate=0., d_state=16)
		self.GeneBiSS = BiSSBlock(hidden_dim=opt.path_out, drop_path=0., norm_layer=nn.LayerNorm, attn_drop_rate=0., d_state=16)
		
		self.fc_dim = opt.dim_fc
		self.fc_dropout = opt.dropout_fc
		fc1 = nn.Sequential(nn.Linear(opt.lin_in, opt.fc_dim), nn.ELU(), nn.AlphaDropout(p=self.fc_dropout, inplace=True))
		fc2 = nn.Sequential(nn.Linear(opt.fc_dim, opt.lin_out), nn.ELU(), nn.AlphaDropout(p=self.fc_dropout, inplace=True))
		self.encoder = nn.Sequential(fc1, fc2)
		
		self.k = k
		self.act = act
		self.opt = opt
		self.n_classes = n_classes
		self.classifier = nn.Linear(opt.lin_out, n_classes)
	
	def forward(self, path, omic, edge):
		gene = self.gene_net(x=omic, adj=edge)
		path = self.path_layer(path)
		
		B_gene, N_gene, C_gene = gene.shape
		B_path, N_path, C_path = path.shape
		num_batch = (N_path + N_gene - 1) // N_gene  # Top-K Differential Expressed Genes(DEGs): K = 200
		
		TotalFusion = []
		BiFuison_res = 0
		TriFuison_res = 0
		for i in range(num_batch):
			gene_batch = gene
			start_idx = i * N_gene
			end_idx = min(start_idx + N_gene, N_path)
			path_batch = path[:, start_idx:end_idx, :]
			if N_path < N_gene:
				pad_size = N_gene - N_path
				padding = torch.zeros(B_path, pad_size, C_path, device='cuda')
				path_batch = torch.cat((path_batch, padding), dim=1)
			
			BiFuison_batch, BiFuison_res = self.BiFusion(path_batch, BiFuison_res, gene_batch)
			pathBiSS_batch, _ = self.PathBiSS(path_batch)
			geneBiSS_batch, _ = self.GeneBiSS(gene_batch)
			TriFuison_batch, TriFuison_res = self.TriFusion(pathBiSS_batch, TriFuison_res, geneBiSS_batch, BiFuison_batch)
			
			TotalFusion.append(TriFuison_batch.cpu())
			
		fusion = torch.cat(TotalFusion, dim=1).to('cuda')
		fusion = fusion[:, :N_path, :]
		fusion = fusion + path
		fusion = fusion.mean(dim=1)
		
		out = self.encoder(fusion)
		logits = self.classifier(out)
		Y_hat = torch.topk(logits, 1, dim=1)[1]
		Y_prob = F.softmax(logits, dim=1)
		hazards = torch.sigmoid(logits)
		S = torch.cumprod(1 - hazards, dim=1)
		risk = -torch.sum(S, dim=1)
		
		return hazards, S, Y_hat
	
	def __hasattr__(self, name):
		if '_parameters' in self.__dict__:
			_parameters = self.__dict__['_parameters']
			if name in _parameters:
				return True
		if '_buffers' in self.__dict__:
			_buffers = self.__dict__['_buffers']
			if name in _buffers:
				return True
		if '_modules' in self.__dict__:
			modules = self.__dict__['_modules']
			if name in modules:
				return True
		return False
	