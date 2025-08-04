import torch.nn.functional as F

from utils.utils_models import *

# ******  Low Graph Convolution  ******


class LowGraph(nn.Module):
	def __init__(self, opt, omic_dim, dropout_gene, alpha):
		super(LowGraph, self).__init__()
		self.dropout_gene = dropout_gene
		self.act = define_act_layer(act_type=opt.act_type_gene)
		
		self.nhids = [32, 64]
		self.nheads = [2, 1]
		
		self.attentions1 = [GraphAttentionLayer(
			omic_dim, self.nhids[0], dropout_gene=dropout_gene, alpha=alpha, concat=True) for _ in
			range(self.nheads[0])]
		for i, attention1 in enumerate(self.attentions1):
			self.add_module('attention1_{}'.format(i), attention1)
		
		self.attentions2 = [GraphAttentionLayer(
			self.nhids[0] * self.nheads[0], self.nhids[1], dropout_gene=dropout_gene, alpha=alpha, concat=True) for _ in
			range(self.nheads[1])]
		for i, attention2 in enumerate(self.attentions2):
			self.add_module('attention2_{}'.format(i), attention2)
		
		self.dropout_layer = nn.Dropout(p=self.dropout_gene)
	
	def forward(self, x, adj):
		batch = torch.linspace(0, x.size(0) - 1, x.size(0), dtype=torch.long)
		batch = batch.unsqueeze(1).repeat(1, x.size(1)).view(-1).cuda()
		
		# Get the gene data from columns 1 to Top-K: the index is zero-based, and the dimension is (B, K).
		x = x[:, 0:200, :]
		
		x = self.dropout_layer(x)
		x_layer1 = torch.cat([att(x, adj) for att in self.attentions1], dim=-1)
		
		x = self.dropout_layer(x_layer1)
		x_layer2 = torch.cat([att(x, adj) for att in self.attentions2], dim=-1)
		
		# Concatenate the output of each layer: [bs, N, 128]
		x = torch.cat([x_layer1, x_layer2], dim=-1)
		
		return x


# ******  Mid Graph Convolution  ******
class MidGraph(nn.Module):
	def __init__(self, opt, omic_dim, dropout_gene, alpha):
		super(MidGraph, self).__init__()
		self.dropout_gene = dropout_gene
		self.act = define_act_layer(act_type=opt.act_type_gene)
		
		"""
		nhids[] Hierarchical design can effectively learn and capture different levels of abstract features in the data.
		nheads[] Different attention heads in each layer improve the representation ability of local relations with different graph structures.
		"""
		# Hidden Dimensions in Graph Attention Layers.
		self.nhids = [32, 32, 64]
		# The number of attention heads in the graph attention layer.
		self.nheads = [2, 2, 2]
		
		# Layer 1: Each attention head is added to the model as a submodule, and the corresponding metrics are obtained
		self.attentions1 = [GraphAttentionLayer(
			omic_dim, self.nhids[0], dropout_gene=dropout_gene, alpha=alpha, concat=True) for _ in
			range(self.nheads[0])]
		for i, attention1 in enumerate(self.attentions1):
			self.add_module('attention1_{}'.format(i), attention1)
		
		# Layer 2: The input dimension is the sum of the output dimensions of the first layer
		self.attentions2 = [GraphAttentionLayer(
			self.nhids[0] * self.nheads[0], self.nhids[1], dropout_gene=dropout_gene, alpha=alpha, concat=True) for _ in
			range(self.nheads[1])]
		for i, attention2 in enumerate(self.attentions2):
			self.add_module('attention2_{}'.format(i), attention2)
		
		# Layer 3: The input dimension is the sum of the output dimensions of the second layer
		self.attentions3 = [GraphAttentionLayer(
			self.nhids[1] * self.nheads[1], self.nhids[2], dropout_gene=dropout_gene, alpha=alpha, concat=True) for _ in
			range(self.nheads[2])]
		for i, attention3 in enumerate(self.attentions3):
			self.add_module('attention3_{}'.format(i), attention3)
		
		self.dropout_layer = nn.Dropout(p=self.dropout_gene)
	
	def forward(self, x, adj):
		batch = torch.linspace(0, x.size(0) - 1, x.size(0), dtype=torch.long)
		batch = batch.unsqueeze(1).repeat(1, x.size(1)).view(-1).cuda()
		
		# Get the gene data from columns 1 to Top-K: the index is zero-based, and the dimension is (B, K).
		x = x[:, 0:200, :]
		
		# Graph Attention Layer 1: [bs, N, d1*h1]
		x = self.dropout_layer(x)
		x_layer1 = torch.cat([att(x, adj) for att in self.attentions1], dim=-1)
		
		# Graph Attention Layer 2: [bs, N, d2*h2]
		x = self.dropout_layer(x_layer1)
		x_layer2 = torch.cat([att(x, adj) for att in self.attentions2], dim=-1)
		
		# Graph Attention Layer 3: [bs, N, d3*h3]
		x = self.dropout_layer(x_layer2)
		x_layer3 = torch.cat([att(x, adj) for att in self.attentions3], dim=-1)
		
		# Concatenate the output of each layer: [bs, N, 256]
		x = torch.cat([x_layer1, x_layer2, x_layer3], dim=-1)
		
		return x


# ****** High Graph Convolution  ******
class HighGraph(nn.Module):
	def __init__(self, opt, omic_dim, dropout_gene, alpha):
		super(HighGraph, self).__init__()
		self.dropout_gene = dropout_gene
		self.act = define_act_layer(act_type=opt.act_type_gene)

		self.nhids = [32, 64, 64]
		self.nheads = [4, 2, 4]
		
		self.attentions1 = [GraphAttentionLayer(
			omic_dim, self.nhids[0], dropout_gene=dropout_gene, alpha=alpha, concat=True) for _ in
			range(self.nheads[0])]
		for i, attention1 in enumerate(self.attentions1):
			self.add_module('attention1_{}'.format(i), attention1)
		
		self.attentions2 = [GraphAttentionLayer(
			self.nhids[0] * self.nheads[0], self.nhids[1], dropout_gene=dropout_gene, alpha=alpha, concat=True) for _ in
			range(self.nheads[1])]
		for i, attention2 in enumerate(self.attentions2):
			self.add_module('attention2_{}'.format(i), attention2)
		
		self.attentions3 = [GraphAttentionLayer(
			self.nhids[1] * self.nheads[1], self.nhids[2], dropout_gene=dropout_gene, alpha=alpha, concat=True) for _ in
			range(self.nheads[2])]
		for i, attention3 in enumerate(self.attentions3):
			self.add_module('attention3_{}'.format(i), attention3)
		
		self.dropout_layer = nn.Dropout(p=self.dropout_gene)
	
	def forward(self, x, adj):
		batch = torch.linspace(0, x.size(0) - 1, x.size(0), dtype=torch.long)
		batch = batch.unsqueeze(1).repeat(1, x.size(1)).view(-1).cuda()
		
		# Get the gene data from columns 1 to Top-K: the index is zero-based, and the dimension is (B, K).
		x = x[:, 0:200, :]
		
		x = self.dropout_layer(x)
		x_layer1 = torch.cat([att(x, adj) for att in self.attentions1], dim=-1)
		
		x = self.dropout_layer(x_layer1)
		x_layer2 = torch.cat([att(x, adj) for att in self.attentions2], dim=-1)
		
		x = self.dropout_layer(x_layer2)
		x_layer3 = torch.cat([att(x, adj) for att in self.attentions3], dim=-1)
		
		# Concatenate the output of each layer: [bs, N, 512]
		x = torch.cat([x_layer1, x_layer2, x_layer3], dim=-1)
		
		return x


# ****** Graph Attention Layer ******
class GraphAttentionLayer(nn.Module):
	def __init__(self, in_features, out_features, dropout_gene, alpha, concat=True):
		super(GraphAttentionLayer, self).__init__()
		self.dropout_gene = dropout_gene
		self.in_features = in_features
		self.out_features = out_features
		self.alpha = alpha
		self.concat = concat
		
		self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
		nn.init.xavier_uniform_(self.W.data, gain=1.414)
		
		self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
		nn.init.xavier_uniform_(self.a.data, gain=1.414)
		
		self.leakyrelu = nn.LeakyReLU(self.alpha)
		self.dropout_layer = nn.Dropout(p=self.dropout_gene)
	
	def forward(self, input, adj):
		"""
        input: mini-batch input. size: [batch_size, num_nodes, node_feature_dim]
        adj:   adjacency matrix. size: [num_nodes, num_nodes].  need to be expanded to batch_adj later.
        """
		h = torch.matmul(input, self.W)  # Multiply and add input features with weights [bs, N, F]
		bs, N, _ = h.size()              # Returns batch size, number of nodes, and feature dimension
		
		"""
        1. Make N copies in the third dimension (feature dimension) to obtain a tensor of size (bs, N, N * F), changing the shape to (bs, N * N, F).
        2. Replicating N times in the second dimension (number of nodes) yields a tensor of size (bs, N * N, F)
        3. The two tensors are concatenated in the last dimension to obtain a tensor of size (bs, N * N, F + F), which changes the shape to (bs, N, -1, 2 * self.out_features).
        """
		a_input = torch.cat([h.repeat(1, 1, N).view(bs, N * N, -1), h.repeat(1, N, 1)], dim=-1).view(bs, N, -1, 2 * self.out_features)
		e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))
		# batch_adj = torch.unsqueeze(adj, 0).repeat(bs, 1, 1)
		batch_adj = adj.repeat(bs, 1, 1)
		
		zero_vec = -9e15 * torch.ones_like(e)
		attention = torch.where(batch_adj > 0, e, zero_vec)
		attention = self.dropout_layer(F.softmax(attention, dim=-1))  #
		h_prime = torch.bmm(attention, h)
		
		if self.concat:
			return F.elu(h_prime)
		else:
			return h_prime
	
	def __repr__(self):
		return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
	