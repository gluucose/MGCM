import numbers

# from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mamba_simple import *


def to_3d(x):
	return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
	return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
	def __init__(self, normalized_shape):
		super(BiasFree_LayerNorm, self).__init__()
		if isinstance(normalized_shape, numbers.Integral):
			normalized_shape = (normalized_shape,)
		normalized_shape = torch.Size(normalized_shape)

		assert len(normalized_shape) == 1

		self.weight = nn.Parameter(torch.ones(normalized_shape))
		self.normalized_shape = normalized_shape

	def forward(self, x):
		sigma = x.var(-1, keepdim=True, unbiased=False)
		return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
	def __init__(self, normalized_shape):
		super(WithBias_LayerNorm, self).__init__()
		if isinstance(normalized_shape, numbers.Integral):
			normalized_shape = (normalized_shape,)
		normalized_shape = torch.Size(normalized_shape)

		assert len(normalized_shape) == 1

		self.weight = nn.Parameter(torch.ones(normalized_shape))
		self.bias = nn.Parameter(torch.zeros(normalized_shape))
		self.normalized_shape = normalized_shape

	def forward(self, x):
		mu = x.mean(-1, keepdim=True)
		sigma = x.var(-1, keepdim=True, unbiased=False)
		return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


def L1_norm(source_en_a, source_en_b):
	result = []
	narry_a = source_en_a
	narry_b = source_en_b

	dimension = source_en_a.shape

	# caculate L1-norm
	temp_abs_a = torch.abs(narry_a)
	temp_abs_b = torch.abs(narry_b)
	_l1_a = torch.sum(temp_abs_a, dim=1)
	_l1_b = torch.sum(temp_abs_b, dim=1)

	_l1_a = torch.sum(_l1_a, dim=0)
	_l1_b = torch.sum(_l1_b, dim=0)
	with torch.no_grad():
		l1_a = _l1_a.detach()
		l1_b = _l1_b.detach()

	# caculate the map for source images
	mask_value = l1_a + l1_b

	mask_sign_a = l1_a / mask_value
	mask_sign_b = l1_b / mask_value

	array_MASK_a = mask_sign_a
	array_MASK_b = mask_sign_b
	for i in range(dimension[0]):
		for j in range(dimension[1]):
			temp_matrix = array_MASK_a * narry_a[i, j, :, :] + array_MASK_b * narry_b[i, j, :, :]
			result.append(temp_matrix)

	result = torch.stack(result, dim=-1)

	result = result.reshape((dimension[0], dimension[1], dimension[2], dimension[3]))

	return result


class LayerNorm(nn.Module):
	def __init__(self, dim, LayerNorm_type):
		super(LayerNorm, self).__init__()
		if LayerNorm_type == 'BiasFree':
			self.body = BiasFree_LayerNorm(dim)
		else:
			self.body = WithBias_LayerNorm(dim)

	def forward(self, x):
		if len(x.shape) == 4:
			h, w = x.shape[-2:]
			return to_4d(self.body(to_3d(x)), h, w)
		else:
			return self.body(x)


# **********************   Bi-Interactive Mamba  *****************
class BiMamba(nn.Module):
	def __init__(self, dim):
		super(BiMamba, self).__init__()
		self.cross_mamba = Mamba(dim, bimamba_type="v3")
		self.norm1 = LayerNorm(dim, 'with_bias')
		self.norm2 = LayerNorm(dim, 'with_bias')

	def forward(self, M1, fusion_resi, M2):
		fusion_resi = M1 + fusion_resi
		M1 = self.norm1(fusion_resi)
		M2 = self.norm2(M2)
		fusion = self.cross_mamba(self.norm1(M1), extra_emb=self.norm2(M2))

		return fusion, fusion_resi


# **********************  Tri-Interactive Mamba  *****************
class TriMamba(nn.Module):
	def __init__(self, dim):
		super(TriMamba, self).__init__()
		self.cross_mamba = Mamba(dim, bimamba_type="m3")
		self.norm1 = LayerNorm(dim, 'with_bias')
		self.norm2 = LayerNorm(dim, 'with_bias')
		self.norm3 = LayerNorm(dim, 'with_bias')

	def forward(self, M1, fusion_resi, M2, fusion):
		fusion_resi = fusion + fusion_resi
		fusion = self.norm1(fusion_resi)
		M2 = self.norm2(M2)
		M1 = self.norm3(M1)
		fusion = self.cross_mamba(self.norm1(fusion), extra_emb1=self.norm2(M2), extra_emb2=self.norm3(M1))

		return fusion, fusion_resi
	