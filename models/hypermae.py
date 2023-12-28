# This code contains hypermae model that can plugged into the MAE repo to replace https://github.com/facebookresearch/mae/blob/main/models_mae.py
# We use the Meta Platform's MAE codebase (https://github.com/facebookresearch/mae.git) for pretraining and finetuning
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block
import math
import einops
from util.pos_embed import get_2d_sincos_pos_embed
import torch.nn.functional as F
from util import misc
import numpy as np
import time

class PaCoMlp(nn.Module):
	def __init__(self, pre_mod_depth, post_mod_depth, in_dim, out_dim, hidden_dim, use_pe, pe_dim, out_bias=0, pe_sigma=1024):
		super().__init__()
		self.use_pe = use_pe
		self.pe_dim = pe_dim
		self.pe_sigma = pe_sigma
		self.hidden_dim = hidden_dim
		self.pre_mod_depth = pre_mod_depth
		self.post_mod_depth = post_mod_depth
		self.param_shapes = dict()
		if use_pe:
			last_dim = in_dim * pe_dim
		else:
			last_dim = in_dim
		depth = pre_mod_depth + post_mod_depth
		self.mlp = nn.ModuleList()
		for i in range(depth):
			cur_dim = hidden_dim if i < depth - 1 else out_dim
			self.param_shapes[f'w{i}'] = (last_dim, cur_dim)
			self.mlp.append(nn.Linear(last_dim, cur_dim))
			last_dim = cur_dim
		self.relu = nn.ReLU()
		self.M = None
		self.out_bias = out_bias 

	def set_weight_matrix(self, M):
		self.M = M

	def convert_posenc(self, x):
		w = torch.exp(torch.linspace(0, np.log(self.pe_sigma), self.pe_dim // 2, device=x.device)) 
		x = torch.matmul(x.unsqueeze(-1), w.unsqueeze(0)).view(*x.shape[:-1], -1) 
		x = torch.cat([torch.cos(np.pi * x), torch.sin(np.pi * x)], dim=-1) 
		return x

	def modulate(self, x):
		x = torch.matmul(x, self.M)
		return x

	def forward(self, x):
		# [LZ] x shape: BxHxWx2
		B, query_shape = x.shape[0], x.shape[1: -1]
		x = x.view(B, -1, x.shape[-1]) # BxHWx2
		if self.use_pe:
			x = self.convert_posenc(x) # BxHWx2Pe_dim
		# Pre-modulation MLP layers
		for i in range(self.pre_mod_depth):
			x = self.mlp[i](x)
			if i < self.pre_mod_depth - 1:
				x = self.relu(x)
		# Modulation
		x = self.modulate(x)
		# Post-modulation MLP layers
		for i in range(self.post_mod_depth):
			x = self.mlp[self.pre_mod_depth + i](x)
			if i < self.post_mod_depth - 1:
				x = self.relu(x)
			else:
				x = x + self.out_bias
		x = x.view(B, *query_shape, -1) 
		return x

class MaskedAutoencoderViTINR(nn.Module):
	def __init__(self, img_size=224, patch_size=16, in_chans=3,
				 embed_dim=1024, depth=24, num_heads=16,
				 pacomlp=None, n_weight_tokens=8, 
				 mlp_ratio=4., feat2W_depth=4, norm_layer=nn.LayerNorm, norm_pix_loss=False):
		super().__init__()

		# --------------------------------------------------------------------------
		# MAE encoder specifics
		self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
		num_patches = self.patch_embed.num_patches
		self.num_heads = num_heads

		self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
		self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

		self.blocks = nn.ModuleList([
			Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
			for i in range(depth)])
		self.norm = norm_layer(embed_dim)

		# --------------------------------------------------------------------------
		# Decoder specifics
		self.pacomlp = pacomlp
		pacomlp_dim = self.pacomlp.hidden_dim

		self.feat2W_blocks = nn.ModuleList([
			Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
			for i in range(feat2W_depth)])

		print("pacomlp", pacomlp)
		self.weight_tokens = nn.Parameter(torch.randn(n_weight_tokens, embed_dim))
		self.linear_proj_pacomlp_dim = nn.Linear(embed_dim, pacomlp_dim)
		self.transformation_matrix = nn.Parameter(torch.randn(pacomlp_dim, n_weight_tokens))

		self.initialize_weights()
		self.norm_pix_loss = norm_pix_loss

		# --------------------------------------------------------------------------

	def initialize_weights(self):
		# initialization
		# initialize (and freeze) pos_embed by sin-cos embedding
		pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
		self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

		# initialize patch_embed like nn.Linear (instead of nn.Conv2d)
		w = self.patch_embed.proj.weight.data
		torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

		# timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
		torch.nn.init.normal_(self.cls_token, std=.02)
		torch.nn.init.normal_(self.weight_tokens, std=.02)
		torch.nn.init.normal_(self.transformation_matrix, std=.02)

		# initialize nn.Linear and nn.LayerNorm
		self.apply(self._init_weights)

	def _init_weights(self, m):
		if isinstance(m, nn.Linear):
			# we use xavier_uniform following official JAX ViT:
			torch.nn.init.xavier_uniform_(m.weight)
			if isinstance(m, nn.Linear) and m.bias is not None:
				nn.init.constant_(m.bias, 0)
		elif isinstance(m, nn.LayerNorm):
			nn.init.constant_(m.bias, 0)
			nn.init.constant_(m.weight, 1.0)

	def patchify(self, imgs):
		"""
		imgs: (N, 3, H, W)
		x: (N, L, patch_size**2 *3)
		"""
		p = self.patch_embed.patch_size[0]
		assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

		h = w = imgs.shape[2] // p
		x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
		x = torch.einsum('nchpwq->nhwpqc', x)
		x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
		return x

	def unpatchify(self, x):
		"""
		x: (N, L, patch_size**2 *3)
		imgs: (N, 3, H, W)
		"""
		p = self.patch_embed.patch_size[0]
		h = w = int(x.shape[1]**.5)
		assert h * w == x.shape[1]
		
		x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
		x = torch.einsum('nhwpqc->nchpwq', x)
		imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
		return imgs

	def random_masking(self, x, mask_ratio):
		"""
		Perform per-sample random masking by per-sample shuffling.
		Per-sample shuffling is done by argsort random noise.
		x: [N, L, D], sequence
		"""
		N, L, D = x.shape  # batch, length, dim
		len_keep = int(L * (1 - mask_ratio))
		
		noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
		
		# sort noise for each sample
		ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
		ids_restore = torch.argsort(ids_shuffle, dim=1)

		# keep the first subset
		ids_keep = ids_shuffle[:, :len_keep]
		x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

		# generate the binary mask: 0 is keep, 1 is remove
		mask = torch.ones([N, L], device=x.device)
		mask[:, :len_keep] = 0
		# unshuffle to get the binary mask
		mask = torch.gather(mask, dim=1, index=ids_restore)

		return x_masked, mask, ids_restore

	def forward_encoder_feat2w(self, x, mask_ratio):
		# embed patches
		x = self.patch_embed(x)

		# add pos embed w/o cls token
		x = x + self.pos_embed[:, 1:, :]

		# masking: length -> length * mask_ratio
		x, mask, ids_restore = self.random_masking(x, mask_ratio)

		# append cls token
		cls_token = self.cls_token + self.pos_embed[:, :1, :]
		cls_tokens = cls_token.expand(x.shape[0], -1, -1)
		x = torch.cat((cls_tokens, x), dim=1)

		# apply encoder Transformer blocks
		for blk in self.blocks:
			x = blk(x)

		# process weight tokens
		B = x.shape[0]
		weight_tokens = einops.repeat(self.weight_tokens, 'n d -> b n d', b=B)
		x = torch.cat([x, weight_tokens], dim=1)

		for blk in self.feat2W_blocks:
			x = blk(x)
		x = self.linear_proj_pacomlp_dim(x)
		trans_out = x[:, -len(self.weight_tokens):, :]
		transformation_matrix = einops.repeat(self.transformation_matrix, 'n d -> b n d', b=B)
		M = torch.matmul(transformation_matrix, trans_out)
		self.pacomlp.set_weight_matrix(M)

		return self.pacomlp, mask
		
	def forward_pacomlp(self, x, pacomlp):
		x = self.patchify(x)
		h = w = int(x.shape[1]**.5)
		B = x.shape[0]
		coord = misc.make_coord_grid([h, w], (-1, 1), device=x.device)
		coord = einops.repeat(coord, 'h w d -> b h w d', b=B)
		coord = coord.reshape(coord.shape[0], -1, coord.shape[-1])
		pred = pacomlp(coord) 
		return pred

	def forward_loss(self, imgs, pred, mask):
		"""
		imgs: [N, 3, H, W]
		pred: [N, L, p*p*3]
		mask: [N, L], 0 is keep, 1 is remove, 
		"""
		target = self.patchify(imgs)
		if self.norm_pix_loss:
			mean = target.mean(dim=-1, keepdim=True)
			var = target.var(dim=-1, keepdim=True)
			target = (target - mean) / (var + 1.e-6)**.5

		loss = (pred - target) ** 2
		loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

		loss = (loss * mask).sum() / mask.sum()  # mean loss on removed/masked patches

		if self.norm_pix_loss:
			pred = pred * (var.sqrt() + 1e-6) + mean
		return loss, pred

	def forward(self, imgs, mask_ratio=0.75):
		pacomlp, mask = self.forward_encoder_feat2w(imgs, mask_ratio)
		pred = self.forward_pacomlp(imgs, pacomlp)  
		loss, pred = self.forward_loss(imgs, pred, mask)
		return loss, pred, mask, None

def hyper_mae_vit_tiny(**kwargs):
	pacomlp = PaCoMlp(pre_mod_depth=1, post_mod_depth=4, in_dim=2, out_dim=768, hidden_dim=256, use_pe=True, pe_dim=128, out_bias=0.5, pe_sigma=1024)
	model = MaskedAutoencoderViTINR(
		patch_size=16, embed_dim=192, depth=12, num_heads=3,
		pacomlp=pacomlp, n_weight_tokens=16,
	mlp_ratio=4, feat2W_depth=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
	return model

def hyper_mae_vit_base(**kwargs):
	pacomlp = PaCoMlp(pre_mod_depth=1, post_mod_depth=4, in_dim=2, out_dim=768, hidden_dim=256, use_pe=True, pe_dim=128, out_bias=0.5, pe_sigma=1024)
	model = MaskedAutoencoderViTINR(
		patch_size=16, embed_dim=768, depth=12, num_heads=12,
		pacomlp=pacomlp, n_weight_tokens=16,
	mlp_ratio=4, feat2W_depth=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
	return model



