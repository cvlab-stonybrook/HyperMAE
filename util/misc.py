# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import builtins
import datetime
import os
import time
from collections import defaultdict, deque
from pathlib import Path

import torch
import torch.distributed as dist
from torch._six import inf
import torch.nn.functional as F
import torch.nn as nn
from sklearn.cluster import KMeans
import numpy as np
import math

class SmoothedValue(object):
	"""Track a series of values and provide access to smoothed values over a
	window or the global series average.
	"""

	def __init__(self, window_size=20, fmt=None):
		if fmt is None:
			fmt = "{median:.4f} ({global_avg:.4f})"
		self.deque = deque(maxlen=window_size)
		self.total = 0.0
		self.count = 0
		self.fmt = fmt

	def update(self, value, n=1):
		self.deque.append(value)
		self.count += n
		self.total += value * n

	def synchronize_between_processes(self):
		"""
		Warning: does not synchronize the deque!
		"""
		if not is_dist_avail_and_initialized():
			return
		t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
		dist.barrier()
		dist.all_reduce(t)
		t = t.tolist()
		self.count = int(t[0])
		self.total = t[1]

	@property
	def median(self):
		d = torch.tensor(list(self.deque))
		return d.median().item()

	@property
	def avg(self):
		d = torch.tensor(list(self.deque), dtype=torch.float32)
		return d.mean().item()

	@property
	def global_avg(self):
		return self.total / self.count

	@property
	def max(self):
		return max(self.deque)

	@property
	def value(self):
		return self.deque[-1]

	def __str__(self):
		return self.fmt.format(
			median=self.median,
			avg=self.avg,
			global_avg=self.global_avg,
			max=self.max,
			value=self.value)


class MetricLogger(object):
	def __init__(self, delimiter="\t"):
		self.meters = defaultdict(SmoothedValue)
		self.delimiter = delimiter

	def update(self, **kwargs):
		for k, v in kwargs.items():
			if v is None:
				continue
			if isinstance(v, torch.Tensor):
				v = v.item()
			assert isinstance(v, (float, int))
			self.meters[k].update(v)

	def __getattr__(self, attr):
		if attr in self.meters:
			return self.meters[attr]
		if attr in self.__dict__:
			return self.__dict__[attr]
		raise AttributeError("'{}' object has no attribute '{}'".format(
			type(self).__name__, attr))

	def __str__(self):
		loss_str = []
		for name, meter in self.meters.items():
			loss_str.append(
				"{}: {}".format(name, str(meter))
			)
		return self.delimiter.join(loss_str)

	def synchronize_between_processes(self):
		for meter in self.meters.values():
			meter.synchronize_between_processes()

	def add_meter(self, name, meter):
		self.meters[name] = meter

	def log_every(self, iterable, print_freq, header=None):
		i = 0
		if not header:
			header = ''
		start_time = time.time()
		end = time.time()
		iter_time = SmoothedValue(fmt='{avg:.4f}')
		data_time = SmoothedValue(fmt='{avg:.4f}')
		space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
		log_msg = [
			header,
			'[{0' + space_fmt + '}/{1}]',
			'eta: {eta}',
			'{meters}',
			'time: {time}',
			'data: {data}'
		]
		if torch.cuda.is_available():
			log_msg.append('max mem: {memory:.0f}')
		log_msg = self.delimiter.join(log_msg)
		MB = 1024.0 * 1024.0
		for obj in iterable:
			data_time.update(time.time() - end)
			yield obj
			iter_time.update(time.time() - end)
			if i % print_freq == 0 or i == len(iterable) - 1:
				eta_seconds = iter_time.global_avg * (len(iterable) - i)
				eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
				if torch.cuda.is_available():
					print(log_msg.format(
						i, len(iterable), eta=eta_string,
						meters=str(self),
						time=str(iter_time), data=str(data_time),
						memory=torch.cuda.max_memory_allocated() / MB))
				else:
					print(log_msg.format(
						i, len(iterable), eta=eta_string,
						meters=str(self),
						time=str(iter_time), data=str(data_time)))
			i += 1
			end = time.time()
		total_time = time.time() - start_time
		total_time_str = str(datetime.timedelta(seconds=int(total_time)))
		print('{} Total time: {} ({:.4f} s / it)'.format(
			header, total_time_str, total_time / len(iterable)))


def setup_for_distributed(is_master):
	"""
	This function disables printing when not in master process
	"""
	builtin_print = builtins.print

	def print(*args, **kwargs):
		force = kwargs.pop('force', False)
		force = force or (get_world_size() > 8)
		if is_master or force:
			now = datetime.datetime.now().time()
			builtin_print('[{}] '.format(now), end='')  # print with time stamp
			builtin_print(*args, **kwargs)

	builtins.print = print


def is_dist_avail_and_initialized():
	if not dist.is_available():
		return False
	if not dist.is_initialized():
		return False
	return True


def get_world_size():
	if not is_dist_avail_and_initialized():
		return 1
	return dist.get_world_size()


def get_rank():
	if not is_dist_avail_and_initialized():
		return 0
	return dist.get_rank()


def is_main_process():
	return get_rank() == 0


def save_on_master(*args, **kwargs):
	if is_main_process():
		torch.save(*args, **kwargs)


def init_distributed_mode(args):
	if args.dist_on_itp:
		args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
		args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
		args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
		args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
		os.environ['LOCAL_RANK'] = str(args.gpu)
		os.environ['RANK'] = str(args.rank)
		os.environ['WORLD_SIZE'] = str(args.world_size)
		# ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
	elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
		args.rank = int(os.environ["RANK"])
		args.world_size = int(os.environ['WORLD_SIZE'])
		args.gpu = int(os.environ['LOCAL_RANK'])
	elif 'SLURM_PROCID' in os.environ:
		args.rank = int(os.environ['SLURM_PROCID'])
		args.gpu = args.rank % torch.cuda.device_count()
	else:
		print('Not using distributed mode')
		setup_for_distributed(is_master=True)  # hack
		args.distributed = False
		return

	args.distributed = True

	torch.cuda.set_device(args.gpu)
	#args.dist_backend = 'nccl'
	print('| distributed init (rank {}): {}, gpu {}'.format(
		args.rank, args.dist_url, args.gpu), flush=True)
	print(args.dist_url, args.world_size, args.dist_backend)
	torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
										 world_size=args.world_size, rank=args.rank)
	print("entering barrier")
	torch.distributed.barrier()
	print("out of barrier")
	setup_for_distributed(args.rank == 0)


def init_distributed_mode_v2(args):
	args.distributed = True
	args.gpu = args.rank
	torch.multiprocessing.set_start_method('fork', force=True)
	# suppress printing if not master

	torch.cuda.set_device(args.gpu)
	#args.dist_backend = 'nccl'
	print('| distributed init (rank {}): {}, gpu {}'.format(
		args.rank, args.dist_url, args.gpu), flush=True)
	print(args.dist_url, args.world_size, args.dist_backend)
	torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
										 world_size=args.world_size, rank=args.rank)
	print("entering barrier")
	torch.distributed.barrier()
	print("out of barrier")
	if (args.gpu != 0 or args.rank != 0):
		def print_pass(*args):
			pass
		builtins.print = print_pass
	#setup_for_distributed(args.rank == 0)


class NativeScalerWithGradNormCount:
	state_dict_key = "amp_scaler"

	def __init__(self):
		self._scaler = torch.cuda.amp.GradScaler()

	def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
		self._scaler.scale(loss).backward(create_graph=create_graph)
		if update_grad:
			if clip_grad is not None:
				assert parameters is not None
				self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
				norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
			else:
				self._scaler.unscale_(optimizer)
				norm = get_grad_norm_(parameters)
			self._scaler.step(optimizer)
			self._scaler.update()
		else:
			norm = None
		return norm

	def state_dict(self):
		return self._scaler.state_dict()

	def load_state_dict(self, state_dict):
		self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
	if isinstance(parameters, torch.Tensor):
		parameters = [parameters]
	parameters = [p for p in parameters if p.grad is not None]
	norm_type = float(norm_type)
	if len(parameters) == 0:
		return torch.tensor(0.)
	device = parameters[0].grad.device
	if norm_type == inf:
		total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
	else:
		total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
	return total_norm


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler):
	output_dir = Path(args.output_dir)
	epoch_name = str(epoch)
	if loss_scaler is not None:
		checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
		for checkpoint_path in checkpoint_paths:
			to_save = {
				'model': model_without_ddp.state_dict(),
				'optimizer': optimizer.state_dict(),
				'epoch': epoch,
				'scaler': loss_scaler.state_dict(),
				'args': args,
			}

			save_on_master(to_save, checkpoint_path)
	else:
		client_state = {'epoch': epoch}
		model.save_checkpoint(save_dir=args.output_dir, tag="checkpoint-%s" % epoch_name, client_state=client_state)


def load_model(args, model_without_ddp, optimizer, loss_scaler):
	if args.resume:
		if args.resume.startswith('https'):
			checkpoint = torch.hub.load_state_dict_from_url(
				args.resume, map_location='cpu', check_hash=True)
		else:
			checkpoint = torch.load(args.resume, map_location='cpu')
		model_without_ddp.load_state_dict(checkpoint['model'])
		print("Resume checkpoint %s" % args.resume)
		if 'optimizer' in checkpoint and 'epoch' in checkpoint and not (hasattr(args, 'eval') and args.eval):
			optimizer.load_state_dict(checkpoint['optimizer'])
			args.start_epoch = checkpoint['epoch'] + 1
			if 'scaler' in checkpoint:
				loss_scaler.load_state_dict(checkpoint['scaler'])
			print("With optim & sched!")

def make_coord_grid(shape, range, device=None):
	"""
		Args:
			shape: tuple
			range: [minv, maxv] or [[minv_1, maxv_1], ..., [minv_d, maxv_d]] for each dim
		Returns:
			grid: shape (*shape, )
	"""
	l_lst = []
	for i, s in enumerate(shape):
		l = (0.5 + torch.arange(s, device=device)) / s
		if isinstance(range[0], list) or isinstance(range[0], tuple):
			minv, maxv = range[i]
		else:
			minv, maxv = range
		l = minv + (maxv - minv) * l
		l_lst.append(l)
	grid = torch.meshgrid(*l_lst, indexing='ij')
	grid = torch.stack(grid, dim=-1)
	return grid


def all_reduce_mean(x):
	world_size = get_world_size()
	if world_size > 1:
		x_reduce = torch.tensor(x).cuda()
		dist.all_reduce(x_reduce)
		x_reduce /= world_size
		return x_reduce.item()
	else:
		return x


def off_diagonal(x):
	n, m = x.shape
	assert n == m
	return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def vic_reg(x):
	std_x = torch.mean(torch.sqrt(x.var(dim=0) + 0.0001))
	cov_x = (x.T @ x) / (x.shape[0] - 1)
	cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
		x.shape[-1])
	return std_x.item(), cov_loss.item()

class KoLeoLoss(nn.Module):
	"""Kozachenko-Leonenko entropic loss regularizer from Sablayrolles et al. - 2018 - Spreading vectors for similarity search"""

	def __init__(self):
		super().__init__()
		self.pdist = nn.PairwiseDistance(2, eps=1e-8)

	def pairwise_NNs_inner(self, x):
		"""
		Pairwise nearest neighbors for L2-normalized vectors.
		Uses Torch rather than Faiss to remain on GPU.
		"""
		# parwise dot products (= inverse distance)
		dots = torch.mm(x, x.t())
		n = x.shape[0]
		dots.view(-1)[:: (n + 1)].fill_(-1)  # Trick to fill diagonal with -1
		# max inner prod -> min distance
		_, I = torch.max(dots, dim=1)  # noqa: E741
		return I

	def forward(self, student_output, eps=1e-8):
		"""
		Args:
			student_output (BxD): backbone output of student
		"""
		with torch.cuda.amp.autocast(enabled=False):
			student_output = F.normalize(student_output, eps=eps, p=2, dim=-1)
			I = self.pairwise_NNs_inner(student_output)  # noqa: E741
			distances = self.pdist(student_output, student_output[I])  # BxD, BxD -> B
			loss = -torch.log(distances + eps).mean()
		return loss.item()


def evaluate_kmeans_entropy(features, nclasses, dist_sign, use_norm):
	entropy_list = []
	if use_norm == True:
		features = F.normalize(features, eps=1e-8, p=2, dim=-1)
	features = features.numpy()
	for rand_state in range(0, 5):
		kmeans_fit = KMeans(n_clusters=nclasses,random_state=rand_state, n_init=10).fit(features)
		centers = kmeans_fit.cluster_centers_  # 100 X D 
		features = features[:, :, np.newaxis]  
		centers = np.transpose(centers)[np.newaxis, :, :] # torch.bmm
		centers = np.repeat(centers, repeats=features.shape[0], axis=0)
		#print("features shape", features.shape)
		#print("centers shape", centers.shape)
		dist = (features - centers) ** 2
		dist = np.sum(dist, axis=1)
		dist = torch.tensor(dist)
		class_prob = torch.nn.functional.softmax(dist_sign*dist, dim=-1)
		avg_class_prob = torch.mean(class_prob, dim=0)
		#print("avg_class_prob", avg_class_prob) 
		#target_prob = 0.01
		entropy = - (1/nclasses) * torch.log2(avg_class_prob)
		#print(avg_class_prob)
		entropy = torch.sum(entropy)
		entropy = entropy.item()
		#print("entropy", entropy) #!!!!!!
		features = features[:, :, 0]
		entropy_list.append(entropy)
	final_entropy = sum(entropy_list) / len(entropy_list)
	print("entropy", entropy_list, "mean_entropy", entropy)
	return final_entropy

def hypo_initialize_params(params, init_type, **kwargs):
    fan_in, fan_out = params.shape[0], params.shape[1]
    if init_type is None or init_type == "normal":
        nn.init.normal_(params)
    elif init_type == "kaiming_uniform":
        nn.init.kaiming_uniform_(params, a=math.sqrt(5))
    elif init_type == "uniform_fan_in":
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(params, -bound, bound)
    elif init_type == "zero":
        nn.init.zeros_(params)
    elif "siren" == init_type:
        assert "siren_w0" in kwargs.keys() and "is_first" in kwargs.keys()
        w0 = kwargs["siren_w0"]
        if kwargs["is_first"]:
            w_std = 1 / fan_in
        else:
            w_std = math.sqrt(6.0 / fan_in) / w0
        nn.init.uniform_(params, -w_std, w_std)
    else:
        raise NotImplementedError


def hypo_create_params_with_init(shape, init_type="normal", include_bias=False, bias_init_type="zero", **kwargs):
    if not include_bias:
        params = torch.empty([shape[0], shape[1]])
        hypo_initialize_params(params, init_type, **kwargs)
        return params
    else:
        params = torch.empty([shape[0] - 1, shape[1]])
        bias = torch.empty([1, shape[1]])

        hypo_initialize_params(params, init_type, **kwargs)
        hypo_initialize_params(bias, bias_init_type, **kwargs)
        return torch.cat([params, bias], dim=0)

def hypo_create_activation(act):
    if act == "relu":
        activation = nn.ReLU()
    elif act == 'silu':
        activation = nn.SiLU()
    else:
        raise NotImplementedError
    return activation