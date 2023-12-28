# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch

sys.path.append('..')
import util.misc as misc
import util.lr_sched as lr_sched
import torch.nn.functional as F
import wandb 
import torchvision

class Pretrain():
	def train_one_epoch(self, model: torch.nn.Module,
					data_loader: Iterable, optimizer: torch.optim.Optimizer,
					device: torch.device, epoch: int, loss_scaler,
					log_writer=None,
					args=None):
		model.train(True)
		metric_logger = misc.MetricLogger(delimiter="  ")
		metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
		header = 'Epoch: [{}]'.format(epoch)
		print_freq = 20

		accum_iter = args.accum_iter

		optimizer.zero_grad()

		if log_writer is not None:
			print('log_dir: {}'.format(log_writer.log_dir))

		for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

		# we use a per iteration (instead of per epoch) lr scheduler
			if data_iter_step % accum_iter == 0:
				lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

			samples = samples.to(device, non_blocking=True)

			with torch.cuda.amp.autocast():
				loss, _, _, _ = model(samples, mask_ratio=args.mask_ratio)
		

			loss_value = loss.item()

			if not math.isfinite(loss_value):
				print("Loss is {}, stopping training".format(loss_value))
				sys.exit(1)


			loss /= accum_iter
			loss_scaler(loss, optimizer, parameters=model.parameters(),
					update_grad=(data_iter_step + 1) % accum_iter == 0)
			if (data_iter_step + 1) % accum_iter == 0:
				optimizer.zero_grad()

			torch.cuda.synchronize()

			metric_logger.update(loss=loss_value)

			lr = optimizer.param_groups[0]["lr"]
			metric_logger.update(lr=lr)

			loss_value_reduce = misc.all_reduce_mean(loss_value)
			if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
				""" We use epoch_1000x as the x-axis in tensorboard.
				This calibrates different curves when batch size changes.
				"""
				epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
				log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
				log_writer.add_scalar('lr', lr, epoch_1000x)

			if  args.log_to_wandb and args.gpu==0:
				niters = epoch * len(data_loader) + data_iter_step
				wandb.log(
						{
						"lr": lr,
						"Loss": loss.item()
						},
						step=niters,
					)



		# gather the stats from all processes
		metric_logger.synchronize_between_processes()
		print("Averaged stats:", metric_logger)
		return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


	def val_one_epoch(self, model: torch.nn.Module,
					val_data_loader: Iterable,
					device: torch.device, epoch: int, niters: int,
					log_writer=None, 
					args=None): 
	
		if (epoch+1) % 50 != 0 or args.gpu != 0 or args.vis == 0:
			return None

		model.eval()

		for i, data_item in enumerate(val_data_loader):
			samples, index = data_item
			samples = samples.cuda(device, non_blocking=True)
			B = 4
			samples = samples[:B]
			with torch.no_grad():
				loss, pred, mask, _ = model(samples, mask_ratio=0.75)

			mask = mask[:, :, None].expand(-1, -1, 768)
			pred = self.unpatchify(pred, 16)
			mask = self.unpatchify(mask, 16)

			n_per_row = B
			x_masked = samples * (1-mask)
			print(samples.shape, x_masked.shape, pred.shape)
			image = torch.cat([samples, x_masked, pred], dim=0)
			grid_of_images = torchvision.utils.make_grid(image, nrow=n_per_row)
			grid_of_images.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
			print(grid_of_images.shape)
			if  args.log_to_wandb:
				print("wandb logging")
				vis_grid = wandb.Image(grid_of_images, caption=f"iter{niters:06d}")
				#print(vis_grid.size)

				wandb.log(
				{
					f'vis': vis_grid,
				},
				step=niters,
				)
			break

	def unpatchify(self, x, p=16):
		"""
		x: (N, L, patch_size**2 *3)
		imgs: (N, 3, H, W)
		"""
		h = w = int(x.shape[1]**.5)
		assert h * w == x.shape[1]
		
		x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
		x = torch.einsum('nhwpqc->nchpwq', x)
		imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
		return imgs

pretrain = Pretrain()

