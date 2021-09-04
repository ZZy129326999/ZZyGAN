import torch 
import torch.nn as nn 
import torchvision.utils 
from torch.nn.parallel import DistributedDataParallel as NativeDDP

import timm
from timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.models import create_model, resume_checkpoint, load_checkpoint, convert_splitbn_model, model_parameters
from timm.utils import *
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import ApexScaler, NativeScaler

from apex import amp
from apex.parallel import DistributedDataParallel as ApexDDP
from apex.parallel import convert_syncbn_model

from types import SimpleNamespace
import numpy as np 
import random 
import matplotlib.pyplot as plt
import os 
import argparse
import logging
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model

    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

torch.backends.cudnn.benchmark = True

from timm.data.dataset import ImageDataset

config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Dataset / Model parameters
parser.add_argument('--data_dir', metavar='DIR', default='./data/images_original/',
                    help='path to dataset')
parser.add_argument('--dataset', '-d', metavar='NAME', default='',
                    help='dataset type (default: ImageFolder/ImageTar if empty)')
parser.add_argument('--train-split', metavar='NAME', default='train',
                    help='dataset train split (default: train)')
parser.add_argument('--val-split', metavar='NAME', default='validation',
                    help='dataset validation split (default: validation)')
parser.add_argument('--model', default='resnet50', type=str, metavar='MODEL',
                    help='Name of model to train (default: "volo_d1"')
parser.add_argument('--pretrained', action='store_true', default=True,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
parser.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')
parser.add_argument('--num-classes', type=int, default=10, metavar='N',
                    help='number of label classes (Model default if None)')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--img-size', type=int, default=224, metavar='N',
                    help='Image patch size (default: None => model default)')
parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N',
                    help='Input all image dimensions (d h w, e.g. --input-size 3 224 224),'
                         ' uses model default if empty')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop percent (for validation only)')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('-b', '--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 16)')
parser.add_argument('-vb', '--validation-batch-size-multiplier', type=int, default=1, metavar='N',
                    help='ratio of validation batch size to training batch size (default: 1)')

# Optimizer parameters
parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "adamw"')
parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: None, use opt default)')
parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                    help='Optimizer Betas (default: None, use opt default)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='Optimizer momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=0.05,
                    help='weight decay (default: 0.05)')
parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')
parser.add_argument('--clip-mode', type=str, default='norm',
                    help='Gradient clipping mode. One of ("norm", "value", "agc")')

# Learning rate schedule parameters
parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "cosine"')
parser.add_argument('--lr', type=float, default=1.6e-3, metavar='LR',
                    help='learning rate (default: 1.6e-3)')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                    help='learning rate cycle len multiplier (default: 1.0)')
parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                    help='learning rate cycle limit')
parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 300)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')  # None
parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=20, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')

# Augmentation & regularization parameters
parser.add_argument('--no-aug', action='store_true', default=False,
                    help='Disable all training augmentation, override other train aug args')
parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                    help='Random resize scale (default: 0.08 1.0)')
parser.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
                    help='Random resize aspect ratio (default: 0.75 1.33)')
parser.add_argument('--hflip', type=float, default=0.5,
                    help='Horizontal flip training aug probability')
parser.add_argument('--vflip', type=float, default=0.,
                    help='Vertical flip training aug probability')
parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". (default: rand-m9-mstd0.5-inc1)'),
parser.add_argument('--aug-splits', type=int, default=0,
                    help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
parser.add_argument('--jsd', action='store_true', default=False,
                    help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                    help='Random erase prob (default: 0.25)')
parser.add_argument('--remode', type=str, default='pixel',
                    help='Random erase mode (default: "pixel")')
parser.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
parser.add_argument('--resplit', action='store_true', default=False,
                    help='Do not random erase first (clean) augmentation split')
parser.add_argument('--mixup', type=float, default=0.0,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix', type=float, default=0.0,
                    help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
parser.add_argument('--mixup-prob', type=float, default=1.0,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup-mode', type=str, default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                    help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
parser.add_argument('--smoothing', type=float, default=0.1,
                    help='Label smoothing (default: 0.1)')
parser.add_argument('--train-interpolation', type=str, default='random',
                    help='Training interpolation (random, bilinear, bicubic default: "random")')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
parser.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                    help='Drop path rate (default: None)')
parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')

# Batch norm parameters (only works with gen_efficientnet based models currently)
parser.add_argument('--bn-tf', action='store_true', default=False,
                    help='Use Tensorflow BatchNorm defaults for models that support it (default: False)')
parser.add_argument('--bn-momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
parser.add_argument('--bn-eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')
parser.add_argument('--sync-bn', action='store_true',
                    help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
parser.add_argument('--dist-bn', type=str, default='',
                    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
parser.add_argument('--split-bn', action='store_true',
                    help='Enable separate BN layers per augmentation split.')

# Model Exponential Moving Average
parser.add_argument('--model-ema', action='store_true', default=False,
                    help='Enable tracking moving average of model weights')
parser.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                    help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
parser.add_argument('--model-ema-decay', type=float, default=0.99992,
                    help='decay factor for model weights moving average (default: 0.99992)')

# Misc
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')
parser.add_argument('--checkpoint-hist', type=int, default=100, metavar='N',
                    help='number of checkpoints to keep (default: 10)')  # 10
parser.add_argument('-j', '--workers', type=int, default=1, metavar='N',
                    help='how many training processes to use (default: 1)')  # 8
parser.add_argument('--save-images', action='store_true', default=False,
                    help='save images of input bathes every log interval for debugging')
parser.add_argument('--amp', action='store_true', default=False,
                    help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
parser.add_argument('--apex-amp', action='store_true', default=True,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native-amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
parser.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                    help='Best metric (default: "top1"')
parser.add_argument('--tta', type=int, default=0, metavar='N',
                    help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                    help='use the multi-epochs-loader to save time at the beginning of every epoch')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')

# Token labeling

parser.add_argument('--token-label', action='store_true', default=False,
                    help='Use dense token-level label map for training')
parser.add_argument('--token-label-data', type=str, default='./label_top5_train_nfnet', metavar='DIR',
                    help='path to token_label data')
parser.add_argument('--token-label-size', type=int, default=14, metavar='N',
                    help='size of result token label map')
parser.add_argument('--dense-weight', type=float, default=0.5,
                    help='Token labeling loss multiplier (default: 0.5)')
parser.add_argument('--cls-weight', type=float, default=1.0,
                    help='Cls token prediction loss multiplier (default: 1.0)')
parser.add_argument('--ground-truth', action='store_true', default=False,
                    help='mix ground truth when use token labeling')

# Finetune
parser.add_argument('--finetune', default='', type=str, metavar='PATH',
                    help='path to checkpoint file (default: none)')


datapath = r'./data/images_original'


def main():
	args = parser.parse_args()
	args.distributed = False
	if 'WORLD_SIZE' in os.environ:
		args.distributed = int(os.environ['WORLD_SIZE']) > 1
	if args.distributed:
		args.device = 'cuda:%d' % args.local_rank
		torch.cuda.set_device(args.local_rank) 
		args.world_size = int(os.environ['WORLD_SIZE'])
		args.rank = int(os.environ['RANK'])
		torch.distributed.init_process_group(backend='nccl', init_method='env://', rank=args.rank, world_size=args.world_size)
		args.world_size = torch.distributed.get_world_size()
		args.rank = torch.distributed.get_rank()
		
	use_amp = None
	if args.amp:
		# `--amp` chooses native amp before apex (APEX ver not actively maintained)
		if has_native_amp:
			args.native_amp = True
		elif has_apex:
			args.apex_amp = True
	if args.apex_amp and has_apex:
		use_amp = 'apex'
	elif args.native_amp and has_native_amp:
		use_amp = 'native'
		
	print(args.device, "=================================================================")
	print(args.device, "Loading dataset:", datapath)
	dataset = ImageDataset(datapath)
	print(args.device, len(dataset), dataset.parser.class_to_idx)
	dataset_train = create_dataset("train", root=datapath, split="train", is_training=True)
	loader_train = create_loader(dataset_train, input_size=(3, 224, 224), batch_size=32, is_training=True, no_aug=True, distributed=args.distributed)
	dataset_val = create_dataset("val", root=datapath, split="val", is_training=False)
	loader_val = create_loader(dataset_val, input_size=(3, 224, 224), batch_size=32, is_training=False, no_aug=False, distributed=args.distributed)
	print(args.device, "Successfully loaded dataset:", datapath)
	print(args.device, "Training set:", len(dataset_train), "   Validation set:", len(dataset_val))
	print(args.device, "=================================================================")
	
	model = create_model(args.model, pretrained=args.pretrained, num_classes=args.num_classes)
	model.cuda()
	if args.channels_last:
		model = model.to(memory_format=torch.channels_last)
	if args.distributed and args.sync_bn:
		assert not args.split_bn
		if has_apex and use_amp != 'native':
			# Apex SyncBN preferred unless native amp is activated
			model = convert_syncbn_model(model)
		else:
			model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
	
	
	train_loss_fn = SoftTargetCrossEntropy().cuda()
	validate_loss_fn = nn.CrossEntropyLoss().cuda()
			
	optimizer = create_optimizer(args, model)
	amp_autocast = suppress  # do nothing
	loss_scaler = None
	if use_amp == 'apex':
		model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
		loss_scaler = ApexScaler()
	elif use_amp == 'native':
		amp_autocast = torch.cuda.amp.autocast
		loss_scaler = NativeScaler()
	if args.distributed:
		if has_apex and use_amp != 'native':
			# Apex DDP preferred unless native amp is activated
			model = ApexDDP(model, delay_allreduce=True)
		else:
			model = NativeDDP(model, device_ids=[args.local_rank])  # can use device str in Torch >= 1.1
			# NOTE: EMA model does not need to be wrapped by DDP
	print(args.device, use_amp)
	
# 	loss = train_loss_fn
# 	optimizer.zero_grad()
# 	if loss_scaler is not None:
# 		loss_scaler(loss, optimizer,
# 				clip_grad=args.clip_grad, clip_mode=args.clip_mode,
# 				parameters=model_parameters(model, exclude_head='agc' in args.clip_mode),
# 				create_graph=second_order)
# 	else:
# 		loss.backward(create_graph=second_order)
# 		if args.clip_grad is not None:
# 				dispatch_clip_grad(model_parameters(model, exclude_head='agc' in args.clip_mode),
# 							value=args.clip_grad, mode=args.clip_mode)
# 		optimizer.step()
	
			
	# optionally resume from a checkpoint
	resume_epoch = None
	if args.resume:
		resume_epoch = resume_checkpoint(
			model,
			args.resume,
			optimizer=None if args.no_resume_opt else optimizer,
			loss_scaler=None if args.no_resume_opt else loss_scaler,
			log_info=args.local_rank == 0)

	# setup exponential moving average of model weights, SWA could be used here too
	model_ema = None
	if args.model_ema:
		# Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
		model_ema = ModelEmaV2(
			model,
			decay=args.model_ema_decay,
			device='cpu' if args.model_ema_force_cpu else None)
		if args.resume:
			load_checkpoint(model_ema.module, args.resume, use_ema=True)
	
	# setup learning rate schedule and starting epoch
	lr_scheduler, num_epochs = create_scheduler(args, optimizer)
	start_epoch = 0
	if args.start_epoch is not None:
		# a specified start_epoch will always override the resume epoch
		start_epoch = args.start_epoch
	elif resume_epoch is not None:
		start_epoch = resume_epoch
	if lr_scheduler is not None and start_epoch > 0:
		lr_scheduler.step(start_epoch)

	

if __name__ == '__main__':
	main()