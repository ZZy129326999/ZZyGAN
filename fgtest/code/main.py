from __future__ import print_function
import torch
import torchvision.transforms as transforms

import argparse
import os
import random
import sys
import pprint
import datetime
import dateutil.tz
import time
import pickle

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
print('dir_path =', dir_path)
sys.path.append(dir_path)
# sys.exit()

from miscc.config import cfg, cfg_from_file


def parse_args():
    parser = argparse.ArgumentParser(description='Train a FDGAN network')
#     parser.add_argument('--cfg', dest='cfg_file', help='optional config file', default='', type=str)  # cfg/Grape_proGAN.yml
    parser.add_argument('--gpu', dest='gpu_id', type=str, default='0')
    parser.add_argument('--dataroot', dest='dataroot', type=str, default='../data/')
    parser.add_argument('--dataset', dest='dataset', type=str, default='Fungus')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
if __name__ == "__main__":
    args = parse_args()
#     if args.cfg_file is not None:
#         cfg_from_file(args.cfg_file)
 
    if args.gpu_id != '-1':
        cfg.GPU_ID = args.gpu_id
    else:
        cfg.CUDA = False

    if args.dataset != 'Grape1':
        cfg.DATASET_NAME = args.dataset  # ./dataroot/dataset/dirs(001. 002. 003.)
    if args.dataroot != '../data/':
        cfg.DATA_DIR = args.dataroot
    if cfg.TRAIN.FLAG:
        print('Using config:')
        pprint.pprint(cfg)

    if not cfg.TRAIN.FLAG:
        args.manualSeed = 9987   # Change this to have different random seed during evaluation
    elif args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
        now = datetime.datetime.now(dateutil.tz.tzlocal())
        timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
        if not os.path.exists('seeds'):
            os.makedirs('seeds')
        with open('seeds/Seed%s.txt'.format(timestamp), 'w') as p:
            p.write(str(args.manualSeed))
        
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)
    
    # Evaluation part
    if not cfg.TRAIN.FLAG:
        from trainer import FineGAN_evaluator as evaluator
        algo = evaluator()
        algo.evaluate_finegan()

    # Training part
    else:
        now = datetime.datetime.now(dateutil.tz.tzlocal())
        timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
        output_dir = '../output/%s_%s' % (cfg.DATASET_NAME, timestamp)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        bshuffle = True

        # Get data loader
        imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM-1))
        print(imsize)  # 64*(2**2) == 64*4 = 256
        image_transform = transforms.Compose([
            transforms.Resize(int(imsize * 76 / 64)), 
            transforms.RandomCrop(imsize),
            transforms.RandomHorizontalFlip()])
        """
        image_transform = Compose(
            Scale(size=64, interpolation=PIL.Image.BILINEAR)
            RandomCrop(size=(64, 64), padding=None)
            RandomHorizontalFlip(p=0.5)
        )
        """
#         sys.exit()
        from datasets import Dataset
        dataset = Dataset(cfg.DATA_DIR,
                              base_size=cfg.TREE.BASE_SIZE,
                              transform=image_transform)
        assert dataset
        num_gpu = len(cfg.GPU_ID.split(','))
        print("---num_gpu =", num_gpu)  # 1 gpu
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=cfg.TRAIN.BATCH_SIZE * num_gpu,
            drop_last=True, shuffle=bshuffle, num_workers=int(cfg.WORKERS))
        print('---num_workers =', cfg.WORKERS)  # workers = 6


        from trainer import FineGAN_trainer as trainer
#         torch.cuda.empty_cache()
        algo = trainer(output_dir, dataloader, imsize)

        torch.cuda.empty_cache()
        start_t = time.time()
        algo.train()
        end_t = time.time()
        print('Total time for training:', end_t - start_t)
