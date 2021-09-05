from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
from easydict import EasyDict as edict


__C = edict()
cfg = __C

# Dataset name: Grapes
__C.DATASET_NAME = 'Grape11'
__C.CONFIG_NAME = 'zzy'
__C.DATA_DIR = '../data/'
__C.SAVE_DIR = 'outputs'
__C.GPU_ID = '0'
__C.CUDA = True

__C.WORKERS = 6

__C.TREE = edict()
__C.TREE.BRANCH_NUM = 3  # operactor numbers  fang da 2** ji bei 1
__C.TREE.BASE_SIZE = 64  # fundimentional image size
__C.SUPER_CATEGORIES = 3  
__C.FINE_GRAINED_CATEGORIES = 3  # cateoriges
__C.TEST_CHILD_CLASS = 0
__C.TEST_PARENT_CLASS = 0
__C.TEST_BACKGROUND_CLASS = 0
__C.TIED_CODES = True

# Test options
__C.TEST = edict()

# Training options
__C.TRAIN = edict()
__C.TRAIN.BATCH_SIZE = 16  # batch size
__C.TRAIN.BG_LOSS_WT = 10 
__C.TRAIN.VIS_COUNT = 64  # generator images numbers
__C.TRAIN.MAX_EPOCH = 30000
__C.TRAIN.HARDNEG_MAX_ITER = 1500
__C.TRAIN.SNAPSHOT_INTERVAL = 1000
__C.TRAIN.SNAPSHOT_INTERVAL_HARDNEG = 1000
__C.TRAIN.DISCRIMINATOR_LR = 2e-4
__C.TRAIN.GENERATOR_LR = 2e-4
__C.TRAIN.FLAG = True
__C.TRAIN.NET_G = '../output/chair_2021_05_20_12_44_43/Model/netG_1000.pth'
__C.TRAIN.NET_D = '../output/chair_2021_05_20_12_44_43/Model/netD'
# __C.TRAIN.NET_G = ''  # if GPU is enough, please // it
# __C.TRAIN.NET_D = ''  # if GPU is enough, please // it

# Modal options
__C.GAN = edict()
__C.GAN.DF_DIM = 64
__C.GAN.GF_DIM = 64
__C.GAN.Z_DIM = 100
__C.GAN.NETWORK_TYPE = 'default'
__C.GAN.R_NUM = 2




def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.iteritems():
        # a must specify keys that are in b
        if not b.has_key(k):
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)
