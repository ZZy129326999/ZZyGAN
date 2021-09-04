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
from trainer import * 

from miscc.config import cfg, cfg_from_file



class FineGAN_evaluator(object):

    def __init__(self, dirr, name):

        self.save_dir = os.path.join('ZZYdata11/', dirr)
        mkdir_p(self.save_dir)
        s_gpus = cfg.GPU_ID.split(',')
        self.gpus = [int(ix) for ix in s_gpus]
        self.num_gpus = len(self.gpus)
        torch.cuda.set_device(self.gpus[0])
        cudnn.benchmark = True
        self.batch_size = cfg.TRAIN.BATCH_SIZE * self.num_gpus
        self.name = str(name).zfill(8)


    def evaluate_finegan(self):
        with torch.no_grad():
            if cfg.TRAIN.NET_G == '':
                print('Error: the path for model not found!')
            else:
                # Build and load the generator
                netG = G_NET()
                netG.apply(weights_init)
                netG = torch.nn.DataParallel(netG, device_ids=self.gpus)
                model_dict = netG.state_dict()

                state_dict = \
                    torch.load(cfg.TRAIN.NET_G,
                               map_location=lambda storage, loc: storage)

                state_dict = {k: v for k, v in state_dict.items() if k in model_dict}

                model_dict.update(state_dict)
                netG.load_state_dict(model_dict)
                print('Load ', cfg.TRAIN.NET_G)

                # Uncomment this to print Generator layers
                # print(netG)

                nz = cfg.GAN.Z_DIM
                noise = torch.FloatTensor(self.batch_size, nz)
                noise.data.normal_(0, 1)

                if cfg.CUDA:
                    netG.cuda()
                    noise = noise.cuda()

                netG.eval()

                background_class = cfg.TEST_BACKGROUND_CLASS 
                parent_class = cfg.TEST_PARENT_CLASS 
                child_class = cfg.TEST_CHILD_CLASS
                bg_code = torch.zeros([self.batch_size, cfg.FINE_GRAINED_CATEGORIES])
                p_code = torch.zeros([self.batch_size, cfg.SUPER_CATEGORIES])
                c_code = torch.zeros([self.batch_size, cfg.FINE_GRAINED_CATEGORIES])

                for j in range(self.batch_size):
                    bg_code[j][background_class] = 1
                    p_code[j][parent_class] = 1
                    c_code[j][child_class] = 1

                fake_imgs, fg_imgs, mk_imgs, fgmk_imgs = netG(noise, c_code, p_code, bg_code) # Forward pass through the generator
#                 summary(netG, [(16, 100), (16, 3), (16, 1), (16, 3)])
#                 if not os.path.exists(self.save_dir + '/Cbackground/'):
#                     os.makedirs(self.save_dir + '/Cbackground/')
                if not os.path.exists(self.save_dir + '/cfinal/'):
                    os.makedirs(self.save_dir + '/cfinal/')
                if not os.path.exists(self.save_dir + '/pfinal/'):
                    os.makedirs(self.save_dir + '/pfinal/') 
                if not os.path.exists(self.save_dir + '/pforeground/'):
                    os.makedirs(self.save_dir + '/pforeground/')
                if not os.path.exists(self.save_dir + '/cforeground/'):
                    os.makedirs(self.save_dir + '/cforeground/')
                if not os.path.exists(self.save_dir + '/cmask/'):
                    os.makedirs(self.save_dir + '/cmask/')
                    os.makedirs(self.save_dir + '/cfmask/')
                if not os.path.exists(self.save_dir + '/pmask/'):
                    os.makedirs(self.save_dir + '/pmask/')
                    os.makedirs(self.save_dir + '/pfmask/')
                if not os.path.exists(self.save_dir + '/bg/'):
                    os.makedirs(self.save_dir + '/bg/')
                self.save_image(fake_imgs[0][0], self.save_dir + '/bg/' + self.name, 'background')
                self.save_image64(fake_imgs[1][0], self.save_dir + '/pfinal/' + self.name, 'parent_final')
                self.save_image64(fake_imgs[2][0], self.save_dir + '/cfinal/' + self.name, 'child_final')  # 'child_final'
                self.save_image64(fg_imgs[0][0], self.save_dir + '/pforeground/' + self.name, 'parent_foreground')
                self.save_image64(fg_imgs[1][0], self.save_dir + '/cforeground/' + self.name, 'child_foreground')
                self.save_image64(mk_imgs[0][0], self.save_dir + '/pmask/' + self.name, 'parent_mask')
                self.save_image64(mk_imgs[1][0], self.save_dir + '/cmask/' + self.name, 'child_mask')
                self.save_image64(fgmk_imgs[0][0], self.save_dir + '/pfmask/' + self.name, 'parent_foreground_masked')
                self.save_image64(fgmk_imgs[1][0], self.save_dir + '/cfmask/' + self.name, 'child_foreground_masked')


    def save_image(self, images, save_dir, iname):
        
        img_name = '%s.jpg' % (save_dir)
        full_path = os.path.join(save_dir, img_name)
        
        if (iname.find('mask') == -1) or (iname.find('foreground') != -1):
            img = images.add(1).div(2).mul(255).clamp(0, 255).byte()
            ndarr = img.permute(1, 2, 0).data.cpu().numpy()
            im = Image.fromarray(ndarr)
            im.save(img_name)

        else:
            img = images.mul(255).clamp(0, 255).byte()
            ndarr = img.data.cpu().numpy()
            ndarr = np.reshape(ndarr, (ndarr.shape[-1], ndarr.shape[-1], 1))
            ndarr = np.repeat(ndarr, 3, axis=2)
            im = Image.fromarray(ndarr)
            im.save(img_name)
            
    def save_image64(self, images, save_dir, iname):
        
        img_name = '%s.jpg' % (save_dir)
        full_path = os.path.join(save_dir, img_name)
        
        if (iname.find('mask') == -1) or (iname.find('foreground') != -1):
            img = images.add(1).div(2).mul(255).clamp(0, 255).byte()
            ndarr = img.permute(1, 2, 0).data.cpu().numpy()
            im = Image.fromarray(ndarr).resize((64, 64))
            im.save(img_name)

        else:
            img = images.mul(255).clamp(0, 255).byte()
            ndarr = img.data.cpu().numpy()
            ndarr = np.reshape(ndarr, (ndarr.shape[-1], ndarr.shape[-1], 1))
            ndarr = np.repeat(ndarr, 3, axis=2)
            im = Image.fromarray(ndarr).resize((64, 64))
            im.save(img_name)



# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
if __name__ == "__main__":
    classes = ['BlackMeaslesFungus', 'BlackRotFungus', 'LeafBlightFungus']
    cfg.GPU_ID = '0'
    cfg.DATASET_NAME = 'Grape11'  # ./dataroot/dataset/dirs(001. 002. 003.)
    cfg.DATA_DIR = '../data/'
        
    cfg.TEST_CHILD_CLASS = 0
    cfg.TEST_BACKGROUND_CLASS = 0
    cfg.TRAIN.NET_G = '../output/chair_2021_05_20_12_53_42/Model/netG_9000.pth'
    cfg.TRAIN.NET_D = '../output/chair_2021_05_20_12_53_42/Model/netD'
        
    
    print('Using config:')
    pprint.pprint(cfg)
    
    for i in range(0, 10000, 1):  # 0, 1
        random.seed(i)
        torch.manual_seed(i)
        if cfg.CUDA:
            torch.cuda.manual_seed_all(i)
        print(i, i//3 + 1 + 10000)
#         from trainer import FineGAN_evaluator as evaluator
        algo = FineGAN_evaluator(classes[1], i//3 + 1 + 10000)
        algo.evaluate_finegan()