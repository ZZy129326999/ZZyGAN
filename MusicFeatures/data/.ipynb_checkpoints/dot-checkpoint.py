import os
import sys 
from PIL import Image 
import numpy as np
import torch
from torchvision.transforms import ToPILImage

path = r'./images/'

def conver(path):
    path_ = os.path.join('./genres', path.split('images/')[-1])  # images/ / is note
    return path_

def multi(path1, path2, train_val, dirname, file):
    image1 = Image.open(path1).convert('RGB').resize((224, 224))
    image2 = Image.open(path2).convert('RGB').resize((224, 224))
    ts1 = np.array(image1)
    ts2 = np.array(image2)
    for i, xs in enumerate(ts1):
        for j, x in enumerate(xs):
            for k, xx in enumerate(x):
                if ts1[i,j,k] < ts2[i,j,k]:
                    ts1[i,j,k] = ts2[i,j,k] 
    ts1 = torch.from_numpy(ts1)
#     ts1 = ts1/224/255
    ts1 = ts1.permute(2, 0, 1)
    print(ts1.shape)
    newimage = ToPILImage()(ts1)
    print(os.path.join('./maximages/', train_val, dirname, file))
    save_path = os.path.join('./maximages/', train_val, dirname)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    newimage.save(os.path.join('./maximages/', train_val, dirname, file))
    

if __name__ == '__main__':
    A = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    A = torch.from_numpy(A)
    B = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    B = torch.from_numpy(B)
#     print(A)
#     print(A@B)
#     multi(path, conver(path))
    for dirpath, dirnames, filenames in os.walk(path):
        if dirpath is path: 
            continue
            
        for file in filenames:
            if '.ipynb' in file:
                continue
            img_path = os.path.join(dirpath, file)
            dirname = dirpath.split('/')[-1]
            train_val = dirpath.split("/")[-2]
            multi(img_path, conver(img_path), train_val, dirname, file)










