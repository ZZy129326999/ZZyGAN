import torch 

path = r"GANss/MusicFeatures/pytorch-image-models/output/train/20210725-052720-resnet50-224/model_best.pth.tar"
data_path = r'./data/images_general/'

model = torch.load(path)


