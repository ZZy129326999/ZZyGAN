import torch 
import timm 
from timm.models import create_model 
from tlt.utils import load_pretrained_weights
from timm.data import create_transform
from PIL import Image
import models 

classes = ['BlackMeasles', 'BlackRot', 'LeafBlight']
path = r'./checkpoint/d1_224_84.2.pth.tar'
path = r'./output/train/volo_d1_Grape/model_best.pth.tar'
# model_dict = torch.load(path, map_location='cuda')
# for k, v in model_dict.items():
#     print(k)
img_path = r'./data/Grape/train/BlackMeasles/a_s00000005.jpg'

model = create_model('volo_d1', pretrained=True)
load_pretrained_weights(model, checkpoint_path=path, strict=False, num_classes=3)
transform = create_transform(input_size=224, crop_pct=model.default_cfg['crop_pct'])
model.eval()

img = Image.open(img_path).resize((224, 224)).convert('RGB')
input_image = transform(img).unsqueeze(0) 
output = model(input_image)
print(output)
print(output.argmax())
print(classes[int(output.argmax())])
