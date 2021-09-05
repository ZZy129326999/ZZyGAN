import models 
import sys
from PIL import Image 
from tlt.utils import load_pretrained_weights 
from timm.data import create_transform

def main():
	model = models.volo_d1(img_size=224)
	load_pretrained_weights(model=model, checkpoint_path=r'./output/train/20210728-021806-volo_d1-224/model_best.pth.tar')
	model.eval()
	transform = create_transform(input_size=224, crop_pct=model.default_cfg['crop_pct'])
	image = Image.open(r'./data/RiceDiseases/val/BacterialLeafBlight/BacterialLeafBlight_10.jpg')
	input_image = transform(image).unsqueeze(0) 
	x_cls, x_aux, (bbx1, bby1, bbx2, bby2) = model(input_image)
	print('finished')
	print(f'Prediction: {int(pred.argmax())}.')
	print(bbx1, bby1, bbx2, bby2)

if __name__ == '__main__':
	main()
	sys.exit(0)