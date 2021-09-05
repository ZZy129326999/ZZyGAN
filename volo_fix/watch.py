import os 
import shutil 

for root, dirs, files in os.walk("../MusicFeatures/data/", topdown=False):
    for name in files:
        print(os.path.join(root, name))
    for name in dirs:
        print(os.path.join(root, name))
        if '.ipynb_checkpoints' in os.path.join(root, name): 
            shutil.rmtree(os.path.join(root, name))
        