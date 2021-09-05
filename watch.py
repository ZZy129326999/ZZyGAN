import os 
import shutil 
import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--path', type=str, default = './')
args = parser.parse_args()
for root, dirs, files in os.walk(args.path, topdown=False):
#     for name in files:
#         print(os.path.join(root, name))
    for name in dirs:
#         print(os.path.join(root, name))
        if '.ipynb_checkpoints' in os.path.join(root, name): 
            shutil.rmtree(os.path.join(root, name))
#         if '.git' in os.path.join(root, name): 
#             shutil.rmtree(os.path.join(root, name))
        