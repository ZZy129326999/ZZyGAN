# Usual Libraries
import pandas as pd
import numpy as np
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt

# Librosa (the mother of audio files)
import librosa
import librosa.display
import warnings
warnings.filterwarnings('ignore')

import os
import sys
general_path = './'
cla = list(os.listdir(f'{general_path}/genres_original/'))

n_fft = 2048 # FFT window size
hop_length = 512 # number audio of frames between STFT columns (looks like a good default)

def changename(name):
    name = name.split('.')[0]
    for i, s in enumerate(name):
        if s > '9':
            continue
        else:
            name = '.'.join([name[:i], name[i:]])
            break
    return name + '.wav'


clas = ['blues', 'metal', 'hiphop', 'pop', 'classical', 'rock', 'disco', 'jazz', 'reggae', 'country']

def savegenre(dataset_path):
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        if dirpath is not dataset_path:
            dirpath_comp = dirpath.split("/")
            label = dirpath_comp[-1]
            dirpath_comp = dirpath.split("/")
            train_val = dirpath_comp[-2]

            for f in filenames:
                file_path = os.path.join(dirpath, f)
                wav_path = os.path.join('./genres_original/', label, changename(f))
                y, sr = librosa.load(wav_path)
                y, _ = librosa.effects.trim(y)

                S = librosa.feature.melspectrogram(y, sr=sr)
                S_DB = librosa.amplitude_to_db(S, ref=np.max)
                fig = plt.figure(figsize = (3, 3), dpi=100)
                librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, cmap = 'cool')
                ax = plt.subplot()
                ax.set_xticks([])
                ax.set_yticks([])
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                plt.margins(0, 0)
                save_path = os.path.join('./genres/', train_val, label, f[:-4]+'.jpg')
                print(save_path)
                if not os.path.exists(os.path.join('./genres/', train_val, label)):
                    os.makedirs(os.path.join('./genres/', train_val, label))
                plt.savefig(save_path)


dataset_path = r'./images/'
if __name__ == '__main__':
    savegenre(dataset_path)