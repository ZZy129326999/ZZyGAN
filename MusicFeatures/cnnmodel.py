# Usual Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn

# Librosa (the mother of audio files)
import librosa
import librosa.display
import IPython.display as ipd
import warnings
warnings.filterwarnings('ignore')

import os
import json 
general_path = './data'
print(list(os.listdir(f'{general_path}/genres_original/')))

# Importing 1 file
y, sr = librosa.load(f'{general_path}/genres_original/reggae/reggae.00036.wav')

# Trim leading and trailing silence from an audio signal (silence before and after the actual audio)
audio_file, _ = librosa.effects.trim(y)

# the result is an numpy ndarray
print('Audio File:', audio_file, '\n')
print('Audio File shape:', np.shape(audio_file))

data = {
        "mapping": [],
        "mfcc": {},
        "labels": [],
    }
imgpath = r'./data/images_original/train/'
clas = ['blues', 'metal', 'hiphop', 'pop', 'classical', 'rock', 'disco', 'jazz', 'reggae', 'country']
# for cla_idx, cla in enumerate(clas):
#     names = os.listdir(os.path.join(imgpath+cla))
#     names = [name.split('.')[0] for name in names] 
#     for idx, name in enumerate(names):
#         for i, s in enumerate(name):
#             if s > '9':
#                 continue
#             else:
#                 name = '.'.join([name[:i], name[i:]])
#                 names[idx] = name
#                 break
#     print(names)
#     for name in names:
#         y, sr = librosa.load(f'{general_path}/genres_original/'+cla+'/'+name+'.wav') 
#         audio_file, _ = librosa.effects.trim(y)
#         mfccs = librosa.feature.mfcc(audio_file, sr=sr)
#         print('mfccs shape:', mfccs.shape)
        

#Displaying  the MFCCs:
# plt.figure(figsize = (16, 6))
# librosa.display.specshow(mfccs, sr=sr, x_axis='time', cmap = 'cool');

def changename(name):
    name = name.split('.')[0]
    for i, s in enumerate(name):
        if s > '9':
            continue
        else:
            name = '.'.join([name[:i], name[i:]])
            break
    return name + '.wav'

def savejson(dataset_path, json_path):
    data = {
        "mapping": [],
        "mfcc": {},
        "labels": {},
    }
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        if dirpath is not dataset_path:
            dirpath_comp = dirpath.split("/")
            semantic_label = dirpath_comp[-1]
            data["mapping"].append(semantic_label)

            for f in filenames:
                file_path = os.path.join(dirpath, f)
                wav_path = os.path.join('./data/genres_original/', semantic_label, changename(f))
                signal,sr = librosa.load(wav_path)
                audio_file, _ = librosa.effects.trim(signal)
                mfccs = librosa.feature.mfcc(audio_file, sr=sr)
                print(wav_path, mfccs.shape)
                data["mfcc"][wav_path] = mfccs.tolist()
                data["labels"][wav_path] = i-1

    with open(json_path,"w") as f:
        json.dump(data,f,indent=4)
 

            
dataset_path = r'./data/images_original/train/'  # "./data/genres_original/"
json_path = r'Mfccdata.json'
# if __name__ == '__main__':
#     savejson(dataset_path, json_path)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set_style('whitegrid')
import warnings
warnings.filterwarnings('ignore')
import sklearn.metrics as skm
import sklearn.model_selection as skms
import sklearn.preprocessing as skp
from sklearn.metrics import confusion_matrix
import random
import librosa, IPython
import librosa.display as lplt
seed = 12
np.random.seed(seed)
df = pd.read_csv('./data/features_3_sec.csv')
df.head()
print("Dataset has",df.shape)
print("Count of Positive and Negative samples")
df.label.value_counts().reset_index()
audio_fp = './data/genres_original/blues/blues.00000.wav'
audio_data, sr = librosa.load(audio_fp)
audio_data, _ = librosa.effects.trim(audio_data)
# Default FFT window size
n_fft = 2048 # window size
hop_length = 512 # window hop length for STFT

stft = librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length)
stft_db = librosa.amplitude_to_db(stft, ref=np.max)


# map labels to index
label_index = dict()
index_label = dict()
for i, x in enumerate(df.label.unique()):
    label_index[x] = i
    index_label[i] = x
print(label_index)
print(index_label)
# update labels in df to index
df.label = [label_index[l] for l in df.label]

# shuffle samples
df_shuffle = df.sample(frac=1, random_state=seed).reset_index(drop=True)
# remove irrelevant columns
df_shuffle.drop(['filename', 'length'], axis=1, inplace=True)
df_y = df_shuffle.pop('label')
df_X = df_shuffle

# split into train dev and test
X_train, df_test_valid_X, y_train, df_test_valid_y = skms.train_test_split(df_X, df_y, train_size=0.4, random_state=seed, stratify=df_y)
X_dev, X_test, y_dev, y_test = skms.train_test_split(df_test_valid_X, df_test_valid_y, train_size=0.5, random_state=seed, stratify=df_test_valid_y)

print(f"Train set has {X_train.shape[0]} records out of {len(df_shuffle)} which is {round(X_train.shape[0]/len(df_shuffle)*100)}%")
print(f"Dev set has {X_dev.shape[0]} records out of {len(df_shuffle)} which is {round(X_dev.shape[0]/len(df_shuffle)*100)}%")
print(f"Test set has {X_test.shape[0]} records out of {len(df_shuffle)} which is {round(X_test.shape[0]/len(df_shuffle)*100)}%")

print(y_train.value_counts()[0]/y_train.shape[0]*100)
print(y_dev.value_counts()[0]/y_dev.shape[0]*100)
print(y_test.value_counts()[0]/y_test.shape[0]*100)
scaler = skp.StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_dev = pd.DataFrame(scaler.transform(X_dev), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_train.columns)

import tensorflow as tf
print("TF version:-", tf.__version__)
import tensorflow.keras as k
tf.random.set_seed(seed)
ACCURACY_THRESHOLD = 0.94
class myCallback(k.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('val_accuracy') > ACCURACY_THRESHOLD):
            print("\n\nStopping training as we have reached %2.2f%% accuracy!" %(ACCURACY_THRESHOLD*100))   
            self.model.stop_training = True

def trainModel(model, epochs, optimizer):
    batch_size = 128
    callback = myCallback()
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics='accuracy'
    )
    return model.fit(X_train, y_train, validation_data=(X_dev, y_dev), epochs=epochs, 
                     batch_size=batch_size, callbacks=[callback])
def plotHistory(history):
    print("Max. Validation Accuracy",max(history.history["val_accuracy"]))
    pd.DataFrame(history.history).plot(figsize=(12,6))
    plt.show()

model_1 = k.models.Sequential([
    k.layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    k.layers.Dense(128, activation='relu'),
    k.layers.Dense(64, activation='relu'),
    k.layers.Dense(10, activation='softmax'),
])
# print(model_1.summary())
# model_1_history = trainModel(model=model_1, epochs=70, optimizer='adam')

model_2 = k.models.Sequential([
    k.layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
    k.layers.Dropout(0.2),
    
    k.layers.Dense(256, activation='relu'),
    k.layers.Dropout(0.2),

    k.layers.Dense(128, activation='relu'),
    k.layers.Dropout(0.2),

    k.layers.Dense(64, activation='relu'),
    k.layers.Dropout(0.2),

    k.layers.Dense(10, activation='softmax'),
])
# print(model_2.summary())
# model_2_history = trainModel(model=model_2, epochs=100, optimizer='adam')

# plotHistory(model_2_history)

model_4 = k.models.Sequential([
    k.layers.Dense(1024, activation='relu', input_shape=(X_train.shape[1],)),
    k.layers.Dropout(0.3),
    
    k.layers.Dense(512, activation='relu'),
    k.layers.Dropout(0.3),

    k.layers.Dense(256, activation='relu'),
    k.layers.Dropout(0.3),

    k.layers.Dense(128, activation='relu'),
    k.layers.Dropout(0.3),

    k.layers.Dense(64, activation='relu'),
    k.layers.Dropout(0.3),

    k.layers.Dense(10, activation='softmax'),
])
# strategy=tf.distribute.MirroredStrategy()
# with strategy.scope():
#     model_4 = model_4
# print(model_4.summary())
# model_4_history = trainModel(model=model_4, epochs=500, optimizer='rmsprop')
# model_4.save('MusicCNN.h5')
# plotHistory(model_4_history)
# test_loss, test_acc  = model_4.evaluate(X_test, y_test, batch_size=128)
# print("The test Loss is :",test_loss)
# print("\nThe Best test Accuracy is :",test_acc*100)

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,accuracy_score,recall_score,classification_report,confusion_matrix
import itertools
import os

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, name='Gaussnb'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

#     print(cm)

    plt.figure(figsize = (9, 7), dpi=100)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)

#     fig, ax = plt.subplots()
#     ax.set_xticks(np.arange(cm.shape[1]+1)-.5, minor=True)
#     ax.set_yticks(np.arange(cm.shape[0]+1)-.5, minor=True)
#     ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.2)
#     ax.tick_params(which="minor", bottom=False, left=False)
#     im = ax.imshow(cm)
#     ax.set_ylim(len(cm)-0.5, -0.5)
    
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)  # rotation=45
    plt.yticks(tick_marks, classes)
    plt.ylim(len(cm) - 0.5, -0.5)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    if not os.path.exists(f'./resulImgs/{name}cm.jpg'):
        plt.savefig(f'./resulImgs/Cm{name}.jpg')

def confusion_matrix_(y_pred, y):
    cm = confusion_matrix(y, y_pred, labels=range(10))
    cm = cm.astype(np.float32)
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    ss0 = "TP, TN, FP, FN: " + str(TP.mean()) + " " +  str(TN.mean()) + " " + str(FP.mean()) + " " + str(FN.mean()) + '\n'
    print("TP, TN, FP, FN: ", TP.mean(), TN.mean(), FP.mean(), FN.mean())
    TPR = TP/(TP+FN) 
    TNR = TN/(TN+FP)
    PPV = TP/(TP+FP)
    REC = TP/(TP+FN)
    ACC = (TP+TN)/(TP+FN+TN+FP)
    ss = 'Sensivity: %.2f%%'%(TPR.mean()*100) + '\n' + \
         'Specificity: %.2f%%'%(TNR.mean()*100) + '\n' + \
         'Precision: %.2f%%'%(PPV.mean()*100) + '\n' + 'Pecall: %.2f%%'%(REC.mean()*100) + '\n'
         
    
    print('Sensivity: %.2f%%'%(TPR.mean()*100))
    print('Specificity: %.2f%%'%(TNR.mean()*100))
    print('Precision: %.2f%%'%(PPV.mean()*100))
    print('Pecall: %.2f%%'%(REC.mean()*100))
#     print('Accuracy: %.2f%%'%(ACC.mean()*100))
#     print('F1:%.2f%%'%(100*(2*PPV.mean()*REC.mean())/(PPV.mean()+REC.mean())))
        
def CalculationResults(val_y,y_val_pred,simple = False,\
                       target_names = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"], name='SGD'):
    # 计算检验
    F1_score = f1_score(val_y,y_val_pred, average='macro')
    if simple:
        return F1_score
    else:
        acc = accuracy_score(val_y,y_val_pred)
        recall_score_ = recall_score(val_y,y_val_pred, average='macro')
        confusion_matrix_ = confusion_matrix(val_y,y_val_pred)
        class_report = classification_report(val_y, y_val_pred)
        ss1 = 'f1:%.2f%%'%(F1_score*100) + '\n' + 'Accuarcy:%.2f%%'%(acc*100) + '\n'
        print('f1:%.2f%%'%(F1_score*100))
        print('Accuarcy:%.2f%%'%(acc*100))
        print('f1_score:',F1_score,'\nACC_score:',acc,'\nrecall:',recall_score_)
#         print('\n----class report ---:\n',class_report)
#         print('----confusion matrix ---:\n',confusion_matrix_)

        # 画混淆矩阵
            # 画混淆矩阵图
        plt.figure()
        plot_confusion_matrix(confusion_matrix_, classes=target_names,
                              title=f'Confusion matrix of {name}', name=name)
#         plt.show()
        return F1_score,acc,recall_score_,confusion_matrix_,class_report

from tensorflow.keras.models import load_model
model_4 = load_model("MusicCNN.h5")
y_pred = model_4.predict(X_test, batch_size=64, verbose=1)
y_pred = np.argmax(y_pred, axis=1)
confusion_matrix_(y_test, y_pred)
CalculationResults(y_test, y_pred, name='DenseCNN')



