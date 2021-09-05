import os
import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import librosa
import librosa.display
import json
from tensorflow.keras import Sequential
# from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.layers import *
from tensorflow.keras import optimizers
import tensorflow.keras as keras
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense
from sklearn.metrics import confusion_matrix

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

dataset_path = "./data/genres_original/"
json_path = r"data.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 30 # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION
num_mfcc=13
n_fft=2048
hop_length=512
num_segments = 10
samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

# dictionary to store mapping, labels, and MFCCs
data = {
    "mapping": [],
    "labels": [],
    "mfcc": []
}

def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048,
             hop_length=512, num_segments=5):
    # Data storage dictionary
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": [],
    }
    samples_ps = int(SAMPLES_PER_TRACK/num_segments) # ps = per segment
    expected_vects_ps = math.ceil(samples_ps/hop_length)
    
    # loop through all the genres
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        # ensuring not at root
        if dirpath is not dataset_path:
            # save the semantic label
            dirpath_comp = dirpath.split("/")
            semantic_label = dirpath_comp[-1]
            data["mapping"].append(semantic_label)
#             print(f"Processing: {semantic_label}")

            # process files for specific genre
            for f in filenames:
                if(f==str("jazz.00054.wav")):
                    # As librosa only read files <1Mb
                    continue
                else:
                    # load audio file
                    file_path = os.path.join(dirpath, f)
                    signal,sr = librosa.load(file_path,sr=SAMPLE_RATE)
                    for s in range(num_segments):
                        start_sample = samples_ps * s
                        finish_sample = start_sample + samples_ps

                        mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample],
                                                    sr = sr,
                                                    n_fft = n_fft,
                                                    n_mfcc = n_mfcc,
                                                    hop_length = hop_length)

                        mfcc = mfcc.T

                        # store mfcc if it has expected length 
                        if len(mfcc)==expected_vects_ps:
                            data["mfcc"].append(mfcc.tolist())
                            data["labels"].append(i-1)
#                             print(f"{file_path}, segment: {s+1}")
    
    with open(json_path,"w") as f:
        json.dump(data,f,indent=4)
    

# load data
def load_data(dataset_path):
    with open(dataset_path,"r") as f:
        data = json.load(f)
    
    # Convert list to numpy arrays
    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])    
    
    return inputs,targets

def ANN(inputs):
    model = Sequential()
    model.add(Flatten(input_shape=(inputs.shape[1], inputs.shape[2])))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model

def plot_history(hist):
    plt.figure(figsize=(20,15))
    fig, axs = plt.subplots(2)
    # accuracy subplot
    axs[0].plot(hist.history["accuracy"], label="train accuracy")
    axs[0].plot(hist.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")
    
    # Error subplot
    axs[1].plot(hist.history["loss"], label="train error")
    axs[1].plot(hist.history["val_loss"], label="test error")    
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")
    
def Myplot(model, hist):
    plt.plot(hist.history["accuracy"],'r',label='Training accuracy')
    plt.plot(hist.history["val_accuracy"],'b',label='Validation accuracy')
    val_acc=history['val_accuracy']
    max_val_acc_index=np.argmax(val_acc)
    plt.plot(max_val_acc_index+1,val_acc[max_val_acc_index],'ks')
    show_max='['+str(max_val_acc_index)+','+str(format(val_acc[max_val_acc_index],'.2f'))+']'
    plt.annotate(show_max, xytext=(-40,-30),xy=(max_val_acc_index+1,val_acc[max_val_acc_index]),
                 textcoords='offset points',arrowprops=dict(arrowstyle='->'))
    plt.xlabel('Epochs')
    plt.ylabel('Accuarcy')
    plt.title('Training and validation accuracy of our model')
    plt.legend(loc=3)
    plt.grid(axis="y",linewidth=1)
    plt.grid(axis="x",linewidth=1)
    plt.savefig(model+'ACC.tif')
    plt.savefig(model+'ACC.png')
    plt.clf()
    
    plt.plot(hist.history["loss"],'r',label='Training loss')
    plt.plot(hist.history["val_accuracy"],'b',label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and validation loss of our model')
    plt.legend(loc=2)
    plt.grid(axis="y",linewidth=1)
    plt.grid(axis="x",linewidth=1)
    plt.savefig(model+'LOSS.tif')
    plt.savefig(model+'LOSS.png')
    plt.clf()
    
    
    
    
    
    plt.savefig("ANN.png")

    
def prepare_dataset(test_size, validation_size):
    X,y = load_data(r"./data.json")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = validation_size)
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_val, X_test, y_train, y_val, y_test

def CNN(input_shape):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation = "relu", input_shape = input_shape))
    model.add(MaxPool2D((3, 3), strides=(2, 2), padding="same"))
    model.add(BatchNormalization())

    model.add(Conv2D(32, (3, 3), activation = "relu"))
    model.add(MaxPool2D((3, 3), strides=(2, 2), padding="same"))
    model.add(BatchNormalization())

    model.add(Conv2D(32, (2, 2), activation = "relu"))
    model.add(MaxPool2D((2, 2), strides=(2, 2), padding="same"))
    model.add(BatchNormalization())

    model.add(Conv2D(16, (1, 1), activation = "relu"))
    model.add(MaxPool2D((1, 1), strides=(2, 2), padding="same"))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(10, activation="softmax"))
    return model

def plot_metrics(history):
    metrics =  ['loss', 'acc', 'f1_m', 'precision_m', 'recall_m']
    colors = ['b', 'm', 'c', 'r', 'pink']
    for n, metric in enumerate(metrics):
        name = metric
        if n >= 2:
            name = name[:-2]
        plt.plot(history.epoch, history.history[metric], color=colors[n], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric], color=colors[n], linestyle="--", label='Val')
        tmp = history.history[metric]
        max_tmp_index=np.argmax(tmp)
        plt.plot(max_tmp_index,tmp[max_tmp_index],'ks')
        show_max='['+str(max_tmp_index)+','+str(format(tmp[max_tmp_index],'.2f'))+']'
        plt.annotate(show_max, xytext=(-40,-30),xy=(max_tmp_index,tmp[max_tmp_index]),
                     textcoords='offset points',arrowprops=dict(arrowstyle='->'))
        plt.xlabel('Epoch')
        plt.ylabel(name)
        Max = tmp[max_tmp_index]
        tmp = history.history['val_'+metric]
        max_tmp_index=np.argmax(tmp)
        Max = max(Max, tmp[max_tmp_index]) + 0.5
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'acc':
            plt.ylim([0,1])
        else:
            plt.ylim([0,1])
        plt.legend()
        plt.title(name)
        plt.savefig("res_"+name+'.png')
        plt.savefig("res_"+name+'.tif')
        plt.clf()

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,accuracy_score,recall_score,classification_report,confusion_matrix
import itertools  

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

    print(cm)

    plt.figure(figsize = (9, 7), dpi=100)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    
#     fig, ax = plt.subplots()
#     im = ax.imshow(cm)
#     ax.set_ylim(len(cm)-0.5, -0.5)
    
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)  # rotation=45
    plt.yticks(tick_marks, classes)
#     plt.ylim(len(cm) - 0.5, -0.5)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    if not os.path.exists(f'./resulImgs/Cm{name}.jpg'):
        plt.savefig(f'./resulImgs/Cm{name}.jpg')

def confusion_matrix_(y_pred, y):
    cm = confusion_matrix(y, y_pred, labels=range(10))
    cm = cm.astype(np.float32)
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    ss0 = "TP, TN, FP, FN: " + str(TP.mean()) + " " +  str(TN.mean()) + " " + str(FP.mean()) + " " + str(FN.mean()) + '\n'
#     f.write(ss0)
    print("TP, TN, FP, FN: ", TP.mean(), TN.mean(), FP.mean(), FN.mean())
    TPR = TP/(TP+FN) 
    TNR = TN/(TN+FP)
    PPV = TP/(TP+FP)
    REC = TP/(TP+FN)
    ACC = (TP+TN)/(TP+FN+TN+FP)
    ss = 'Sensivity: %.2f%%'%(TPR.mean()*100) + '\n' + \
         'Specificity: %.2f%%'%(TNR.mean()*100) + '\n' + \
         'Precision: %.2f%%'%(PPV.mean()*100) + '\n' + 'Pecall: %.2f%%'%(REC.mean()*100) + '\n'
         
#     f.write(ss)
    
    print('Sensivity: %.2f%%'%(TPR.mean()*100))
    print('Specificity: %.2f%%'%(TNR.mean()*100))
    print('Precision: %.2f%%'%(PPV.mean()*100))
    print('Pecall: %.2f%%'%(REC.mean()*100))
#     print('Accuracy: %.2f%%'%(ACC.mean()*100))
    print('F1:%.2f%%'%(100*(2*PPV.mean()*REC.mean())/(PPV.mean()+REC.mean())))
        
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
#         f.write(ss1)
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

if __name__ == "__main__":
#     save_mfcc(dataset_path,json_path,num_segments=10)
    
    inputs,targets = load_data(r"./data.json")
    input_train, input_test, target_train, target_test = train_test_split(inputs, targets, test_size=0.3)
    print(input_train.shape, target_train.shape)
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Create 2 virtual GPUs with 1GB memory each
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024),
                 tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)
    
    tf.debugging.set_log_device_placement(True)

    strategy = tf.distribute.MirroredStrategy()
#     with strategy.scope():
# #         inputs = tf.keras.layers.Input(shape=(1,))
# #         predictions = tf.keras.layers.Dense(1)(inputs)
# #         model = tf.keras.models.Model(inputs=inputs, outputs=predictions)
# #         model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(learning_rate=0.2))
#         # ANN
#         model = ANN(inputs)
#         adam = optimizers.Adam(lr=1e-4)
#         model.compile(optimizer=adam, loss="sparse_categorical_crossentropy", metrics=METRICS)
#         hist = model.fit(input_train, target_train, validation_data = (input_test,target_test), epochs = 50, batch_size = 32)
#         plot_history(hist)
#         test_error, test_accuracy = model.evaluate(input_test, target_test, verbose=1)
#         print(f"Test accuracy: {test_accuracy}")
        
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_dataset(0.25, 0.2)
    input_shape = (X_train.shape[1],X_train.shape[2],X_train.shape[3])
    print(input_shape)
    
    
    
    from keras import backend as K
    def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
    def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    def f1_m(y_true, y_pred):
        precision = precision_m(y_true, y_pred)
        recall = recall_m(y_true, y_pred)
        return 2*((precision*recall)/(precision+recall+K.epsilon()))

    # compile the model
    model = CNN(input_shape)
    model.compile(optimizer=optimizers.Adam(lr=1e-4), loss="categorical_crossentropy", metrics=['acc', f1_m, precision_m, recall_m])
    
    # categorial_CE one-hot
    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)
#     y_test = to_categorical(y_test)

    # fit the model
#     history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=64)
#     plot_metrics(history)
#     model.save("MusicCNN.h5")

    # evaluate the model
#     loss, accuracy, f1_score, precision, recall = model.evaluate(X_test, y_test, verbose=0)
    
#     with strategy.scope():
    if False:
        model = CNN(input_shape)
        adam = optimizers.Adam(lr=1e-4)
        model.compile(optimizer=adam, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        hist = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=40, batch_size=32)
        model.save("CNN.h5")
        
    model = load_model("CNN.h5")
#     test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
#     print(f"Test accuracy: {test_accuracy}")
    from sklearn.metrics import classification_report

    y_pred = model.predict(X_test, batch_size=64, verbose=1)
    y_pred_bool = np.argmax(y_pred, axis=1)

#     print(classification_report(y_test, y_pred_bool)))
    confusion_matrix_(y_test, y_pred_bool)
    CalculationResults(y_test, y_pred_bool, name='CNN')
#     tn, fp, fn, tp = confusion_matrix(y_pred_bool, y_test, labels=[0,1,2,3,4,5,6,7,8,9]).ravel()
#     print("TN, FP, FN, TP: ", tn, fp, fn, tp)
    
#     cm = confusion_matrix(y_pred_bool, y_test, labels=range(10))
     
#     print('\n')
#     print("pred", y_test.shape , "cm:", cm.shape)
#     print(cm)
#     cm = cm.astype(np.float32)
#     FP = cm.sum(axis=0) - np.diag(cm)
#     FN = cm.sum(axis=1) - np.diag(cm)
#     TP = np.diag(cm)
#     TN = cm.sum() - (FP + FN + TP)
#     print("TP, TN, FP, FN: ", TP.mean(), TN.mean(), FP.mean(), FN.mean())
#     TPR = TP/(TP+FN) # Sensitivity/ hit rate/ recall/ true positive rate
#     TNR = TN/(TN+FP) # Specificity/ true negative rate
#     PPV = TP/(TP+FP) # Precision/ positive predictive value
#     NPV = TN/(TN+FN) # Negative predictive value
#     FPR = FP/(FP+TN) # Fall out/ false positive rate
#     FNR = FN/(TP+FN) # False negative rate
#     FDR = FP/(TP+FP) # False discovery rate
#     REC = TP/(TP+FN) # accuracy of each class
#     print('Sensivity: %.2f%%'%(TPR.mean()*100))
#     print('Specificity: %.2f%%'%(TNR.mean()*100))
#     print('Precision: %.2f%%'%(PPV.mean()*100))
#     print('Pecall: %.2f%%'%(REC.mean()*100))
    
    
    
    

    
    
    
    
