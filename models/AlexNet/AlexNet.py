import keras
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
# from tensorflow.keras.utils import np_utils
from tensorflow.keras.layers import Convolution2D, Activation, MaxPooling2D, Flatten, Dense, Dropout, Input, Reshape, Lambda, Conv2D
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,LearningRateScheduler,ReduceLROnPlateau
import time

layers=keras.layers
models=keras.models
utils=keras.utils

import tensorflow.keras
import tensorflow as tf

path = '../../data/Fungus/'
path2='./'
import os 
if not os.path.exists(path2):
    os.makedirs(path2)
x_train=np.load(path+'x_train.npy')
y_train=np.load(path+'y_train.npy')
x_validata=np.load(path+'x_validata.npy')
y_validata=np.load(path+'y_validata.npy')
config=tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True      # TensorFlow按需分配显存
config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 指定显存分配比例

def Alexnet():
    #input_tensor = Input(shape=(64, 64, 3))
    model = Sequential()

    # 第一层卷积层：42 个卷积核，大小为 5∗5, relu 激活函数
    model.add(Conv2D(42, kernel_size=(5, 5), strides=(2, 2), activation='relu', padding='valid', input_shape=(196, 196, 3)))
    # 第二层池化层：最大池化，步长为2
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 第三层卷积层：74 个卷积核，大小为 5*5，relu 激活函数
    model.add(Conv2D(74, kernel_size=(5, 5), strides=(2, 2), padding='valid', activation='relu'))
    # 第四层池化层：最大池化，步长为2
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 第五层卷积层：74 个卷积核，大小为 5*5，relu 激活函数
    model.add(Conv2D(74, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'))
    # 第六层池化层：最大池化，步长为2
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 将参数进行扁平化，在 LeNet5 中称之为卷积层，实际上这一层是一维向量，和全连接层一样
    model.add(Flatten())
    # 随机丢弃
    model.add(Dropout(0.2))
    # 输出层 输出5类，用 softmax 激活函数计算分类概率
    model.add(Dense(3, activation='softmax'))
    # 设置损失函数和优化器配置
    model.compile(loss=keras.metrics.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=0.001), metrics=['acc'])
    print(model.summary())
    return model


alexmodel = Alexnet()
history=alexmodel.fit(x_train, y_train, batch_size=16, epochs=200, validation_data=(x_validata,y_validata))
history=history.history
acc=history['acc']
loss=history['loss']
val_acc=history['val_acc']
val_loss=history['val_loss']
epochs=range(1,len(acc)+1)
plt.plot(epochs,acc,'r',label='Training accuracy')
plt.plot(epochs,val_acc,'b',label='Validation accuracy')
max_val_acc_index=np.argmax(val_acc)
plt.plot(max_val_acc_index+1,val_acc[max_val_acc_index],'ks')
show_max='['+str(max_val_acc_index)+','+str(format(val_acc[max_val_acc_index],'.2f'))+']'
plt.annotate(show_max, xytext=(-40,-30),xy=(max_val_acc_index+1,val_acc[max_val_acc_index]),textcoords='offset points',arrowprops=dict(arrowstyle='->'))
plt.xlabel('Epochs')
plt.ylabel('Accuarcy')
plt.title('Training and validation accuracy of AlexNet on group data (c)')
plt.legend(loc=3)
plt.grid(axis="y",linewidth=1)
plt.grid(axis="x",linewidth=1)
plt.savefig(path2+'AlexNet_acc(zzyc).tif')
plt.savefig(path2+'AlexNet_acc(zzyc).png')
plt.clf()
plt.plot(epochs,loss,'r',label='Training loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and validation loss of AlexNet on group data (c)')
# Training and Validation loss of ReSNet50 on original images
plt.legend(loc=2)
plt.grid(axis="y",linewidth=1)
plt.grid(axis="x",linewidth=1)
plt.savefig(path2+'AlexNet_loss(zzyc).tif')
plt.savefig(path2+'AlexNet_loss(zzyc).png')
alexmodel.save(path2+'AlexNet(zzyc).h5')
# model.save(path2+'model_AlexNet.h5')