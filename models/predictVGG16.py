import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential
# from tensorflow.keras.applications.imagenet_utils import (decode_predictions,
#                                                preprocess_input)
from tensorflow.keras.layers import (Conv2D, Dense, Flatten, GlobalAveragePooling2D,
                          GlobalMaxPooling2D, Input, MaxPooling2D)
from tensorflow.keras.preprocessing import image
# from tensorflow.keras.utils.data_utils import get_file
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np
import os
import datetime
import tensorflow as tf
import csv
path='../data/'
model_path='./VGG_16/VGG16(xxxc).h5'
img_path='../data/images/images/'
csv_path='../data/'
def read_csv(name):
    with open(name,'r') as f:
        csv_f=csv.reader(f)
        i=0
        img_names=[]
        labels=[]
        for row in csv_f:
            if i==0:
                i+=1
                continue
            img_names.append(row[0])
            labels.append(row[1])
        labels=tf.keras.utils.to_categorical(labels,3)
    return img_names,labels
def parse(name,label):
    img=tf.io.read_file(img_path+name)
    img=tf.image.decode_jpeg(img,channels=3)
    img=tf.image.convert_image_dtype(img,dtype=tf.float32)
    return img,label
test_names,test_labels=read_csv(csv_path+'test.csv')
test_db=tf.data.Dataset.from_tensor_slices((test_names,test_labels))
test_db=test_db.map(parse).batch(16).repeat()


def VGG16(num_classes):
    image_input = Input(shape = (64,64,3))

    # 224,224,3 -> 112,112,64
    x = Conv2D(64,(3,3),activation = 'relu',padding = 'same',name = 'block1_conv1')(image_input)
    x = Conv2D(64,(3,3),activation = 'relu',padding = 'same', name = 'block1_conv2')(x)
    x = MaxPooling2D((2,2), strides = (2,2), name = 'block1_pool')(x)

    # 第二个卷积部分
    # 112,112,64 -> 56,56,128
    x = Conv2D(128,(3,3),activation = 'relu',padding = 'same',name = 'block2_conv1')(x)
    x = Conv2D(128,(3,3),activation = 'relu',padding = 'same',name = 'block2_conv2')(x)
    x = MaxPooling2D((2,2),strides = (2,2),name = 'block2_pool')(x)

    # 第三个卷积部分
    # 56,56,128 -> 28,28,256
    x = Conv2D(256,(3,3),activation = 'relu',padding = 'same',name = 'block3_conv1')(x)
    x = Conv2D(256,(3,3),activation = 'relu',padding = 'same',name = 'block3_conv2')(x)
    x = Conv2D(256,(3,3),activation = 'relu',padding = 'same',name = 'block3_conv3')(x)
    x = MaxPooling2D((2,2),strides = (2,2),name = 'block3_pool')(x)

    # 第四个卷积部分
    # 28,28,256 -> 14,14,512
    x = Conv2D(512,(3,3),activation = 'relu',padding = 'same', name = 'block4_conv1')(x)
    x = Conv2D(512,(3,3),activation = 'relu',padding = 'same', name = 'block4_conv2')(x)
    x = Conv2D(512,(3,3),activation = 'relu',padding = 'same', name = 'block4_conv3')(x)
    x = MaxPooling2D((2,2),strides = (2,2),name = 'block4_pool')(x)

    # 第五个卷积部分
    # 14,14,512 -> 7,7,512
    x = Conv2D(512,(3,3),activation = 'relu',padding = 'same', name = 'block5_conv1')(x)
    x = Conv2D(512,(3,3),activation = 'relu',padding = 'same', name = 'block5_conv2')(x)
    x = Conv2D(512,(3,3),activation = 'relu',padding = 'same', name = 'block5_conv3')(x)    
    x = MaxPooling2D((2,2),strides = (2,2),name = 'block5_pool')(x)

    # 分类部分
    # 7,7,512 -> 25088 -> 4096 -> 4096 -> num_classes
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(num_classes, activation='softmax', name='predictions')(x)
    
    model = Model(image_input,x,name = 'vgg16')
    return model

model = VGG16(3)
model.compile(optimizer='adadelta',loss='categorical_crossentropy',metrics=['acc'])
loss,acc=model.evaluate(test_db,steps=2400//16)

