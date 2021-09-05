import numpy as np
from tensorflow.keras.applications import ResNet50,InceptionResNetV2,DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop,Adadelta
from tensorflow.keras import activations
from tensorflow.keras.models import load_model,Model
from tensorflow.keras import callbacks
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
# from tensorflow.keras.utils.vis_utils import plot_model
import os
import datetime
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import load_model
# import sys
# sys.path.append(r"C:\Users\709\Desktop\DenseNet-Keras-master")
# import  densenet121
import csv
import tensorflow as tf
path='../data/'
model_path='./DenseNet_121/DenseNet_121(xxxc).h5'
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

def dense():
    DenseNet=DenseNet121(weights=None,include_top=False,input_shape=(64,64,3))
    DenseNet.trainable=True
    # for layer in DenseNet.layers:
    #     if layer.name=='conv2_block2_1_conv':
    #         layer.trainable=True
    #         print(layer.name+"is trainable")
    #     if layer.name=='conv1/conv':
    #         layer.trainable=True
    #         print(layer.name+"is trainable")
    #     if layer.name=='conv3_block3_1_conv':
    #         layer.trainable=True
    #         print(layer.name+"is trainable")
    #     if layer.name=='conv2_block3_1_conv':
    #         layer.trainable=True
    #         print(layer.name+"is trainable")
    # plot_model(DenseNet,path2+'model_DenseNet121.png',show_layer_names=True)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
    os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1"
    input_=layers.Input((64,64,3))
    x=DenseNet(input_)
    x=layers.Flatten()(x)
    x=layers.Dense(64,activation='relu',kernel_regularizer=regularizers.l2(0.00005))(x)
    x=layers.Dropout(0.5)(x)
    output=layers.Dense(3,activation='softmax')(x)
    model=Model(input_, output)
    model.compile(optimizer='adadelta',loss='categorical_crossentropy',metrics=['acc'])
    return model

model = dense()
model.compile(optimizer='adadelta',loss='categorical_crossentropy',metrics=['acc'])
loss,acc=model.evaluate(test_db,steps=2400//16)

