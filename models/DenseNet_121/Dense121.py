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
img_path='../../data/images/images/'
csv_path='../../data/'
path2='./'
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
train_names,train_labels=read_csv(csv_path+'train.csv')
val_names,val_labels=read_csv(csv_path+'valid.csv')
test_names,test_labels=read_csv(csv_path+'test.csv')
def parse(name,label):
    img=tf.io.read_file(img_path+name)
    img=tf.image.decode_jpeg(img,channels=3)
    img=tf.image.convert_image_dtype(img,dtype=tf.float32)
    return img,label
train_db=tf.data.Dataset.from_tensor_slices((train_names,train_labels))
train_db=train_db.map(parse).batch(32).repeat()
val_db=tf.data.Dataset.from_tensor_slices((val_names,val_labels))
val_db=val_db.map(parse).batch(32).repeat()
test_db=tf.data.Dataset.from_tensor_slices((test_names,test_labels))
test_db=test_db.map(parse).batch(32).repeat()

#DenseNet=DenseNet121(weights='imagenet',include_top=False,input_shape=(196,196,3))
def dense():
    DenseNet=DenseNet121(weights='imagenet',include_top=False,input_shape=(64,64,3))
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
    input_=layers.Input((64,64,3))
    x=DenseNet(input_)
    x=layers.Flatten()(x)
    x=layers.Dense(64,activation='relu',kernel_regularizer=regularizers.l2(0.00005))(x)
    x=layers.Dropout(0.5)(x)
    output=layers.Dense(3,activation='softmax')(x)
    model=Model(input_, output)
    model.compile(optimizer='adadelta',loss='categorical_crossentropy',metrics=['acc'])
    return model

strategy=tf.distribute.MirroredStrategy()
with strategy.scope():
    model=dense()
history=model.fit(train_db,epochs=1000,steps_per_epoch=7200//32,validation_data=val_db,validation_steps=2400//32)
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
plt.annotate(show_max, xytext=(-20,-30),xy=(max_val_acc_index+1,val_acc[max_val_acc_index]),textcoords='offset points',arrowprops=dict(arrowstyle='->'))
plt.xlabel('Epochs')
plt.ylabel('Accuarcy')
plt.title('Training and validation accuracy of DenseNet-121 on group data (c)')
plt.legend(loc=3)
plt.grid(axis="y",linewidth=1)
plt.grid(axis="x",linewidth=1)
plt.savefig(path2+'DenseNet-121_acc(c).tif')
plt.savefig(path2+'DenseNet-121_acc(c).png')
plt.clf()
plt.plot(epochs,loss,'r',label='Training loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and validation loss of DenseNet-121 on group data (c)')
plt.legend(loc=2)
plt.grid(axis="y",linewidth=1)
plt.grid(axis="x",linewidth=1)
plt.savefig(path2+'DenseNet-121_loss(c).tif')
plt.savefig(path2+'DenseNet-121_loss(c).png')
model.save(path2+'DenseNet_121(c).h5')
loss,acc=model.evaluate(test_db,steps=2400//32)