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
import matplotlib.pyplot as plt


WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'

import os 
import csv
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
train_db=train_db.map(parse).batch(16).repeat()
val_db=tf.data.Dataset.from_tensor_slices((val_names,val_labels))
val_db=val_db.map(parse).batch(16).repeat()
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

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
    os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1"
    strategy=tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = VGG16(3)
    model.compile(optimizer='adadelta',loss='categorical_crossentropy',metrics=['acc'])
    history=model.fit(train_db,epochs=1000,steps_per_epoch=7200//16,validation_data=val_db,validation_steps=2400//16)
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
    plt.title('Training and validation accuracy of VGG-16 on group data (c)')
    plt.legend(loc=3)
    plt.grid(axis="y",linewidth=1)
    plt.grid(axis="x",linewidth=1)
    plt.savefig(path2+'VGG16_acc(c).tif')
    plt.savefig(path2+'VGG16_acc(c).png')
    plt.clf()
    plt.plot(epochs,loss,'r',label='Training loss')
    plt.plot(epochs,val_loss,'b',label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and validation loss of VGG-16 on gsroup data (c)')
    plt.legend(loc=2)
    plt.grid(axis="y",linewidth=1)
    plt.grid(axis="x",linewidth=1)
    plt.savefig(path2+'VGG16_loss(c).tif')
    plt.savefig(path2+'VGG16_loss(c).png')
    
    loss,acc=model.evaluate(test_db,steps=2400//16)
    model.save(path2+'VGG16(c).h5')
#     weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5', WEIGHTS_PATH, cache_subdir='models')
#     model.load_weights(weights_path)

#     img_path = 'elephant.jpg'
#     img = image.load_img(img_path, target_size=(224, 224))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)
#     print('Input image shape:', x.shape)

#     preds = model.predict(x)
#     print('Predicted:', decode_predictions(preds))
