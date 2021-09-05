import numpy as np
from tensorflow.keras.applications import ResNet50,InceptionResNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop,Adadelta
from tensorflow.keras import activations
from tensorflow.keras.models import load_model,Model
from tensorflow.keras import callbacks
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
import os
import datetime
import tensorflow as tf
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
input_=layers.Input((64,64,3))

res50=ResNet50(weights='imagenet',
                include_top=False,
                input_shape=(64,64,3))
'''
res50=ResNet50(weights='imagenet',
                include_top=False,
                input_shape=(196,196,3))
'''
res50.trainable=True#221 321 331
# for layer in res50.layers:
#     if layer.name=='conv1_conv':
#         layer.trainable=True
#         print(layer.name+" is trainable")
#     # if layer.name=='conv2_block1_0_conv':
#     #     layer.trainable=True
#     #     print(layer.name+' is trainable')
#     if layer.name=='conv2_block2_1_conv':
#         layer.trainable=True
#         print(layer.name+' is trainable')
#     # if layer.name=='conv2_block1_1_relu':
#     #     layer.trainable=True
#     #     print(layer.name+' is trainable')
#     if layer.name=='conv3_block2_1_conv':
#         layer.trainable=True
#         print(layer.name+' is trainable')
#     # if layer.name=='conv2_block1_2_relu':
#     #     layer.trainable=True
#     #     print(layer.name+' is trainable')
#     if layer.name=='conv3_block3_1_conv':
#         layer.trainable=True
#         print(layer.name+' is trainable')
x=res50(input_)
# res50.summary()
x=layers.Flatten()(x)
x=layers.Dense(64,activation='relu',kernel_regularizer=regularizers.l2(0.0001))(x) #l2范数正则化，系数0.002，没有过拟合，最后5次准确率91%+-1.5%
x=layers.Dropout(0.4)(x)
output=layers.Dense(3,activation='softmax')(x)
model=Model(input_,output)
model.compile(optimizer='adadelta',loss='categorical_crossentropy',metrics=['acc'])# 等sgd 训练后，尝试Adadelta以及牛顿动量法
#res50.summary()
history=model.fit(train_db,epochs=200,steps_per_epoch=7200//16,validation_data=val_db,validation_steps=2400//16)
history=history.history
acc=history['acc']
loss=history['loss']
val_acc=history['val_acc']
val_loss=history['val_loss']
# np.save('E:/apple/resnet/acc.npy',acc)
# np.save('E:/apple/resnet/val_acc.npy',val_acc)
# np.save('E:/apple/resnet/loss.npy',loss)
# np.save('E:/apple/resnet/val_loss.npy',val_loss)
epochs=range(1,len(acc)+1)
plt.plot(epochs,acc,'r',label='Training accuracy')
plt.plot(epochs,val_acc,'b',label='Validation accuracy')
max_val_acc_index=np.argmax(val_acc)
plt.plot(max_val_acc_index+1,val_acc[max_val_acc_index],'ks')
show_max='['+str(max_val_acc_index)+','+str(format(val_acc[max_val_acc_index],'.2f'))+']'
plt.annotate(show_max, xytext=(-40,-30),xy=(max_val_acc_index+1,val_acc[max_val_acc_index]),textcoords='offset points',arrowprops=dict(arrowstyle='->'))
plt.xlabel('Epochs')
plt.ylabel('Accuarcy')
plt.title('Training and validation accuracy of ResNet-50 on group data (xxxc)')
plt.legend(loc=3)
plt.grid(axis="y",linewidth=1)
plt.grid(axis="x",linewidth=1)
plt.savefig(path2+'ResNet_50_acc(xxxc).tif')
plt.savefig(path2+'ResNet_50_acc(xxxc).png')
plt.clf()
plt.plot(epochs,loss,'r',label='Training loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and validation loss of ResNet-50 on group data (xxxc)')
plt.legend(loc=2)
plt.grid(axis="y",linewidth=1)
plt.grid(axis="x",linewidth=1)
plt.savefig(path2+'ResNet_50_loss(xxxc).tif')
plt.savefig(path2+'ResNet_50_loss(xxxc).png')

loss,acc=model.evaluate(test_db,steps=2400//16)
model.save('ResNet50(xxxc).h5')
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
