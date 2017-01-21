#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 18:57:50 2017

@author: waffleboy
"""
#==============================================================================
#                       Temp bug fix
import tensorflow as tf
tf.python.control_flow_ops = tf
#==============================================================================


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.advanced_activations import PReLU
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
import cv2
import pandas as pd
import glob
import random
import numpy as np

def load_label_dic(csv_file):
    df = pd.read_csv(csv_file)
    trainID = dict(zip(df.name, df.label)) 
    return trainID
    
def get_pic_name(picture_link):
    return picture_link[picture_link.rfind('/')+1:]
    
label_dic = load_label_dic("labels.csv")
TRAINING_DIRECTORY = "train"
batch_size = 20
nb_classes = len(label_dic["label"].unique())
nb_epoch = 50
useValidation = True
SIZE = (160,160)

#==============================================================================
#                                Load Images
#==============================================================================
def change_labels_to_numeric(labels):
    unique = np.unique(labels)
    mapper = {v:k for k,v in enumerate(unique)}
    for i in range(len(labels)):
        labels[i] = mapper[labels[i]]
    return labels,mapper

images_and_labels = []
files = glob.glob(TRAINING_DIRECTORY+'/*.jpg')
print("Beginning to load pictures into memory")
counter = 0
for file in files:
    counter += 1
    if counter % 500 == 0:
        print("Loaded {} pictures".format(counter))
    pic = cv2.imread(file)
    picname = get_pic_name(file)
    label = label_dic[picname]
    images_and_labels.append([pic,label])
    
random.shuffle(images_and_labels)

trainImgs = np.array([x[0] for x in images_and_labels])
labels = np.array([x[1] for x in images_and_labels])
labels, label_mapper_dic = change_labels_to_numeric(labels)
#==============================================================================
#                                   Models
#==============================================================================

img_channels = 3 #RGB
img_rows, img_cols = SIZE[0],SIZE[1]

def custom_model(img_channels,img_rows,img_cols):    
    model = Sequential()
    # 1st layer
    model.add(Convolution2D(32, 3, 3, border_mode='same',init='glorot_normal',
                            input_shape=(img_channels, img_rows, img_cols)))
    model.add(Activation('relu'))
    # second layer
    model.add(Convolution2D(32, 3, 3,init='glorot_normal'))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.20))
    # 3rd layer
    model.add(Convolution2D(64, 3, 3, border_mode='same',init='glorot_normal'))
    model.add(PReLU())
    model.add(Convolution2D(64, 3, 3,init='glorot_normal',))
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.20))
    #4th layer
    model.add(Convolution2D(64, 3, 3,init='glorot_normal'))
    model.add(PReLU())
    model.add(Convolution2D(64, 3, 3,init='glorot_normal'))
    model.add(PReLU())
    model.add(ZeroPadding2D())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 5th layer
    model.add(Convolution2D(32, 3, 3,init='glorot_normal'))
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    # 5th and 6th normal layer
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(PReLU())
    model.add(Dense(1024))
    model.add(Dropout(0.3))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    return model
    
#==============================================================================
#                                   train
#==============================================================================

def splitTrainTest(trainXData,trainYData,test_size=0.1):
    from sklearn.cross_validation import train_test_split
    print('Splitting Data')
    X_train, X_test, y_train, y_test = train_test_split(trainXData,trainYData,test_size=test_size)
    return X_train, X_test, y_train, y_test 
    
#==============================================================================
#                                   run
#==============================================================================

if useValidation:
    dataset, X_test, target, y_test = splitTrainTest(trainImgs,labels)
    X_train, X_val, y_train,y_val = splitTrainTest(dataset,target)
else:
    X_train, X_test, y_train,y_test = splitTrainTest(trainImgs,labels)

# the data, shuffled and split between train and test sets
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = custom_model(img_channels,img_rows,img_cols)

earlyStopping = EarlyStopping(monitor= 'val_loss',patience = 5,verbose = 0,mode='auto')

sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

# Cause idk why must reshape
X_train = X_train.reshape((-1,img_channels,SIZE[0],SIZE[1]))
X_test = X_test.reshape((-1,img_channels,SIZE[0],SIZE[1]))

# convert to float and normalize. just do it.
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

if useValidation:
    Y_val =  np_utils.to_categorical(y_val, nb_classes)
    X_val = X_val.reshape((-1,img_channels,SIZE[0],SIZE[1]))
    X_val = X_val.astype('float32')
    X_val /= 255
    

data_augmentation = False #does fancy processing stuff!
if not data_augmentation:
    print('Not using data augmentation.')
    if useValidation:
        model.fit(X_train, Y_train,
                  batch_size=batch_size,
                  nb_epoch=nb_epoch,
                  validation_data=(X_val, Y_val),
                  shuffle=True,
                  callbacks = [earlyStopping])
    else:
        model.fit(X_train, Y_train,
                  batch_size=batch_size,
                  nb_epoch=nb_epoch,
                  validation_data=(X_test, Y_test),
                  shuffle=True,
                  callbacks = [earlyStopping])
else:
    print('Using real-time data augmentation.')

    # this will do preprocessing and realtime data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=5,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(X_train)
    if useValidation:
        model.fit_generator(datagen.flow(X_train, Y_train,
                        batch_size=batch_size),
                        samples_per_epoch=X_train.shape[0],
                        nb_epoch=nb_epoch,
                        validation_data=(X_val, Y_val),
                        callbacks = [earlyStopping])
    else:
    # fit the model on the batches generated by datagen.flow()
        model.fit_generator(datagen.flow(X_train, Y_train,
                            batch_size=batch_size),
                            samples_per_epoch=X_train.shape[0],
                            nb_epoch=nb_epoch,
                            validation_data=(X_test, Y_test),
                            callbacks = [earlyStopping])

## FOR TEST IMGS.
if useValidation:
    from sklearn import metrics
    testpred = model.predict_classes(X_test)
    Y_test = Y_test.argmax(1)
    print('Test accuracy is :'+str(metrics.accuracy_score(Y_test,testpred)))
