#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 18:57:50 2017

@author: waffleboy
"""

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
    