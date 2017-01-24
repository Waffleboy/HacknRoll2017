#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 18:57:50 2017

@author: waffleboy
"""

#==============================================================================
#                    Temp keras bugfix
import tensorflow as tf
tf.python.control_flow_ops = tf
#==============================================================================

from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.advanced_activations import PReLU
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from sklearn import metrics
from sklearn.externals import joblib
import cv2
import pandas as pd
import glob
import random
import numpy as np

random.seed(4)

def load_label_dic(csv_file):
    df = pd.read_csv(csv_file)
    trainID = dict(zip(df.name, df.label)) 
    return trainID
    
def get_pic_name(picture_link):
    return picture_link[picture_link.rfind('/')+1:]

#==============================================================================
#                           Settings
#==============================================================================
label_dic = load_label_dic("labels.csv") #CSV containing labels for training data
TRAINING_DIRECTORY = "train" #folder containing training images
batch_size = 20 
nb_epoch = 25
SIZE = (120,120) # input picture dimensions
img_channels = 3 #RGB
#==============================================================================
img_rows, img_cols = SIZE[0],SIZE[1]
nb_classes = len(np.unique(np.array(list(label_dic.values()))))

#==============================================================================
#                                Load Images
#==============================================================================
def change_labels_to_numeric(labels):
    unique = np.unique(labels)
    mapper = {v:k for k,v in enumerate(unique)}
    for i in range(len(labels)):
        labels[i] = mapper[labels[i]]
    return labels,mapper

def load_images_and_labels():
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
        images_and_labels.append([pic,label,picname])
    random.shuffle(images_and_labels)
    return images_and_labels

def split_imageslabels_to_arrays(images_and_labels):
    trainImgs = np.array([x[0] for x in images_and_labels])
    labels = np.array([x[1] for x in images_and_labels])
    picnames = np.array([x[2] for x in images_and_labels])
    return trainImgs,labels,picnames
#==============================================================================
#                                   Models
#==============================================================================

def custom_model(img_channels,img_rows,img_cols):    
    model = Sequential()
    # 1st layer
    model.add(Convolution2D(32, 3, 3, input_shape=(3, SIZE[0],SIZE[1])))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    #last
    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512))
    model.add(Activation('sigmoid'))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    return model


def VGG_16(weights_path=None):
    global SIZE
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,SIZE[0],SIZE[1])))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

#    model.add(ZeroPadding2D((1,1)))
#    model.add(Convolution2D(512, 3, 3, activation='relu'))
#    model.add(ZeroPadding2D((1,1)))
#    model.add(Convolution2D(512, 3, 3, activation='relu'))
#    model.add(ZeroPadding2D((1,1)))
#    model.add(Convolution2D(512, 3, 3, activation='relu'))
#    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model
    
#==============================================================================
#                             Split Data
#==============================================================================

def split_data_and_reshape_master(trainImgs,labels,useValidation = True):
    if useValidation:
        return split_data_and_reshape_with_validation(trainImgs,labels)
    return split_data_and_reshape_without_validation(trainImgs,labels)
    
def split_data_and_reshape_with_validation(trainImgs,labels):
    global img_channels,SIZE
    dataset, X_test, target, y_test = splitTrainTest(trainImgs,labels)
    X_train, X_val, y_train,y_val = splitTrainTest(dataset,target)
    
    X_train, X_test, Y_train,Y_test = reshape_and_normalize_all(X_train, X_test, y_train,y_test)
    X_val,Y_val = reshape_and_normalize(X_val,y_val)
    return X_train,X_test,Y_train,Y_test,X_val,Y_val
    
def split_data_and_reshape_without_validation(trainImgs,labels):
    global img_channels,SIZE
    X_train, X_test, y_train,y_test = splitTrainTest(trainImgs,labels)
     # convert class vectors to binary class matrices
    X_train, X_test, Y_train,Y_test = reshape_and_normalize_all(X_train, X_test, y_train,y_test)
    return X_train, X_test, Y_train,Y_test
  
def reshape_and_normalize_all(X_train, X_test, y_train,y_test):
    X_train,Y_train = reshape_and_normalize(X_train,y_train)
    X_test,Y_test = reshape_and_normalize(X_test,y_test)
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    return X_train, X_test, Y_train,Y_test
    
def reshape_and_normalize(train_x,target_y):
    target_y = np.array(list(map(lambda x:int(x),target_y)))
    targetY =  np_utils.to_categorical(target_y, nb_classes)
    train_x = train_x.reshape((-1,img_channels,SIZE[0],SIZE[1]))
    train_x = train_x.astype('float32')
    train_x /= 255
    return train_x,targetY
    
def splitTrainTest(trainXData,trainYData,test_size=0.1):
    from sklearn.cross_validation import train_test_split
    print('Splitting Data')
    X_train, X_test, y_train, y_test = train_test_split(trainXData,trainYData,test_size=test_size)
    return X_train, X_test, y_train, y_test 
    
#==============================================================================
#                           Model Training
#==============================================================================
    
def train_model_without_augmentation(model,X_train, X_test, Y_train,Y_test,\
                                     earlyStopping,X_val = None,Y_val = None):
    print("Training model without augmentation")
    global nb_epoch,batch_size
    if X_val is not None:
        model.fit(X_train, Y_train,
                          batch_size=batch_size,
                          nb_epoch=nb_epoch,
                          validation_data=(X_val, Y_val),
                          shuffle=True,
                          callbacks = [earlyStopping])
        return model
    model.fit(X_train, Y_train,
                      batch_size=batch_size,
                      nb_epoch=nb_epoch,
                      validation_data=(X_test, Y_test),
                      shuffle=True,
                      callbacks = [earlyStopping])
    return model
    
def train_model_with_augmentation(model,X_train, X_test, Y_train,Y_test,\
                                     earlyStopping,X_val = None,Y_val = None):
    global nb_epoch,batch_size
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
    if X_val is not None:
        model.fit_generator(datagen.flow(X_train, Y_train,
                        batch_size=batch_size),
                        samples_per_epoch=X_train.shape[0],
                        nb_epoch=nb_epoch,
                        validation_data=(X_val, Y_val),
                        callbacks = [earlyStopping])
        return model
    # fit the model on the batches generated by datagen.flow()
    model.fit_generator(datagen.flow(X_train, Y_train,
                        batch_size=batch_size),
                        samples_per_epoch=X_train.shape[0],
                        nb_epoch=nb_epoch,
                        validation_data=(X_test, Y_test),
                        callbacks = [earlyStopping])
    return model
    
#==============================================================================
#                                misc
#==============================================================================
def find_test_accuracy(model,X_test,Y_test):
    testpred = model.predict_classes(X_test)
    Y_test = Y_test.argmax(1)
    print('Test accuracy is :'+str(metrics.accuracy_score(Y_test,testpred)))
    return testpred
    
def save_model(model,name):
    model.save('{}.h5'.format(name))  # creates a HDF5 file 

# Keras does not use categorical targets. save the mapping from categorical to numerical
def save_mapping(label_mapper_dic,save_model_to):
    joblib.dump(label_mapper_dic,"{}.pkl".format('saved_models/'+save_model_to +'_mapping'))

def save_model_if_specified(model,model_name,label_mapper_dic):
    if model_name:
        print("Saving model to {}".format('saved_models/'+model_name))
        save_model(model,model_name)
        save_mapping(label_mapper_dic)
    return

#==============================================================================
#                               Main
#==============================================================================
def run(use_validation = True,save_model_to = ''):
    global img_channels,img_rows,img_cols
    images_and_labels = load_images_and_labels()
    trainImgs,labels,picnames = split_imageslabels_to_arrays(images_and_labels)
    labels, label_mapper_dic = change_labels_to_numeric(labels)
    model = custom_model(img_channels,img_rows,img_cols)
    #model = VGG_16()
    if use_validation:
        X_train,X_test,Y_train,Y_test,X_val,Y_val = split_data_and_reshape_master(trainImgs,labels)
    else:
        X_train,X_test,Y_train,Y_test = split_data_and_reshape_master(trainImgs,labels,False)
    earlyStopping = EarlyStopping(monitor= 'val_loss',patience = 5,verbose = 0,mode='auto')
    sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    if use_validation:
        model = train_model_without_augmentation(model,X_train,X_test,\
                                             Y_train,Y_test,earlyStopping,
                                             X_val,Y_val)
    else:
        model = train_model_without_augmentation(model,X_train,X_test,\
                                             Y_train,Y_test,earlyStopping)
    find_test_accuracy(model,X_test,Y_test)
    save_model_if_specified(model,save_model_to,label_mapper_dic)
    return model
    
if __name__ == '__main__':
    run()