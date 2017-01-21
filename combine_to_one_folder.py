#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 16:44:48 2017

Description: This file combines the different labelled folders into 
            one folder, and generates a CSV Of filename : label in the 
            current directory
            
@author: waffleboy
"""
import os
import preprocessor
import glob
import pandas as pd
import cv2

ROOT_FOLDER = "processed_pictures" #where the original training images are
SAVE_TO_FOLDER = "train" #where the new images will be

def check_and_create_save_folder(SAVE_TO_FOLDER):
    if not os.path.exists(SAVE_TO_FOLDER):
        os.mkdir(SAVE_TO_FOLDER)

def save_to_folder(img,picturename):
    global SAVE_TO_FOLDER
    cv2.imwrite(SAVE_TO_FOLDER+'/'+picturename, img)
    
def get_pic_name(picture_link):
    return picture_link[picture_link.rfind('/')+1:]
    
def generate_csv(label_dic):
    a = pd.Series(index=range(len(label_dic)),dtype=str)
    b = pd.Series(index=range(len(label_dic)),dtype=str)
    counter = 0
    for key,value in label_dic.items():
        a.set_value(counter,key)
        b.set_value(counter,value)
        counter += 1
    df = pd.DataFrame()
    df["name"] = a
    df["label"] = b
    df.to_csv("labels.csv",index=False)

check_and_create_save_folder(SAVE_TO_FOLDER)
counter = 0
label_dic = {}
for root, dirs, files in os.walk(ROOT_FOLDER, topdown=False):
    for folder in dirs:
        link = ROOT_FOLDER+'/'+folder + '/*.jpg'
        pictures = glob.glob(link)
        for picture_link in pictures:
            pic_name = get_pic_name(picture_link)
            picture = preprocessor.read_image(picture_link)
            label_dic[pic_name] = folder
            save_to_folder(picture,pic_name)
            counter += 1
            if counter % 2000 == 0:
                print("Transferred {} images".format(counter))
            
generate_csv(label_dic)
