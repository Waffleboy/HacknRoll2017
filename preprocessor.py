#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 17:26:59 2017

Description: This file processes the raw images. It resizes, flips, rotates.

@author: waffleboy
"""

import cv2
import os,glob

BASE_FOLDER = "raw_images"
SAVE_TO_FOLDER = "processed_pictures"
DIMENSIONS = (120,120)

def run():
    global BASE_FOLDER
    check_and_create_save_folder(SAVE_TO_FOLDER)
    process_all_images(BASE_FOLDER)
    
def check_and_create_save_folder(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)

def process_all_images(BASE_FOLDER):
    global SAVE_TO_FOLDER
    for root, dirs, files in os.walk(BASE_FOLDER, topdown=False):
        for folder in dirs:
            link = BASE_FOLDER+'/'+folder + '/*.jpg'
            pictures = glob.glob(link)
            save_to_folder = SAVE_TO_FOLDER + '/'+folder
            check_and_create_save_folder(save_to_folder)
            for picture_link in pictures:
                process(picture_link,save_to_folder)
    return
    
    
def process(image_link,save_to_folder):
    global DIMENSIONS
    try:
        picname_without_extension = get_picname_without_extension(image_link)
        img = read_image(image_link)
        img = resize_image(img,DIMENSIONS)
        flip_image_all_directions(img,picname_without_extension,save_to_folder)
        rotate_image_all_directions(img,picname_without_extension,save_to_folder)
        save_image(save_to_folder,img,get_pic_name(image_link))
    except:
        print("Error in file {}".format(image_link))

def get_pic_name(picture_link):
    return picture_link[picture_link.rfind('/')+1:]
    
def get_picname_without_extension(picname):
    picname = get_pic_name(picname)
    return picname[:picname.find('.')]
    
    
def read_image(image_link):
    return cv2.imread(image_link)
    
# dimensions is a tuple.
def resize_image(img,dimensions):
    return cv2.resize(img, dimensions) 

def save_image(prev_path,img,picname):
    cv2.imwrite(prev_path+'/'+picname, img)

def flip_image_all_directions(img,picname_without_extension,save_to_folder):
    hori_flip_img=img.copy()
    verti_flip_img=img.copy()
    hori_flip_img=cv2.flip(img,1)
    verti_flip_img=cv2.flip(img,0)
    hori_verti_flip_img = hori_flip_img.copy()
    hori_verti_flip_img = cv2.flip(verti_flip_img,1) 
    save_image(save_to_folder,hori_flip_img,picname_without_extension +'_horizontal.jpg')
    save_image(save_to_folder,verti_flip_img,picname_without_extension + '_vertical.jpg')
    save_image(save_to_folder,hori_verti_flip_img,picname_without_extension + '_horivertical.jpg')
    
    
def rotate_image_all_directions(img,picname_without_extension,save_to_folder):
    lst = []
    for i in range(3):
        lst.append(img.copy())
    degree = 0
    for i in range(len(lst)):
        entry = lst[i]
        degree += 90
        (h, w) = entry.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, degree, 1.0)
        rotated = cv2.warpAffine(entry, M, (w, h))
        lst[i] = rotated
    degree = 0
    for entry in lst:
        degree += 90
        save_image(save_to_folder,entry,picname_without_extension+'rotated{}.jpg'.format(degree))
    
if __name__ == '__main__':
    run()

