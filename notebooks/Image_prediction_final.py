#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess


#load imgclf_resnet50_cifar10_v1
imgclf_resnet50_model = keras.models.load_model("/workspaces/MS_DF_ComputerVision_Project/models/imgclf_resnet50_cifar10_v1.keras")


#STILL TO GET
#load imgclf_effnetb0_cifar10_v1.keras
#imgclf_effnetb0_model = keras.models.load_model("imgclf_effnetb0_cifar10_v1.keras")



class_names = {
    0:"airplane", 
    1:"automobile", 
    2:"bird", 
    3:"cat", 
    4:"deer",
    5:"dog", 
    6:"frog", 
    7:"horse", 
    8:"ship", 
    9:"truck"
}


# Image Preprocessing

# Upload image
img = cv2.imread('/workspaces/MS_DF_ComputerVision_Project/images/airplane.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


#resize and padding


height, width, ch = img.shape

w_longest =  width > height
max_side = np.max([height, width])
min_side = np.min([height, width])

resized_max_side = 32
resized_min_side = int(32 * min_side / max_side) if height != width else 32

dsize = (resized_max_side, resized_min_side) if w_longest else (resized_min_side, resized_max_side)
img_resized = cv2.resize(img, dsize, interpolation=cv2.INTER_AREA)

padd_tuple = (int(np.ceil((32 - resized_min_side)/2)), int((32 - resized_min_side)/2), 0, 0)

if w_longest:
  top, bottom, left, right = padd_tuple
else:
  left, right, top, bottom = padd_tuple


img_prepr = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)


#Model Preprocessing

img_prepr_resnet = resnet_preprocess(img_prepr)
img_prepr_resnet = np.array([img_prepr_resnet])

img_prepr_effnet = efficientnet_preprocess(img_prepr)
img_prepr_effnet = np.array([img_prepr_effnet])


#Prediction

#prediction with imgclf_resnet50_model
class_prob_resnet = imgclf_resnet50_model.predict(img_prepr_resnet)
class_pred_resnet = np.argmax(class_prob_resnet, axis=1)



#prediction with imgclf_effnetb0_model
#class_prob_effnet = imgclf_effnetb0_model.predict(img_prepr_effnet)
#class_pred_effnet = np.argmax(class_prob_effnet, axis=1)



print("imgclf_resnet50_cifar10_v1:", class_names[int(class_pred_resnet[0])])
#print("imgclf_effnetb0_cifar10_v1:", class_names[int(class_pred_effnet[0])])

