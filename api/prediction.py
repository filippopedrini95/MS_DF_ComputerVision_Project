#!/usr/bin/env python
# coding: utf-8

import numpy as np
import tensorflow as tf
import keras
from keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess

def predict(img_prepr):
    if type(img_prepr) != np.ndarray:
        print("provide an image")
        return 0    

    #load imgclf_resnet50_cifar10_v1
    imgclf_resnet50_model = keras.models.load_model("/workspaces/MS_DF_ComputerVision_Project/models/imgclf_resnet50_cifar10_v1.keras")

    #load imgclf_effnetb0_cifar10_v1.keras
    imgclf_effnetb0_model = keras.models.load_model("/workspaces/MS_DF_ComputerVision_Project/models/imgclf_effnetb0_cifar10_v1.keras")

    #classes dict
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
    class_prob_effnet = imgclf_effnetb0_model.predict(img_prepr_effnet)
    class_pred_effnet = np.argmax(class_prob_effnet, axis=1)

    print("imgclf_resnet50_cifar10_v1:", class_names[int(class_pred_resnet[0])])
    print("imgclf_effnetb0_cifar10_v1:", class_names[int(class_pred_effnet[0])])

