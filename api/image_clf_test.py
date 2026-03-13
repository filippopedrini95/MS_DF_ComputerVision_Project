import cv2
from preprocess import image_preprocess
import imgclf_resnet50_cifar10_v1_model 
import imgclf_effnetb0_cifar10_v1_model
import sys
from pathlib import Path

# Upload image
img = cv2.imread('/Users/filippopedrini/PROGRAMMING/MS_DF_ComputerVision_Project/images/cat.jpeg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_prep = image_preprocess(img)
a, b = imgclf_resnet50_cifar10_v1_model.predict(img_prep)

print(a, b)

