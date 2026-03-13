import cv2
from preprocess import image_preprocess
from prediction import predict
import sys

# Upload image
img = cv2.imread(f'/workspaces/MS_DF_ComputerVision_Project/images/{sys.argv[1]}')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_prep = image_preprocess(img)
predict(img_prep)
