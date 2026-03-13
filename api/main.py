from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2

from .preprocess import image_preprocess
from . import resnet50_cifar10_v1
from . import effnetb0_cifar10_v1

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    #read image bites
    contents = await file.read()

    #transform into numpy array
    np_arr = np.frombuffer(contents, np.uint8)

    #decode np_arr to image using cv2 functions
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #preprocess the image to fit cifar10 images size
    img = image_preprocess(img)

    #do predictions with both models
    resnet_pred_class, resnet_prob = resnet50_cifar10_v1.predict(img)
    effnet_pred_class, effnet_prob = effnetb0_cifar10_v1.predict(img)
    
    #return prediction results
    return {
        "imgclf_resnet50_cifar10_v1": {
            "predicted_class": resnet_pred_class,
            "confidence": float(resnet_prob)
        },
        "imgclf_effnetb0_cifar10_v1": {
            "predicted_class": effnet_pred_class,
            "confidence": float(effnet_prob)
        }
    }
