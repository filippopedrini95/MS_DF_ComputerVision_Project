from pathlib import Path
import numpy as np
import tensorflow as tf
import keras
from keras.applications.efficientnet import preprocess_input as efficientnet_preprocess

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

#models directory path
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"


def predict(img_prepr):
    if type(img_prepr) != np.ndarray:
        print("provide an image")
        return 0    

    #load imgclf_effnetb0_cifar10_v1.keras
    imgclf_effnetb0_model = keras.models.load_model(f"{MODELS_DIR}/imgclf_effnetb0_cifar10_v1.keras")

    #Model Preprocessing
    img_prepr_effnet = efficientnet_preprocess(img_prepr)
    img_prepr_effnet = np.array([img_prepr_effnet])

    #prediction with imgclf_effnetb0_model
    class_prob_effnet = imgclf_effnetb0_model.predict(img_prepr_effnet)
    class_pred_effnet = np.argmax(class_prob_effnet, axis=1)

    result = (class_names[int(class_pred_effnet[0])], np.max(class_prob_effnet))
    return result