from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # leggi i byte dell'immagine
    contents = await file.read()

    # converti in array numpy
    np_arr = np.frombuffer(contents, np.uint8)

    # decodifica immagine con OpenCV
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    return {
        "filename": file.filename,
        "shape": img.shape
    }