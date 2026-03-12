#!/usr/bin/env python
# coding: utf-8

#Libraries
from pathlib import Path
import time
import pandas as pd
import numpy as np
import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.datasets import cifar10
from sklearn.metrics import classification_report
from colorama import Fore, Style

def main():
    #Import Dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print(Fore.CYAN + "\ncifar10 dataset imported" + Style.RESET_ALL)

    #image preprocessing to match EfficientNet input requirements
    x_test_effnetb0 = efficientnet_preprocess(x_test)
    x_train_effnetb0 = efficientnet_preprocess(x_train)
    print(Fore.CYAN + "\nimages processed to match ResNet input requirements\n" + Style.RESET_ALL)

    #validation set
    x_train_effnetb0, x_val, y_train_effnetb0, y_val = train_test_split(
        x_train_effnetb0,
        y_train,
        test_size=0.2,
        random_state=42,
        shuffle=True,
        stratify=y_train
    )

    #import EfficientNetB0 for feature extraction
    pretrained_model = keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(32,32,3),
        pooling="max",
        name="effnetb0",
    )

    pretrained_model.trainable = False # disable EfficientNetB0 parameter optimization

    #define the structure of the CNN
    cnn_model = Sequential([
        pretrained_model,
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(10, activation='softmax')
    ])

    #compile the model
    cnn_model.compile(optimizer="adam",
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    #define early stopping
    early_stop = EarlyStopping(
        monitor="val_loss",         # monitors validation loss function
        patience=4,                 # number of steps with no improvements in "val_loss" before stopping the fitting
        restore_best_weights=True   # in case the further "patience" steps made the performance worse, it restores the best weights
    )

    #model fitting
    print(Fore.CYAN + "\nmodel defined and ready for training\n" + Style.RESET_ALL)
    start_time = time.time()

    history = cnn_model.fit(
        x=x_train_effnetb0,
        y=y_train_effnetb0,
        epochs= 50,
        batch_size=64,
        validation_data=(x_val, y_val),
        shuffle=True,
        callbacks=[early_stop],
        verbose=1
    )

    end_time = time.time()
    print(Fore.GREEN + f"\ntraining completed - duration:{(end_time - start_time)/60:.1f} min" + Style.RESET_ALL)

    #save the model after training
    BASE_DIR = Path(__file__).resolve().parent.parent
    MODELS_DIR = BASE_DIR / "models"
    MODELS_DIR.mkdir(exist_ok=True)
    model_path = MODELS_DIR / "imgclf_effnetb0_cifar10_v1.keras"
    cnn_model.save(model_path)
    print(Fore.GREEN + f"\nmodel saved at: {model_path}"  + Style.RESET_ALL)

    #model prediction
    print(Fore.CYAN + "\nready for prediction on test set\n" + Style.RESET_ALL)
    y_pred_probs = cnn_model.predict(x_test_effnetb0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    #model evaluation parameters
    test_loss, test_accuracy = cnn_model.evaluate(x_test_effnetb0, y_test)

    #print model performance on test 
    print("\n===================================================================")
    print("MODEL PERFORMANCE ON TEST SET")
    print("===================================================================")

    print("\naccuracy:", test_accuracy)
    print("loss:", test_loss, end="\n\n")

    class_names = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]

    print(classification_report(y_test, y_pred, target_names=class_names))
    print("===================================================================")

if __name__ == '__main__':
    main()