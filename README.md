# Computer Vision Image Classifier API

## 1. Description
This is a personal project implementing a computer vision image classifier using transfer learning on the CIFAR-10 dataset.  
It trains two CNN models, ResNet50 and EfficientNetB0, by freezing the pre-trained feature extractor and training only a custom dense layer on top, optimized for 10 classes.

The models achieve the following test set accuracies:

- ResNet50-based: ~0.658
- EfficientNetB0-based: ~0.602

The trained models have been deployed within the API (built using FastAPI) so it takes an image as input, applies preprocessing, runs predictions on both models, and returns a JSON containing the predicted classes and confidence scores for both models at once.

The project is fully containerized with Docker, allowing the API to run without installing Python or libraries locally.

## 2. Project Structure

The project is organized as follows:

```
.
├── Dockerfile
├── README.md
│
├── api
│   ├── effnetb0_cifar10_v1.py
│   ├── main.py
│   ├── preprocess.py
│   └── resnet50_cifar10_v1.py
│
├── images
│   ├── airplane.jpg
|   └── cat.jpeg
│
├── models
│   ├── imgclf_effnetb0_cifar10_v1.keras
│   └── imgclf_resnet50_cifar10_v1.keras
│
├── notebooks
│   ├── 0_EDA_hyperparam_tuning.ipynb
│   ├── 1_train_resnet50_cifar10.ipynb
│   ├── 2_train_effnetb0_cifar10.ipynb
│   ├── 3_inference_draft.ipynb
│   └── 4_inference_final.ipynb
│
├── pyproject.toml
│
├── training
│   ├── train_effnetb0-cifar10.py
│   └── train_resnet50-cifar10.py
│
└── uv.lock
```

### Brief explanation of each folder/file

- **Dockerfile** → Instructions to build the Docker image for the API  
- **README.md** → Project documentation (this file)  
- **api/** → FastAPI application code and model prediction scripts  
  - `main.py` → FastAPI app and endpoints  
  - `preprocess.py` → Image preprocessing functions  
  - `effnetb0_cifar10_v1.py` → EfficientNetB0-based model prediction function  
  - `resnet50_cifar10_v1.py` → ResNet50-based model prediction function
- **images/** → a few sample images that can be used to test the models 
- **models/** → Trained model files saved in Keras format  
- **notebooks/** → Jupyter notebooks used for EDA, hyperparameter tuning, model training and inference drafts  
- **pyproject.toml** → Dependency and environment management for the project (used with `uv`)  
- **training/** → Python scripts for training individual models outside the notebooks  
- **uv.lock** → Lock file for reproducible environment managed by `uv`




## 3. Requirements / How to run

- Python >= 3.12
- Dependencies managed with `uv` (`pyproject.toml` and `uv.lock`)
- Key Python libraries: colorama, FastAPI, Keras, numpy, opencv-python-headless, pandas, pathlib, pydantic, python-multipart, scikit-learn, TensorFlow, uvicorn
- Additional development libraries: Jupyter, matplotlib, seaborn

### To run the API locally:

1. Install dependencies: `uv sync`  
2. Start the server: `uvicorn api.main:app --port 8000`  
3. Access the API at: `http://localhost:8000`

### Alternatively, you can run the API via Docker:

1. Build the image: `docker build -t my-api-image .`  
2. Run the container: `docker run -p 8000:8000 my-api-image`  
3. Access the API at `http://localhost:8000`



## 4. Model Architecture & Training

The two convolutional neural network (CNN) models in this project are built using **transfer learning**. Transfer learning allows the use of pre-trained networks as feature extractors, which reduces training time and improves performance when working with smaller datasets like CIFAR-10.

The first model uses **ResNet50** as the feature extractor. ResNet50 is a deep CNN originally trained on ImageNet that includes residual connections to facilitate the training of very deep networks. In this project, the top (classification) layers of ResNet50 are removed (`include_top=False`) and the network is used only to extract features from the input images. The ResNet50 weights are **frozen** to prevent them from being updated during training, so only the new dense layers added on top are trained.

The second model uses **EfficientNetB0** as the feature extractor. EfficientNetB0 is a CNN designed to achieve a good balance between accuracy and efficiency by scaling depth, width, and resolution systematically. Like with ResNet50, the top layers are removed and the pre-trained weights are frozen, leaving only the custom dense layers trainable.

For both models, the **dense layer architecture** added on top of the feature extractors consists of:

- a first Dense layer with 128 units and ReLU activation  
- a Dropout layer (0.2) to reduce overfitting  
- a second Dense layer with 32 units and ReLU activation  
- a final Dense layer with 10 units and softmax activation to produce class probabilities for the 10 CIFAR-10 classes  

**Early stopping** is used during training, monitoring the validation loss with a patience of 4 epochs. This stops training if the validation loss does not improve for four consecutive epochs, and restores the weights of the best-performing epoch to avoid overfitting.

The models are compiled using the Adam optimizer with sparse categorical crossentropy loss, and accuracy as the metric. Training is performed with a batch size of 64, for up to 20 epochs for ResNet50 and 50 epochs for EfficientNetB0, using the provided training and validation sets.


## 5. Model Performance

The two models were evaluated on the CIFAR-10 test set. The ResNet50-based model achieved an accuracy of approximately 65.8%, while the EfficientNetB0-based model reached around 60.2%. The following tables summarize precision, recall, and f1-score for each class.

### ResNet50-based model

**Accuracy:** 0.658  
**Loss:** 0.989  

| Class       | Precision | Recall | F1-score | Support |
|------------|-----------|--------|----------|---------|
| airplane   | 0.72      | 0.69   | 0.71     | 1000    |
| automobile | 0.74      | 0.69   | 0.71     | 1000    |
| bird       | 0.61      | 0.55   | 0.58     | 1000    |
| cat        | 0.52      | 0.51   | 0.51     | 1000    |
| deer       | 0.58      | 0.61   | 0.60     | 1000    |
| dog        | 0.69      | 0.54   | 0.61     | 1000    |
| frog       | 0.64      | 0.77   | 0.70     | 1000    |
| horse      | 0.65      | 0.71   | 0.68     | 1000    |
| ship       | 0.73      | 0.77   | 0.75     | 1000    |
| truck      | 0.69      | 0.74   | 0.71     | 1000    |

### EfficientNetB0-based model

**Accuracy:** 0.602  
**Loss:** 1.120  

| Class       | Precision | Recall | F1-score | Support |
|------------|-----------|--------|----------|---------|
| airplane   | 0.67      | 0.62   | 0.65     | 1000    |
| automobile | 0.64      | 0.71   | 0.67     | 1000    |
| bird       | 0.52      | 0.50   | 0.51     | 1000    |
| cat        | 0.39      | 0.55   | 0.45     | 1000    |
| deer       | 0.56      | 0.55   | 0.55     | 1000    |
| dog        | 0.60      | 0.44   | 0.51     | 1000    |
| frog       | 0.64      | 0.73   | 0.68     | 1000    |
| horse      | 0.74      | 0.59   | 0.66     | 1000    |
| ship       | 0.72      | 0.67   | 0.70     | 1000    |
| truck      | 0.67      | 0.67   | 0.67     | 1000    |

These metrics are consistent with a small-scale proof-of-concept transfer learning approach. The goal of this project is to demonstrate the full pipeline—from feature extraction and dense layer training to API integration and containerization—rather than to maximize classification accuracy.




## 6. API Usage

The project exposes a single endpoint `/predict` through the FastAPI API. This endpoint accepts a POST request containing an image file. The request must be sent as `multipart/form-data` with the image attached.

For example, using `curl`, sending the image `cat.jpeg` stored in the `images/` folder:

```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@images/cat.jpeg;type=image/jpeg'
```

When the request is sent, the API will return a JSON response containing the predicted class and confidence score for both models at once. An example response is:

```json
{
  "imgclf_resnet50_cifar10_v1": {
    "predicted_class": "cat",
    "confidence": 0.6586176156997681
  },
  "imgclf_effnetb0_cifar10_v1": {
    "predicted_class": "cat",
    "confidence": 0.540245771408081
  }
}
```

If the API is running inside a Docker container, make sure the container’s port is mapped correctly (for example, using `docker run -p 8000:8000 my-api-image`). Then, the API can be accessed from `http://localhost:8000/predict` on the host machine.

This simple setup allows you to send any image and receive predictions from both models in a single call, demonstrating the full inference pipeline with preprocessing and model execution.


## 7. Notes / Future Work

**Current Limitations:**

- The model accuracies are moderate (ResNet50 ~66%, EfficientNetB0 ~60%), which makes the system functional for demonstration purposes but not suitable for critical applications or production scenarios.
- Certain classes perform worse than others, as seen in the precision, recall, and f1-score metrics—particularly for the “cat” and “bird” categories—indicating that some image types are harder for the models to classify accurately.    
- Both models use frozen feature extractors, meaning only the top dense layers were trained. This approach does not fully exploit the power of transfer learning and may limit performance.  


**Potential Improvements / Future Work:**

- Experiment with fine-tuning the feature extractor layers of both ResNet50 and EfficientNetB0 to potentially improve overall accuracy and class-specific performance.  
- Test additional pre-trained models or custom CNN architectures to explore whether alternative feature extractors or deeper networks can yield better results on CIFAR-10.  
- Develop a simple web interface for the API, allowing users to upload images and see predictions directly, without requiring `curl` commands or Python scripts, making the system more user-friendly and self-contained.

## 8. Contact

**Author:** Filippo Pedrini\
**Contact:** filippopedrini95@gmail.com