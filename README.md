# Face-Mask-Detection-using-CNN

## Description
During the COVID-19 pandemic, the WHO made wearing masks compulsory to protect against the virus. This project develops a real-time Face Mask Detector using Python. The system detects whether the person in the image is wearing a mask or not. We train the face mask detector model using Keras and OpenCV.

## Dataset
The dataset consists of 7553 images from Kaggle:
- Total images: 7553
  - Masked images: 3725
  - Unmasked images: 3828

### Download the Dataset
You can download the dataset from Kaggle [here](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset).

## Model
The model is built using a Convolutional Neural Network (CNN) architecture. The CNN captures spatial hierarchies in images, making it suitable for image classification tasks.

### Architecture
- Input Layer: Image input layer that accepts images of size 128x128x3 (height, width, channels).
- Convolutional Layers: Several convolutional layers followed by ReLU activation functions.
- Pooling Layers: MaxPooling layers to reduce the spatial dimensions of the feature maps.
- Fully Connected Layers: Dense layers for final classification.
- Output Layer: A dense layer with a two neurons and a sigmoid activation function for binary classification (masked or unmasked).

### Performance
- Training accuracy: 99%
- Validation accuracy: 96%
- Test accuracy: 95%

## Dependencies
The following libraries and frameworks are required to run this project:
- numpy
- tensorflow
- keras
- matplotlib
- pandas
- opencv-python (cv2)

## Installation
To install the necessary dependencies, run the following command:
```bash
pip install numpy tensorflow keras matplotlib pandas opencv-python
