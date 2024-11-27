# Sign Language Recognition

## Overview

This project aims to recognize sign language gestures using computer vision and deep learning techniques. The system uses a Convolutional Neural Network (CNN) to classify images of hand gestures into different sign language symbols. It utilizes libraries like **OpenCV**, **Keras**, **TensorFlow**, and **CVZone** to process images and train the model.

## Motivation

Sign language is an essential communication tool for the hearing-impaired community. By automating the recognition of sign language gestures, this project facilitates better communication between deaf and non-deaf individuals. The goal is to create a real-time sign language interpreter using deep learning techniques.

## Methodology

The Sign Language Recognition system follows these steps:

- **Data Collection**: The dataset of sign language gestures is collected using a camera and stored in the `Data` folder.
- **Pre-processing**: The images are processed using **OpenCV** for resizing and augmentation to prepare the data for training.
- **Model Training**: A Convolutional Neural Network (CNN) is trained using the pre-processed data. The model is saved as a `.h5` file in the `model` folder.
- **Prediction**: The model is used to predict the sign language gestures from input images.

## Repository Structure

- **Data/**: Contains the dataset of images used for training and testing the model.
- **Model/**: Contains the saved trained model (`keras_model.h5`) and the labels file (`labels.txt`).
- **README.md**: This file with project details.
- **datacollections.py**: Script used to collect and capture hand gesture data for training.
- **test.py**: Script that loads the trained model and performs real-time sign language recognition.
- **requirements.txt**: List of required dependencies for running the project.

### Implementation Steps

### 1. Collecting Data
Run the datacollections.py script to capture hand gestures and save them in the Data folder.

```bash
python datacollections.py

### 2. Training the Model
Once the data is collected, train the CNN model using the keras_model.h5 file. This step uses Keras and TensorFlow to train the model on the dataset of hand gestures.

```bash
python train_model.py

### 3. Testing the Model
After training the model, use the test.py script to perform real-time sign language recognition using a webcam. The model will predict the gestures shown in front of the camera.

```bash
python test.py

### Acknowledgments
- The model uses a custom Convolutional Neural Network (CNN) architecture built with Keras and TensorFlow.
- The dataset used in this project contains various sign language gestures collected from a camera.
- The OpenCV and CVZone libraries are used for image processing and real-time camera interaction.

