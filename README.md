# CIFAR-10 Image Classification

This repository contains implementations of image classification on the CIFAR-10 dataset using three different models: Artificial Neural Network (ANN), Convolutional Neural Network (CNN), and Support Vector Machine (SVM). I've also showcased the classification accuracy of each model.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Models](#models)
  - [Artificial Neural Network (ANN)](#artificial-neural-network-ann)
  - [Convolutional Neural Network (CNN)](#convolutional-neural-network-cnn)
  - [Support Vector Machine (SVM)](#support-vector-machine-svm)
- [Results](#results)
- [Setup and Usage](#setup-and-usage)

## Introduction
The goal of this project is to classify images from the CIFAR-10 dataset into one of the 10 classes using different machine learning models. We compare the performance of ANN, CNN, and SVM models.

## Dataset
The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes. There are 50,000 training images and 10,000 test images. The classes are:

1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

## Models

### Artificial Neural Network (ANN)
Artificial Neural Networks (ANNs) are a type of neural network that mimics the way humans learn. They consist of layers of neurons where each neuron is connected to every neuron in the next layer. ANNs are effective for a variety of tasks but can be less efficient for image data due to the high number of parameters involved.

### Convolutional Neural Network (CNN)
Convolutional Neural Networks (CNNs) are a specialized type of neural network designed specifically for processing structured grid data such as images. They are highly effective for image classification tasks due to their ability to automatically and adaptively learn spatial hierarchies of features through backpropagation by using multiple building blocks, such as convolution layers, pooling layers, and fully connected layers. CNNs have fewer parameters than fully connected networks with the same number of hidden units, making them easier to train and less prone to overfitting.

### Support Vector Machine (SVM)
Support Vector Machines (SVMs) are supervised learning models used for classification and regression analysis. They are effective in high-dimensional spaces and are still efficient when the number of dimensions exceeds the number of samples. SVMs are particularly suited for binary classification tasks. For multi-class classification, as in the CIFAR-10 dataset, a one-vs-all or one-vs-one approach can be used. In this project, we use HOG (Histogram of Oriented Gradients) features extracted from the images to train the SVM.

## Results
The classification accuracy for each model on the CIFAR-10 test set is as follows:

- **SVM**: 52.99%
- **CNN**: 70.08%
- **ANN**: 49.41%


## Setup and Usage
Follow these steps to set up and run the project:

1. Clone the repository:
    ```bash
    git clone https://github.com/yashnilapwar/cifar10-classification-using-SVM-ANN-and-CNN.git
    cd cifar10-classification
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the training scripts for each model:
    - For ANN:
        ```bash
        python train_ann.py
        ```
    - For CNN:
        ```bash
        python train_cnn.py
        ```
    - For SVM:
        ```bash
        python train_svm.py
        ```

