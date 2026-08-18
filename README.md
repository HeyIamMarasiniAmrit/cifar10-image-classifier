# CIFAR-10 CNN Classifier

A simple yet effective Convolutional Neural Network built with TensorFlow/Keras to classify images from the CIFAR-10 dataset.

## Overview
This project implements a lightweight CNN that classifies 32×32 color images into 10 categories:
- airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

## Model Architecture
- **Conv2D** (32 filters, 3×3) → ReLU → MaxPooling
- **Conv2D** (64 filters, 3×3) → ReLU → MaxPooling
- Flatten
- Dense (64 units) → ReLU
- Dense (10 units) → Softmax

## Results
- Training for 10 epochs
- Test Accuracy: ~70-75% (depending on run)
- Loss and accuracy curves included

## Requirements
```bash
tensorflow>=2.10
matplotlib
numpy
