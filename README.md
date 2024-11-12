# RAPID Submission: Image Classification and Data Handling with PyTorch

This repository contains three Python scripts, each demonstrating different tasks related to image classification using PyTorch. Below is a detailed description of each script and the tasks performed within them.

## Files in the Repository

1. [cifar10_training.py](fleet-file:///home/chirag/Documents/Projects/RAPID_submission/cifar10_training.py?hostId=grhptnvil43tde7csg1t&root=%2F&type=file)
2. [custom_dataset_loader.py](fleet-file:///home/chirag/Documents/Projects/RAPID_submission/custom_dataset_loader.py?hostId=grhptnvil43tde7csg1t&root=%2F&type=file)
3. [transfer_learning_resnet.py](fleet-file:///home/chirag/Documents/Projects/RAPID_submission/transfer_learning_resnet.py?hostId=grhptnvil43tde7csg1t&root=%2F&type=file)

## cifar10_training.py

This script demonstrates how to train a simple Convolutional Neural Network (CNN) on the CIFAR-10 dataset using PyTorch.

### Features:
- **Dataset Loading**:
  - Utilizes `torchvision.datasets.CIFAR10` to download and load CIFAR-10 dataset.
  - Applies normalization to the dataset.
- **Neural Network Definition**:
  - Defines a simple CNN with two convolutional layers and two fully connected layers.
  - Uses ReLU activation and max pooling.
- **Training Configuration**:
  - Uses the Adam optimizer and Cross-Entropy Loss.
  - Trains the model for a specified number of epochs (10 epochs in this case).
  - Logs the loss value at regular intervals.
- **Model Evaluation**:
  - Tests the trained model on the CIFAR-10 test set.
  - Reports the final accuracy of the model on the test set.

## custom_dataset_loader.py

This script showcases how to create a custom dataset class for loading a simple CSV file containing image filenames and labels.

### Features:
- **CSV Reading**:
  - Reads a CSV file with image filenames and labels using `pandas`.
- **Custom Dataset Class**:
  - Inherits from `torch.utils.data.Dataset`.
  - Loads images from the file paths provided in the CSV.
  - Applies specified transformations to the images.
- **DataLoader**:
  - Uses `torch.utils.data.DataLoader` to create a data loader for batching and shuffling the custom dataset.
  - Demonstrates iterating through the DataLoader to fetch batches of images and their corresponding labels.

## transfer_learning_resnet.py

This script demonstrates how to use transfer learning with a pre-trained ResNet model for a custom classification task.

### Features:
- **Loading Pre-trained Model**:
  - Loads a pre-trained ResNet-18 model from `torchvision.models`.
- **Model Modification**:
  - Modifies the final fully connected layer to match the number of classes in the custom dataset (10 classes in this example).
- **Freezing Pre-trained Layers**:
  - Freezes the weights of all pre-trained layers so that only the final layer's weights are updated during training.
- **Dataset Loading**:
  - Uses CIFAR-10 dataset for the illustration but can be replaced with any custom dataset.
  - Applies suitable transformations, including resizing and normalization, expected by ResNet.
- **Training Configuration**:
  - Uses the Adam optimizer and Cross-Entropy Loss.
  - Trains the model for a specified number of epochs (10 epochs in this case).
  - Logs the loss value at regular intervals.
- **Model Evaluation**:
  - Tests the trained model on the dataset's test set.
  - Reports the final accuracy of the model on the test set.

## Getting Started

### Prerequisites

Ensure you have the following libraries installed:
- `torch`
- `torchvision`
- `pandas`

You can install them using pip:
```bash
pip install torch torchvision pandas
```

and then just execute the scripts one by one