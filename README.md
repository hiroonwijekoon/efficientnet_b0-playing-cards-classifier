# Playing Card Classifier using Convolutional Neural Networks (CNNs)

This project aims to classify playing cards using deep learning techniques, specifically Convolutional Neural Networks (CNNs). The model is trained on a dataset containing images of various playing cards.

## Table of Contents

- [Introduction](#introduction)
- [Dataset Overview](#dataset-overview)
- [Model Architecture](#model-architecture)
- [Training Process](#training-process)
- [Model Evaluation](#model-evaluation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [References](#references)

## Introduction

In this project, we develop a CNN-based model to classify playing cards into different categories. The model is trained on a dataset consisting of images of playing cards from various decks and suits. The goal is to accurately identify the type of playing card depicted in a given image.

## Dataset Overview

The dataset used for training and evaluation contains a diverse collection of playing card images. It includes images of cards from different decks, suits, and ranks. The dataset is preprocessed to ensure consistency in image size and format.

- Dataset Size: 7794 images
- Classes: 53 (One class for each type of playing card)
- Image Size: 224 x 224 pixels (RGB format)
- Train-Validation-Test Split: 7624 images / 265 images / 265 images

For more information about the dataset, refer to [this link](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification).

## Model Architecture

The model architecture has been updated to utilize a pre-trained EfficientNet model for improved performance and accuracy. Here's an overview of the new model architecture:

### Base Model:

- **EfficientNet-B0**: The model utilizes a pre-trained EfficientNet-B0 as the base model for feature extraction. This pre-trained model helps in leveraging transfer learning for better feature representations.

### Custom Layers:

1. **Features Extraction**:

   - The base model's children layers, excluding the last classification layer, are used for feature extraction.
   - Output size from EfficientNet-B0 feature extraction: 1280.

2. **Classifier**:
   - **Flatten Layer**: Flattens the output feature maps into a 1D tensor.
   - **Fully Connected Layer**: A linear layer that maps the 1280 features to the number of classes (53).

### Forward Pass:

- The input images are passed through the EfficientNet-B0 feature extractor.
- The extracted features are then flattened.
- The flattened features are passed through the fully connected layer to get the class logits.

Overall, the `CardClassifierCNN` architecture employs a pre-trained EfficientNet-B0 for feature extraction and a custom classifier to predict the class of the playing card.

## Training Process

The model is trained using the Adam optimizer with the Cross-Entropy Loss function. Training is performed over multiple epochs, with early stopping implemented to prevent overfitting. Training progress and performance metrics are monitored using validation data.

For detailed information about the training process, refer to the Training Process section in the code.

## Model Evaluation

After training, the model is evaluated on a separate test set to assess its performance. The evaluation includes metrics such as accuracy, precision, recall, and F1-score. Additionally, qualitative assessment is performed by visualizing predictions on sample test images.

For more details, refer to the Model Evaluation section in the code.

## Usage

To use the model for inference, follow these steps:

1. Install the required dependencies (specified in the Dependencies section).
2. Clone the repository to your local machine.
3. Download the dataset and place it in the appropriate directory.
4. Run the provided scripts or execute the code in your preferred environment.

## Dependencies

Ensure you have the following dependencies installed:

- PyTorch version 2.1.2
- Torchvision version 0.16.2
- Numpy version 1.26.3
- Pandas version 2.1.4
- Matplotlib
- scikit-learn

## Contributing

Contributions to this project are welcome. Feel free to open issues, submit pull requests, or provide feedback on the existing implementation.

## References

- [Kaggle Dataset](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification)
- [Related Paper](https://www.kaggle.com/code/robikscube/train-your-first-pytorch-model-card-classifier)
