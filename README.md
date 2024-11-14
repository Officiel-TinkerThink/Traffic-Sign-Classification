# Traffic Sign Classification

This project develops a machine learning model capable of classifying traffic sign images into one of 43 categories using a Convolutional Neural Network (CNN).

## Objective

The goal of this project is to build an AI system that can identify traffic signs in images using a convolutional neural network (CNN) trained with PyTorch.

## Background

In the development of self-driving cars, one of the main challenges is computer vision – the ability for a car to understand its environment from images. Recognizing and distinguishing road signs, such as stop signs, speed limits, and yield signs, is a key aspect of this task.

In this project, we use the German Traffic Sign Recognition Benchmark (GTSRB) dataset, which consists of thousands of labeled images across 43 traffic sign categories. The task is to classify these road signs based on their image representations using a CNN built with TensorFlow.

## Specifications

### Data Loading (`load_data`)

- **Function Description**: `load_data(data_dir)`
  - The function should load the dataset from the given directory (`data_dir`) and return a tuple containing the image arrays and their corresponding labels.
  - The directory structure will include subdirectories for each class, named from `0` to `NUM_CATEGORIES-1`, each containing images of the respective category.
  - The images should be resized to dimensions `(IMG_WIDTH, IMG_HEIGHT)` to fit the neural network input.
  - Return two lists:
    - `images`: A list of image arrays (numpy.ndarrays).
    - `labels`: A list of integers representing the category labels for each image.
  - Ensure platform independence when handling paths by using `os.sep` and `os.path.join` instead of hardcoding path separators.

### Model Creation (`get_model`)

- **Function Description**: `get_model()`
  - The function should return a compiled neural network model.
  - The input layer should accept images with the shape `(IMG_WIDTH, IMG_HEIGHT, 3)` (RGB channels).
  - The output layer should have `NUM_CATEGORIES` units, each corresponding to one of the 43 traffic sign classes.
  - You are encouraged to experiment with various layers and architectures:
    - Convolutional and pooling layers (number and size of filters, pool sizes).
    - Fully connected layers (number and size of hidden layers).
    - Dropout for regularization.

## Architecture Features

## Adding Convolutional Layers

- The introduction of convolutional layers improved performance by allowing the model to learn multi-pixel features rather than relying on raw pixel intensity data.
   
## Pooling Layers

- Pooling layers help reduce image size by downsampling, making the model less sensitive to the exact location of features.

## Dropout

- Dropout was added to reduce overfitting by randomly deactivating certain neurons during training.

## Fully Connected NN
- on top of the convolution, I give Fully Connected Neural Network for the Architecture to learn after the feature given by convolutional layer


## Conclusion

By applying various techniques like convolutional layers, pooling layers, and dropout, we were able to achieve substantial improvements in the model's ability to classify traffic signs. The best model achieved approximately **92% accuracy**, showing that the addition of multi-layer convolutions and pooling led to better generalization on the test dataset.

## Usage:

Requires Python(3) and the python package installer pip(3) to run.

First install requirements:

$pip(3) install -r requirements.txt

Download the GTSRB dataset from https://cdn.cs50.net/ai/2020/x/projects/5/gtsrb.zip

Run the training and testing script:

$python3 traffic.py data_directory [model_name_to_save_model.h5]

## Acknowledgements:

Data provided by [J. Stallkamp, M. Schlipsing, J. Salmen, and C. Igel. The German Traffic Sign Recognition Benchmark: A multi-class classification competition. In Proceedings of the IEEE International Joint Conference on Neural Networks, pages 1453–1460. 2011](http://benchmark.ini.rub.de/index.php?section=gtsrb&subsection=dataset#Acknowledgements)
This project is licensed under the MIT License.
