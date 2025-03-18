# CAT_DOG
This project is focused on building an image classification model using TensorFlow 2.0 and Keras to distinguish between images of cats and dogs. The goal is to create a convolutional neural network (CNN) that can classify these images with an accuracy of at least 63%, with extra credit for achieving 70% accuracy.
Key Steps:

    Data Preprocessing:
        You will be given a dataset containing images of cats and dogs. The dataset is organized into three directories: train, validation, and test.
        Your task is to create image generators for each dataset using ImageDataGenerator and convert the images into floating-point tensors with rescaled pixel values between 0 and 1.

    Model Creation:
        You will build a CNN using Keras Sequential model, which includes layers like Conv2D and MaxPooling2D to process the image data.
        After the convolutional layers, you will add a fully connected layer activated by the ReLU function.
        You will compile the model using an optimizer and loss function, and track accuracy during training with metrics=['accuracy'].

    Data Augmentation:
        Since the training dataset is small, there is a risk of overfitting. To address this, you will use data augmentation by applying random transformations to the training images, creating more diverse data for training.

    Model Training:
        The model will be trained using the fit method, where you'll pass in the training and validation data, set the number of epochs, and track the training and validation accuracy.

    Model Evaluation:
        After training, you'll use the trained model to make predictions on test images. These predictions will determine if an image contains a cat or a dog.
        You will visualize the results by plotting the test images alongside the model's predicted probabilities.

    Challenge Evaluation:
        To pass the challenge, your model must correctly classify test images with an accuracy of at least 63%. If you achieve 70% accuracy, you will receive extra credit.

Course Context:

This project is part of a machine learning curriculum aimed at teaching students how to build and deploy a basic deep learning model for image classification. It provides hands-on experience with TensorFlow, Keras, and Convolutional Neural Networks (CNNs), which are fundamental in computer vision tasks.
