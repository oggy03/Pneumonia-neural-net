import cv2
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split

""" 
    normal = 0
    pneumonia = 1
    pneumonia data from https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
"""


EPOCHS = 10
IMG_WIDTH = 50
IMG_HEIGHT = 50
NUM_CATEGORIES = 2
TEST_SIZE = 0.4
TRAIN_PATH = os.path.join("chest_xray", "train")


def main():
    # Get image arrays and labels for all image files
    images, labels = load_data(TRAIN_PATH)

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test, y_test, verbose=2)

    # save model
    model.save("pneumonia.h5")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    # initialize list of images and labels
    images = list()
    labels = list()

    # loop through all categories
    for category in os.listdir(data_dir):
        if category == ".DS_Store":
            continue

        # get path to directory of category
        directory = os.path.join(data_dir, str(category))

        # loop through images in directory
        for file in os.listdir(directory):
            if file == ".DS_Store" or not file.endswith(".jpeg"):
                continue
            # read image as numpy array
            path_to_image = os.path.join(directory, file)
            image = cv2.imread(path_to_image)
            if image is not None:
                resized_image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
                # save image and category
                images.append(resized_image)
                labels.append(category)

    to_return = (images, labels)
    return to_return


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    # create sequential model
    model = tf.keras.models.Sequential([

        # Convolutional layer learning filters
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),

        # use max-pooling to reduce image size
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Convolutional layer learning filters
        tf.keras.layers.Conv2D(
            65, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),

        # use max-pooling to reduce image size
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # flatten images
        tf.keras.layers.Flatten(),

        # hidden layers
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(75, activation="relu"),
        # dropout to prevent overfitting
        tf.keras.layers.Dropout(0.5),

        # output layer
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")

    ])

    # adam is a type of stochastic gradient descent
    # accuracy is the way the model is judged - like a loss function but not used in training
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    main()
