import tensorflow as tf
import numpy as np
import cv2
import os

# load model, get path to images
model_dir = os.path.join("models", "pneumonia.h5")
model = tf.keras.models.load_model(model_dir)
SUB_DIR = "val/1"
DIR = os.path.join("chest_xray", SUB_DIR)
CONVERT = {1: "PNEUMONIA", 0: "CLEAR"}
TEST = False


def main():
    print("checking...")

    # if testing directory calculate score
    if os.path.isdir(DIR) and TEST:
        Classification, Score = get_score(DIR)
        for item in Classification:
            print(CONVERT[item])
        print("SCORE: ", Score*100, "%", sep="")

    # check image(s) if not for testing
    else:
        for im in os.listdir(DIR):
            # skips .DS_Store files
            if im == ".DS_Store":
                continue

            # classifies image
            Classification = classify(os.path.join(DIR, im))
            print(CONVERT[Classification])


def classify(im):
    """ classifies image as pneumonia or not """

    # open and resize image
    image = cv2.imread(im)
    image = cv2.resize(image, (30, 30))
    image = tf.cast(image, tf.float32)

    # test image through neural network
    classification = model.predict(
        [np.array(image).reshape(1, 30, 30, 3)]
    ).argmax()
    return classification


def get_score(directory):
    """ gets the accuracy of the model if labelled testing data is used """

    correct = 0
    incorrect = 0
    classifications = list()

    # loops through folders in given directory
    for folder in os.listdir(directory):
        # skips .DS_Store files
        if folder == ".DS_Store":
            continue
        for im in os.listdir(os.path.join(directory, folder)):
            if im == ".DS_Store":
                continue

            # classifies image
            classification = classify(os.path.join(directory, folder, im))
            classifications.append(classification)

            # count number of correct and incorrect classifications
            if classification == int(folder):
                correct += 1
            else:
                incorrect += 1

    # calculate percentage score
    score = correct / (correct + incorrect)
    return classifications, score


if __name__ == "__main__":
    main()
