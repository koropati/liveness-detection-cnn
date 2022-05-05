import tensorflow as tf
import numpy as np
from tensorflow import keras
import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

model = keras.models.load_model('my_model.h5')


def predictImage(filename):
    img1 = image.load_img(filename, target_size=(250, 250))

    plt.imshow(img1)

    Y = image.img_to_array(img1)

    X = np.expand_dims(Y, axis=0)
    val = model.predict(X)
    print(val)
    if val == 1:
        plt.xlabel("REAL", fontsize=30)
    elif val == 0:
        plt.xlabel("FAKE", fontsize=30)


def main():
    # predictImage("dataset\\val\\fake\\fake_8.jpg")
    # predictImage("dataset\\val\\fake\\fake_39.jpg")
    # predictImage("dataset\\val\\fake\\fake_13.jpg")
    # predictImage("dataset\\val\\fake\\fake_15.jpg")
    
    predictImage("dataset\\val\\real\\real_79.jpg")
    predictImage("dataset\\val\\real\\real_151.jpg")
    predictImage("dataset\\val\\real\\real_120.jpg")
    predictImage("dataset\\val\\real\\real_90.jpg")


if __name__ == '__main__':
    main()
