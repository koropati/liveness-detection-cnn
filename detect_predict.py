import tensorflow as tf
import numpy as np
from tensorflow import keras
import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import argparse
from imutils import face_utils
import dlib

# construct argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True,
                help="path to trained model")
ap.add_argument("-i", "--input", type=str, required=True,
                help="path to input image")
ap.add_argument("-o", "--output", type=str, required=True,
                help="path to output image")
args = vars(ap.parse_args())


model = keras.models.load_model(args["model"])
detector = dlib.get_frontal_face_detector()
widthImageOut = 250

# python detect_predict.py -m my_model.h5 -i test1.jpg -o result1.jpg
# python detect_predict.py -m my_model.h5 -i test2.jpg -o result2.jpg


def predictImage(input_image):
    Y = image.img_to_array(input_image)
    X = np.expand_dims(Y, axis=0)
    val = model.predict(X)
    print("VALUE: {}".format(val))
    return val

def findFace(path, outPath):
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    for rect in rects:
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        print("X Y W H: {} {} {} {}".format(x, y, w, h))
        cropImage = image[y:y+h, x:x+w]
        resizedImage = cv2.resize(cropImage, [widthImageOut, widthImageOut])
        im_rgb = cv2.cvtColor(resizedImage, cv2.COLOR_BGR2RGB)
        result = predictImage(im_rgb)
        
        if result >= 1:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, "ASLI (REAL)", (x, y),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(image, "PALSU (FAKE)", (x, y),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imwrite(outPath, image)


def main():
    findFace(args["input"],args["output"])

if __name__ == '__main__':
    main()
