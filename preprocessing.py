from imutils import face_utils
import dlib
import cv2
import glob
import os
from os.path import exists

detector = dlib.get_frontal_face_detector()
widthImageOut = 250
outPath = "data\\clean"
inputPath = "data\\original"


def renameImage(path):
    for path, subdirs, files in os.walk(path):
        counter = 1
        for name in files:
            extension = name.split(".")[-1].lower()
            if extension != "jpg":
                continue
            if not exists(os.path.join(path, os.path.basename(path) + "_" + str(counter) + "." + extension)):
                os.rename(os.path.join(path, name), os.path.join(
                    path, os.path.basename(path) + "_" + str(counter) + "." + extension))
            counter += 1


def findFace(path, outPath):
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    for rect in rects:
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        print("X Y W H: {} {} {} {}".format(x, y, w, h))
        cropImage = image[y:y+h, x:x+w]
        resizedImage = cv2.resize(cropImage, [widthImageOut, widthImageOut])
        outPathImage = outPath + "\\" + "\\".join(path.split("\\")[2:])
        print(outPathImage)
        cv2.imwrite(outPathImage, resizedImage)


def main():
    renameImage(inputPath)
    for imagePath in glob.iglob(f'{inputPath}/*/*'):
        if (imagePath.endswith(".jpg")):
            findFace(imagePath,outPath)


if __name__ == '__main__':
    main()
