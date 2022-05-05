# python stream.py -m my_model.h5

from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image


ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True,
                help="path to trained model")
ap.add_argument("-v", "--video", type=str, default="",
	help="path to input video file")
args = vars(ap.parse_args())


print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
model = keras.models.load_model(args["model"])
widthImageOut = 250

print("[INFO] starting video stream thread...")
vs = FileVideoStream(args["video"]).start()
fileStream = True
vs = VideoStream(src=0).start()

fileStream = False
time.sleep(1.0)


def predictImage(input_image):
    Y = image.img_to_array(input_image)
    X = np.expand_dims(Y, axis=0)
    val = model.predict(X)
    # print("VALUE: {}".format(val))
    return val


while True:
    if fileStream and not vs.more():
        break

    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    for rect in rects:
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cropImage = frame[y:y+h, x:x+w]
        resizedImage = cv2.resize(cropImage, [widthImageOut, widthImageOut])
        im_rgb = cv2.cvtColor(resizedImage, cv2.COLOR_BGR2RGB)
        result = predictImage(im_rgb)

        if result >= 1:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "ASLI (REAL)", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, "PALSU (FAKE)", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # show the frame
    cv2.imshow("CNN DewokLivenessNET Detection", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
