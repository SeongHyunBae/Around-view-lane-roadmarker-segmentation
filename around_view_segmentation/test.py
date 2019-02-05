import numpy as np
import cv2
import copy
import glob
from keras.models import load_model
from model import *

test_path = glob.glob('./dataset/test/*.jpg')

model = fcn()
model.load_weights('./checkpoint/checkpoint-03-0.9650.hdf5', by_name=True)

for path in test_path:
    frame = cv2.imread(path)


    frame = cv2.resize(frame, (1280, 720))

    if frame is not None:
        frame = cv2.resize(frame, (224, 360), interpolation=cv2.INTER_AREA)
        original = copy.copy(frame)
        h, w, c = frame.shape
        frame = np.reshape(frame, (1, h, w, c))
        frame = np.array(frame, dtype=np.float64)
        frame /= 255
        prediction =  model.predict(frame)[0] * 255

        cv2.imshow('original', original)
        cv2.imshow('prediction', prediction)
        cv2.waitKey(0)

    else:
        break





