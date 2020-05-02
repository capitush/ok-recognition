import random
import string

import cv2
import os
import numpy as np
import torch
from PIL import Image

rect_length = 300

min_YCrCb = np.array([0,133,77],np.uint8)
max_YCrCb = np.array([235,173,127],np.uint8)

iterations = 3


def numpy_to_pil_image(numpy_array):
    return Image.fromarray(numpy_array)

def get_area(cnt):
    return cv2.contourArea(cnt)

def get_center_contour(cnt):
    x, y, w, h = cv2.boundingRect(cnt)
    return int(x+w/2), int(y+h/2)

def get_contour():
    image_YCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(image_YCrCb)
    # cv2.equalizeHist(channels[0], channels[0])
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe.apply(channels[0], channels[0])

    # blur = cv2.GaussianBlur(channels[0], (5, 5), 0)
    # ret3, channels[0] = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cv2.merge(channels, image_YCrCb)

    # WEIRD MACHINE LEARNING TO GET COLOR OF FACE

    # print(weird_cascade(frame))
    skinRegionYCrCb = cv2.inRange(image_YCrCb, min_YCrCb, max_YCrCb)

    skinRegionYCrCb = cv2.erode(skinRegionYCrCb, np.ones((3,3),np.uint8), iterations=iterations)
    skinRegionYCrCb = cv2.dilate(skinRegionYCrCb, np.ones((3,3),np.uint8), iterations=iterations)
    skinRegionYCrCb = cv2.dilate(skinRegionYCrCb, np.ones((3,3),np.uint8), iterations=iterations)
    skinRegionYCrCb = cv2.erode(skinRegionYCrCb, np.ones((3,3),np.uint8), iterations=iterations)

    skinRegionYCrCb = cv2.GaussianBlur(skinRegionYCrCb, (5,5), 100)

    skinYCrCb = cv2.bitwise_and(frame, frame, mask = skinRegionYCrCb)

    # Our operations on the frame come here

    contours, hierarchy = cv2.findContours(skinRegionYCrCb, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # cv2.rectangle(frame, (0,0), (300,500-20), [0,0,255], 5)
    good_contours = []
    for cnt in contours:
        x, y = get_center_contour(cnt)
        if x < rect_length:
            # cv2.circle(frame, (x, y), 5, [255, 0, 0], -1)
            # cv2.putText(frame, str(get_area(cnt)), (x,y), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            good_contours.append(cnt)
    good_contours.sort(key=get_area, reverse=True)
    try:
        return good_contours[0]
    except:
        return None


def random_string(string_length=10):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(string_length))


cap = cv2.VideoCapture(0)


while True:

    ret, frame = cap.read()
    name = random_string(8)
    cv2.imwrite("Training_data/Images/i{}.png".format(name), frame)
    try:
        height, width, channels = frame.shape
        blank_image = np.zeros((height, width, channels), np.uint8)
        cv2.rectangle(blank_image, (0,0), (width,height), (255,0,0), -1)
        cv2.drawContours(blank_image, [get_contour()], -1, (123, 255, 123), -1)
        cv2.imwrite("Training_data/oldMasks/i{}.png".format(name), blank_image)
    except cv2.error:
        pass
    cv2.imshow('a', cv2.flip(blank_image, 1))
    # img = numpy_to_pil_image(frame)
    cv2.waitKey(1) & 0xFF