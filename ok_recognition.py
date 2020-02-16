import numpy as np
import cv2
import time
import pyautogui
from matplotlib import pyplot as plt
# from skimage.filters import threshold_yen
# from skimage.exposure import rescale_intensity
# from skimage.io import imread, imsave

cap = cv2.VideoCapture(0)

iterations = 4

rect_length = 300

multiplier = 2

min_YCrCb = np.array([0,133,77],np.uint8)
max_YCrCb = np.array([235,173,127],np.uint8)

min_hsv = np.array([108, 23, 82], np.uint8)
max_hsv = np.array([179, 255, 255], np.uint8)
prev_x, prev_y = None, None
dx, dy = None, None
ok_time = time.time()
pressed = True
just_appeared = True

pyautogui.PAUSE = 0


def map(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)


def gamma_correct(gamma, frame):
    lookUpTable = np.empty((1, 256), np.uint8)
    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    return cv2.LUT(frame, lookUpTable)


def get_area(cnt):
    return cv2.contourArea(cnt)


def get_center_contour(cnt):
    x, y, w, h = cv2.boundingRect(cnt)
    return int(x+w/2), int(y+h/2)


def get_brightness(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, _, v = cv2.split(hsv)
    return int(np.average(v.flatten()))
# import pygame
# pygame.init()
# size = width, height = 200, 200
# surface = pygame.display.set_mode(size)

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
range = 20


def get_average_color(img):
    y_cr_cb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    y, cr, cb = cv2.split(y_cr_cb)
    averages = []
    for color in (y,cr,cb):
        averages.append(int(np.median(color.flatten())))
    return averages


def weird_cascade(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=5,
        minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # print("Found {0} faces!".format(len(faces)))
    # Draw a rectangle around the faces
    for xf, yf, wf, hf in faces:
        xf = faces[0][0]
        yf = faces[0][1]
        wf = faces[0][2]
        hf = faces[0][3]
        cv2.rectangle(frame, (xf, yf), (xf + wf, yf + hf), (255, 0, 0), 1)
    # Display the resulting frame
        averages = get_average_color(frame[yf:yf + hf, xf:xf + wf])
    try:
        min_YCrCb = np.array([averages[0] - range, averages[1] - range, averages[2] - range], np.uint8)
        max_YCrCb = np.array([averages[0] + range, averages[1] + range, averages[2] + range], np.uint8)
    except:
        min_YCrCb = np.array([0, 133, 77], np.uint8)
        max_YCrCb = np.array([235, 173, 127], np.uint8)
    print(min_YCrCb, max_YCrCb)
    return min_YCrCb, max_YCrCb

active = True

if not active:
    print("not active")
while True:
    # Capture frame-by-frame

    # for event in pygame.event.get():
    #     if event.type == pygame.QUIT:
    #         done = True
    #     mouse = list(pygame.mouse.get_pos())
    #     alpha = map(mouse[0], 0, 200, 0, 3)
    #     beta = map(mouse[1], 0, 200, 0, 100)
    #     print(alpha, beta)
    # pygame.display.flip()

    ret, frame = cap.read()

    # yen_threshold = threshold_yen(frame)
    # frame = rescale_intensity(frame, (0, yen_threshold), (40, 230))

    # brightness = map(get_brightness(frame), 70, 134, 0.7, 1)
    # frame = gamma_correct(brightness, frame)

    # gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image_YCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(image_YCrCb)
    cv2.imshow('before', channels[0])
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

    cv2.rectangle(frame, (0,0), (300,500-20), [0,0,255], 5)
    good_contours = []
    for cnt in contours:
        x, y = get_center_contour(cnt)
        if x < rect_length:
            cv2.circle(frame, (x, y), 5, [255, 0, 0], -1)
            # cv2.putText(frame, str(get_area(cnt)), (x,y), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            good_contours.append(cnt)
    good_contours.sort(key=get_area, reverse=True)
    try:
        hand_contour = good_contours[0]
        cv2.drawContours(frame, hand_contour, -1, (123, 255, 123), 3)
    except:
        hand_contour = None
    very_good_contours = []
    for cnt in good_contours:
        x, y = get_center_contour(cnt)
        inside = False
        try:
            index_1 = hierarchy[0][contours.index(cnt)][3]
            index_2 = contours.index(hand_contour)
            if index_1 == index_2:
                inside = True
        except ValueError:
            inside = True
        # if cv2.contourArea(cnt) / (3.1415 * radius * radius) > 0.65 and get_area(cnt)/get_area(hand_contour) > 1/16:
        # and get_area(cnt)/get_area(hand_contour) > 1/16
        if inside and get_area(cnt) / get_area(hand_contour) > 1/16 * get_area(hand_contour)/(rect_length*200):
            cv2.circle(frame, (x, y), 5, [0, 0, 255], -1)
            very_good_contours.append(cnt)
        elif inside:
            cv2.circle(frame, (x, y), 5, [0, 255, 0], -1)
    try:
        ok_contour = very_good_contours[0]
    except:
        ok_contour = None

    if ok_contour is not None:
        x, y = get_center_contour(ok_contour)
        topmost = tuple(hand_contour[hand_contour[:, :, 1].argmin()][0])
        if just_appeared:
            dx, dy = 0, 0
            ty = 0
            just_appeared = False
        else:
            dx = x - prev_x
            dy = y - prev_y
            ty = topmost[1] - prev_ty
            # if ty > 28:
            #     # Implemented index finger right click activation
            #     pyautogui.click(button='right')
            #     print(ty)
            if abs(ty) > 25 and active:
                pyautogui.scroll(-int(ty*12))
                cv2.circle(frame, topmost, 5, [0, 50, 0], -1)
        prev_x, prev_y = x, y
        prev_ty = topmost[1]
    else:
        just_appeared = True

    if ok_contour is not None and dx is not None:
        if True:
            # Move mouse
            if active and not time.time() - ok_time < 0.2 and dx < 200 and dy < 200:
                pyautogui.move(-dx*multiplier, dy*multiplier)
            if pressed is False:
                ok_time = time.time()
                pressed = True
    elif pressed is True:
        pressed = False
        destiny = time.time() - ok_time
        if 0.16 < destiny < 0.4 and active:
            print(destiny)
            # Click
            pyautogui.click()

    # Display the resulting frame
    cv2.imshow('FaceDetection', cv2.flip(frame, 1))
    cv2.imshow('cool', cv2.flip(skinYCrCb, 1))
    # cv2.imshow('a', cv2.flip(imageYCrCb, 1))
    # cv2.imshow('AAAA', cv2.flip(hsvImg, 1))
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        min_YCrCb, max_YCrCb = weird_cascade(frame)
    elif key == ord('w'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
