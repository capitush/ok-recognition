import numpy as np
import cv2
import time
import pyautogui

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

    ret, frame = cap.read()  # Get the frame from the camera, stored at the variable frame

    image_YCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)  # Convert the frame from RGB to YCrCb

    #  Apply histogram equalization to the frame on the LUNA channel
    channels = cv2.split(image_YCrCb)
    cv2.imshow('grayscale', channels[0])  # Display the grayscale LUNA channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe.apply(channels[0], channels[0])

    cv2.merge(channels, image_YCrCb)  # Apply the equalized LUNA channel to the rest of the frame

    # Set the lower and upper bounds for human skin tone in YCrCb
    min_YCrCb = np.array([0, 133, 77], np.uint8)
    max_YCrCb = np.array([235, 173, 127], np.uint8)
    # Create a binary image containing only the pixels that fit between the human skin tone bounds
    skinRegionYCrCb = cv2.inRange(image_YCrCb, min_YCrCb, max_YCrCb)

    # Apply morphological transformations (closing then opening) in order to remove noise
    skinRegionYCrCb = cv2.erode(skinRegionYCrCb, np.ones((3,3),np.uint8), iterations=iterations)
    skinRegionYCrCb = cv2.dilate(skinRegionYCrCb, np.ones((3,3),np.uint8), iterations=iterations)
    skinRegionYCrCb = cv2.dilate(skinRegionYCrCb, np.ones((3,3),np.uint8), iterations=iterations)
    skinRegionYCrCb = cv2.erode(skinRegionYCrCb, np.ones((3,3),np.uint8), iterations=iterations)
    # Also apply Gaussian blur
    skinRegionYCrCb = cv2.GaussianBlur(skinRegionYCrCb, (5,5), 100)

    # Combine the binary skin tone image with the RGB image to create an RGB image containing only skin tone pixels
    skinYCrCb = cv2.bitwise_and(frame, frame, mask = skinRegionYCrCb)

    # Find the contours in the image
    contours, hierarchy = cv2.findContours(skinRegionYCrCb, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the bounding box in which the contours are detected
    cv2.rectangle(frame, (0,0), (rect_length,500-20), [0,0,255], 5)
    good_contours = []  # An array in which only the contours in the bounding box are included
    for cnt in contours:  # Put into good_contours only the contours that are in the bounding box
        x, y = get_center_contour(cnt)
        if x < rect_length:
            cv2.circle(frame, (x, y), 5, [255, 0, 0], -1)
            # cv2.putText(frame, str(get_area(cnt)), (x,y), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            good_contours.append(cnt)
    # Sort the contours by area, so that the contour in index 0 is the largest contour
    good_contours.sort(key=get_area, reverse=True)
    try:
        hand_contour = good_contours[0]  # The largest contour is assumed to be the contour of the hand being detected
        cv2.drawContours(frame, hand_contour, -1, (123, 255, 123), 3)  # Draw the largest contour
    except:
        hand_contour = None  # If no contour appears in the bounding box
    very_good_contours = []  # An array including all contours that are inside the largest hand contour
    for cnt in good_contours:
        x, y = get_center_contour(cnt)
        inside = False
        try:
            # Find the contour whose father is the hand contour
            index_1 = hierarchy[0][contours.index(cnt)][3]
            index_2 = contours.index(hand_contour)
            if index_1 == index_2:
                inside = True
        except ValueError:
            inside = True

        # Only include that contour inside of the list if it's area is under certain criteria
        if inside and get_area(cnt) / get_area(hand_contour) > 1/16 * get_area(hand_contour)/(rect_length*200):
            cv2.circle(frame, (x, y), 5, [0, 0, 255], -1)  # Draw a red dot in the middle of that contour
            very_good_contours.append(cnt)
        elif inside:
            cv2.circle(frame, (x, y), 5, [0, 255, 0], -1)  # Draw a green dot if it doesnt fit the area criteria
    try:
        ok_contour = very_good_contours[0]  # The largest contour that fits the above criteria is the hole contour
    except:
        ok_contour = None

    if ok_contour is not None:
        x, y = get_center_contour(ok_contour)
        # Get the topmost location of the hand contour
        topmost = tuple(hand_contour[hand_contour[:, :, 1].argmin()][0])
        # If the program has just been initialized, initialize ty and dx, dy
        if just_appeared:
            dx, dy = 0, 0
            ty = 0
            just_appeared = False
        # Else, calculate dx and dy (dx and dy are the change in position of the ok contour center,
        # ty is the position of the topmost coordinate of the hand contour)
        else:
            dx = x - prev_x
            dy = y - prev_y
            ty = topmost[1] - prev_ty
            if abs(ty) > 25 and active:  # If ty has changed a lot, scroll the mouse wheel
                pyautogui.scroll(-int(ty*12))
                cv2.circle(frame, topmost, 5, [0, 50, 0], -1)
        prev_x, prev_y = x, y
        prev_ty = topmost[1]
    else:
        just_appeared = True

    if ok_contour is not None and dx is not None:
        if True:
            # Move the mouse according to dx and dy, unless the ok contour reappeared recently
            if active and not time.time() - ok_time < 0.2 and dx < 200 and dy < 200:
                pyautogui.move(-dx*multiplier, dy*multiplier)
            # If it didn't, remember the time so we can know if it will reappear quickly in the future
            if pressed is False:
                ok_time = time.time()
                pressed = True
    elif pressed is True:
        pressed = False
        # If the contour did reappear recently, click the mouse.
        destiny = time.time() - ok_time
        if 0.16 < destiny < 0.4 and active:
            print(destiny)
            # Click
            pyautogui.click()

    cv2.imshow('FaceDetection', cv2.flip(frame, 1))  # Display the frame containing the hand contours drawn
    cv2.imshow('No skin', cv2.flip(skinYCrCb, 1))  # Display the frame only containing human skin
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        # Enable the experimental night adaptation mode on the press of the "Q" button,
        # which changes the skin tone colors if the program is having a hard time detecting hands under difficult
        # light conditions.
        min_YCrCb, max_YCrCb = weird_cascade(frame)
    elif key == ord('w'):
        # Close the program on the press of the "W" button.
        break

# When the program is closed, release the capture and close all windows.
cap.release()
cv2.destroyAllWindows()
