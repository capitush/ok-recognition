import random
import string
import cv2
import numpy as np

rect_length = 300

min_YCrCb = np.array([0, 133, 77], np.uint8)
max_YCrCb = np.array([235, 173, 127], np.uint8)

iterations = 3


def get_area(cnt):
    return cv2.contourArea(cnt)


def get_center_contour(cnt):
    x, y, w, h = cv2.boundingRect(cnt)
    return int(x + w / 2), int(y + h / 2)


def get_contour(frame):
    image_YCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)  # Convert the input frame from RGB to YCrCb
    # Apply histogram equalization:
    channels = cv2.split(image_YCrCb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe.apply(channels[0], channels[0])
    cv2.merge(channels, image_YCrCb)

    # Filter non-human colors:
    skinRegionYCrCb = cv2.inRange(image_YCrCb, min_YCrCb, max_YCrCb)
    # Apply morphology:
    skinRegionYCrCb = cv2.erode(skinRegionYCrCb, np.ones((3, 3), np.uint8), iterations=iterations)
    skinRegionYCrCb = cv2.dilate(skinRegionYCrCb, np.ones((3, 3), np.uint8), iterations=iterations)
    skinRegionYCrCb = cv2.dilate(skinRegionYCrCb, np.ones((3, 3), np.uint8), iterations=iterations)
    skinRegionYCrCb = cv2.erode(skinRegionYCrCb, np.ones((3, 3), np.uint8), iterations=iterations)
    # Apply gaussian blur:
    skinRegionYCrCb = cv2.GaussianBlur(skinRegionYCrCb, (5, 5), 100)

    # Find the contours in the frame
    contours, hierarchy = cv2.findContours(skinRegionYCrCb, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour in the red rectangle:
    good_contours = []
    for cnt in contours:
        x, y = get_center_contour(cnt)
        if x < rect_length:
            good_contours.append(cnt)
    # Sort them by area:
    good_contours.sort(key=get_area, reverse=True)
    # Draw the red rectangle:
    cv2.rectangle(frame, (0, 0), (rect_length, 500 - 20), [0, 0, 255], 5)
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
    root = "Training_data"
    ret, pic = cap.read()
    name = random_string(8)  # Get a random name for the image
    cv2.imwrite(root + "/Images/i{}.png".format(name), pic)  # Save the original image in "Images"
    try:
        height, width, channels = pic.shape  # Get the correct dimensions of the image
        blank_image = np.zeros((height, width, channels), np.uint8)  # Make an empty black picture
        # Draw the contour of the hand on the empty picture. The contours color is the binary image color of 1:
        cv2.drawContours(blank_image, [get_contour(pic)], -1, (1, 1, 1), -1)
        cv2.imwrite(root + "/Masks/i{}.png".format(name), blank_image)  # Save the binary image in "masks"
    except cv2.error:
        pass
    cv2.imshow('camera', cv2.flip(pic, 1))
    cv2.imshow('output', cv2.flip(blank_image, 1))
    cv2.waitKey(1) & 0xFF
