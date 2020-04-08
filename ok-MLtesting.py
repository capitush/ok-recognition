import numpy as np
import cv2
import time
import pyautogui
import torch
from PIL import Image
import torchvision.transforms.transforms as Trans
from matplotlib import cm

model = torch.load("model.pt")
model.eval()


def get_prediction(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = Image.fromarray(image)
    transform = Trans.Compose([Trans.ToTensor()])  # Defing PyTorch Transform
    img = transform(img).cuda()  # Apply the transform to the image
    pred = model([img])  # Pass the image to the model
    masks = pred[0]['masks']
    masks = masks.cpu().detach().numpy()
    for i in range(len(masks)):
        mask = masks[i]
        mask = mask[0]
        mask = np.uint8(cm.gist_earth(mask) * 255)
        return mask


cap = cv2.VideoCapture(0)

iterations = 4

rect_length = 300

multiplier = 2

min_YCrCb = np.array([0, 133, 77], np.uint8)
max_YCrCb = np.array([235, 173, 127], np.uint8)

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


def get_area(cnt):
    return cv2.contourArea(cnt)


def get_center_contour(cnt):
    x, y, w, h = cv2.boundingRect(cnt)
    return int(x + w / 2), int(y + h / 2)


def get_brightness(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, _, v = cv2.split(hsv)
    return int(np.average(v.flatten()))


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

    prediction = get_prediction(frame)

    # Display the resulting frame
    cv2.imshow('FaceDetection', cv2.flip(prediction, 1))
    cv2.imshow('Normal', cv2.flip(frame, 1))
    key = cv2.waitKey(1) & 0xFF
    if key == ord('w'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
