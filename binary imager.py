import cv2
import os
root = 'Training_data'
imgs = list(sorted(os.listdir(os.path.join(root, "Masks"))))

for path in imgs:
    mask = cv2.imread('Training_data/Masks/' + path)
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    thresh = 127
    im_bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)[1]
    cv2.imwrite('Training_data/Masks/' + path, im_bw)