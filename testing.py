import os
import numpy as np
from PIL import Image
import cv2
root = 'Training_data'
imgs = list(sorted(os.listdir(os.path.join(root, "Images"))))
masks = list(sorted(os.listdir(os.path.join(root, "Masks"))))
img_path = os.path.join(root, "Images", imgs[1])
mask_path = os.path.join(root, "Masks", masks[1])
img = Image.open(img_path).convert("RGB")
# note that we haven't converted the mask to RGB,
# because each color corresponds to a different instance
# with 0 being background
mask = Image.open(mask_path)
# mask.show()
# convert the PIL Image into a numpy array
mask = np.array(mask)
# gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
# thresh = 127
# im_bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)[1]
# Image.fromarray(im_bw).show()
# instances are encoded as different colors
obj_ids = np.unique(mask)
# split the color-encoded mask into a set
# of binary masks
pos = np.where(mask)
xmin = np.min(pos[1])
xmax = np.max(pos[1])
ymin = np.min(pos[0])
ymax = np.max(pos[0])
print(([xmin, ymin, xmax, ymax]))
# for color in obj_ids:
#
# get bounding box coordinates for each mask
# for data in masks:
#     Image.fromarray(data, 'RGB').show()