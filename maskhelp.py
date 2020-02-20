import cv2
import numpy as np
import os
root = 'Training_data'
imgs = list(sorted(os.listdir(os.path.join(root, "Images"))))
masks = list(sorted(os.listdir(os.path.join(root, "Masks"))))
newMasks = list(sorted(os.listdir(os.path.join(root, "newMasks"))))
# for path in imgs:
#     image = cv2.imread(root + "/Masks/" + path)
#     for y in range(len(image)):
#         for x in range(len(image[y])):
#             pixel = image[y][x]
#             if list(pixel) == [255, 255, 255]:
#                 image[y][x] = [1, 1, 1]
#     cv2.imwrite('newMasks/{}'.format(path), image)

#
# image0 = cv2.imread(root + "/newMasks/" + imgs[0])
# image1 = cv2.imread("PedMasks/" + imgs1[0])
# im = [image1]
# for image in im:
#     for y in range(len(image)):
#         for x in range(len(image[y])):
#             pixel = image[y][x]
#             if list(pixel) != [0,0,0]:
#                 print(pixel, x, y)
#     print("AAAAAA")

for name in imgs:
    if name not in masks:
        print(name)
        os.remove(root + '/' + "Images/" + name)