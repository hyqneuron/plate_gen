
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from plate import get_clean_plate

def simple_expand(img):
    H,W = img.shape[0], img.shape[1]
    H_new = int(H * 1.3)
    W_new = int(W * 1.3)
    img_new = np.zeros((H_new, W_new, img.shape[2]), img.dtype)
    x1 = int((W_new - W) / 2)
    y1 = int((H_new - H) / 2)
    x2 = x1 + W
    y2 = y1 + H
    img_new[y1:y2, x1:x2] = img
    return img_new, (x1,y1,x2,y2)

def get_random_transform(img, (x1,y1,x2,y2)):
    H = y2 - y1
    W = x2 - x1
    # a = random.uniform(0.7, 1.0)   # horozontal shift of pt2
    # b = random.uniform(-0.1, 0.1)  # vertical shift of pt2
    # c = 1 # random.uniform(0.8, 1.2)
    src = np.asarray([(x1, y1),(x2, y1),(x2, y2),(x1, y2)]).astype(np.float32)
    dst = np.asarray([(x1, y1),(x2, y1),(x2, y2),(x1, y2)]).astype(np.float32)
    #dst = np.asarray([(0.0, 0.0),(a, 0.0+b),(a, (1.0+b)*c),(0.0, 1.0)]).astype(np.float32)
    # vertically shift the right vertical
    ax = [1,2] if random.randint(0,1) else [0,3]
    a = random.uniform(-0.3*H, 0.3*H)
    dst[ax[0]][1] += a
    dst[ax[1]][1] += a
    # vertically shift the bottom-right
    b = random.uniform(-0.3*H, 0.00)
    dst[ax[1]][1] += b
    # horizontally shift the bottom horizontal
    c = random.uniform(-0.1*H, 0.1*H)
    dst[2][0] += c
    dst[3][0] += c
    transform = cv2.getPerspectiveTransform(src, dst)
    return transform

def get_xy_minmax(chars):
    x1 = 999999
    y1 = 999999
    x2 = 0
    y2 = 0
    for char in chars:
        for corner in char['corners']:
            x, y = corner
            x1 = min(x, x1)
            y1 = min(y, y1)
            x2 = max(x, x2)
            y2 = max(y, y2)
    return x1,y1,x2,y2

def perspective_transform(img, chars):
    """
    1. compute a transform with random rotation, and stretch
    """
    # img_expanded, (x1,y1,x2,y2) = simple_expand(img)
    x1,y1,x2,y2 = get_xy_minmax(chars)
    trans = get_random_transform(img, (x1,y1,x2,y2))
    warped_img = cv2.warpPerspective(img, trans, (img.shape[1], img.shape[0]))
    for char in chars:
        for corner in char['corners']:
            # print(corner)
            a = [corner[0], corner[1], 1]
            transformed_corner = np.matmul(trans, np.asarray(a).reshape(3,1))
            corner[0] = transformed_corner[0] / transformed_corner[2]
            corner[1] = transformed_corner[1] / transformed_corner[2]
            # print(corner)
    return warped_img, chars

def test():
    global warped_img
    warped_img, _ = perspective_transform(img, chars)
    plt.imshow(warped_img)
    plt.show()

if __name__ == "__main__":
    img, chars = get_clean_plate()
    warped_img, _ = perspective_transform(img, chars)
    plt.imshow(warped_img)
    plt.show()
