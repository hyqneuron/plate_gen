import cv2
import numpy as np
import random
from plate import get_clean_plate, get_background_img
from noise import distort_plate
from matplotlib import pyplot as plt

def clip(img):
    img[img <= 0] = 0
    img[img >= 255] = 255
    return img

def shift_coords(offsets, coords):
    for char in coords:
        char_coords = char['corners']
        for point in char_coords:
            point[0] += offsets[0]
            point[1] += offsets[1]
    return coords

# bg must be bigger than img
def add_background(img, coords):
    bg_shape = list(img.shape)
    bg_shape[0] += random.randint(10, 50)
    bg_shape[1] += random.randint(10, 50)
    bg = np.random.rand(*bg_shape) * 255.0
    h_bounds = bg.shape[0] - img.shape[0]
    w_bounds = bg.shape[1] - img.shape[1]
    ul_h = random.randint(0, h_bounds)
    ul_w = random.randint(0, w_bounds)
    br_h = ul_h + img.shape[0]
    br_w = ul_w + img.shape[1]
    bg[ul_h:br_h, ul_w:br_w, :] = img
    offsets = [ul_w, ul_h]
    coords = shift_coords(offsets, coords)
    bg = clip(bg)
    bg = bg.astype(np.uint8)
    return bg, coords

def blend(bg, img, x1,y1):
    H,W = img.shape[:2]
    mask = img != 0
    inv_mask = 1-mask
    region = bg[y1:y1+H, x1:x1+W]
    region[:] = mask * img + inv_mask * region
    return bg


def add_background(img, coords):
    """
    Use a background image of fixed aspect-ratio (3:1)
    - In the case of 1-line plate, add vertical padding
    - In the case of 2-line plate, add horizontal padding
    """
    H,W = img.shape[:2]
    if W/float(H) > 2.35: # likely 1-line
        W_new = int(W*1.32)
        H_new = int(W_new/3)
    else: # likely 2-line
        H_new = int(H*1.2)
        W_new = int(H_new*3)
    assert H_new > H and W_new > W
    #if random.randint(0,1):
        #bg = np.random.rand(H_new, W_new, 3)*255.0
    #else:
    if H_new <= 150 and W_new < 300:
        bg = get_background_img(W_new, H_new)
    else:
        bg = np.zeros((H_new, W_new, 3), np.uint8) # get_background_img can only handle limited size
    x1 = random.randint(0, W_new - W)
    y1 = random.randint(0, H_new - H)
    # bg[y1:y1+H, x1:x1+W] = img
    bg = blend(bg, img, x1, y1)
    coords = shift_coords([x1,y1], coords)
    bg = clip(bg)
    bg = bg.astype(np.uint8)
    return bg, coords


# bg = np.random.rand(100, 100, 3) * 255
# img = np.ones((50, 50, 3)) * 255
# coords = [0, 0, 0, 0]
# out, coords = add_background(img, bg, coords)
# out = out.astype(np.uint8)
# print coords
# plt.imshow(out)
# plt.show()



