import cv2
import random
import numpy as np
from plate import get_clean_plate
from matplotlib import pyplot as plt

def blur(img):
    """
    Adding normal or median blur to image
    """
    if random.uniform(0, 1) < 0.7:
        bk_size = random.randint(1,3) * 2 + 1
        img = cv2.blur(img, (bk_size, bk_size))
    else:
        img = clip(img)
        img = img.astype(np.uint8)
        img = cv2.medianBlur(img, 3)
    return img

def clip(img):
    img[img <= 0] = 0
    img[img >= 255] = 255
    return img

def invert(img):
    img = img.astype(np.float32)
    img = 255 - img
    img = np.minimum(img, 250)
    img = img.astype(np.uint8)
    return img

def add_back_brightness(img):
    """
    Create some background brightness so that background does not appear entirely black
    """
    brightness = random.uniform(0.2, 0.5)*255
    img = np.maximum(brightness, img)
    return img

clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
def clahe(img):
    """
    Use clahe as a means to create blur and high-frequency noise
    """
    clahe_img = clahe_obj.apply(img.mean(2).astype(np.uint8))
    img[:,:] = np.expand_dims(clahe_img, 2)
    return img


def block_shift(img):
    """
    Make everything look fat by shifting and pasting the image
    """
    H,W = img.shape[:2]
    img[1:H, 1:W] = np.maximum(img[1:H, 1:W], img[0:H-1, 0:W-1])
    img[1:H, 0:W] = np.maximum(img[1:H, 0:W], img[0:H-1, 0:W])
    return img

def dilate(img):
    img = cv2.dilate(img, np.ones((3,3), np.uint8), iterations=1)
    return img

def get_spot_noise(H,W):
    a = (np.random.rand(H,W) > 0.9).astype(np.float32)
    b = cv2.dilate(a, np.ones((3,3), np.float32), iterations=1)
    return 1 - b

def add_spot_noise(img):
    """
    add spot-like noise 
    """
    H, W = img.shape[:2]
    spot_noise = np.expand_dims(get_spot_noise(H,W), 2)
    return clip(img.astype(np.float32) + spot_noise * random.uniform(-30, 30))

# img range [0, 255]
# img is 3 channel
# img output range [0, 255]
def distort_plate(img):
    """
    2 types of transforms:
    - applicable in 0-255: block_shift, dilate, clahe
    - applicable after
    - shape-morphing: block_shift, dilate, clahe
    - brightness morphing: multiply_by_uniform

    non-inversion:
    - block_shift, dilate, clahe, blur, add_spot_noise

    inversion:
    - add_back_brightness
    - coahe, blur, add_spot_noise, uniform_mult

    """
    if random.randint(0,1): #no inversion
        # fattening
        if random.uniform(0,1) < 0.3:
            img = block_shift(img)
        if random.randint(0,1):
            img = dilate(img)
        if random.randint(0,1):
            img = clahe(img)
        img = img.astype(np.float32)

        # non-fattening
        if random.uniform(0, 1) < 0.9:
            img = blur(img)
        if True:
            img = add_back_brightness(img)
        if random.random() < 0.7:
            img = add_spot_noise(img)
        if True:
            img = img * random.uniform(0.3, 1.0)
        if random.random() < 0.1:
            img = add_spot_noise(img)
    else: # inversion
        if random.uniform(0, 1) < 0.9:
            img = blur(img)
        if True:
            img = add_back_brightness(img)
        if random.random() < 0.7:
            img = add_spot_noise(img)
        img = invert(img)
        if True:
            img = img * random.uniform(0.3, 1.0)
        if random.random() < 0.1:
            img = add_spot_noise(img)

    img = clip(img)
    img = np.maximum(img, 1)
    img = img.astype(np.uint8)
    return img

