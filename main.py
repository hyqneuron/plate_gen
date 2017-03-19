"""
1. get clean plate
2. blur/ add noise, add borders
3. slap background on
4. perspective transform
5. output

get_clean_plate returns:
    (img, [CharCoord])
    CharCoord:{
        corners: [pt1,pt2,pt3,pt4], where ptx = (x,y)
        class: char_id
    }

TODOs:
* Reduce distance between 2 lines
* Add plate border
* Use real backgrounds
- Use stronger noise on plate
  - random thin wires laying across
  - random spots
- Use multiple fonts

"""
from matplotlib import pyplot as plt
from plate import get_clean_plate
from noise import distort_plate
from background import add_background
from transform import perspective_transform
import numpy as np
import cv2

def showimg(img):
    plt.imshow(img)
    plt.show()

def get_sample():
    img, coords = get_clean_plate()
    img = distort_plate(img) # returns np.uint8
    img, coords = perspective_transform(img, coords)
    img, coords = add_background(img, coords) # returns np.uint8
    return img, coords

def test(showheat=False):
    global img, coords
    img, coords = get_sample()
    if showheat:
        showimg(heat_chars(img, coords))
    else:
        showimg(img)
    # new_img, new_chars = add_background(img, chars)
    # new_img, new_chars = perspective_transform(new_img, new_chars)

def heat_chars(img, coords):
    jet = np.zeros(img.shape, img.dtype)
    jet[:,:] = [0,0,255]
    for char in coords:
        pts = np.asarray(char['corners']).astype(np.int32)
        cv2.fillPoly(jet, [pts], (255,0,0))
    return jet/2 + img/2

def output(N):
    for i in range(N):
        img, coords = get_sample()
        cv2.imwrite('output/{}.jpg'.format(i), img)

if __name__ == '__main__':
    test()

