import cv2
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw

dir_patches = '/home/noid/data/car_backgrounds/patches/'

def charid_to_char(charid):
    """
    0-9 the digits
    10 space
    11-36 letters
    """
    assert 0 <= charid <= 36, 'accept only 0~36, got '+str(charid)
    if charid <= 9:
        return chr(charid+48)
    elif charid == 10:
        return ' '
    else:
        return chr(charid-10+64)

_char_to_charid = {charid_to_char(charid):charid for charid in range(0, 37)}

def char_to_charid(char):
    return _char_to_charid[char]


"""
TODO
* increase frequency of 2, S
- change white plates
- find thin font
"""
def rand_lettid():
    """
    things that get messed up: [Q, W, V, M, N]
    """
    val = random.uniform(0,1)
    if val < 0.2:
        candidates = ['Q', 'W', 'V', 'M', 'N']
        return char_to_charid(candidates[random.randint(0,len(candidates)-1)])
    else:
        charid = random.randint(11,36) 
        while charid_to_char(charid) in 'IO':
            charid = random.randint(11,36) 
    return charid

def rand_digitid():
    if random.uniform(0,1) < 0.1:
        return 2
    else:
        return random.randint(0,9)


def show_img(np_img):
    # converts a np array image to PIL Image then show it. Better than matplotlib.pyplot.imshow
    rgb_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(np.uint8(rgb_img))
    img.show()


"""
Gutenberg: 40 removed as it has an ugly 1
"""
font_files = ['UKFont.ttf','Kenteken.ttf','Kenteken.ttf','LicensePlate.ttf','MANDATOR.ttf', 
              'NAM54.ttf', ]
font_off   = [4,  0,  0,  4,  0,  0 ]
font_sizes = [40, 28, 28, 30, 25, 28]
fonts = [ImageFont.truetype(os.path.dirname(os.path.realpath(__file__))+'/'+filename, fontsize)
        for filename, fontsize in zip(font_files, font_sizes)]
fontsize = 40 # reference size
# ukfont = ImageFont.truetype(os.path.dirname(os.path.realpath(__file__))+"/UKFont.ttf", fontsize)
font_charid_imgs = {}
for i in range(len(fonts)):
    font_charid_imgs[i] = dict({})

def get_char_img(charid, fontidx=4):
    """
    get np array image for a single charid, along with width and min max info
    """
    charid_imgs = font_charid_imgs[fontidx]
    if charid in charid_imgs: 
        return charid_imgs[charid]
    char = charid_to_char(charid)
    height = int(fontsize * 0.9)
    width  = int(fontsize * 0.6)
    img  = Image.new("RGB", (width, height), "black")
    draw = ImageDraw.Draw(img)
    draw.text((2,font_off[fontidx]), char, font=fonts[fontidx])
    # convert to np array image
    np_img = np.asarray(img)
    # take one channel to compute mask
    np_img0 = np_img[:,:,0].max(0)
    np_mask = np_img0 >= 255
    np_hehe = np_mask * np.arange(1,width+1)
    w_max = np_hehe.max() - 1 # inclusive
    np_hehe[np_hehe==0]=1000
    w_min = np_hehe.min() - 1 # inclusive
    if char==' ': # give width to space
        w_max = int(6*fontsize/60)
        w_min = 0
    np_selected = np_img[:, w_min:w_max+1, :]
    width = w_max - w_min + 1
    charid_imgs[charid] = (np_selected, width)
    assert width == np_selected.shape[1]
    return charid_imgs[charid]

def gen_top_charid_sequence():
    """ top sequence of a 2-line plate """
    # 2 or 3 letters
    sequence = []
    length = 3 if random.uniform(0,1) < 0.9 else 2
    for i in xrange(length):
        sequence.append(rand_lettid())
    return sequence

def gen_bottom_charid_sequence():
    """ bottom sequence of a 2-line plate """
    # 4 digits followed by one letter, sometimes 4 digits are replaced with 1-3 digits
    sequence = []
    if random.uniform(0,1) < 0.9:
        length = 4
    else:
        length = random.randint(1,3)
    for i in xrange(length):
        sequence.append(rand_digitid())
    sequence.append(10)
    sequence.append(rand_lettid())
    return sequence

def gen_flat_charid_sequence():
    """ sequence of a 1-line plate """
    return gen_top_charid_sequence() + [10] + gen_bottom_charid_sequence()

def gen_img_from_seq(charid_sequence, fontidx=-1):
    height = int(fontsize * 0.9)
    width  = int(fontsize * 0.6) * len(charid_sequence) + 10
    np_paste_background = np.zeros((height, width, 3), np.uint8)
    padding = 6
    w_begin = padding
    box_info = []
    for charid in charid_sequence:
        if charid==10:
            w_begin += random.randint(3, 6)
            continue # skip space
        if fontidx >= 0:
            pass
        elif random.uniform(0,1) < 0.3:
            fontidx = 0 # default to UKFont.ttf
        else:
            fontidx = random.randint(0,len(font_files)-1)
        np_char_img, width = get_char_img(charid, fontidx)
        region = np_paste_background[:, w_begin:w_begin+width, :]
        np_paste_background[:, w_begin:w_begin+width, :] = np_char_img
        # create bbox info
        box_info.append({
            'label' : charid, # 1-9, 10, 11-36
            'left'  : w_begin,
            'top'   : 1,
            'height': height,
            'width' : width
        })
        w_begin += width + 3
    w_begin += 5
    full_region = np_paste_background[:, 0:w_begin, :]
    return full_region, box_info

def gen_1line_img():
    sequence = gen_flat_charid_sequence()
    return gen_img_from_seq(sequence)

def gen_2line_img():
    seq_top = gen_top_charid_sequence()
    seq_bot = gen_bottom_charid_sequence()

    img_top, box_info_top = gen_img_from_seq(seq_top)
    img_bot, box_info_bot = gen_img_from_seq(seq_bot)

    H_top, W_top = img_top.shape[0], img_top.shape[1]
    H_bot, W_bot = img_bot.shape[0], img_bot.shape[1]

    H_new = H_top + H_bot
    W_new = max(W_top, W_bot, int(fontsize*0.6)*4)

    img_new = np.zeros((H_new, W_new, 3), img_top.dtype)
    off_top = (W_new - W_top)/2
    off_bot = (W_new - W_bot)/2
    img_new[0    :H_top      , off_top:off_top+W_top, :] = img_top
    s = 7  # upward shift of bottom image
    S_top = H_top - s
    region = img_new[S_top:S_top+H_bot, off_bot:off_bot+W_bot, :]
    region[:] = np.maximum(region, img_bot)
    for box in box_info_top:
        box['left'] += off_top
    for box in box_info_bot:
        box['left'] += off_bot
        box['top'] += S_top
    return  img_new, box_info_top + box_info_bot

def add_plate_border(img, box_info):
    """ add horizontal padding to simulate plates with big space """
    if random.randint(0,1):
        padding = random.randint(3,20)
        shape = list(img.shape)
        shape[1] += 2*padding
        img_new = np.zeros(shape, img.dtype)
        img_new[:, padding:padding+img.shape[1]] = img
        for box in box_info:
            box['left'] += padding
        img = img_new
    """ add bright border to image """
    H,W = img.shape[:2]
    thickness = random.randint(1,3)
    cv2.rectangle(img, (0,0), (W-thickness,H-thickness), (255,255,255), thickness)
    return img, box_info


def gen_sgseq_img(return_box_info=False):
    """
    Generate a black-white SG license plate-looking image without anything else
    Returns the np array image, along with box_info[{left, top, height, width, label}]
    """
    if random.randint(0,1):
        full_region, box_info = gen_1line_img()
    else:
        full_region, box_info = gen_2line_img()
    if random.randint(0,1):
        full_region, box_info = add_plate_border(full_region, box_info)
    if return_box_info:
        return full_region, box_info
    else:
        return full_region


def tonp(arr):
    return np.asarray(arr, np.uint8)

plate_colors = [
    [tonp([0, 0, 0]), tonp([255,255,255])], # black -white
    [tonp([66, 179, 244]), tonp([0,0,0])],  # yellow-black
]

def colorize(img):
    """
    colorize a black-white image by setting background and foreground color
    """
    assert img.dtype == np.uint8
    mask_fore = np.asarray(img, np.float32) / 255
    mask_back = 1 - mask_fore
    # randomly pick a background-foreground color
    colors = plate_colors[random.randint(0,1)]
    return np.asarray(mask_fore * colors[1] + mask_back * colors[0], np.uint8)

def get_background_img(W, H):
    """
    Returns a WxH background images loaded from patches located in a directory
    """
    num_patches = 179917
    patch_idx = random.randint(1, num_patches)
    img_back  = cv2.imread('/home/noid/data/car_backgrounds/patches/{}.jpg'.format(patch_idx))
    # crop and return
    assert W <= img_back.shape[1], '{}<{}'.format(img_back.shape[1], W)
    assert H <= img_back.shape[0], '{}<{}'.format(img_back.shape[0], W)
    crop_x = random.randint(0, img_back.shape[1] - W)
    crop_y = random.randint(0, img_back.shape[0] - H)
    return img_back[crop_y:crop_y+H, crop_x:crop_x+W, :]

def corrupt_blend(background_img, foreground_img, paste_x, paste_y):
    paste_W = foreground_img.shape[1]
    paste_H = foreground_img.shape[0]
    blend_ratio = random.random() * 0.4 + 0.1 # [0.1, 0.5]
    blended_source = background_img[paste_y:paste_y + paste_H, paste_x:paste_x + paste_W, :].copy()
    background_img[paste_y:paste_y + paste_H, paste_x:paste_x + paste_W, :] = (
            blend_ratio * blended_source + (1-blend_ratio) * foreground_img)

def gen_sgseq_full_img(return_box_info=False):
    """
    Generate a full 256x128 image with sgseq_img randomly placed on background_img
    """
    while True:
        sgseq_img, box_info = gen_sgseq_img(return_box_info=True)
        sgseq_img = colorize(sgseq_img)
        seq_W = sgseq_img.shape[1]
        seq_H = sgseq_img.shape[0]
        target_W = 256
        target_H = 128
        assert seq_W <= target_W
        assert seq_H <= target_H
        paste_x = random.randint(0, target_W - seq_W)
        paste_y = random.randint(0, target_H - seq_H)
        # paste 
        background_img = get_background_img(target_W, target_H)
        corrupt_blend(background_img, sgseq_img, paste_x, paste_y)
        if return_box_info:
            for box in box_info:
                box['top']  += paste_y
                box['left'] += paste_x
            return background_img, box_info
        else:
            return background_img

def gen_batch(N):
    """
    generate N samples using gen_sgseq_full_img
    """
    list_box_info = []
    # save the images first
    for i in xrange(N):
        img, box_info = gen_sgseq_full_img(return_box_info=True)
        cv2.imwrite('output/{}.jpg'.format(i+1), img)
        list_box_info.append(box_info)
        if i % 100 == 0:
            print(i)
    # convert list_box_info to 2 np arrays and save
    X_refs = np.zeros((N, 8), np.uint32)
    X_info = np.zeros((N* 8, 5), np.float32)
    char_index = 0
    for i, box_info in enumerate(list_box_info):
        for j, box in enumerate(box_info):
            char_index += 1
            X_info[char_index-1][0] = box['top']
            X_info[char_index-1][1] = box['left']
            X_info[char_index-1][2] = box['height']
            X_info[char_index-1][3] = box['width']
            X_info[char_index-1][4] = box['label']
            X_refs[i][j] = char_index
    np.save('X_refs.np', X_refs)
    np.save('X_info.np', X_info)

def test():
    show_img(gen_sgseq_full_img())

def get_clean_plate():
    img, bboxes = gen_sgseq_img(True)
    list_char_coord = []
    for box in bboxes:
        x1 = box['left']
        y1 = box['top'] + 2
        x2 = x1 + box['width']
        y2 = y1 + box['height'] - 7
        if box['label'] not in [1]:
            x1 += 2
            x2 -= 2
        list_char_coord.append({
            'class': box['label'],
            'corners':[[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
        })
    
    return img, list_char_coord

def test_font(fontidx):
    return gen_img_from_seq(map(char_to_charid, '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'), fontidx=fontidx)

if __name__ == "__main__":
    test()
    
