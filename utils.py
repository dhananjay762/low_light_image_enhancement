import numpy as np
import cv2 as cv
from math import log10, sqrt
from skimage.metrics import structural_similarity as ssim
from skimage import io, img_as_float
import imquality.brisque as brisque


def load_img(file, flag):
    if flag==1:
        im = cv.imread(file)
        return im
    else:
        im = cv.imread(file)
        im_hls = cv.cvtColor(im, cv.COLOR_BGR2HLS)
        im = cv.split(im_hls)[1]
        return np.array(im, dtype='float32')/255.0


def save_img(filepath, result_1, result_2=None):
    result_1 = np.squeeze(result_1)
    result_2 = np.squeeze(result_2)
    if not result_2.any():
        cat_image = result_1
    else:
        cat_image = np.concatenate([result_1, result_2], axis=1)
        cat_image = cat_image*255
    cv.imwrite(filepath, cat_image)


def data_aug(image, mode):
    if mode==0:
        return image                  #return original
    elif mode==1:
        return np.flipud(image)       #flip up and down
    elif mode==2:
        return np.rot90(image)        #rotate counterwise 90-degree
    elif mode==3:
        image = np.rot90(image)
        return np.flipud(image)       #rotate 90-degree and flip up&down
    elif mode==4:
        return np.rot90(image, k=2)   #rotate 180-degree
    elif mode==5:
        image = np.rot90(image, k=2)
        return np.flipud(image)       #rotate 180-degree and flip up&down
    elif mode==6:
        return np.rot90(image, k=3)   #rotate 270-degree
    elif mode==7:
        image = np.rot90(image, k=3)
        return np.flipud(image)       #rotate 270-degree and flip up&down


def get_attention_map(image_path):
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    illum_channel = 1-(img/255.0)
    return illum_channel

def psnr_score(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    if mse==0:
        return 100
    max_pixel = 255.0
    psnr = np.round(20*log10(max_pixel/sqrt(mse)), 3)
    return psnr


def ssim_score(image1, image2):
    image1_gray = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
    image2_gray = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
    (score, diff) = ssim(image1_gray, image2_gray, full=True)
    diff = (diff*255).astype('uint8')
    return np.round(score, 4)

def brisque_score(image_path):
    img = img_as_float(io.imread(image_path, as_gray=True))
    score = brisque.score(img)
    return score
