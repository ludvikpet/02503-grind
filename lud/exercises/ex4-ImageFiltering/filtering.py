import os
import pydicom as dicom
import cv2
from skimage.util import img_as_float, img_as_ubyte
from scipy.ndimage import correlate
from skimage import io
from skimage.color import rgb2gray, rgb2hsv
from skimage.filters import threshold_otsu, median, gaussian, prewitt_h, prewitt_v, prewitt
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

#####################
# HELPERS
#####################

# Get input data
in_dir = 'data'

# Read Gaussian image and convert to grayscale
gauss_img = io.imread(os.path.join(in_dir, 'Gaussian.png'))
gray_gauss = img_as_ubyte(rgb2gray(gauss_img))

# Read saltNpepper img
saltNpepper = io.imread(os.path.join(in_dir, "SaltPepper.png"))
gray_saltNpepper = img_as_ubyte(rgb2gray(saltNpepper))

# Car image
car_img = io.imread(os.path.join(in_dir, "car.png"))
gray_car = img_as_ubyte(rgb2gray(car_img))

# Donald image
donald_img = io.imread(os.path.join(in_dir, "donald_1.png"))
gray_donald = img_as_ubyte(rgb2gray(donald_img))

def show_ref_and_res(ref_img, res_img, title = [], rows=1):
    fig, ax = plt.subplots(rows, 2, figsize=(8,4))
    for i, row in enumerate(range(rows)):
        ax[row,0].imshow(ref_img)
        ax[row,0].set_title('Reference image')
        ax[row,1].imshow(res_img[i])
        if title:
            ax[row,1].set_title(title[i])
                # Image becomes less sharpened -> more blurry. Transition areas become expanded,
                # highlighting the areas "difference" color
    plt.show()

#####################
# HELPERS
#####################

# Simple image
input_img = np.arange(25).reshape(5,5)

# Simple filter
weights = [[0,1,0],
           [1,2,1],
           [0,1,0]]
def print_init():
    print(f"input image:\n{ input_img }")
    print(f"Weights:\n{weights}")
print_init()

# Correlate image with the weights
res_img = correlate(input_img, weights)

def ex1():
    print(res_img[3,3])

# ex1() # 108 = weight matrix x 3,3 pixel

# ZERO PADDING IMG BOUNDARIES
def ex2():
    res_img = correlate(input_img, weights, mode="constant", cval=10) # cval sets off-the-edge pixels to have a constant value, e.g. 10
    print(res_img)
    res_img_ref = correlate(input_img, weights, mode='reflect') # Reflect boundary values outside of boundary
    print(res_img_ref)
# ex2()


def ex3(size):
    # Mean filter w/ normalized weights
    weights = np.ones((size,size))
    weights = weights / np.sum(weights) # Normalize weights

    res_img = correlate(gray_gauss, weights, mode='reflect')
   
    show_ref_and_res(gray_gauss, res_img, title='Mean filter')
# ex3(size=5)

def ex4(size:int, ref_img):
    footprint = np.ones((size, size))
    res_img = median(ref_img, footprint)
    show_ref_and_res(ref_img, res_img, title='Median filter')
    # Larger size will favor transition areas in image, where larger values are located,
    # increasing effects from these areas (lower right vs. lower left transition to top).
    # Center becomes more uniform as kernel size increases.

    # Ex 5: When applied to 'SaltPepper.png', noise is removed! If done using the mean
    # filter instead however, noise is smeared across the image instead of removed.
# ex4(5, gray_saltNpepper)

def ex6(sigma: int, ref_img):
    gauss_img = gaussian(ref_img, sigma)
    show_ref_and_res(ref_img, gauss_img, title=[f'Gaussian kernel with sigma = {sigma}'])
    # Transitions are rather stable between areas, however, image becomes more blurry
    # when increasing sigma (makes sense, as we increase s>). Noise is removed.
# ex6(sigma=5, ref_img=gray_gauss)

# Ex 7
def ex7(ref_img):
    for i in [1,5,10,20]:
        gauss_img = gaussian(ref_img, sigma=i)
        matrix = np.ones((i,i))
        median_img = median(ref_img, matrix)
        mean_img = correlate(ref_img, matrix)
        show_ref_and_res(ref_img, [gauss_img, median_img, mean_img], title=[f'Gauss, i={i}',f'Median, i={i}',f'Mean, i={i}'], rows=3)
        # Gauss becomes blurry when increasing i,
        # Median sharpens image but becomes quite blurry when increasing i,
        # Mean adds noise as soon as i > 1
# ex7(gray_car)

def ex8(ref_img):
    fix, ax = plt.subplots(1,4, figsize=(10,4))
    
    pv = prewitt_v(ref_img)
    ph = prewitt_h(ref_img)
    p = prewitt(ref_img)

    ax[0].imshow(ref_img)
    ax[1].imshow(pv)
    ax[1].set_title('Vertical prewitt')
    print(f'Min: {pv.min()}, max: {pv.max()}')
    ax[2].imshow(ph)
    ax[2].set_title('Horizontal prewitt')
    print(f'Min: {ph.min()}, max: {ph.max()}')
    ax[3].imshow(p)
    ax[3].set_title('Prewitt')
    plt.show()
# ex8(gray_donald)

def ex10(x:int):
    CT = img_as_ubyte(rgb2gray(io.imread(os.path.join(in_dir, 'ElbowCTSlice.png'))))

    _, ax = plt.subplots(2,3)
    for i in range(2):
        if i == 0:
            post_CT = gaussian(CT, x)
        else:
            post_CT = median(CT, np.ones((x,x)))


        grads = prewitt(post_CT)
        T = threshold_otsu(grads)
        bin_grads = (grads > T)


        ax[i,0].imshow(CT)
        ax[i,1].imshow(grads)
        ax[i,2].imshow(bin_grads, vmin=bin_grads.min(), vmax=bin_grads.max(), cmap='terrain')
    plt.show()
    # Median filter is very good at identifying the general structure of the elbow,
    # however, if we would like to also identify some of the surrounding tissue, then
    # gauss is better. In general, gauss performs best using smaller sigma values.
    # This is the opposite case for median.
# ex10(3)

