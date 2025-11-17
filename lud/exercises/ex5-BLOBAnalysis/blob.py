from skimage import io, color, morphology
from skimage.util import img_as_float, img_as_ubyte
import matplotlib.pyplot as plt
import numpy as np
import math
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage import segmentation
from skimage import measure
from skimage.color import label2rgb
import os
from skimage.morphology import erosion, dilation, opening, closing
from skimage.morphology import disk 

def show_comparison(original, modified, modified_name):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,
                                   sharey=True)
    ax1.imshow(original)
    ax1.set_title('Original')
    ax1.axis('off')
    ax2.imshow(modified)
    ax2.set_title(modified_name)
    ax2.axis('off')
    io.show()

in_dir = 'data'

# Lego images
lego = io.imread(os.path.join(in_dir, 'lego_4_small.png'))
gray_lego = rgb2gray(lego)
T = threshold_otsu(gray_lego)
bin_lego = (gray_lego < T) # Foreground = blobs

def ex1():
    show_comparison(lego, bin_lego, 'Binary lego 4')
# ex1()

def ex2(show, bin_img):
    orig_bin = bin_img
    bin_lego = segmentation.clear_border(bin_img)
    if show:
        show_comparison(orig_bin, bin_lego, 'No border')
        # When referring to 'clear border', we mean remove object at image border,
        # not edges, FYI.
    return bin_lego
# bin_lego = ex2(False, bin_lego)

# Ex 3 -> why does it say exactly which operations I should perform to solve this?
def ex3(orig_img, show=False):
    footprint = disk(5)
    bin_img = closing(orig_img, footprint)
    bin_img = opening(bin_img, footprint)
    if show:
        show_comparison(orig_img, bin_img, 'Morphed binary image')
    # This exercise is just a plugNplay -> reader doesn't need to know what is occurring
    return bin_img
# bin_lego = ex3(bin_lego)

def ex4(img_open):
    label_img = measure.label(img_open)
    n_labels = label_img.max()
    print(f"Number of labels: {n_labels}")
        # Again, solution given to reader
    return label_img
# label_img = ex4(bin_lego)

def ex5(img):
    rgb_proc = label2rgb(img)
    show_comparison(img, rgb_proc, 'RGB label image')
# ex5(label_img)

def ex6(label_img):
    region_props = measure.regionprops(label_img)
    areas = np.array([prop.area for prop in region_props])
    plt.hist(areas, bins=50)
    plt.show()
        # Measure blob features - in this case, we visualize the area of BLOBs
        # Why can the student not do this themselves?
# ex6(label_img)

img_org = io.imread(in_dir + '/Sample E2 - U2OS DAPI channel.tiff')
# slice to extract smaller image
img_small = img_org[700:1200, 900:1400]
img_gray = img_as_ubyte(img_small) 

def show_med():
    minv,maxv = 0,70
    io.imshow(img_gray, vmin=minv, vmax=maxv)
    plt.title('DAPI Stained U2OS cell nuclei')
    io.show()

    plt.hist(img_gray.ravel(), bins=256, range=(minv,maxv))
    io.show()
# show_med()

T = 10 # From histogram inspection
# bin_img = (img_gray > T)
def ex8(img_gray,bin_img):
    # bin_img = erosion(bin_img, disk(2))
    show_comparison(img_gray, bin_img, f'Binary image with threshold {T}')
# ex8(img_gray,bin_img)

def ex9(bin_img, show=False):
    bin_n_b = segmentation.clear_border(bin_img)
    label_img = measure.label(bin_n_b)
    print(f'Number of labels: {label_img.max()}')

    image_label_overlay = label2rgb(label_img)
    if show:
        show_comparison(bin_img , image_label_overlay, 'Found BLOBS')
    return label_img
# label_img = ex9(bin_img)

def ex10(label_img, show=False):
    region_props = measure.regionprops(label_img)
    areas = np.array([prop.area for prop in region_props])
    if show:
        plt.hist(areas.ravel(), bins=256)
        io.show()
            # Some noise is definitely present, which is also evident visually with some
            # BLOBs having merged.
    return region_props
# region_props = ex10(label_img)

def ex11(label_img, region_props, show=False):
    # Student should do this themselves.
    min_area = 0
    max_area = 200

    # Create a copy of the label_img
    label_img_filter = label_img
    for region in region_props:
        # Find the areas that do not fit our criteria
        if region.area > max_area or region.area < min_area:
            # set the pixels in the invalid areas to background
            for cords in region.coords:
                label_img_filter[cords[0], cords[1]] = 0
    # Create binary image from the filtered label image
    i_area = label_img_filter > 0
    if show:
        show_comparison(img_small, i_area, 'Found nuclei based on area')
# ex11(label_img, region_props, show=True)

# BLOB features
# perimeters = np.array([prop.perimeter for prop in region_props])
# areas = np.array([prop.area for prop in region_props])
# circularity = np.array([(4 * math.pi * area) / math.pow(peri,2) for area, peri in zip(areas, perimeters)])
# T_circ = 0.8 # Found from hist

def ex12(region_props):
    plt.scatter(areas, perimeters)
    plt.show()
# ex12(region_props)

def ex13(show=False,show2=False):
    if show2:
        plt.hist(circularity,bins=256)
        plt.show()

    # Create a copy of the label_img
    label_img_filter = label_img
    for region in region_props:
        prop_circ = (4*math.pi*region.area) / math.pow(region.perimeter,2)
        # Find the areas that do not fit our criteria
        if  prop_circ > 1 or prop_circ< 0.8:
            # set the pixels in the invalid areas to background
            for cords in region.coords:
                label_img_filter[cords[0], cords[1]] = 0
    # Create binary image from the filtered label image
    i_area = label_img_filter > 0
    if show:
        show_comparison(img_small, i_area, 'Found nuclei based on area')
# ex13(show=True)


def ex14(areas, circularity):
    plt.scatter(areas, circularity)
    plt.show()
        # More defined pattern emerges, as it is shown, that nuclei with small areas are
        # also circular, which is the range we're looking at. What is interesting is,
        # that the circularity measure identifies one nuclei more, that is discarded
        # when using the chosen threshold, derived from the area histogram.
# ex14()

def ex15(in_path):
    in_img = io.imread(os.path.join(in_dir, in_path))
    gray_img = rgb2gray(in_img[200:700, 0:500])
    T = threshold_otsu(gray_img)
    bin_img = (gray_img > T) # Foreground = blobs
    show_comparison(gray_img, bin_img, 'New binary image')

    label_img = ex9(bin_img, show=True)
    region_props = ex10(label_img)

    # Get blob features
    perimeters = np.array([prop.perimeter for prop in region_props])
    areas = np.array([prop.area for prop in region_props])
    circularity = np.array([(4 * math.pi * area) / math.pow(peri,2) for area, peri in zip(areas, perimeters)])

    # Show areas, circ hist:
    ex14(areas, circularity)

    # Perform thresholding based on this info:
    Amin, Amax = 35, 200
    Cmin, Cmax = 0.6, 1.2

    # Filter based on thresholding
    label_img_filter = label_img
    for region in region_props:
        prop_circ = (4*math.pi*region.area) / math.pow(region.perimeter,2)
        # Find the areas that do not fit our criteria
        if  prop_circ > Cmax or prop_circ< Cmin or region.area > Amax or region.area < Amin:
            # set the pixels in the invalid areas to background
            for cords in region.coords:
                label_img_filter[cords[0], cords[1]] = 0
    # Create binary image from the filtered label image
    new_img = label_img_filter > 0
    show_comparison(in_img, new_img, 'Found nuclei based on area and circularity')


in_path = 'Sample G1 - COS7 cells DAPI channel.png'
ex15(in_path)

# Why is exercise 17 treated as if we don't know this???
