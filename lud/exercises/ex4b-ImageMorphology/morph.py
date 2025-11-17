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
from skimage.morphology import erosion, dilation, opening, closing
from skimage.morphology import disk 

in_dir = 'data'

# Lego images
lego = io.imread(os.path.join(in_dir, 'lego_5.png'))
gray_lego = rgb2gray(lego)
T = threshold_otsu(gray_lego)
bin_lego = (gray_lego > T)

# New lego image
lego_2 = io.imread(os.path.join(in_dir, 'lego_7.png'))
gray_lego_2 = rgb2gray(lego_2)
T = threshold_otsu(gray_lego_2)
bin_lego_2 = (gray_lego_2 > T)

# Tough lego
tough_lego = io.imread(os.path.join(in_dir, 'lego_9.png'))
gray_tough = rgb2gray(tough_lego)
T = threshold_otsu(gray_tough)
bin_tough = (gray_tough > T)

def show_ref_and_res(ref_img, res_img, bin_img, title = [], rows=1, disk_size=[]):
    fig, ax = plt.subplots(rows, 3, figsize=(8,4))
    if rows > 1:
        for i, row in enumerate(range(rows)):
            ax[row,0].imshow(ref_img)
            ax[row,0].set_title('Reference image')
            ax[row,1].imshow(res_img[i], cmap=plt.cm.gray)
            if title:
                ax[row,1].set_title(title[i])
                    # Image becomes less sharpened -> more blurry. Transition areas become expanded,
                    # highlighting the areas "difference" color
            ax[row,2].imshow(bin_img[i])
            if disk_size:
                ax[row,2].set_title(f'Binary image, eroded/diluted w/ size = {disk_size[i]}')
            else:
                ax[row,2].set_title('Binary image')
    else:
        ax[0].imshow(ref_img)
        ax[0].set_title('Reference image')
        ax[1].imshow(res_img, cmap=plt.cm.gray)
        if title:
            ax[1].set_title(title)
                # Image becomes less sharpened -> more blurry. Transition areas become expanded,
                # highlighting the areas "difference" color
        ax[2].imshow(bin_img)
        ax[2].set_title('Binary image')
    plt.show()

def compute_outline(bin_img):
    """
    Computes the outline of a binary image
    """
    footprint = disk(1)
    dilated = dilation(bin_img, footprint)
    outline = np.logical_xor(dilated, bin_img)
    return outline

def ex1():
    show_ref_and_res(lego, [gray_lego], [bin_lego], ['Gray lego'])
# ex1()

def ex2(bin_img, morph):
    imgs = []
    sizes = [2,5,10]
    for size in [2,5,10]:
        footprint = disk(size)
        eroded = morph(bin_img, footprint)
        imgs.append(eroded)

    show_ref_and_res(lego, [gray_lego,gray_lego,gray_lego], imgs, ['Gray lego','Gray lego','Gray lego'], rows=3, disk_size=sizes)

# ex2(bin_lego, lambda x,y: erosion(x,y)) # A few blobs will be present outside of the lego brick @ disk=10, however, all inide is eroded at this time

# EX 3:
# ex2(bin_lego, lambda x,y: dilation(x,y)) # At 2, dilution is still okay. Afterwards, not so much.

# EX 4:
# ex2(bin_lego, lambda x,y: opening(x,y)) # Opening, meaning erode -> dilute, perfectly segments @ 10. Before, artifacts still occur
# EX 5:
# ex2(bin_lego, lambda x,y: closing(x,y)) # Closing, dilute -> erode, makes blobs that are quite large when size = 10

def ex6(ref, gray, bin_img):
    outline = compute_outline(bin_img)
    show_ref_and_res(ref, gray, outline, 'Gray lego')
    # Using a naive outline, the boundary is quite will segmented.

# ex6(lego, gray_lego,bin_lego)

def ex7(ref, gray, bin_img, open_val, closing_val, eros=None, dila=None):
    open = opening(bin_img, disk(open_val)) if open_val > 0 else bin_img
    close = opening(open, disk(closing_val)) if closing_val > 0 else open
    if dila:
        close = dilation(close, disk(dila))
    if eros:
        close = erosion(close, disk(eros))
    outline = compute_outline(close)

    show_ref_and_res(open, close, outline)
        # Employing this compound operation, the lego is mostly perfectly segmented,
        # leading to a mostly perfect boundary. Problem: three blobs below real object.
        # Why does this work so well? Opening is used to REMOVE small objects, which it
        # is mostly successful to do. Closing is intended to fill holes without removing
        # objects, which removes the blobs in the middle, as everything is joined into a
        # single object.

# ex7(lego, gray_lego, bin_lego, open_val=1, closing_val=15)

def ex8(ref, gray, bin_img):
    outline = compute_outline(bin_img)

    show_ref_and_res(ref, bin_img, outline)
        # Image is easy to threshold, as there's no noise in the background.
        # Otsu thresholding hence performs well. Computing the outline, there's still
        # some noise that needs to be handled.

# ex8(lego_2, gray_lego_2, bin_lego_2)

# EX 9
# ex7(lego_2,gray_lego_2, bin_lego_2, open_val=1, closing_val=15)
    # Using the exact same values and in Ex7, the legos are able to be perfectly segmented.

def ex10():    
    lego_2 = io.imread(os.path.join(in_dir, 'lego_3.png'))
    gray_lego_2 = rgb2gray(lego_2)
    T = threshold_otsu(gray_lego_2)
    bin_lego_2 = (gray_lego_2 > T)
    ex7(lego_2,gray_lego_2, bin_lego_2, open_val=1, closing_val=15)
# ex10()
    # Same applies here - only a slight artifact in the lower right lego brick.


def ex11(ref, gray, bin_img):
    outline = compute_outline(bin_img)

    show_ref_and_res(ref, bin_img, outline)
# ex11(tough_lego, gray_tough, bin_tough)
    # No outline is produced over the two areas where bricks are touching.
    # Objects are quite well segmented, however.

# EX12
# ex7(tough_lego,gray_tough, bin_tough, open_val=0, closing_val=15)
    # We can use the same value again (15), which does the trick.

# EX13
# ex7(tough_lego,gray_tough, bin_tough, open_val=0, closing_val=15, eros=0, dila=55) # EROS AND DILA ARE OPPOSITE! (same with everyting else ...)

# EX14
# ex7(tough_lego,gray_tough, bin_tough, open_val=0, closing_val=15, eros=10, dila=55)

# Ex 15, 16
def ex15(size):    
    puzzle = io.imread(os.path.join(in_dir, 'puzzle_pieces.png'))
    gray_puzzle = rgb2gray(puzzle)
    T = threshold_otsu(gray_puzzle)
    bin_puzzle = (gray_puzzle > T)
    closed = opening(bin_puzzle, disk(size))
    show_ref_and_res(puzzle, bin_puzzle, closed)
ex15(size=25)

