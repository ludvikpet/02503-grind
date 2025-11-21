import os
import matplotlib.pyplot as plt
import numpy as np
import sys
from skimage import io
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
from loguru import logger
import cv2

logger.remove()
logger.add(sys.stderr, format="<level>{level}</level> | <level>{message}</level>")

def compute_integral_image(integral_img, y0, x0, w, h):
    """
        Computes the integral image within the bounding box represented by (y0,x0,w,h).

        :param numpy.ndarray integral_img:      The integral image input. Dimensions need to supercede those of the bounding box.
        :param int y0:                          Initial starting y position of bounding box.
        :param int x0:                          Initial starting x position of bounding box.
        :param int w:                           Width of bounding box.
        :param int h:                           Height of bounding box.
    """
    
    # Retrieve corners of bounding box
    pos1, pos2, pos3, pos4 = (y0-1, x0-1), (y0-1, x0-1+w-1),(y0-1+h-1, x0-1), (y0-1+h-1, x0-1+w-1)

    # Compute integral image
    A = integral_img[pos1[0], pos1[1]]
    B = integral_img[pos2[0], pos2[1]]
    C = integral_img[pos3[0], pos3[1]]
    D = integral_img[pos4[0], pos4[1]]
    return D + A - B - C

def compute_diagonal_feat(integral_img, start_positions, w_reg, h_reg):
    """
        Computes the diagonal Haar feature of a 
    """

    # Compute integral image for each square region
    regions = []
    for y,x in start_positions:
        reg_integral = compute_integral_image(integral_img, int(y), int(x), w_reg, h_reg)
        regions.append(reg_integral)
   
    # Compute Haar feature
    white_pixels = regions[1] + regions[2]
    black_pixels = regions[0] + regions[3]
    haar_feat =  white_pixels - black_pixels

    return haar_feat

if __name__ == '__main__':

    # Load both the stage image and the reference image
    img_path = sys.argv[1]
    stage_img = io.imread(img_path)
    ref_path = sys.argv[2]
    ref_img = io.imread(ref_path)
    ref_img = img_as_ubyte(rgb2gray(ref_img)) # Want the image to be grayscaled

    # Look at the stage image shape
    logger.info(stage_img.shape)
        # Image has been upscaled by 10

    # Rescale image down to w_window x h_window
    w_enlargened, h_enlargened = (stage_img.shape[0],stage_img.shape[0])
    ROI_feat = 2
    ROI_img = stage_img[:, ROI_feat*w_enlargened:(ROI_feat+1)*w_enlargened]
    resized_img = cv2.resize(ROI_img, (w_enlargened//10,h_enlargened//10), interpolation = cv2.INTER_AREA)
    io.imsave('test_data/resized_img.jpg', resized_img)

    _, axes = plt.subplots(ncols=3, figsize=(8,8))

    axes[0].imshow(ROI_img, cmap='gray')
    axes[0].set_title('Haar image pre rescaling')
    axes[1].imshow(resized_img, cmap='gray')
    axes[1].set_title('Haar image post rescaling')
    axes[2].imshow(ref_img, cmap='gray')
    axes[2].set_title('Reference image')
    plt.show()
        # Contrast has been removed from original annotation image. Just a note

    # EX 2: Identify ROI pixels
    plt.hist(resized_img.ravel(), bins=256)
    plt.show()
        # Clearly bucketizes the ROI into separate 0,255 bins
    
    # Retrieve all pixels with boundary values
    ys, xs = np.where((resized_img == 0) | (resized_img == 255))
    coords = np.vstack((ys, xs)) # ys,xs due to standard convention of packages
    logger.info(coords.shape)

    bounds = (coords.min(axis=1), coords.max(axis=1))
    print(bounds)

    ymin, xmin, ymax, xmax = np.concatenate((coords.min(axis=1), coords.max(axis=1)))
    io.imshow(resized_img[ymin-1:ymax+2,xmin-1:xmax+2])
    plt.show()

    # EX 3: Compute the integral image of the sliced image
    sliced_img = ref_img[ymin-1:ymax+2,xmin-1:xmax+2]
    # sliced_img = resized_img[ymin-1:ymax+2,xmin-1:xmax+2]

    # Integral image options
    integral_img = cv2.integral(sliced_img) # Padds boundary
    integral_img = np.cumsum(np.cumsum(sliced_img, axis=0), axis=1)

    # Visualize the sliced image along with its integral counterpart
    fig, axes = plt.subplots(ncols=2, figsize=(8,8))
    axes[0].imshow(sliced_img, cmap='gray')
    img = axes[1].imshow(integral_img, cmap='viridis')
    for (i, j), val in np.ndenumerate(integral_img):
        axes[1].text(j, i, f"{val}", ha='center', va='center', color='white', fontsize=10)
    fig.colorbar(img, label='Cumulative pixel value', ax=axes[1])
    plt.show()

    # EX4: Compute Haar feature

    # Now we need to separate Haar regions accordingly. In our case, we're working with a
    # four rectangle diagonal feature, for which we'll compute below:

    # Retrieve 
    w, h = integral_img.shape[1], integral_img.shape[0]
    w_reg, h_reg = w//2, h//2

    # As regions are standard uniform, we may retrieve start positions of each region like:
    start_positions = np.array([
        [(y, x) for x in np.linspace(start=1, stop=w_reg, num=2)]
        for y in np.linspace(start=1, stop=h_reg, num=2)]
    ).reshape(-1, 2)
    
    # Having retrieved the starting positions, we now compute the diagonal Haar feature:
    haar_feat = compute_diagonal_feat(integral_img, start_positions, w_reg, h_reg)
    logger.info(f"The computed Haar feature of image slice = {haar_feat}")
