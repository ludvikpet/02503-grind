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

def compute_edges(img):

    pv = prewitt_v(img)
    pv = 
    ph = prewitt_h(img)
    p = prewitt(img)

    coords = np.argwhere(p > 0)
    # print(pv)
    # print(p)
    # print(p.max())
    # print(coords)
    # print(coords.shape)
    y0,x0 = coords.min(axis=0)
    y1,x1 = coords.max(axis=0)
    print(f"First pos = ({x0},{y0}) | Second pos = ({x1},{y1})")
    # plt.imshow(p, cmap="gray")
    # plt.show()

    _, ax = plt.subplots(1,2, figsize=(12,12))
    # ax[0,0].imshow(img, cmap="gray")
    # ax[0,1].imshow(p, cmap="gray")
    ax[0].imshow(pv, cmap="gray")
    ax[1].imshow(ph, cmap="gray")
    plt.show()
    return p


def capture(vid_path):

    cap = cv2.VideoCapture(vid_path)

    xmin, ymin = 1738, 823
    xmax, ymax = 2116, 1326
    T = 235
    run = True
    warmup_frames = 100
    for i in range(warmup_frames):
        ret, frame = cap.read()

    while run:
        ret, frame = cap.read()

        # if frame is not read -> return
        if not ret:
            print("Can't receive frame")
            return

        cropped_frame = frame[ymin:ymax, xmin:xmax]

        gray_img = img_as_ubyte(rgb2gray(cropped_frame))
        bin_img = gray_img > T
        p = compute_edges(bin_img)





if __name__ == "__main__":
    vid_path = "larger_height_far_away_focus.mp4" 
    capture(vid_path)




