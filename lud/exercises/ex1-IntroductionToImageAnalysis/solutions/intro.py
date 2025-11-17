import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pydicom as dicom
import numpy as np
from skimage import color, io, measure
from skimage.util import img_as_ubyte
from skimage.measure import profile_line
from skimage.transform import rescale, resize

# Directory containing data and images
in_dir = "data/"

# X-ray image
im_name = "metacarpals.png"

# Read the image.
# Here the directory and the image name is concatenated
# by "+" to give the full path to the image.
im_org = io.imread(in_dir + im_name)

print(f"Image shape: { im_org.shape }, image pixel type: {im_org.dtype}")

# Display image
io.imshow(im_org)
plt.title('Metacarpal image')
plt.savefig('solutions/metacarpal.png')

# Display image with colormap
io.imshow(im_org, cmap="jet") # Examples of other color maps: cool, hot, pink, copper
plt.title('Metacarpal image')
plt.savefig('solutions/metacarpal_colormap.png')
# Second try with cool cmap
io.imshow(im_org, cmap="cool")
plt.title('Metacarpal image (cool)')
plt.savefig('solutions/metacarpal_cool.png')

# Grayscale image
io.imshow(im_org, vmin=20, vmax=170) # Here, we enhance contrast by setting all pixels to 0 with p_x<=20, and those to 255 that have p_x>=170
plt.title('Metacarpal image (with gray level scaling)')
plt.savefig('solutions/metacarpal_scale.png')

############################################
#                HISTOGRAMS                #
############################################

plt.hist(im_org.ravel(), bins=256)
plt.title('Image histogram')
io.show()

# How plt.hist works:
h = plt.hist(im_org.ravel(), bins = 256)
# h is a list of tuples, where the first element h[0][i] for each tuple is the COUNT within bin i and h[1][i] is the BIN_EDGE

# Bin edges for bin i can then be found as follows:
i = 40
i_count = h[0][i]
left_edge = h[1][i]
right_edge = h[1][i+1]
print(f"Left edge: {left_edge}, right_edge: {right_edge}")

# Alternative way of calling hist:
count, x, _ = plt.hist(im_org.ravel(), bins=256)

# TODO 9: find most common range of intensities
common_bin = np.argmax(h[0][:])
print(f"Most common bin: {common_bin}, with value: {h[0][common_bin]}")


############################################
# PIXEL VALUES                             #
############################################

#TODO 10: pixel value at (r,c)=(110, 90)
p = im_org[110,90]

#TODO 11: what does this operation do? (on slicing)
im_org[:30] = 0
io.imshow(im_org)
io.show()
    # Slices the image such, that all rows up to row 30 retrieve value == 0

# TODO 12: where are the values 1 and where are they 0 for the following masking?:
mask = im_org > 150
io.imshow(mask)
io.show()
    # Only pixels with values > 150 have 1, else 0

# TODO 13: What does this code do?:
im_org[mask] = 255
io.imshow(im_org)
io.show()
    # Resets the mask to uint format, having range (0,255), where pixels with value 255 previously had value 1

############################################
# COLOR IMAGES                             #
############################################

# TODO 14: read and print image dims, pixel type of image: ardeche.jpg:
rgb_img = io.imread(in_dir + 'ardeche.jpg')
print(f'RGB image shape: {rgb_img.shape}, dtype: {rgb_img.dtype}')

# TODO 15: RGB values at (110,90)?:
print(f'Values at position (110,90): {rgb_img[110, 90, :]}')

# TODO 16: color upper half of photo green:
rgb_img[:int( rgb_img.shape[0]/2 ), :, 0] = 0
rgb_img[:int( rgb_img.shape[0]/2 ), :, 1] = 255
rgb_img[:int( rgb_img.shape[0]/2 ), :, 2] = 0
io.show()
