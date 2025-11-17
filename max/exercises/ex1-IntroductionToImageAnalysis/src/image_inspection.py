from skimage import io 
import matplotlib.pyplot as plt 
in_dir = "data/"
im_org = io.imread(in_dir+"DTUSign1.jpg")
r_comp = im_org[:, :, 0]
plt.imshow(r_comp)
plt.title('DTU sign image (Red)')
plt.show()


in_dir = "data/"
im_org = io.imread(in_dir+"metacarpals.png")
plt.imshow(im_org)
plt.show()
