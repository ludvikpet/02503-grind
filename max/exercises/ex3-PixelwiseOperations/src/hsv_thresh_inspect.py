from skimage import color, io
import matplotlib.pyplot as plt  

im_org = io.imread("data/DTUSigns2.jpg")

hsv_img = color.rgb2hsv(im_org)
hue_img = hsv_img[:, :, 0]
value_img = hsv_img[:, :, 2]
fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(8, 2))
ax0.imshow(im_org)
ax0.set_title("RGB image")
ax0.axis('off')
ax1.imshow(hue_img, cmap='hsv')
ax1.set_title("Hue channel")
ax1.axis('off')
ax2.imshow(value_img)
ax2.set_title("Value channel")
ax2.axis('off')

fig.tight_layout()
#fig.savefig("im_for_thresh.png")
plt.show()

# plt.imshow(hue_img,cmap="gray")
# plt.savefig("hue_thresh.png")

# plt.imshow(value_img,cmap="gray")
# plt.savefig("value_thresh.png")

###this can only be solved using inspection with python, we need the intensities of the images 
